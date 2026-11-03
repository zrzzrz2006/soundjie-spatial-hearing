"""Assemble a create-only RAM patch; never mount/write SD or contact the board."""
import argparse
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import numpy as np
from scripts.host.deploy_soundjie_siren5 import ROOT, CHECKPOINT, POLICY, read, write, sha, sha_bytes, check_files
from scripts.host.build_soundjie_dual_ui_patch import TOOLBIN, lin


def archive(payload,path):
    for name in payload:
        if not name or '\\' in name or ':' in name or name.startswith('/') or '..' in PurePosixPath(name).parts:
            raise ValueError('unsafe archive member')
    with tarfile.open(path,'x:gz') as tar:
        for name,data in sorted(payload.items()):
            item=tarfile.TarInfo('sj-siren5/'+name);item.size=len(data);item.mtime=0
            item.mode=0o755 if name.startswith('runtime/') or name.endswith('.sh') else 0o644
            tar.addfile(item,io.BytesIO(data))
    with tarfile.open(path,'r:gz') as tar:
        if len(tar.getmembers())!=len(payload):raise ValueError('archive member count differs')
        for name,data in payload.items():
            if tar.extractfile('sj-siren5/'+name).read()!=data:raise ValueError('archive roundtrip mismatch')


def check_classifier_contract(report,model_sha):
    fields=dict(line.split('=',1) for line in report.splitlines() if '=' in line)
    if fields.get('PHYSICAL_OUTPUTS')!='6' or fields.get('MODEL_SHA256')!=model_sha:
        raise ValueError('classifier physical count/model mismatch')


def probe_payloads(input_dir,fixed):
    import soundfile as sf
    from scripts.host.logmel_reference import load_frontend_config
    from scripts.host.prepare_soundjie_d3 import compute_feature
    source=read(input_dir/'replay-sources.json');features=np.load(input_dir/'replay-inputs.npy',allow_pickle=False)
    logits=np.load(fixed/'replay-fixed.npy',allow_pickle=False);payload={};records=[]
    frontend=load_frontend_config(ROOT/'config/logmel_frontend_v1.json')
    for e in source['events']:
        if e['purpose']!='validation':continue
        path=Path(e['captured_wav'])
        if sha(path)!=e['captured_sha256']:raise ValueError('captured PCM drift')
        samples,rate=sf.read(path,dtype='int16')
        if rate!=16000 or samples.ndim!=1:raise ValueError('capture format differs')
        for i,w in enumerate(source['windows']):
            if w['event_index']!=e['event_index'] or w['domain']!='acoustic':continue
            cut=samples[w['start_sample']:w['start_sample']+16000]
            if len(cut)!=16000:raise ValueError('probe window incomplete')
            feat=compute_feature(cut.astype(np.float64)/32768.,frontend)
            if not np.allclose(feat,features[i],atol=1e-6,rtol=0):raise ValueError('PCM/features mismatch')
            name='probe/event%02d-window%03d.s16'%(e['event_index'],i);data=cut.astype('<i2').tobytes();payload[name]=data
            records.append(dict(file=name,sha256=sha_bytes(data),window=w,recording_id=e['recording_id'],purpose='validation',
                original_target=e['target_index'],caffe_fixed_logits=logits[i].tolist(),
                attribution={k:e[k] for k in ('title','author','original_url','license','license_url','competition_use')},
                transformation='excerpt, resampled to 16kHz, loudspeaker playback captured with XVF3800, mono PCM window'))
    if len(records)!=28:raise ValueError('28 validation acoustic windows required')
    payload['probe/reference.json']=json.dumps(dict(scope='development playback PCM; compare physical DPU, not independent acceptance',
        records=records),ensure_ascii=False,indent=2).encode('utf-8')
    return payload


def build(root,output,test_log):
    if output.exists():raise FileExistsError(output)
    compiled=root/'deployment-build-b/build';fixed=root/'activation-eval-a/result';input_dir=root/'deployment-input-a'
    cm=read(compiled/'compiled-manifest.json');fm=read(fixed/'fixed-evaluation.json');assessment=read(root/'deployment-assessment.json')
    check_files(input_dir,read(input_dir/'input-manifest.json')['artifacts']);check_files(fixed,fm['artifacts'])
    if cm['checkpoint_sha256']!=CHECKPOINT or cm['policy_sha256']!=POLICY or sha(input_dir/'candidate-policy.json')!=POLICY:
        raise ValueError('candidate mismatch')
    if fm['compiled_manifest_sha256']!=sha(compiled/'compiled-manifest.json') or assessment['fixed_manifest_sha256']!=sha(fixed/'fixed-evaluation.json'):
        raise ValueError('assessment chain mismatch')
    for name,h in cm['artifacts'].items():
        if sha(compiled/name)!=h:raise ValueError('compiled hash drift')
    log=test_log.read_text(encoding='utf-8-sig')
    if not re.search(r'Ran \d+ tests',log) or not log.rstrip().endswith('OK'):raise ValueError('green regression log required')
    model=compiled/'compiled/dpu_soundjie_d3.elf';payload={};reports={}
    for kind,name in [('classifier','soundjie_d4_classifier'),('runtime','soundjie_d4_runtime'),('ui','soundjie_hdmi_ui')]:
        folder=root/'arm-ui-b' if kind=='ui' else root/'arm-a'/kind
        binary=folder/name;report=(folder/'build-report.txt').read_text()
        for required in ('CPU_ARCH=v7','HARD_FLOAT=yes','MAX_GLIBC=2.17','SHA256='+sha(binary)):
            if required not in report:raise ValueError('ARM build report mismatch '+name)
        if kind=='classifier':check_classifier_contract(report,sha(model))
        reports[name]=report;payload['runtime/'+name]=binary.read_bytes()
    payload['model/dpu_soundjie_d3.elf']=model.read_bytes()
    payload['model/candidate-policy.json']=(input_dir/'candidate-policy.json').read_bytes()
    payload['model/compiled-manifest.json']=(compiled/'compiled-manifest.json').read_bytes()
    payload['config/class-contract.txt']=b'soundjie-siren5-v1\n'
    payload['config/classes.json']=(ROOT/'config/soundjie_siren5_classes.json').read_bytes()
    payload['assets/hdmi/soundjie_glyph_atlas.bin']=(ROOT/'assets/hdmi/soundjie_siren5_glyph_atlas.bin').read_bytes()
    for name in ('run_soundjie_d4_session.sh','run_soundjie_tv_session.sh','prepare_soundjie_siren5.sh','probe_soundjie_siren5.sh'):
        payload['board/'+name]=(ROOT/'scripts/board'/name).read_bytes().replace(b'\r\n',b'\n')
    payload.update(probe_payloads(input_dir,fixed))
    output.mkdir(parents=True,exist_ok=False)
    # Preserve the classifier's linked DPU sections/symbols. Strip only CPU/UI executables.
    for name in ('soundjie_d4_runtime','soundjie_hdmi_ui'):
        source=output/(name+'.unstripped');source.write_bytes(payload['runtime/'+name]);dest=output/name
        subprocess.run(['wsl','-d','Ubuntu','--',TOOLBIN+'arm-linux-gnueabihf-strip','--strip-unneeded','-o',lin(dest),lin(source)],check=True,capture_output=True)
        payload['runtime/'+name]=dest.read_bytes()
    for name in ('soundjie_d4_runtime','soundjie_hdmi_ui','soundjie_d4_classifier'):
        binary=output/name
        if not binary.exists():binary.write_bytes(payload['runtime/'+name])
        elf=subprocess.check_output(['wsl','-d','Ubuntu','--',TOOLBIN+'arm-linux-gnueabihf-readelf','-h','-A','-l','--version-info',lin(binary)]).decode('utf-8')
        if not(re.search(r'Machine:\s+ARM',elf) and 'Tag_CPU_arch: v7' in elf and 'Tag_ABI_VFP_args: VFP registers' in elf and '/lib/ld-linux-armhf.so.3' in elf):raise ValueError('ARM ABI mismatch')
        if max(tuple(map(int,m)) for m in re.findall(r'GLIBC_(\d+)\.(\d+)',elf))>(2,17):raise ValueError('GLIBC too new')
        payload['evidence/'+name+'-readelf.txt']=elf.encode()
    sources=['src/'+name for name in ('soundjie_class_contract.h','soundjie_d4_classifier.cpp','soundjie_d4_runtime.cpp','soundjie_d4_state.h','soundjie_d4_state.cpp','soundjie_hdmi_ui.cpp')]
    report=dict(schema=1,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        display_contract='soundjie-siren5-v1',physical_output_count=6,display_output_count=5,
        checkpoint_sha256=CHECKPOINT,model_sha256=sha(model),build_reports=reports,test_log_sha256=sha(test_log),
        assessment_sha256=sha(root/'deployment-assessment.json'),source_sha256={n:sha(ROOT/n) for n in sources},
        board_tested=False,independent_acceptance=False,autostart=False,writes_sd=False)
    payload['manifest.json']=json.dumps(report,indent=2,ensure_ascii=False).encode()
    payload['evidence/regression.log']=test_log.read_bytes();payload['evidence/assessment.json']=(root/'deployment-assessment.json').read_bytes()
    payload['SHA256SUMS']=''.join(sha_bytes(data)+'  '+name+'\n' for name,data in sorted(payload.items())).encode()
    target=output/'SJS5A.TGZ';archive(payload,target)
    (output/'SJS5A.SHA').write_text(sha(target)+'  SJS5A.TGZ\n',encoding='ascii')
    write(output/'package-manifest.json',dict(**report,archive=str(target),archive_sha256=sha(target),bytes=target.stat().st_size,
        payload_sha256={n:sha_bytes(v) for n,v in payload.items()}))
    print(str(target));print('SHA256='+sha(target));print('BYTES='+str(target.stat().st_size))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--test-log',type=Path,required=True)
    a=p.parse_args();build(a.root,a.output,a.test_log)
