"""Create-only physical-five RAM package, with exact PCM-to-feature parity fixtures."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import subprocess
import tarfile
import numpy as np
from scripts.host.build_soundjie_dual_ui_patch import ROOT,TOOLBIN,lin
from scripts.host.prepare_soundjie_d3 import decode_audio,resample_mono_16k,compute_feature
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.deploy_alert4_candidate import staged_inputs,check_compiled,check_files
from scripts.host.audit_alert4_dpu_layout import audit as audit_output_layout

def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sections(data):
    if data[:6]!=b'\x7fELF\x01\x01':raise ValueError('ELF32 LE required')
    off=struct.unpack_from('<I',data,32)[0];size,n,idx=struct.unpack_from('<HHH',data,46)
    hs=[struct.unpack_from('<10I',data,off+i*size) for i in range(n)]
    st=data[hs[idx][4]:hs[idx][4]+hs[idx][5]]
    return {st[h[0]:st.find(b'\0',h[0])].decode():data[h[4]:h[4]+h[5]] for h in hs}

def preserve_dpu(before,after):
    a,b=sections(before),sections(after);names=[k for k in a if k.startswith('.deephi.')]
    if len(names)!=6 or '.symtab' not in b or any(a[k]!=b.get(k) for k in names):raise ValueError('DPU section loss')
    return {n:sha(a[n]) for n in names}

def pcm_fixture(row,sources,frontend):
    src=sources[(row['recording_id'],row['parent_sha256'])];path=Path(src['path'])
    if sha(path.read_bytes())!=row['parent_sha256']:raise ValueError('source drift')
    samples=resample_mono_16k(decode_audio(path))
    full=np.clip(np.rint(samples*32768),-32768,32767).astype('<i2')
    if sha(full.tobytes())!=row['pcm_sha256_16k']:raise ValueError('decoded PCM drift')
    start=row['start_sample_16k'];a=max(0,start);b=min(len(samples),start+16000)
    if b<=a:raise ValueError('empty source window')
    window=np.zeros(16000);window[a-start:b-start]=samples[a:b]
    pcm=np.clip(np.rint(window*32768),-32768,32767).astype('<i2')
    feature=compute_feature(pcm.astype(np.float64)/32768.,frontend)
    if sha(feature.tobytes())!=row['feature_sha256']:raise ValueError('PCM/frontend differs from exported tensor')
    return pcm.tobytes(),src

def build(root,output,test_log,hop,arm_root=None):
    if output.exists():raise FileExistsError(output)
    if arm_root is None:raise ValueError('new runtime/ui/doa --arm-root required')
    side=root/'compile-20261031a-sidecar-a';compiled=side/'container-output/build';fixed=side/'container-output/fixed';inputs=side/'input'
    policy=read(side/'thresholds-fixed.json');_,meta=staged_inputs(inputs);cm=check_compiled(inputs,compiled)
    fm=read(fixed/'fixed-evaluation.json');check_files(fixed,fm['artifacts'],{'parity-fixed.npy'})
    model=(compiled/'compiled/dpu_soundjie_d3.elf').read_bytes()
    layout=audit_output_layout(model)
    if policy['contract']!='soundjie-alert4bg-v2' or policy['elf_sha256']!=sha(model):raise ValueError('model/contract drift')
    for name,path in [('assessment_sha256',side/'assessment.json'),('compiled_manifest_sha256',compiled/'compiled-manifest.json'),('fixed_manifest_sha256',fixed/'fixed-evaluation.json')]:
        if sha(path.read_bytes())!=policy[name]:raise ValueError('assessment chain drift '+name)
    if meta['output_shape']!=[1,5] or meta['checkpoint_sha256']!=policy['checkpoint_sha256']:raise ValueError('physical output contract differs')
    log=test_log.read_text(encoding='utf-8-sig')
    if not re.search(r'Ran \d+ tests',log) or not log.rstrip().endswith('OK'):raise ValueError('passing regression log required')
    output.mkdir(parents=True,exist_ok=False);payload={};reports={};dpu_sections={}
    for kind,name in [('classifier','soundjie_d4_classifier'),('runtime','soundjie_d4_runtime'),('ui','soundjie_hdmi_ui'),('doa','xvf3800_doa')]:
        folder=root/'arm-final-a'/'classifier-f' if kind=='classifier' else arm_root/kind
        src=folder/name;report=(folder/'build-report.txt').read_text()
        if kind!='classifier' and b'--log-max-bytes' not in src.read_bytes():raise ValueError('bounded logging binary required '+name)
        for token in ('CPU_ARCH=v7','HARD_FLOAT=yes','MAX_GLIBC=2.17','SHA256='+sha(src.read_bytes())):
            if token not in report:raise ValueError('ARM build metadata mismatch '+name)
        if kind=='classifier' and ('PHYSICAL_OUTPUTS=5' not in report or 'STORAGE_OUTPUTS=6' not in report or 'MODEL_SHA256='+sha(model) not in report):raise ValueError('linked classifier mismatch')
        dest=output/name
        subprocess.run(['wsl','-d','Ubuntu','--',TOOLBIN+'arm-linux-gnueabihf-strip',
            '--strip-debug' if kind=='classifier' else '--strip-unneeded','-o',lin(dest),lin(src)],check=True,capture_output=True)
        data=dest.read_bytes();reports[name]=report
        if kind=='classifier':dpu_sections=preserve_dpu(src.read_bytes(),data)
        elf=subprocess.check_output(['wsl','-d','Ubuntu','--',TOOLBIN+'arm-linux-gnueabihf-readelf','-h','-A','-l','--version-info',lin(dest)]).decode()
        if not(re.search(r'Machine:\s+ARM',elf) and 'Tag_CPU_arch: v7' in elf and '/lib/ld-linux-armhf.so.3' in elf and 'Tag_ABI_VFP_args: VFP registers' in elf):raise ValueError('stripped ARM ABI mismatch')
        if max(tuple(map(int,m)) for m in re.findall(r'GLIBC_(\d+)\.(\d+)',elf))>(2,17):raise ValueError('stripped GLIBC mismatch')
        payload['runtime/'+name]=data;payload['evidence/'+name+'-readelf.txt']=elf.encode()
    payload['model/dpu_soundjie_d3.elf']=model
    payload['model/storage-layout.json']=json.dumps(layout,indent=2).encode()
    payload['model/thresholds-fixed.json']=(side/'thresholds-fixed.json').read_bytes()
    payload['config/class-contract.txt']=b'soundjie-alert4bg-v2\n'
    payload['config/log-policy.txt']=b'bounded-jsonl-v1\n'
    payload['config/alert-thresholds.txt']=(','.join(map(str,policy['alert_thresholds']))+'\n').encode()
    payload['config/hop-samples.txt']=(str(hop)+'\n').encode()
    payload['config/classes.json']=(ROOT/'config/soundjie_alert4bg_classes.json').read_bytes()
    payload['assets/hdmi/soundjie_glyph_atlas.bin']=(ROOT/'assets/hdmi/soundjie_alert4_glyph_atlas.bin').read_bytes()
    for name in ('run_soundjie_d4_session.sh','run_soundjie_tv_session.sh','prepare_soundjie_alert4.sh','probe_soundjie_alert4.sh'):
        payload['board/'+name]=(ROOT/'scripts/board'/name).read_bytes().replace(b'\r\n',b'\n')
    sources={}
    for line in (inputs/'source-manifest.jsonl').read_text(encoding='utf-8').splitlines():
        r=json.loads(line);sources[(r['recording_id'],r['sha256'])]=r
    frontend=load_frontend_config(ROOT/'config/logmel_frontend_v1.json');z=np.load(fixed/'parity-fixed.npy',allow_pickle=False)
    if z.shape!=(10,5) or len(meta['parity_sources'])!=10:raise ValueError('parity shape mismatch')
    records=[]
    for i,row in enumerate(meta['parity_sources']):
        if row['purpose']!='validation' or row['derivation'] is not None:raise ValueError('unexpected parity source')
        data,src=pcm_fixture(row,sources,frontend);name='probe/window-%02d.s16'%i;payload[name]=data
        records.append(dict(file=name,sha256=sha(data),source=row,fixed_logits=z[i].tolist(),
            attribution={k:src.get(k) for k in ('title','author','original_url','license','license_url')}))
    payload['probe/reference.json']=json.dumps(dict(scope='development PCM conversion parity, not independent accuracy',records=records),ensure_ascii=False,indent=2).encode()
    report=dict(schema=2,contract=policy['contract'],physical_outputs=5,logical_output_count=5,storage_output_count=6,alert_rows=4,background_index=4,
        linked_model_sha256=sha(model),checkpoint_sha256=policy['checkpoint_sha256'],thresholds=policy['alert_thresholds'],
        processing_hop_samples=hop,window_samples=16000,log_max_bytes_default=1048576,build_reports=reports,preserved_dpu_sections=dpu_sections,
        test_log_sha256=sha(test_log.read_bytes()),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        writes_sd=False,board_tested=False,independent_acceptance=False)
    payload['manifest.json']=json.dumps(report,ensure_ascii=False,indent=2).encode()
    payload['evidence/regression.log']=test_log.read_bytes()
    payload['SHA256SUMS']=''.join(sha(v)+'  '+k+'\n' for k,v in sorted(payload.items())).encode()
    archive=output/'SJA4A.TGZ'
    with tarfile.open(archive,'x:gz') as tar:
        for name,data in sorted(payload.items()):
            item=tarfile.TarInfo('sj-alert4/'+name);item.size=len(data);item.mtime=0
            item.mode=0o755 if name.startswith('runtime/') or name.endswith('.sh') else 0o644;tar.addfile(item,io.BytesIO(data))
    with tarfile.open(archive) as tar:
        assert len(tar.getmembers())==len(payload)
        for name,data in payload.items():assert tar.extractfile('sj-alert4/'+name).read()==data
    report.update(archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,payload_sha256={n:sha(b) for n,b in payload.items()})
    (output/'package-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(archive=str(archive),sha256=report['archive_sha256'],bytes=report['archive_bytes'])))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--test-log',type=Path,required=True);p.add_argument('--hop',type=int,choices=[4000,6400,8000,12000],default=4000)
    p.add_argument('--arm-root',type=Path,required=True,help='new runtime/ui/doa build directories; classifier remains pinned')
    a=p.parse_args();build(a.root,a.output,a.test_log,a.hop,a.arm_root)
