"""Assemble a create-only complete application; no SD/boot/board mutation."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]

def sha(data):return hashlib.sha256(data).hexdigest()

def read_archive(path,expected_sha,prefix=''):
    data=path.read_bytes()
    if sha(data)!=expected_sha:raise ValueError('archive hash mismatch: '+str(path))
    files={}
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as t:
        for m in t:
            p=PurePosixPath(m.name)
            if '\\' in m.name or p.is_absolute() or '..' in p.parts or not m.name.startswith(prefix):
                raise ValueError('unsafe archive path: '+m.name)
            if m.isdir():continue
            if not m.isfile() or m.size>64*1024*1024:raise ValueError('unexpected archive member')
            name=m.name[len(prefix):]
            if not name or name!=PurePosixPath(name).as_posix() or name in files:
                raise ValueError('duplicate or noncanonical archive path')
            files[name]=t.extractfile(m).read()
            if sum(map(len,files.values()))>128*1024*1024:raise ValueError('archive too large')
    return files

def write_archive(archive,payload):
    with tarfile.open(archive,'x:gz') as t:
        for n,data in sorted(payload.items()):
            item=tarfile.TarInfo('sj-alert4-release/'+n);item.size=len(data);item.mtime=0
            executable=n.startswith(('runtime/','audio/bin/','dnndk/pkgs/bin/')) or n.endswith('.sh')
            item.mode=0o755 if executable else 0o644
            t.addfile(item,io.BytesIO(data))

def assemble(patch_dir,output):
    if output.exists():raise FileExistsError(output)
    sources={
      'base':(ROOT/'artifacts/hdmi-candidate-20261023b/SJD4A.TGZ','7eb0e88b240247a4322c9fe7f2c358211f04f530cc5e4c62badb2643e58e1812',''),
      'tv':(ROOT/'artifacts/tv-ambulance-20261030a/board-patch-v2/SJTV1.TGZ','0fc802118c7592a3616692dfbe1ef8b5b6ba71a51577fbc4c34f0120db806b0d','sj-tv1/'),
      'calibration':(ROOT/'artifacts/tv-ambulance-20261030a/sd-update-v2/SJTVCA1.TGZ','86833ff38e12329c0deeceb9956f4e78bc16213160816d7bc9ec4c491a875d5b','sj-tv1-calibration-v1/'),
    }
    meta=json.loads((patch_dir/'package-manifest.json').read_text(encoding='utf-8'))
    sources['alert4']=(patch_dir/'SJA4A.TGZ',meta['archive_sha256'],'sj-alert4/')
    archives={k:read_archive(*v) for k,v in sources.items()}
    base,tv,cal,patch=[archives[k] for k in ('base','tv','calibration','alert4')]
    expected=meta['payload_sha256']
    if set(patch)!=set(expected) or any(sha(patch[n])!=h for n,h in expected.items()):
        raise ValueError('patch content differs from manifest')
    if meta['contract']!='soundjie-alert4bg-v2' or meta['processing_hop_samples']!=6400:
        raise ValueError('release requires four-alert contract and 400ms hop')
    payload={n:d for n,d in base.items() if n.startswith(('audio/','dnndk/','licenses/'))
             or n in ('runtime/dpu-4.14.0-xilinx.ko','config/soundjie_d4_runtime.tsv','config/logmel_frontend_v1.json')}
    for n,d in tv.items():
        if n.startswith(('tv/','licenses/')):payload[n]=d
    payload['tv/reference.features']=cal['reference.features']
    payload['tv/traffic.bin']=cal['traffic.bin']
    for n,d in patch.items():
        if n in ('SHA256SUMS','manifest.json') or n.startswith('board/prepare_'):continue
        payload[n]=d
    payload['manifest/model-package.json']=patch['manifest.json']
    pinned={'tv/reference.features':'839537e996650ba41dce80f5229774e8978f45dac2aab6922acfbf383ad324b7',
            'tv/traffic.bin':'d153c13085569058067ccc33d45b6093daaaf4abf887fbd69691b6377ee5bff5',
            'model/dpu_soundjie_d3.elf':'bfaf1f963a6cc9b6a00a4023de36385a858c38efc1541a5dafb286205927e8e6',
            'config/logmel_frontend_v1.json':'bd11b2f97b4b82e34d39a78121df92b9d1c3620de1ab296934b04c1ff995a0b1'}
    for n,h in pinned.items():
        if sha(payload[n])!=h:raise ValueError('release asset changed: '+n)
    required=['audio/bin/arecord','tv/cues.tsv','config/soundjie_d4_runtime.tsv',
              'runtime/dpu-4.14.0-xilinx.ko','runtime/xvf3800_doa','runtime/soundjie_d4_runtime',
              'runtime/soundjie_hdmi_ui','runtime/soundjie_d4_classifier','board/run_soundjie_tv_session.sh']
    for n in required:
        if not payload.get(n):raise ValueError('missing release dependency: '+n)
    report=dict(schema=1,application_only=True,board_tested=False,sd_written=False,
        cold_boot_verified=False,boot_files_modified=False,model_changed=False,
        contract=meta['contract'],hop_samples=6400,log_max_bytes_default=1048576,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        source_archives={k:dict(path=str(v[0]),sha256=v[1]) for k,v in sources.items()},pinned_assets=pinned)
    payload['manifest/release.json']=json.dumps(report,ensure_ascii=False,indent=2).encode()
    payload['SHA256SUMS']=''.join(sha(d)+'  '+n+'\n' for n,d in sorted(payload.items())).encode()
    output.mkdir(parents=True,exist_ok=False)
    archive=output/'SJ4R1.TGZ'
    write_archive(archive,payload)
    digest=sha(archive.read_bytes())
    if read_archive(archive,digest,'sj-alert4-release/')!=payload:raise ValueError('readback mismatch')
    (output/'SJ4R1.SHA').write_bytes((digest+'  SJ4R1.TGZ\n').encode())
    report.update(archive_sha256=digest,archive_bytes=archive.stat().st_size,
                  unpacked_bytes=sum(map(len,payload.values())),payload_sha256={n:sha(d) for n,d in payload.items()})
    (output/'release-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(archive=str(archive),sha256=digest,bytes=report['archive_bytes'],unpacked_bytes=report['unpacked_bytes'])))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patch-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assemble(a.patch_dir,a.output)
