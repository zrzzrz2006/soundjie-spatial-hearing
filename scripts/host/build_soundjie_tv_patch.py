"""Create-only TV patch. Does not inspect/mount/write SD or contact a board."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile

ROOT=Path(__file__).resolve().parents[2]
BASE_MODEL='66a317bc2c21e9d155d38e195466219bba0a7e722a01d480dbab95c0d930e7cd'
CLASSIFIER='2e869dd02e22e71251a76de5936c474d89bdc3990f63d559937831e3003a96dd'
def digest(b):return hashlib.sha256(b).hexdigest()
def arm(path,report):
    b=path.read_bytes();fields=dict(s.split('=',1) for s in report.read_text().splitlines() if '=' in s)
    if len(b)<52 or b[:6]!=b'\x7fELF\x01\x01' or struct.unpack_from('<H',b,18)[0]!=40:raise ValueError('not ARM32 little-endian ELF')
    if fields.get('MACHINE')!='ARM' or fields.get('CPU_ARCH')!='v7' or fields.get('HARD_FLOAT')!='yes' or fields.get('ELF_INTERPRETER')!='/lib/ld-linux-armhf.so.3':raise ValueError('incompatible ABI')
    if tuple(map(int,fields['MAX_GLIBC'].split('.')))>(2,17):raise ValueError('GLIBC too new')
    if digest(b)!=fields['SHA256']:raise ValueError('build hash mismatch')
    return b
def main():
    p=argparse.ArgumentParser()
    for name in ('runtime-dir','ui-dir','diagnostic-archive','reference-dir','feature-file','model','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    files={
        'runtime/soundjie_d4_runtime':arm(a.runtime_dir/'soundjie_d4_runtime',a.runtime_dir/'build-report.txt'),
        'runtime/soundjie_hdmi_ui':arm(a.ui_dir/'soundjie_hdmi_ui',a.ui_dir/'build-report.txt'),
        'tv/reference.features':a.feature_file.read_bytes(),'tv/cues.tsv':(a.reference_dir/'cues.tsv').read_bytes(),
        'tv/traffic.bin':a.model.read_bytes(),
        'tv/clip_config.json':(ROOT/'config/soundjie_tv_ambulance_clip.json').read_bytes(),
        'assets/hdmi/soundjie_glyph_atlas.bin':(ROOT/'assets/hdmi/soundjie_tv_glyph_atlas_v3.bin').read_bytes(),
        'tv/label_policy.json':(a.model.parent/'label-policy-reviewed.json').read_bytes(),
        'licenses/gnu-unifont.md':(ROOT/'docs/licenses/gnu-unifont-17.0.04.md').read_bytes(),
    }
    if files['tv/reference.features'][:8]!=b'SJTVF2\0\0' or files['tv/traffic.bin'][:8]!=b'SJTF1\0\0\0':raise ValueError('wrong reference/model schema')
    with tarfile.open(a.diagnostic_archive) as t:
        files['runtime/soundjie_d4_classifier']=t.extractfile('sj-diagnostic/runtime/soundjie_d4_classifier').read()
    if digest(files['runtime/soundjie_d4_classifier'])!=CLASSIFIER:raise ValueError('classifier drift')
    for name in ('prepare_soundjie_tv_overlay.sh','run_soundjie_tv_session.sh','run_soundjie_d4_session.sh'):
        files['board/'+name]=(ROOT/'scripts/board'/name).read_bytes().replace(b'\r\n',b'\n')
    manifest=dict(schema='soundjie-tv1-development',physical_board_tested=False,autostart=False,writes_sd=False,
        base_model_sha256=BASE_MODEL,clip='ambulance_escort_45_75',scope='enrolled 30s clip only; no generic ASR or ambulance accuracy claim',
        runtime_abi=(a.runtime_dir/'build-report.txt').read_text(),ui_abi=(a.ui_dir/'build-report.txt').read_text(),
        files={name:dict(size=len(b),sha256=digest(b)) for name,b in files.items()})
    files['manifest.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8')
    files['SHA256SUMS']=''.join(digest(b)+'  '+name+'\n' for name,b in sorted(files.items())).encode()
    a.output.mkdir(parents=True);stage=a.output/'sj-tv1'
    for name,b in files.items():
        path=stage/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(b)
    archive=a.output/'SJTV1.TGZ'
    with archive.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for name,b in sorted(files.items()):
            info=tarfile.TarInfo('sj-tv1/'+name);info.size=len(b);info.mode=0o755 if name.startswith(('runtime/','board/')) else 0o644;info.mtime=0
            tar.addfile(info,io.BytesIO(b))
    with tarfile.open(archive) as tar:
        for name,b in files.items():
            if tar.extractfile('sj-tv1/'+name).read()!=b:raise ValueError('archive verification failed')
    (a.output/'SJTV1.SHA').write_text(digest(archive.read_bytes())+'  SJTV1.TGZ\n',encoding='ascii')
    (a.output/'build-report.json').write_text(json.dumps(dict(archive_sha256=digest(archive.read_bytes()),archive_bytes=archive.stat().st_size,files=len(files),physical_tested=False),indent=2),encoding='utf-8')
    print((a.output/'build-report.json').read_text())
if __name__=='__main__':main()
