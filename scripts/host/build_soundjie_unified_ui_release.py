"""Build a create-only UI update from the exact verified SD application.

No board, SD, model, classifier, audio or boot mutation is performed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

from scripts.host.assemble_soundjie_alert4_release import read_archive, write_archive

ROOT=Path(__file__).resolve().parents[2]
BASE_SHA='28a34d2c936adb708e5541e4eb21ab6c76491445911bc98cff2e43b9c1c1f640'
TOOLBIN='/mnt/f/MLK/DPU_ZYNQ7000/work/d4-toolchain-linux-20261021a/gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf/bin/'
UI='runtime/soundjie_hdmi_ui'
ATLAS='assets/hdmi/soundjie_glyph_atlas.bin'


def sha(data):return hashlib.sha256(data).hexdigest()


def checksums(payload):
    return ''.join(sha(data)+'  '+name+'\n' for name,data in sorted(payload.items()) if name!='SHA256SUMS').encode()


def audit_abi(text):
    versions=[tuple(map(int,v)) for v in re.findall(r'GLIBC_(\d+)\.(\d+)',text)]
    checks=(re.search(r'Machine:\s+ARM\b',text),re.search(r'Tag_CPU_arch:\s+v7\b',text),
            'Tag_ABI_VFP_args: VFP registers' in text,'/lib/ld-linux-armhf.so.3' in text,
            versions and max(versions)<=(2,24),not re.search(r'Shared library: \[(?:libstdc\+\+|libgcc_s)',text))
    if not all(checks):raise ValueError('ARMv7 Debian 9 ABI audit failed')
    return '.'.join(map(str,max(versions)))


def assemble(base_path,base_sha,ui,atlas,output,audit):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    original=read_archive(Path(base_path),base_sha,'sj-alert4-release/')
    required=(UI,ATLAS,'model/dpu_soundjie_d3.elf','runtime/soundjie_d4_runtime',
              'runtime/soundjie_d4_classifier','config/alert-thresholds.txt',
              'tv/reference.features','tv/traffic.bin','tv/cues.tsv',
              'board/run_soundjie_tv_session.sh','manifest/release.json')
    if any(not original.get(n) for n in required) or not ui or not atlas:
        raise ValueError('base application or update dependencies missing')
    if 'manifest/ui-update.json' in original:raise ValueError('base is already a UI update')
    payload=dict(original);payload[UI]=ui;payload[ATLAS]=atlas
    report=dict(schema=1,ui_layout='unified_live_caption_v2',application_only=True,
        board_tested=False,sd_written=False,cold_boot_verified=False,
        model_changed=False,runtime_changed=False,boot_files_modified=False,
        base_archive_sha256=base_sha,base_ui_sha256=sha(original[UI]),
        base_atlas_sha256=sha(original[ATLAS]),new_ui_sha256=sha(ui),new_atlas_sha256=sha(atlas),
        changed_business_files=sorted([UI,ATLAS]),audit=audit)
    payload['manifest/ui-update.json']=json.dumps(report,ensure_ascii=False,indent=2).encode()
    release=json.loads(original['manifest/release.json'])
    release.update(ui_layout=report['ui_layout'],ui_update_manifest='manifest/ui-update.json',
        base_archive_sha256=base_sha,board_tested=False,sd_written=False,cold_boot_verified=False,
        ui_only_update=True,model_changed=False,runtime_changed=False,
        git_commit=audit.get('git_commit','unit-test'))
    payload['manifest/release.json']=json.dumps(release,ensure_ascii=False,indent=2).encode()
    payload['SHA256SUMS']=checksums(payload)
    allowed={UI,ATLAS,'SHA256SUMS','manifest/release.json'}
    unchanged={n:sha(data) for n,data in original.items() if n not in allowed}
    if any(payload.get(n)!=data for n,data in original.items() if n not in allowed):
        raise ValueError('non-UI business file changed')
    if set(payload)-set(original)!={'manifest/ui-update.json'}:
        raise ValueError('unexpected new payload')
    output.mkdir(parents=True,exist_ok=False)
    full=output/'SJ4U2.TGZ';write_archive(full,payload)
    full_sha=sha(full.read_bytes())
    if read_archive(full,full_sha,'sj-alert4-release/')!=payload:raise ValueError('full archive readback failed')
    patch={n:payload[n] for n in (UI,ATLAS,'manifest/ui-update.json')}
    patch['SHA256SUMS']=checksums(patch)
    patch_path=output/'SJU2.TGZ'
    with tarfile.open(patch_path,'x:gz') as tar:
        for name,data in sorted(patch.items()):
            item=tarfile.TarInfo('sj-unified-ui-v2/'+name)
            item.size=len(data);item.mode=0o755 if name==UI else 0o644;item.mtime=0
            tar.addfile(item,io.BytesIO(data))
    patch_sha=sha(patch_path.read_bytes())
    if read_archive(patch_path,patch_sha,'sj-unified-ui-v2/')!=patch:raise ValueError('patch readback failed')
    report.update(archive_sha256=full_sha,archive_bytes=full.stat().st_size,
        patch_sha256=patch_sha,patch_bytes=patch_path.stat().st_size,
        unchanged_payload_count=len(unchanged),unchanged_payload_sha256=unchanged,
        payload_sha256={n:sha(data) for n,data in payload.items()})
    (output/'SHA256SUMS').write_bytes((full_sha+'  SJ4U2.TGZ\n'+patch_sha+'  SJU2.TGZ\n').encode())
    (output/'release-manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def lin(path):
    p=Path(path).resolve()
    return '/mnt/'+p.drive[0].lower()+p.as_posix().split(':',1)[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--build-dir',type=Path,required=True)
    p.add_argument('--test-log',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    log=a.test_log.read_text(encoding='utf-8-sig')
    passed=re.search(r'Ran (\d+) tests',log)
    if not passed or int(passed[1])<1 or not log.rstrip().endswith('OK'):
        raise ValueError('passing UI regression log required')
    # Verify the original application before creating any output.
    read_archive(a.base,BASE_SHA,'sj-alert4-release/')
    original=a.build_dir/'soundjie_hdmi_ui'
    build=(a.build_dir/'build-report.txt').read_text(encoding='utf-8')
    if 'SHA256='+sha(original.read_bytes()) not in build:raise ValueError('build hash mismatch')
    binary=a.build_dir/'soundjie_hdmi_ui.stripped'
    if binary.exists():raise FileExistsError(binary)
    subprocess.run(['wsl','-d','Ubuntu','--',TOOLBIN+'arm-linux-gnueabihf-strip',
                    '--strip-unneeded','-o',lin(binary),lin(original)],check=True,capture_output=True)
    elf=subprocess.check_output(['wsl','-d','Ubuntu','--',TOOLBIN+'arm-linux-gnueabihf-readelf',
                                 '-h','-A','-l','-d','--version-info',lin(binary)]).decode('utf-8')
    max_glibc=audit_abi(elf)
    atlas=ROOT/'assets/hdmi/soundjie_unified_glyph_atlas.bin'
    audit=dict(max_glibc=max_glibc,readelf_sha256=sha(elf.encode()),
        tests=int(passed[1]),test_log_sha256=sha(a.test_log.read_bytes()),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        ui_source_sha256=sha((ROOT/'src/soundjie_hdmi_ui.cpp').read_bytes()))
    report=assemble(a.base,BASE_SHA,binary.read_bytes(),atlas.read_bytes(),a.output,audit)
    (a.output/'readelf.txt').write_text(elf,encoding='utf-8')
    (a.output/'build-report.txt').write_text(build,encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('archive_sha256','archive_bytes','patch_bytes','unchanged_payload_count','changed_business_files')},indent=2))


if __name__=='__main__':main()
