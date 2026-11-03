#!/usr/bin/env python3
"""Create-only ARM UI/atlas patch, preserving the original model/runtime assets."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parents[2]
TOOLBIN = '/mnt/f/MLK/DPU_ZYNQ7000/work/d4-toolchain-linux-20261021a/gcc-linaro-6.3.1-2017.02-x86_64_arm-linux-gnueabihf/bin/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def lin(path):
    path = path.resolve()
    return '/mnt/' + path.drive[0].lower() + path.as_posix().split(':', 1)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--test-log', type=Path, required=True)
    parser.add_argument('--revision', default='v1')
    args = parser.parse_args()
    if not re.fullmatch(r'v[0-9]+',args.revision):
        raise SystemExit('invalid package revision')
    build = (args.build_dir / 'build-report.txt').read_text(encoding='utf-8')
    original = args.build_dir / 'soundjie_hdmi_ui'
    if not all(v in build for v in ('CPU_ARCH=v7', 'HARD_FLOAT=yes', 'MAX_GLIBC=2.17', 'SHA256='+digest(original.read_bytes()))):
        raise SystemExit('ARM build audit mismatch')
    test_log = args.test_log.read_text(encoding='utf-8')
    if not re.search(r'Ran (\d+) tests', test_log) or not test_log.rstrip().endswith('OK'):
        raise SystemExit('passing regression log required')
    out = args.output_dir
    out.mkdir()  # Refuse to overwrite earlier packages.
    binary = out / 'soundjie_hdmi_ui'
    subprocess.run(['wsl','-d','Ubuntu','--',TOOLBIN+'arm-linux-gnueabihf-strip',
                    '--strip-unneeded','-o',lin(binary),lin(original)],check=True,capture_output=True)
    elf = subprocess.check_output(['wsl','-d','Ubuntu','--',TOOLBIN+'arm-linux-gnueabihf-readelf',
                                   '-h','-A','-l','--version-info',lin(binary)]).decode('utf-8')
    if not (re.search(r'Machine:\s+ARM',elf) and 'Tag_CPU_arch: v7' in elf and
            'Tag_ABI_VFP_args: VFP registers' in elf and '/lib/ld-linux-armhf.so.3' in elf and
            max(tuple(map(int,m)) for m in re.findall(r'GLIBC_(\d+)\.(\d+)',elf)) == (2,17)):
        raise SystemExit('stripped ARM ABI audit failed')
    sources = ['src/'+n for n in ('soundjie_hdmi_ui.cpp','framebuffer_surface.cpp','framebuffer_surface.h',
        'soundjie_d4_doa_stream.cpp','soundjie_d4_doa_stream.h','soundjie_d4_state.cpp','soundjie_d4_state.h',
        'xvf3800_control_protocol.cpp','xvf3800_control_protocol.h','sha256.cpp','sha256.h','utf8_text.h')]
    sources += ['scripts/board/prepare_soundjie_dual_ui.sh','config/soundjie_dual_ui_text.txt']
    report = dict(scope='UI and atlas only; eight-class DPU plus known-clip display',
        toolchain_bin=TOOLBIN,program_sha256=digest(binary.read_bytes()),
        atlas_sha256=digest((ROOT/'assets/hdmi/soundjie_tv_glyph_atlas_v4.bin').read_bytes()),
        tests=int(re.search(r'Ran (\d+) tests',test_log)[1]),test_log_sha256=digest(args.test_log.read_bytes()),
        source_sha256={p:digest((ROOT/p).read_bytes()) for p in sources})
    payloads={'soundjie_hdmi_ui':binary.read_bytes(),
              'soundjie_glyph_atlas.bin':(ROOT/'assets/hdmi/soundjie_tv_glyph_atlas_v4.bin').read_bytes(),
              'prepare.sh':(ROOT/'scripts/board/prepare_soundjie_dual_ui.sh').read_bytes(),
              'build-report.txt':build.encode(), 'report.json':json.dumps(report,indent=2).encode(),
              'readelf.txt':elf.encode()}
    payloads['SHA256SUMS']=''.join(digest(v)+'  '+k+'\n' for k,v in payloads.items()).encode()
    archive = out/('sj-tv1-dual-ui-'+args.revision+'.tgz')
    prefix = 'sj-tv1-dual-ui-patch-'+args.revision+'/'
    with tarfile.open(archive,'x:gz') as tar:
        for name,data in payloads.items():
            info=tarfile.TarInfo(prefix+name)
            info.size=len(data);info.mode=0o755 if name=='soundjie_hdmi_ui' else 0o644;info.mtime=0
            tar.addfile(info,io.BytesIO(data))
    with tarfile.open(archive,'r:gz') as tar:
        for name,data in payloads.items():
            if tar.extractfile(prefix+name).read()!=data:
                raise SystemExit('archive read-back mismatch')
    manifest=dict(bytes=archive.stat().st_size,sha256=digest(archive.read_bytes()),**report)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
