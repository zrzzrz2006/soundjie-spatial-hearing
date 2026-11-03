"""Build a pinned FAT-only startup overlay, without opening any physical disk."""
import argparse
import hashlib
import json
import re
from pathlib import Path

APP_SHA = '0e89305ba1822fe5b8f7a41fb0b6919b09b1212eca0e458b9e51046bb762d7af'
ORIGINAL_ENV_SHA = '900ee46eb0f86c3c90f4f40b0d5c2d05e46df1e06be1f483192084cfcde6bbb3'
REQUIRED_FILES = {
    'BOOT.bin': '992f696208b3cefdecb091dd0f1fb1053e26505199c38e841685d61d8cd53983',
    'uImage': 'ed95081d1cbe16d9b35dcdd5b0e1b0db4063b384c889f138dc5a037193677398',
    'devicetree.dtb': '84a613762b49e32d07223e425f1c2526c872ae832a98986bd814a082286bc701',
    'SJEM1.BIT': '6bc37bf341c549c1c4bfd306f7a3a4275ed07411eaa81fe6992897475af39118',
    'SJEM1.DTB': '721a2d2c228cd7cf958e3de68c38a6db3465741054b2d7101664cf7f0ebecffa',
    'SJ6V1.TGZ': APP_SHA,
    'uEnv.txt': ORIGINAL_ENV_SHA,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_bootargs(script_hash):
    if not re.fullmatch(r'[0-9a-f]{64}', script_hash):
        raise ValueError('invalid script hash')
    # No $, escapes or inner quotes: preserve this whole command as one Linux
    # init argument, including its size= token. U-Boot stores it inside single
    # quotes and Linux strips the one outer double-quoted argument.
    bootstrap = (
        'mount -t tmpfs -o size=256m tmpfs /run && mkdir /run/sj-boot && '
        'mount -t vfat -o ro /dev/mmcblk0p1 /run/sj-boot && '
        'cp /run/sj-boot/SJAUTO1.SH /run/sj-init && '
        'echo ' + script_hash + '  /run/sj-init | sha256sum -c - && '
        'exec /bin/sh /run/sj-init; '
        'echo SJ_BOOTSTRAP_FAILED; while true; do /bin/sh -i; sleep 1; done'
    )
    args = ('console=ttyPS0,115200 earlyprintk root=/dev/mmcblk0p2 ro '
            'rootfstype=ext4 rootflags=noload rootwait consoleblank=0 '
            'vt.global_cursor_default=0 init=/bin/sh -- -c "' + bootstrap + '"')
    if len(args.encode('ascii')) >= 1024:
        raise ValueError('ARM kernel command line too long')
    return args


def make_environment(script_hash):
    env = {
        'optargs': 'console=tty0 consoleblank=0 vt.global_cursor_default=0',
        # The vendor scandev calls these after a returning uenvcmd. Do not
        # silently combine an old DTB/bitstream if any custom step failed.
        **{k: 'false' for k in ('loadkernel','loaddtb','fpga_loadbit','check_ramdisk','baseargs','tryboot')},
        'sj_loadpl': (
            'fatload mmc 0:1 0x10000000 SJEM1.BIT && test ${filesize} = 109c10a && '
            'fpga loadb 0 0x10000000 0x109c10a && '
            'mw.l 0xf8000008 0xdf0d && mw.l 0xf800b000 0x1 && '
            'mw.l 0xf800b014 0xf01 && mw.l 0xf8000004 0x767b'),
        'sj_loadlinux': (
            'fatload mmc 0:1 0x03000000 SJEM1.DTB && test ${filesize} = 28c5 && '
            'fatload mmc 0:1 0x02000000 uImage && test ${filesize} = 3ca858'),
        'sj_args': "setenv bootargs '" + make_bootargs(script_hash) + "'",
        'sj_start': 'mmc rescan && run sj_loadpl && run sj_loadlinux && run sj_args && bootm 0x02000000 - 0x03000000',
        'uenvcmd': 'echo SOUNDJIE_AUTO_V1_2_SECOND_BREAK_WINDOW; if sleep 2; then run sj_start; fi; echo SOUNDJIE_AUTO_STOPPED_USE_SERIAL_OR_RESTORE_UENV',
    }
    return ''.join(k+'='+v+'\n' for k,v in env.items()).encode('ascii')


def build_release(output, app_archive, original_env):
    output = Path(output)
    if output.exists():
        raise FileExistsError('output must be new')
    if sha(Path(app_archive).read_bytes()) != APP_SHA:
        raise ValueError('application archive differs from tested release')
    original = Path(original_env).read_bytes()
    if sha(original) != ORIGINAL_ENV_SHA:
        raise ValueError('original boot environment differs')
    source = Path(__file__).resolve().parents[1]/'board/soundjie_autostart.sh'
    script = source.read_bytes().replace(b'\r\n', b'\n')
    script.decode('ascii')
    if b'\r' in script or APP_SHA.encode() not in script:
        raise ValueError('invalid startup script')
    files = {'SJAUTO1.SH': script, 'uEnv.txt': make_environment(sha(script)), 'SJUBAK1.ENV': original}
    manifest = dict(schema='soundjie-fat-autostart-v1', tested_cold_boot=False,
                    required_files=REQUIRED_FILES,
                    files={n:dict(bytes=len(b),sha256=sha(b)) for n,b in files.items()},
                    bootargs=make_bootargs(sha(script)),
                    board_session='/run/sj-auto-session',
                    activation='Backup verified first; add script; replace uEnv.txt last; no other original file changes')
    output.mkdir(parents=True)
    for name,data in files.items():
        with (output/name).open('xb') as f:
            f.write(data)
    with (output/'manifest.json').open('x',encoding='utf-8') as f:
        json.dump(manifest,f,indent=2)
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--app-archive',required=True,type=Path)
    p.add_argument('--original-env',required=True,type=Path)
    a=p.parse_args()
    r=build_release(a.output,a.app_archive,a.original_env)
    print(json.dumps(dict(output=str(a.output),files=r['files'],bootargs_bytes=len(r['bootargs']))))
