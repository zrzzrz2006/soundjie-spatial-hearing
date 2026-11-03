"""Stage this exact boot overlay; never format, repartition, or write raw media."""
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from build_soundjie_autostart import REQUIRED_FILES, make_environment, sha
from stage_additive_inference_sd import file_sha256, read_volume_serial, read_filesystem_state


def load_release(source):
    source=Path(source)
    m=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if m.get('schema')!='soundjie-fat-autostart-v1' or m.get('required_files')!=REQUIRED_FILES:
        raise ValueError('release contract differs')
    if set(m['files'])!={'uEnv.txt','SJAUTO1.SH','SJUBAK1.ENV'}:
        raise ValueError('unexpected release file names')
    for name,entry in m['files'].items():
        p=source/name
        if p.is_symlink() or p.stat().st_size!=entry['bytes'] or file_sha256(p)!=entry['sha256']:
            raise ValueError('prepared file differs: '+name)
    current=(Path(__file__).resolve().parents[1]/'board/soundjie_autostart.sh').read_bytes().replace(b'\r\n',b'\n')
    script=(source/'SJAUTO1.SH').read_bytes()
    if script!=current or (source/'uEnv.txt').read_bytes()!=make_environment(sha(script)):
        raise ValueError('candidate does not match reviewed startup sources')
    if file_sha256(source/'SJUBAK1.ENV')!=REQUIRED_FILES['uEnv.txt']:
        raise ValueError('original environment backup differs')
    return m


def verify_sd_identity(root):
    root=Path(root).resolve()
    anchor=root.anchor
    if str(root)!=anchor or not re.fullmatch(r'(?:[A-Za-z]:\\|\\\\\?\\Volume\{[0-9a-fA-F-]+\}\\)',anchor):
        raise ValueError('SD volume root required')
    if read_volume_serial(root)!='55DE-86B8':
        raise ValueError('unexpected SD volume serial')
    state=read_filesystem_state(root)
    if not state['clean'] or state['filesystem']!='FAT32':
        raise ValueError('FAT volume does not report Healthy/OK')
    # anchor has been strictly validated above; no arbitrary shell text.
    code=("$target='"+anchor+"'; $p=@(Get-Partition | Where-Object { $_.AccessPaths -contains $target }); "
          "if($p.Count -ne 1){throw 'ambiguous volume mapping'}; $d=Get-Disk -Number $p[0].DiskNumber; "
          "[pscustomobject]@{Serial=[string]$d.SerialNumber;Size=$d.Size;Bus=[string]$d.BusType;"
          "IsSystem=$d.IsSystem;IsBoot=$d.IsBoot;Offset=$p[0].Offset;PartitionSize=$p[0].Size;"
          "Number=$d.Number}|ConvertTo-Json -Compress")
    q=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',code],capture_output=True,text=True,check=True)
    d=json.loads(q.stdout)
    if (d['Serial'].strip()!='121220160204' or d['Size']!=15728640000 or d['Bus']!='USB'
            or d['IsSystem'] or d['IsBoot'] or d['Offset']!=1048576 or d['PartitionSize']!=103809024):
        raise ValueError('physical SD identity/layout differs')
    return dict(device=d,volume_serial='55DE-86B8',filesystem=state)


def snapshot(root):
    found={}
    for p in root.iterdir():
        if p.is_symlink():
            raise ValueError('unexpected link on SD')
        if p.is_file():
            found[p.name]=dict(bytes=p.stat().st_size,sha256=file_sha256(p))
    return found


def write_new_verified(path,data):
    with path.open('xb') as f:
        count=f.write(data)
        if count!=len(data):
            raise OSError('short write')
        f.flush();os.fsync(f.fileno())
    if path.read_bytes()!=data:
        raise OSError('readback differs: '+str(path))


def preflight(source,sd_root,backup,identity_check=verify_sd_identity):
    source=Path(source).resolve();sd_root=Path(sd_root).resolve();backup=Path(backup).resolve()
    if source.is_relative_to(sd_root) or backup.is_relative_to(sd_root):
        raise ValueError('source and backup must be outside SD')
    if backup.exists():
        raise FileExistsError('backup directory must be new')
    manifest=load_release(source)
    identity=identity_check(sd_root)
    before=snapshot(sd_root)
    for name,digest in manifest['required_files'].items():
        if before.get(name,{}).get('sha256')!=digest:
            raise ValueError('original SD file differs: '+name)
    for name in ('SJAUTO1.SH','SJUBAK1.ENV','SJAUTO1.NEW'):
        if (sd_root/name).exists():
            raise FileExistsError('refusing collision: '+name)
    if (sd_root/'uEnv.txt').read_bytes()!=(source/'SJUBAK1.ENV').read_bytes():
        raise ValueError('original uEnv and prepared rollback differ')
    needed=sum(x['bytes'] for x in manifest['files'].values())+1048576
    if shutil.disk_usage(sd_root).free<needed:
        raise OSError('insufficient free space')
    return source,sd_root,backup,manifest,identity,before


def stage_release(source,sd_root,backup,*,identity_check=verify_sd_identity):
    source,sd_root,backup,m,identity,before=preflight(source,sd_root,backup,identity_check)
    backup.mkdir(parents=True)
    # First persist and read back the original environment on the host.
    write_new_verified(backup/'uEnv.txt',(sd_root/'uEnv.txt').read_bytes())
    write_new_verified(backup/'preflight.json',json.dumps(dict(identity=identity,files=before),indent=2).encode())
    identity_check(sd_root)
    for name in ('SJUBAK1.ENV','SJAUTO1.SH'):
        data=(source/name).read_bytes()
        if sha(data)!=m['files'][name]['sha256']:
            raise ValueError('source changed during install: '+name)
        write_new_verified(sd_root/name,data)
    new=(source/'uEnv.txt').read_bytes()
    if sha(new)!=m['files']['uEnv.txt']['sha256']:
        raise ValueError('candidate environment changed')
    write_new_verified(sd_root/'SJAUTO1.NEW',new)
    identity_check(sd_root)
    middle=snapshot(sd_root)
    for name,entry in before.items():
        if middle.get(name)!=entry:
            raise ValueError('original file changed before activation: '+name)
    for name in ('SJUBAK1.ENV','SJAUTO1.SH'):
        if file_sha256(sd_root/name)!=m['files'][name]['sha256']:
            raise ValueError('staged dependency changed before activation')
    if file_sha256(sd_root/'SJAUTO1.NEW')!=m['files']['uEnv.txt']['sha256']:
        raise ValueError('staged configuration changed')
    # Only the requested existing boot configuration is replaced. This is a
    # rename on FAT, NOT a claim of power-failure atomicity. Backups already
    # exist on host and SD. No delete/format or raw-device operation is used.
    os.replace(sd_root/'SJAUTO1.NEW',sd_root/'uEnv.txt')
    with (sd_root/'uEnv.txt').open('r+b') as f:
        f.flush();os.fsync(f.fileno())
    after=snapshot(sd_root)
    expected=(set(before)|{'SJAUTO1.SH','SJUBAK1.ENV'})
    if set(after)!=expected:
        raise ValueError('unexpected root file set after activation')
    for name,entry in before.items():
        if name!='uEnv.txt' and after.get(name)!=entry:
            raise ValueError('unrelated original file changed: '+name)
    for name,entry in m['files'].items():
        if after.get(name)!=entry:
            raise ValueError('installed file differs: '+name)
    result=dict(result='AUTOSTART_FILES_STAGED_NOT_YET_BOOT_TESTED',identity=identity,
                original_environment_backup=str(backup/'uEnv.txt'),
                files=after,unchanged_original_files=len(before)-1,
                rollback='Restore verified SJUBAK1.ENV bytes to uEnv.txt; leave other files in place')
    write_new_verified(backup/'installed.json',json.dumps(result,indent=2).encode())
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--sd-root',type=Path,required=True)
    p.add_argument('--backup',type=Path,required=True)
    p.add_argument('--apply',action='store_true',help='Otherwise preflight only; no writes')
    a=p.parse_args()
    if a.apply:
        r=stage_release(a.source,a.sd_root,a.backup)
        print(json.dumps(dict(result=r['result'],unchanged_original_files=r['unchanged_original_files'])))
    else:
        *_,identity,before=preflight(a.source,a.sd_root,a.backup)
        print(json.dumps(dict(result='READ_ONLY_PREFLIGHT_OK',identity=identity,original_files=len(before))))
