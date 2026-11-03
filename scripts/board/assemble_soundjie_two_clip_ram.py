#!/usr/bin/env python3
"""Python 3.5 RAM-only assembly of a pinned delta and existing SD application."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

def sha(data):
    return hashlib.sha256(data).hexdigest()

def members(path, prefix=None):
    result={}
    with tarfile.open(str(path), 'r:gz') as stream:
        total=0
        for info in stream.getmembers():
            if info.isdir():
                continue
            name=info.name
            rel=PurePosixPath(name)
            if (not info.isfile() or rel.as_posix()!=name or rel.is_absolute() or '..' in rel.parts or
                    '\\' in name or ':' in name or name in result or info.size>67108864):
                raise ValueError('unsafe archive member')
            total+=info.size
            if total>134217728:
                raise ValueError('archive too large')
            if prefix is not None:
                if not name.startswith(prefix+'/'):
                    raise ValueError('archive prefix differs')
                name=name[len(prefix)+1:]
            if name in result:
                raise ValueError('duplicate member')
            result[name]=(stream.extractfile(info).read(),info.mode)
    return result

def main():
    payload, expected, base, dest=map(str,sys.argv[1:])
    dest=Path(dest)
    if not str(dest).startswith('/run/sj6-') or dest.exists() or dest.is_symlink():
        raise ValueError('new /run/sj6-* destination required')
    if dest.parent.resolve()!=Path('/run'):
        raise ValueError('RAM parent required')
    with open('/proc/mounts') as stream:
        if not any(line.split()[1:3]==['/run','tmpfs'] for line in stream):
            raise ValueError('/run is not tmpfs')
    if sha(Path(payload).read_bytes())!=expected:
        raise ValueError('transfer SHA mismatch')
    patch=members(payload)
    meta=json.loads(patch['assembly.json'][0].decode('utf-8'))
    if sha(Path(base).read_bytes())!=meta['base_sha256']:
        raise ValueError('base SHA mismatch')
    original=members(base,'sj-alert4-release')
    files={}
    for name,spec in meta['target'].items():
        rel=PurePosixPath(name)
        if rel.as_posix()!=name or rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
            raise ValueError('unsafe target')
        if any(str(parent) in meta['target'] for parent in rel.parents if str(parent)!='.'):
            raise ValueError('file/ancestor collision')
        entry=patch.get('patch/'+name)
        if entry is None:
            entry=original[name]
            if meta['unchanged'].get(name)!=sha(entry[0]):
                raise ValueError('unverified reuse')
        data,mode=entry
        if len(data)!=spec['size'] or sha(data)!=spec['sha256'] or mode!=spec['mode']:
            raise ValueError('target differs: '+name)
        files[name]=(data,mode)
    dest.mkdir(mode=0o700)
    for name,(data,mode) in sorted(files.items()):
        path=dest/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data)
        path.chmod(mode)
        if sha(path.read_bytes())!=meta['target'][name]['sha256']:
            raise ValueError('RAM readback differs')
    print(json.dumps({'result':'RAM_ASSEMBLY_OK','files':len(files),
                      'target_archive_sha256':meta['target_archive_sha256'],'destination':str(dest)}))

if __name__=='__main__':
    main()
