#!/usr/bin/env python3
"""Compose a new, fully verified RAM release from an existing RAM candidate."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


def main():
    source, update, destination = map(Path, sys.argv[1:])
    for path in (source, update, destination):
        if path.parent.resolve() != Path('/run') or not path.name.startswith('sj6-') or path.is_symlink():
            raise ValueError('direct /run/sj6-* directories required')
    if destination.exists():
        raise ValueError('destination already exists')
    if not any(row.split()[1:3] == ['/run', 'tmpfs'] for row in open('/proc/mounts')):
        raise ValueError('/run is not tmpfs')
    meta = json.loads((update/'target-files.json').read_text(encoding='utf-8'))
    expected = meta['files']
    if not expected or len(expected) > 512:
        raise ValueError('invalid inventory')
    actual = set()
    for path in source.rglob('*'):
        if path.is_symlink():
            raise ValueError('source contains link')
        if path.is_file():
            actual.add(path.relative_to(source).as_posix())
    if actual != set(expected):
        raise ValueError('source inventory differs')
    files = {}
    for name, entry in expected.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or rel.as_posix() != name or '..' in rel.parts or '\\' in name:
            raise ValueError('invalid inventory path')
        patch = update/'patch'/name
        path = patch if patch.exists() else source/name
        if any(p.is_symlink() for p in [path] + list(path.parents)) or not path.is_file():
            raise ValueError('invalid source type')
        data = path.read_bytes()
        if (len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256'] or
                stat.S_IMODE(path.stat().st_mode) != entry['mode']):
            raise ValueError('file differs: ' + name)
        files[name] = data
    destination.mkdir(mode=0o700)
    for name, data in sorted(files.items()):
        path = destination/name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as output:
            output.write(data)
        path.chmod(expected[name]['mode'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]['sha256']:
            raise ValueError('readback mismatch')
    print(json.dumps(dict(result='RAM_RELEASE_FINALIZED', files=len(files), destination=str(destination),
                          target_archive_sha256=meta['target_archive_sha256'])))


if __name__ == '__main__':
    main()
