"""Freeze/verify external candidate source hashes; never edits FPGA or SD files."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create(manifest, paths):
    paths = [Path(path).resolve(strict=True) for path in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('sources must be nonempty and unique')
    rows = [{'path': str(path), 'sha256': digest(path)} for path in paths]
    with Path(manifest).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump({'schema': 1, 'sources': rows, 'board_validated': False}, stream, indent=2)
        stream.write('\n')


def verify(manifest):
    payload = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if payload.get('schema') != 1 or not payload.get('sources'):
        raise ValueError('invalid source binding')
    for row in payload['sources']:
        if digest(Path(row['path'])) != row['sha256']:
            raise ValueError('source drift: ' + row['path'])
    return len(payload['sources'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('create', 'verify'))
    parser.add_argument('manifest', type=Path)
    parser.add_argument('sources', nargs='*', type=Path)
    args = parser.parse_args()
    if args.mode == 'create':
        create(args.manifest, args.sources)
    elif args.sources:
        parser.error('verify does not accept sources')
    print('SOURCE_BINDING_PASS count=' + str(verify(args.manifest)))


if __name__ == '__main__':
    main()
