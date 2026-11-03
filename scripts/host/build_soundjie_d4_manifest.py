"""Create an auditable D4 purpose/domain manifest from frozen D3 splits.

Historical D3 test samples retain their prior exposure; freezing them now does
not turn them into an independently unseen D5 acceptance set.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

PURPOSES = ('train', 'validation', 'demo_calibration', 'demo_shoot', 'd5_holdout')
DOMAINS = ('direct', 'simulated_replay', 'xvf3800_replay')
HEX = re.compile(r'^[0-9a-f]{64}$')


def partition_records(records):
    rows = list(records)
    by_id, groups, hashes = {}, {}, {}
    result = {purpose: [] for purpose in PURPOSES}
    for row in rows:
        rid = row.get('recording_id')
        if not isinstance(rid, str) or not rid or rid in by_id:
            raise ValueError('missing or duplicate recording_id')
        purpose, domain = row.get('purpose'), row.get('domain')
        if purpose not in PURPOSES or domain not in DOMAINS:
            raise ValueError('unsupported purpose/domain')
        group, digest = row.get('source_group'), row.get('sha256')
        if not isinstance(group, str) or not group or not isinstance(digest, str) or not HEX.fullmatch(digest):
            raise ValueError('invalid group/audio hash')
        target = row.get('target_index')
        if type(target) is not int or not 0 <= target < 8:
            raise ValueError('invalid target index')
        for field in ('target_id', 'path', 'license', 'original_url'):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError('missing ' + field)
        if group in groups and groups[group] != purpose:
            raise ValueError('source_group crosses purposes: ' + group)
        if digest in hashes and hashes[digest] != purpose:
            raise ValueError('audio hash crosses purposes: ' + digest)
        groups[group], hashes[digest], by_id[rid] = purpose, purpose, row
        result[purpose].append(row)
    for row in rows:
        parent_id = row.get('parent_recording_id')
        if row['domain'] == 'simulated_replay' and not parent_id:
            raise ValueError('simulated replay requires parent')
        if parent_id is None:
            if row.get('parent_sha256') is not None:
                raise ValueError('parent hash without parent')
            continue
        parent = by_id.get(parent_id)
        if parent is None or parent_id == row['recording_id']:
            raise ValueError('missing or self-referencing parent')
        if row.get('parent_sha256') != parent['sha256']:
            raise ValueError('parent hash mismatch')
        for field in ('source_group', 'purpose', 'target_id', 'target_index'):
            if row[field] != parent[field]:
                raise ValueError('derived record changes parent ' + field)
        if not isinstance(row.get('derivation'), dict) or not row['derivation']:
            raise ValueError('derived record requires recipe')
        seen = {row['recording_id']}
        cursor = parent
        while cursor is not None:
            if cursor['recording_id'] in seen:
                raise ValueError('parent cycle')
            seen.add(cursor['recording_id'])
            cursor = by_id.get(cursor.get('parent_recording_id'))
    return result


def convert_d3(records, source_urls, demo_records=()):
    mapping = {'train': 'train', 'validation': 'validation',
               'test': 'd5_holdout', 'demo': 'demo_shoot'}
    output = []
    combined = list(records)
    for demo in demo_records:
        if demo.get('split', 'demo') != 'demo':
            raise ValueError('demo manifest contains non-demo split')
        combined.append(dict(demo, split='demo'))
    for original in combined:
        if original.get('split') not in mapping:
            raise ValueError('unknown D3 split')
        row = dict(original)
        row.update(domain='direct', purpose=mapping[row['split']],
                   original_url=source_urls.get(row.get('source_id'), ''),
                   parent_recording_id=None, parent_sha256=None,
                   derivation=None, feature_sha256=None, conditions={},
                   prior_exposure='D3_' + row['split'],
                   independent_unseen_acceptance=False)
        output.append(row)
    partition_records(output)
    return output


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--demo', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding='utf-8'))
    inputs = [json.loads(line) for line in args.splits.read_text(encoding='utf-8').splitlines()]
    demo = [json.loads(line) for line in args.demo.read_text(encoding='utf-8').splitlines()]
    rows = convert_d3(inputs,
                      {s['id']: s['official_page'] for s in sources['sources']}, demo)
    # Verify every source before publishing a new manifest; never silently skip drift.
    for row in rows:
        if digest_file(row['path']) != row['sha256']:
            raise ValueError('source audio hash mismatch: ' + row['recording_id'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = args.output_dir / 'manifest.jsonl'
    with manifest.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    report = {'schema': 1, 'manifest_sha256': digest_file(manifest),
              'source_splits_sha256': digest_file(args.splits),
              'source_demo_sha256': digest_file(args.demo),
              'sources_sha256': digest_file(args.sources), 'verified_audio_count': len(rows),
              'counts': dict(Counter(r['purpose'] for r in rows)),
              'class_counts': dict(Counter(r['purpose'] + '/' + r['target_id'] for r in rows)),
              'independent_unseen_acceptance': False,
              'note': 'D3 test data frozen prospectively; new independent sources required for final D5.'}
    with (args.output_dir / 'audit.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
