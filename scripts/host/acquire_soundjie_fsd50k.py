"""Acquire a bounded FSD50K dev subset; never reads old audio or feature caches.

Official metadata controls labels/attribution. Official ZIP central-directory
size and CRC validate individual mirror WAVs. No full audio archive download.
Selection and uploader splits are frozen before downloading or making features.
All writes are exclusive, reread-verified, and restricted to the configured F: root.
Run from the repository: python -B -m scripts.host.acquire_soundjie_fsd50k
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
import unicodedata
from urllib.parse import quote
from urllib.request import Request, urlopen
import wave
import zipfile
import zlib

from scripts.host.build_soundjie_d4_manifest import partition_records


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_verified(path, data):
    """Never replace existing bytes; compare both existing and freshly written files."""
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('existing file differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(data)
            stream.flush()
    if digest(path.read_bytes()) != digest(data):
        raise ValueError('post-write reread hash mismatch: ' + str(path))


def json_bytes(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8')


def get_bytes(url, limit, headers=None):
    req = Request(url, headers={'User-Agent': 'Soundjie-FSD50K-research/1.0', **(headers or {})})
    with urlopen(req, timeout=20) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError('response exceeds bounded download limit')
        return data, response.status, {key.lower(): value for key, value in response.headers.items()}


def validate_range(status, content_range, start, end, data):
    expected = 'bytes %d-%d/' % (start, end)
    if status != 206 or not content_range or not content_range.startswith(expected) or len(data) != end - start + 1:
        raise ValueError('server ignored/misserved HTTP range: %s %s' % (status, content_range))


def range_bytes(url, start, end):
    # Range in query also prevents intermediaries reusing a cached unrelated range.
    url += ('&' if '?' in url else '?') + 'soundjie_range=%d-%d' % (start, end)
    data, status, headers = get_bytes(url, end-start+1, {'Range': 'bytes=%d-%d' % (start, end)})
    validate_range(status, headers.get('content-range'), start, end, data)
    return data


def parse_eocd(tail):
    offset = tail.rfind(b'PK\x05\x06')
    if offset < 0 or len(tail) < offset + 22:
        raise ValueError('missing ZIP end of directory')
    values = struct.unpack_from('<4s4H2LH', tail, offset)
    if values[1] != values[2] or values[5] == 0xffffffff or values[6] == 0xffffffff:
        raise ValueError('unsupported central directory layout')
    return dict(disk=values[1], entries=values[4], size=values[5], offset=values[6])


def parse_central_directory(data):
    entries, offset = {}, 0
    while offset < len(data):
        if data[offset:offset+4] != b'PK\x01\x02' or len(data)-offset < 46:
            raise ValueError('invalid ZIP central directory')
        v = struct.unpack_from('<4s6H3L5H2L', data, offset)
        end = offset + 46 + v[10] + v[11] + v[12]
        if end > len(data):
            raise ValueError('truncated ZIP directory entry')
        name = data[offset+46:offset+46+v[10]].decode('utf-8')
        match = re.fullmatch(r'FSD50K\.dev_audio/(\d+)\.wav', name)
        if match:
            if match[1] in entries or v[2] > 20 or v[3] & 1 or v[4] not in (0, 8):
                raise ValueError('unsupported or duplicate archive member')
            entries[match[1]] = dict(name=name, crc32=v[7], compressed_size=v[8],
                                    uncompressed_size=v[9], disk=v[13], offset=v[16], method=v[4])
        offset = end
    return entries


def verify_member(data, entry):
    if len(data) != entry['uncompressed_size'] or zlib.crc32(data) != entry['crc32']:
        raise ValueError('audio differs from official ZIP size/CRC')


def parse_ground_truth(text):
    rows, seen = [], set()
    for row in csv.DictReader(io.StringIO(text)):
        fid = row['fname']
        labels, mids = row['labels'].split(','), row['mids'].split(',')
        if not re.fullmatch(r'[0-9]+', fid) or fid in seen or len(labels) != len(mids) or not all(labels):
            raise ValueError('invalid/duplicate ID or mismatched labels/mids')
        if row.get('split') not in ('train', 'val'):
            raise ValueError('only official development records are accepted')
        seen.add(fid)
        rows.append(dict(fname=fid, labels=labels, mids=mids, split=row['split']))
    return rows


def classify(labels, config):
    labels, matches = set(labels), []
    for rule in config['classes']:
        if (set(rule['required_all']) <= labels and
                (not rule['required_any'] or labels.intersection(rule['required_any'])) and
                labels <= set(rule['allowed_labels'])):
            matches.append(rule)
    return matches[0] if len(matches) == 1 else None


def exclusions(rows):
    ids, hashes = set(), set()
    for row in rows:
        match = re.fullmatch(r'(?:esc50|fsd50k|freesound):(\d+)', row.get('source_group', ''))
        if match:
            ids.add(match[1])
        for field in ('original_url', 'recording_id', 'source_recording_id'):
            value = row.get(field, '') or ''
            match = re.search(r'(?:freesound\.org/[^\s]*?/sounds/|fsd50k:|freesound:)(\d+)', value)
            if match:
                ids.add(match[1])
        if row.get('sha256'):
            hashes.add(row['sha256'])
    return ids, hashes


def uploader_group(uploader):
    canonical = unicodedata.normalize('NFKC', uploader).strip().casefold()
    if not canonical:
        raise ValueError('missing uploader')
    return 'freesound-uploader:' + quote(canonical, safe='')


def purpose_for(group, config):
    number = int(digest((config['split_seed'] + '\0' + group).encode())[:16], 16) / 2**64
    return 'validation' if number < config['validation_fraction'] else 'train'


def license_name(url, config):
    return config['allowed_licenses'].get(url.replace('http://', 'https://', 1))


def load_metadata(root, config):
    evidence = {}
    for name, md5 in config['metadata_md5'].items():
        path = root / name
        data = path.read_bytes() if path.exists() else get_bytes(config['official_page']+'/files/'+name+'?download=1', 12_000_000)[0]
        if hashlib.md5(data).hexdigest() != md5:
            raise ValueError('official metadata checksum mismatch: ' + name)
        save_verified(path, data)
        evidence[name] = dict(md5=md5, sha256=digest(data), bytes=len(data))
    with zipfile.ZipFile(root / 'FSD50K.ground_truth.zip') as archive:
        rows = parse_ground_truth(archive.read('FSD50K.ground_truth/dev.csv').decode())
        vocab = {row[1]: row[2] for row in csv.reader(io.StringIO(archive.read('FSD50K.ground_truth/vocabulary.csv').decode()))}
    with zipfile.ZipFile(root / 'FSD50K.metadata.zip') as archive:
        meta = json.loads(archive.read('FSD50K.metadata/dev_clips_info_FSD50K.json'))
    for row in rows:
        if any(vocab.get(label) != mid for label, mid in zip(row['labels'], row['mids'])):
            raise ValueError('label/mid disagrees with official vocabulary')
    return rows, meta, evidence


def load_directory(root, config):
    url = config['official_page']+'/files/FSD50K.dev_audio.zip?download=1'
    tail_path, central_path = root/'official-dev-tail.bin', root/'official-dev-central.bin'
    if tail_path.exists():
        tail = tail_path.read_bytes()
    else:
        tail, status, headers = get_bytes(url+'&soundjie_tail=65536', 65536, {'Range': 'bytes=-65536'})
        cr = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', headers.get('content-range', ''))
        if status != 206 or not cr or int(cr[2])+1 != int(cr[3]) or len(tail) != 65536:
            raise ValueError('official suffix range not honored')
        save_verified(tail_path, tail)
    eocd = parse_eocd(tail)
    central = central_path.read_bytes() if central_path.exists() else range_bytes(url, eocd['offset'], eocd['offset']+eocd['size']-1)
    save_verified(central_path, central)
    entries = parse_central_directory(central)
    if len(entries) != 40966:
        raise ValueError('official dev directory count differs from release')
    return entries, dict(eocd=eocd, central_sha256=digest(central), tail_sha256=digest(tail), url=url)


def make_plan(rows, metadata, excluded_ids, config):
    candidates, rejected = [], Counter()
    for row in rows:
        rule = classify(row['labels'], config)
        if rule is None:
            rejected['unselected_or_conflicting_labels'] += 1
            continue
        if row['fname'] in excluded_ids:
            rejected['old_freesound_id_any_purpose'] += 1
            continue
        meta = metadata.get(row['fname'], {})
        license_id = license_name(meta.get('license', ''), config)
        if not license_id or not meta.get('uploader', '').strip():
            rejected['unsupported_license_or_missing_uploader'] += 1
            continue
        group = uploader_group(meta['uploader'])
        candidates.append(dict(freesound_id=row['fname'], original_labels=row['labels'],
                               original_mids=row['mids'], official_split=row['split'],
                               target_index=rule['target_index'], target_id=rule['target_id'],
                               source_group=group, purpose=purpose_for(group, config),
                               author=meta['uploader'], license=license_id, license_url=meta['license'],
                               title=meta.get('title', ''), metadata=meta,
                               original_url='https://freesound.org/people/'+quote(meta['uploader'], safe='')+'/sounds/'+row['fname']+'/'))
    # Round-robin by uploader: increase diversity before taking a second clip.
    buckets = defaultdict(list)
    for row in sorted(candidates, key=lambda r: digest((config['split_seed']+r['freesound_id']).encode())):
        buckets[(row['target_index'], row['source_group'])].append(row)
    selected, counts = [], Counter()
    limits = {r['target_index']: r['limit'] for r in config['classes']}
    for rank in range(config['max_per_uploader_per_class']):
        for key in sorted(buckets, key=lambda k: digest(str(k).encode())):
            if rank < len(buckets[key]) and counts[key[0]] < limits[key[0]]:
                selected.append(buckets[key][rank])
                counts[key[0]] += 1
    return selected, dict(rejected), dict(Counter(str(r['target_index']) for r in candidates))


def audio_quality(data):
    import numpy as np
    with wave.open(io.BytesIO(data), 'rb') as wav:
        channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        pcm = wav.readframes(frames)
    if channels != 1 or width != 2 or rate != 44100 or len(pcm) != frames*2 or not 0.25 <= frames/rate <= 30.1:
        raise ValueError('audio format/duration does not match FSD50K release')
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float64)/32768.0
    rms, peak = float(np.sqrt(np.mean(samples**2))), float(np.max(np.abs(samples)))
    if peak == 0:
        raise ValueError('silent audio')
    return dict(duration_seconds=frames/rate, sample_rate=rate, channels=channels,
                samples=frames, rms_dbfs=float(20*np.log10(rms)), peak=peak,
                clipped_fraction=float(np.mean(np.abs(samples) >= 32767/32768)),
                pcm_sha256=digest(pcm))


def official_member(entry, config):
    """Fallback for a member wholly within one split ZIP segment."""
    part = 'FSD50K.dev_audio.' + ('zip' if entry['disk'] == 5 else 'z%02d' % (entry['disk']+1))
    url = config['official_page']+'/files/'+part+'?download=1'
    header = range_bytes(url, entry['offset'], entry['offset']+29)
    values = struct.unpack('<4s5H3L2H', header)
    if values[0] != b'PK\x03\x04':
        raise ValueError('bad local ZIP header')
    start = entry['offset'] + 30 + values[9] + values[10]
    compressed = range_bytes(url, start, start+entry['compressed_size']-1)
    data = zlib.decompress(compressed, -15) if entry['method'] == 8 else compressed
    verify_member(data, entry)
    return data, url


def acquire_one(row, entry, root, config, revisions):
    path = root/'audio'/(row['freesound_id']+'.wav')
    errors = []
    if path.exists():
        data, source_url = path.read_bytes(), 'existing-file-verified-against-official-crc'
    else:
        data, source_url = None, None
        for repo, revision in revisions.items():
            url = 'https://huggingface.co/datasets/'+repo+'/resolve/'+revision+'/clips/dev/'+row['freesound_id']+'.wav'
            try:
                candidate, status, _ = get_bytes(url, entry['uncompressed_size'])
                if status != 200:
                    raise ValueError('expected complete mirror file')
                verify_member(candidate, entry)
                data, source_url = candidate, url
                break
            except Exception as error:
                errors.append(dict(url=url, error=str(error)))
        if data is None:
            data, source_url = official_member(entry, config)
        save_verified(path, data)
    verify_member(data, entry)
    quality = audio_quality(data)
    expected = digest(data)
    if digest(path.read_bytes()) != expected:
        raise ValueError('audio reread hash mismatch')
    result = {k: v for k, v in row.items() if k != 'metadata'}
    result.update(recording_id='fsd50k:'+row['freesound_id'], source_id='fsd50k',
                  source_recording_id='freesound:'+row['freesound_id'],
                  speaker_or_uploader=row['author'], path=str(path), sha256=expected,
                  split=row['purpose'], domain='direct', parent_recording_id=None,
                  parent_sha256=None, derivation=None, feature_sha256=None, conditions={},
                  prior_exposure='new_official_fsd50k_dev_not_observed_demo',
                  independent_unseen_acceptance=False, download_url=source_url,
                  official_archive_member=entry, quality=quality,
                  weak_clip_label=True, human_audition_performed=False,
                  license_decision='CC0_or_attribution_required_CC_BY_offline_research',
                  competition_use=config['competition_use'])
    return result, errors


def verify_published(root, config_bytes, old_bytes):
    """Read-only complete verification; safe default on rerun of a published set."""
    config = json.loads(config_bytes)
    audit = json.loads((root/'audit.json').read_bytes())
    manifest_bytes = (root/'manifest.jsonl').read_bytes()
    plan_bytes = (root/'selection-plan.json').read_bytes()
    for name, actual in [('manifest_sha256', digest(manifest_bytes)),
                         ('selection_plan_sha256', digest(plan_bytes)),
                         ('config_sha256', digest(config_bytes)),
                         ('old_manifest_sha256', digest(old_bytes))]:
        if audit[name] != actual:
            raise ValueError('published evidence hash mismatch: ' + name)
    if (root/'acquisition-config.json').read_bytes() != config_bytes:
        raise ValueError('archived configuration changed')
    for name, info in audit['metadata'].items():
        if digest((root/name).read_bytes()) != info['sha256']:
            raise ValueError('metadata reread mismatch: ' + name)
    for name, key in [('official-dev-tail.bin', 'tail_sha256'), ('official-dev-central.bin', 'central_sha256')]:
        if digest((root/name).read_bytes()) != audit['official_directory'][key]:
            raise ValueError('official directory reread mismatch')
    rows = [json.loads(line) for line in manifest_bytes.splitlines()]
    old_rows = [json.loads(line) for line in old_bytes.decode('utf-8-sig').splitlines() if line.strip()]
    excluded, old_hashes = exclusions(old_rows)
    selected = {r['freesound_id'] for r in json.loads(plan_bytes)['candidates']}
    pcm_seen = set()
    for row in rows:
        if row['freesound_id'] not in selected or row['freesound_id'] in excluded or row['sha256'] in old_hashes:
            raise ValueError('source inclusion/exclusion violated')
        if row['source_group'] != uploader_group(row['author']) or row['purpose'] != purpose_for(row['source_group'], config):
            raise ValueError('frozen uploader split changed')
        rule = classify(row['original_labels'], config)
        if rule is None or rule['target_index'] != row['target_index']:
            raise ValueError('label mapping changed')
        data = Path(row['path']).read_bytes()
        if digest(data) != row['sha256']:
            raise ValueError('source reread hash mismatch: ' + row['recording_id'])
        verify_member(data, row['official_archive_member'])
        quality = audio_quality(data)
        if quality != row['quality'] or quality['pcm_sha256'] in pcm_seen:
            raise ValueError('audio quality changed or duplicate decoded PCM')
        pcm_seen.add(quality['pcm_sha256'])
    partition_records(old_rows+rows)
    if len(rows) != audit['usable_clips']:
        raise ValueError('published row count changed')
    return dict(stage='verified_existing', usable_clips=len(rows),
                manifest_sha256=digest(manifest_bytes), class_counts=audit['class_counts'],
                split_counts=audit['split_counts'], old_audio_read=False)


def exclude_known_old_uploaders(rows, old_ids, metadata):
    old_groups = {uploader_group(metadata[fid]['uploader']) for fid in old_ids
                  if fid in metadata and metadata[fid].get('uploader')}
    kept = [row for row in rows if row['source_group'] not in old_groups]
    return kept, dict(old_ids_with_uploader=sum(fid in metadata for fid in old_ids),
                      old_ids_without_uploader=sum(fid not in metadata for fid in old_ids),
                      known_old_uploader_groups=sorted(old_groups),
                      excluded_new_ids=[r['freesound_id'] for r in rows if r['source_group'] in old_groups])


def publish_independent(root, old_bytes, verify_only=False):
    """Preserve the acquired pool; publish a stricter subset under independent/."""
    pool_bytes = (root/'manifest.jsonl').read_bytes()
    rows = [json.loads(line) for line in pool_bytes.splitlines()]
    old_rows = [json.loads(line) for line in old_bytes.decode('utf-8-sig').splitlines() if line.strip()]
    old_ids, _ = exclusions(old_rows)
    with zipfile.ZipFile(root/'FSD50K.metadata.zip') as archive:
        metadata = json.loads(archive.read('FSD50K.metadata/dev_clips_info_FSD50K.json'))
        metadata.update(json.loads(archive.read('FSD50K.metadata/eval_clips_info_FSD50K.json')))
    kept, historical = exclude_known_old_uploaders(rows, old_ids, metadata)
    partition_records(old_rows+kept)
    output = root/'independent'
    manifest = b''.join((json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n').encode('utf-8') for row in kept)
    audit = json.loads((root/'audit.json').read_bytes())
    audit.update(usable_clips=len(kept), manifest_sha256=digest(manifest),
                 acquisition_pool_manifest_sha256=digest(pool_bytes), historical_uploader_audit=historical,
                 class_counts=dict(Counter(str(r['target_index']) for r in kept)),
                 split_counts=dict(Counter(r['purpose']+'/'+r['target_id'] for r in kept)),
                 uploader_counts={str(i):len({r['source_group'] for r in kept if r['target_index']==i}) for i in (1,2,3,4,5)},
                 licenses=dict(Counter(r['license'] for r in kept)),
                 note='Recommended training supplement. Removes all known historical uploaders; remaining old uploader identities are unknown. Splits unchanged from pre-download plan.')
    if verify_only:
        if (output/'manifest.jsonl').read_bytes() != manifest or (output/'audit.json').read_bytes() != json_bytes(audit):
            raise ValueError('independent subset/evidence differs from deterministic selection')
    else:
        output.mkdir(exist_ok=True)
        save_verified(output/'manifest.jsonl', manifest)
        save_verified(output/'audit.json', json_bytes(audit))
    return dict(stage='independent_verified' if verify_only else 'independent_published',
                usable_clips=len(kept), manifest_sha256=digest(manifest),
                class_counts=audit['class_counts'], split_counts=audit['split_counts'],
                excluded_known_old_uploader_clips=len(rows)-len(kept),
                old_ids_without_uploader=historical['old_ids_without_uploader'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[2]/'config/soundjie_fsd50k_sources.json')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--verify-only', action='store_true', help='Reread every published audio and evidence file without network access or writes')
    args = parser.parse_args()
    started = time.time()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    root = Path(config['dataset_root'])
    expected_root = Path(r'F:\MLK\DPU_ZYNQ7000\datasets\soundjie_fsd50k_20261025')
    if root.resolve() != expected_root.resolve():
        raise ValueError('write root must be the explicitly authorized new F: directory')
    old_bytes = Path(config['frozen_manifest']).read_bytes()
    if args.verify_only or (root/'audit.json').exists():
        print(json.dumps(verify_published(root, config_bytes, old_bytes)), flush=True)
        if not args.verify_only or (root/'independent').exists():
            print(json.dumps(publish_independent(root, old_bytes, verify_only=args.verify_only)), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    (root/'audio').mkdir(exist_ok=True)
    save_verified(root/'acquisition-config.json', config_bytes)
    old_rows = [json.loads(line) for line in old_bytes.decode('utf-8-sig').splitlines() if line.strip()]
    excluded_ids, old_hashes = exclusions(old_rows)
    if not excluded_ids:
        raise ValueError('no old ESC50 exclusions found; refusing acquisition')
    rows, metadata, evidence = load_metadata(root, config)
    entries, directory_evidence = load_directory(root, config)
    plan_path = root/'selection-plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_bytes())
        if plan['config_sha256'] != digest(config_bytes) or plan['old_manifest_sha256'] != digest(old_bytes):
            raise ValueError('frozen plan inputs changed')
    else:
        selected, rejected, eligible = make_plan(rows, metadata, excluded_ids, config)
        revisions = {}
        for repo in config['mirrors']:
            response = json.loads(get_bytes('https://huggingface.co/api/datasets/'+repo, 20_000_000)[0])
            revisions[repo] = response['sha']
        plan = dict(schema=1, config_sha256=digest(config_bytes), old_manifest_sha256=digest(old_bytes),
                    excluded_old_freesound_ids=sorted(excluded_ids, key=int), old_record_count=len(old_rows),
                    metadata_evidence=evidence, directory_evidence=directory_evidence,
                    mirror_revisions=revisions, candidates=selected, rejected=rejected,
                    eligible_counts=eligible, split_policy='SHA256(seed,NFKC-casefold-uploader), 20% validation; dev only; frozen before audio')
        save_verified(plan_path, json_bytes(plan))
    print(json.dumps(dict(stage='plan_frozen', selected=len(plan['candidates']),
                          counts=dict(Counter(str(r['target_index']) for r in plan['candidates'])),
                          excluded_old_ids=len(excluded_ids))), flush=True)
    successes, failures, fallback_errors = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        tasks = {executor.submit(acquire_one, row, entries[row['freesound_id']], root, config, plan['mirror_revisions']): row for row in plan['candidates']}
        for future in as_completed(tasks):
            row = tasks[future]
            try:
                result, errors = future.result()
                successes.append(result)
                fallback_errors.extend(errors)
            except Exception as error:
                failures.append(dict(freesound_id=row['freesound_id'], target_index=row['target_index'], error=str(error)))
            if (len(successes)+len(failures)) % 25 == 0:
                print(json.dumps(dict(stage='download', passed=len(successes), failed=len(failures))), flush=True)
    retained, seen_hashes, seen_pcm = [], set(old_hashes), set()
    for row in sorted(successes, key=lambda r: int(r['freesound_id'])):
        try:
            reread = Path(row['path']).read_bytes()
            if digest(reread) != row['sha256']:
                raise ValueError('final prepublication reread mismatch')
            verify_member(reread, row['official_archive_member'])
            if row['sha256'] in seen_hashes or row['quality']['pcm_sha256'] in seen_pcm:
                raise ValueError('duplicate audio hash or decoded PCM; source retained but excluded')
            if row['freesound_id'] in excluded_ids:
                raise ValueError('old ID exclusion violated')
            seen_hashes.add(row['sha256'])
            seen_pcm.add(row['quality']['pcm_sha256'])
            retained.append(row)
        except Exception as error:
            failures.append(dict(freesound_id=row['freesound_id'], target_index=row['target_index'], error=str(error)))
    partition_records(retained)
    # Check the combined purpose/hash contract using old JSON only, never old audio.
    partition_records(old_rows+retained)
    manifest_bytes = b''.join((json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n').encode('utf-8') for row in retained)
    save_verified(root/'manifest.jsonl', manifest_bytes)
    save_verified(root/'failures.json', json_bytes(dict(failures=failures, fallback_errors=fallback_errors)))
    report = dict(schema=1, usable_clips=len(retained), downloaded_source_files=len(successes),
                  class_counts=dict(Counter(str(r['target_index']) for r in retained)),
                  split_counts=dict(Counter(r['purpose']+'/'+r['target_id'] for r in retained)),
                  uploader_counts={str(i):len({r['source_group'] for r in retained if r['target_index']==i}) for i in (1,2,3,4,5)},
                  licenses=dict(Counter(r['license'] for r in retained)), failures=len(failures),
                  excluded_old_ids=len(excluded_ids), old_audio_read=False, features_generated=False,
                  manifest_sha256=digest(manifest_bytes), selection_plan_sha256=digest(plan_path.read_bytes()),
                  config_sha256=digest(config_bytes), old_manifest_sha256=digest(old_bytes),
                  official_directory=directory_evidence, metadata=evidence,
                  limitations=['Clip-level weak labels; no positive-event timestamps or human audition.',
                               'Generic Alarm excluded; class 0 not acquired.',
                               'Mirrors verified by official member CRC/size, not whole multi-GB archive MD5.',
                               'Source files retained unchanged; repeated decoded PCM excluded.',
                               'Old Freesound IDs excluded across every purpose; old uploader identities not fully available.',
                               config['competition_use']])
    # Keep the reproducible report independent of elapsed time and download order.
    save_verified(root/'audit.json', json_bytes(report))
    print(json.dumps(publish_independent(root, old_bytes)), flush=True)
    print(json.dumps(dict(stage='complete', elapsed_seconds=round(time.time()-started, 1), **report)), flush=True)


if __name__ == '__main__':
    main()
