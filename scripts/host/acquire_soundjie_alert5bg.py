"""Create a new manifest from 1474 verified existing sources; never download/copy audio."""
import argparse
from collections import Counter
import json
from pathlib import Path
from scripts.host.acquire_soundjie_alert6 import read_rows
from scripts.host.acquire_soundjie_alert4bg import verify_dataset as verify_upstream
from scripts.host.acquire_soundjie_fsd50k import save_verified, json_bytes
from scripts.host.build_soundjie_d4_manifest import digest_file, partition_records
from scripts.host.soundjie_alert5bg import (
    CONTRACT, LABELS, CONFIG_FILE, CLASS_FILE, DATA_ROOT, map_existing, validate_rows, noise_eligible)


def audit_counts(rows):
    return {p:{label:dict(files=len(part),original_recordings=len({r['sampling_identity'] for r in part}),
               source_groups=len({r['source_group'] for r in part}))
               for c,label in enumerate(LABELS)
               for part in [[r for r in rows if r['purpose']==p and r['target_index']==c]]}
            for p in ('train','validation')}


def verify_dataset(root=DATA_ROOT):
    cfg=json.loads(CONFIG_FILE.read_bytes()); upstream=Path(cfg['existing_manifest'])
    if digest_file(upstream)!=cfg['upstream_manifest_sha256']:raise ValueError('upstream manifest drift')
    # This rechecks existing file SHA and official member CRC without downloading.
    verify_upstream(upstream.parent)
    old=read_rows(upstream); rows=read_rows(root/'manifest.jsonl')
    if len(old)!=1474 or rows!=[map_existing(r) for r in old]:
        raise ValueError('source identities, split, or mapping changed')
    validate_rows(rows);partition_records(rows)
    audit=json.loads((root/'audit.json').read_bytes())
    for name,key in [('manifest.jsonl','manifest_sha256'),('attributions.json','attributions_sha256')]:
        if digest_file(root/name)!=audit[key]:raise ValueError('dataset evidence drift '+name)
    if digest_file(CONFIG_FILE)!=audit['config_sha256'] or digest_file(CLASS_FILE)!=audit['class_map_sha256']:
        raise ValueError('configuration drift')
    if audit_counts(rows)!=audit['counts']:raise ValueError('source count drift')
    return audit


def prepare():
    cfg=json.loads(CONFIG_FILE.read_bytes()); upstream=Path(cfg['existing_manifest'])
    if digest_file(upstream)!=cfg['upstream_manifest_sha256']:raise ValueError('upstream manifest drift')
    base=verify_upstream(upstream.parent)
    rows=[map_existing(r) for r in read_rows(upstream)]
    if len(rows)!=1474:raise ValueError('expected all 1474 sources')
    validate_rows(rows);partition_records(rows)
    DATA_ROOT.mkdir(parents=True,exist_ok=False)
    save_verified(DATA_ROOT/'manifest.jsonl',b''.join((json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n').encode() for r in rows))
    save_verified(DATA_ROOT/'attributions.json',(upstream.parent/'attributions.json').read_bytes())
    audit=dict(schema=3,contract=CONTRACT,status='verified_1474_source_relabel_no_weights_reused',
        source_files=len(rows),upstream_manifest=str(upstream),upstream_manifest_sha256=digest_file(upstream),
        upstream_audit_sha256=digest_file(upstream.parent/'audit.json'),upstream_counts=base['counts'],
        manifest_sha256=digest_file(DATA_ROOT/'manifest.jsonl'),attributions_sha256=digest_file(DATA_ROOT/'attributions.json'),
        config_sha256=digest_file(CONFIG_FILE),class_map_sha256=digest_file(CLASS_FILE),counts=audit_counts(rows),
        subtypes=dict(Counter(r['purpose']+'/'+r['negative_subtype'] for r in rows)),
        speech_label_evidence=dict(Counter(r['purpose']+'/'+r['speech_evidence'] for r in rows if r['target_index']==4)),
        noise_bank_sources=sum(noise_eligible(r) for r in rows),
        source_identities_splits_hashes_preserved=True,ordinary_speech_removed_from_background=True,
        crying_screaming_policy='retained as documented nonordinary background; excluded from augmentation noise',
        limitations=base['limitations']+['Inherited ordinary_speech subtype includes laughter, generic voice, whispering and other vocal weak labels; explicit Speech/LibriSpeech assessed separately.',
            'Validation is disjoint from training but reused for epoch/threshold selection, not an independent acceptance test.'],
        d5_audio_read=False,d5_features_read=False,live_replay_used=False,downloaded_files=0)
    save_verified(DATA_ROOT/'audit.json',json_bytes(audit))
    return verify_dataset()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();result=verify_dataset() if args.verify_only else prepare()
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
