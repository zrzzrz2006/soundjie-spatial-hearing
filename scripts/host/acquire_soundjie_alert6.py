"""Create-only expanded six-class dataset. No heldout audio reads or board writes."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
import zipfile

from scripts.host import acquire_soundjie_fsd50k as fsd
from scripts.host.build_soundjie_d4_manifest import partition_records, digest_file
from scripts.host.soundjie_alert6 import LABELS, ROOT, classify_fsd, map_legacy, validate_rows

OUTPUT=Path('F:/MLK/DPU_ZYNQ7000/datasets/soundjie_alert6_20261031')
LIMITS={'alarm':180,'siren':100,'horn':140,'glass':200,'knock':140,'doorbell':80,
        'ordinary_speech':200,'crying_screaming':140,'road_background':120,'ambient':120}


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def select_legacy(rows):
    result=[]
    for purpose in ('train','validation'):
        for target in range(8):
            candidates=[r for r in rows if r['purpose']==purpose and r['target_index']==target]
            candidates.sort(key=lambda r: fsd.digest(('alert6:'+r['recording_id']).encode()))
            cap=(128 if purpose=='train' else 48) if target>=6 else len(candidates)
            result.extend(map_legacy(r) for r in candidates[:cap])
    return result


def plan_fsd(rows, metadata, old, config):
    old_ids,_=fsd.exclusions(old)
    old_groups={fsd.uploader_group(metadata[f]['uploader']) for f in old_ids
                if f in metadata and metadata[f].get('uploader')}
    buckets=defaultdict(list);rejected=Counter()
    for row in rows:
        choice=classify_fsd(row['labels'])
        if choice is None:continue
        index,subtype=choice;fid=row['fname'];meta=metadata[fid]
        lic=fsd.license_name(meta.get('license',''),config)
        if not lic or not meta.get('uploader'):rejected['license_or_uploader']+=1;continue
        group=fsd.uploader_group(meta['uploader'])
        if fid in old_ids or group in old_groups:rejected['old_id_or_known_uploader']+=1;continue
        result=dict(freesound_id=fid,original_labels=row['labels'],original_mids=row['mids'],
                    official_split=row['split'],target_index=index,target_id=LABELS[index],
                    negative_subtype=subtype,source_group=group,purpose=fsd.purpose_for(group,config),
                    author=meta['uploader'],license=lic,license_url=meta['license'],title=meta.get('title',''),
                    original_url='https://freesound.org/people/'+quote(meta['uploader'],safe='')+'/sounds/'+fid+'/')
        buckets[(subtype,group)].append(result)
    for values in buckets.values():values.sort(key=lambda r:fsd.digest(('alert6:'+r['freesound_id']).encode()))
    selected=[];counts=Counter()
    for rank in range(4):
        for key in sorted(buckets,key=lambda k:fsd.digest(str(k).encode())):
            if rank<len(buckets[key]) and counts[key[0]]<LIMITS[key[0]]:
                selected.append(buckets[key][rank]);counts[key[0]]+=1
    return selected,dict(rejected),dict(old_ids_with_uploader=sum(i in metadata for i in old_ids),
             old_ids_without_uploader=sum(i not in metadata for i in old_ids))


def acquire_row(row,entry,root,cache,config,revisions):
    # Existing sources are reused in place only after matching the official member.
    existing=cache/'audio'/(row['freesound_id']+'.wav')
    selected_root=cache if existing.exists() else root
    return fsd.acquire_one(row,entry,selected_root,config,revisions)[0]


def user_rows():
    base=ROOT/'artifacts/user-horns-20261029a'
    result=[]
    for row in read_rows(base/'source-manifest.jsonl'):
        index=2 if row['user_label']=='vehicle_horn' else 5
        path=base/row['wav16k']
        if digest_file(path)!=row['wav16k_sha256']:raise ValueError('user source drift')
        result.append(dict(recording_id='user:'+row['recording_id'],source_id='user_recording',
            source_group=row['source_group'],purpose='train',split='train',domain='direct',
            target_index=index,target_id=LABELS[index],negative_subtype='user_horn' if index==2 else 'road_background',
            path=str(path),sha256=row['wav16k_sha256'],license='user-authorized-project-recording',
            original_url='user-provided:'+row['original_name'],parent_recording_id=None,parent_sha256=None,
            semantic_annotation_source='user',human_audition_performed=False,
            independent_unseen_acceptance=False,prior_exposure='user_adaptation_session',
            temporal_annotations='energy_selection_only_not_manual_onset',duration_seconds=row['duration_s']))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
    config=json.loads((ROOT/'config/soundjie_fsd50k_sources.json').read_text(encoding='utf-8'))
    cache=Path(config['dataset_root']);root=OUTPUT
    if root.resolve().drive.lower()!='f:':raise ValueError('bulk output must stay on F')
    if (root/'audit.json').exists():
        audit=json.loads((root/'audit.json').read_text(encoding='utf-8'))
        if digest_file(root/'manifest.jsonl')!=audit['manifest_sha256']:raise ValueError('published manifest drift')
        print(json.dumps(audit),flush=True);return
    root.mkdir(parents=True,exist_ok=True);(root/'audio').mkdir(exist_ok=True)
    old=read_rows(config['frozen_manifest']);partition_records(old)
    gt,meta,metadata_evidence=fsd.load_metadata(cache,config)
    with zipfile.ZipFile(cache/'FSD50K.metadata.zip') as z:
        meta.update(json.loads(z.read('FSD50K.metadata/eval_clips_info_FSD50K.json')))
    entries,directory=fsd.load_directory(cache,config)
    selected,rejected,historical=plan_fsd(gt,meta,old,config)
    legacy=select_legacy(old);users=user_rows()
    plan_path=root/'selection-plan.json'
    if plan_path.exists():
        plan=json.loads(plan_path.read_text(encoding='utf-8'))
        if plan['candidates']!=selected:raise ValueError('frozen selection changed')
        revisions=plan['revisions']
    else:
        revisions={repo:json.loads(fsd.get_bytes('https://huggingface.co/api/datasets/'+repo,2_000_000)[0])['sha'] for repo in config['mirrors']}
        plan=dict(schema=1,candidates=selected,revisions=revisions,limits=LIMITS,
                  metadata=metadata_evidence,official_directory=directory,rejected=rejected,historical=historical,
                  old_manifest_sha256=digest_file(config['frozen_manifest']),
                  class_map_sha256=digest_file(ROOT/'config/soundjie_alert6_classes.json'),
                  split_seed=config['split_seed'],validation_fraction=config['validation_fraction'])
        fsd.save_verified(plan_path,fsd.json_bytes(plan))
    print(json.dumps(dict(stage='plan',legacy=len(legacy),user=len(users),new_pool=len(selected),
                         selected_counts=dict(Counter(r['negative_subtype'] for r in selected)))),flush=True)
    if args.plan_only:return
    acquired=[];failures=[]
    with ThreadPoolExecutor(max_workers=min(max(args.workers,1),8)) as pool:
        futures={pool.submit(acquire_row,r,entries[r['freesound_id']],root,cache,config,revisions):r for r in selected}
        for i,future in enumerate(as_completed(futures),1):
            try:acquired.append(future.result())
            except Exception as error:failures.append(dict(id=futures[future]['freesound_id'],error=str(error)))
            if i%25==0 or i==len(futures):print(json.dumps(dict(stage='download',done=i,total=len(futures),ok=len(acquired),failed=len(failures))),flush=True)
    # Full-file and PCM duplicates are not independent sources. Do not move a source
    # across the frozen split in order to retain a duplicate.
    seen_hash={r['sha256'] for r in old};seen_pcm=set();duplicates=[];clean=[]
    for row in sorted(acquired,key=lambda r:r['recording_id']):
        pcm=row['quality']['pcm_sha256']
        if row['sha256'] in seen_hash or pcm in seen_pcm:duplicates.append(row['recording_id']);continue
        seen_hash.add(row['sha256']);seen_pcm.add(pcm);clean.append(row)
    combined=sorted(legacy+clean+users,key=lambda r:r['recording_id'])
    validate_rows(combined);partition_records(combined)
    # Check against excluded old purpose groups without relabelling the holdout.
    partition_records(old+clean+users)
    manifest=b''.join((json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n').encode() for r in combined)
    fsd.save_verified(root/'manifest.jsonl',manifest)
    fsd.save_verified(root/'failures.json',fsd.json_bytes(failures))
    fsd.save_verified(root/'attributions.json',fsd.json_bytes([dict(recording_id=r['recording_id'],
        original_url=r['original_url'],license=r['license'],author=r.get('author'),
        title=r.get('title'),license_url=r.get('license_url')) for r in combined]))
    audit=dict(schema=1,status='sources_acquired_not_trained',manifest_sha256=fsd.digest(manifest),
        selection_plan_sha256=digest_file(plan_path),source_count=len(combined),
        legacy_sources=len(legacy),fsd_sources=len(clean),user_adaptation_sources=len(users),
        counts=dict(Counter(r['purpose']+'/'+r['target_id'] for r in combined)),
        subtypes=dict(Counter(r['purpose']+'/'+r['negative_subtype'] for r in combined)),
        group_counts={name:len({r['source_group'] for r in combined if r['target_id']==name}) for name in LABELS},
        duplicates_excluded=duplicates,download_failures=failures,historical_uploader_audit=historical,
        limitations=['weak clip labels; no manual event timestamps','unknown historical uploaders remain',
                      'user horn files share one adaptation session; not independent acceptance',
                      'generic siren labels are not ambulance subclass truth'],
        training_holdout_audio_read=False,competition_use=config['competition_use'])
    fsd.save_verified(root/'audit.json',fsd.json_bytes(audit));print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
