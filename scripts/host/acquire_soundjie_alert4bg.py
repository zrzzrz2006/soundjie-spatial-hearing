"""Create-only v2 data sidecar; reuse only verified files and frozen purposes."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from urllib.parse import quote
import zipfile
from scripts.host import acquire_soundjie_fsd50k as fsd
from scripts.host.acquire_soundjie_alert6 import read_rows
from scripts.host.build_soundjie_d4_manifest import digest_file, partition_records
from scripts.host.soundjie_alert4bg import (
    ROOT,LABELS,CONTRACT,CONFIG_FILE,CLASS_FILE,DATA_ROOT,
    classify_fsd,map_existing,choose_purpose,validate_rows,
)


def known_uploader(row,metadata):
    if row.get('author'):return fsd.uploader_group(row['author'])
    ids,_=fsd.exclusions([row])
    groups={fsd.uploader_group(metadata[i]['uploader']) for i in ids if i in metadata and metadata[i].get('uploader')}
    if len(groups)>1:raise ValueError('one source has multiple uploader identities')
    return next(iter(groups)) if groups else None


def freeze_groups(historical,metadata):
    frozen=defaultdict(set)
    for row in historical:
        group=known_uploader(row,metadata)
        if group:frozen[group].add(row['purpose'])
    return frozen


def plan_sources(existing,historical,gt,metadata,config,fsd_config):
    frozen=freeze_groups(historical,metadata);excluded_ids,_=fsd.exclusions(historical)
    rejected=[];legacy=[]
    for row in existing:
        mapped=map_existing(row);group=known_uploader(row,metadata)
        reason=None
        if row['purpose'] not in ('train','validation'):reason='forbidden_purpose'
        elif row['license'] not in ('CC0-1.0','CC-BY-3.0','CC-BY-4.0','user-authorized-project-recording'):reason='license_filter'
        elif mapped is None:reason='ambiguous_semantics'
        elif group and choose_purpose(group,frozen,row['purpose']) != row['purpose']:reason='historical_uploader_conflict'
        if reason:rejected.append(dict(recording_id=row['recording_id'],reason=reason));continue
        mapped.update(uploader_group=group,source_reuse='existing_read_only_verified',
                      previous_source_group=row['source_group'])
        legacy.append(mapped)
    buckets=defaultdict(list);stats=Counter()
    for row in gt:
        if 'Bark' not in row['labels']:continue
        stats['official_dev_bark']+=1;fid=row['fname'];meta=metadata[fid]
        lic=fsd.license_name(meta.get('license',''),fsd_config)
        if not lic:stats['excluded_license']+=1;continue
        stats['dev_bark_allowed_license']+=1
        if classify_fsd(row['labels']) != (3,'bark'):stats['excluded_mixed_labels']+=1;continue
        if fid in excluded_ids:stats['excluded_historical_id']+=1;continue
        if not meta.get('uploader','').strip():stats['excluded_missing_uploader']+=1;continue
        group=fsd.uploader_group(meta['uploader'])
        purpose=choose_purpose(group,frozen,fsd.purpose_for(group,fsd_config))
        if purpose is None:stats['excluded_historical_uploader_purpose']+=1;continue
        candidate=dict(freesound_id=fid,original_labels=row['labels'],original_mids=row['mids'],
            official_split=row['split'],target_index=3,target_id=LABELS[3],negative_subtype='bark',
            source_group=group,uploader_group=group,purpose=purpose,author=meta['uploader'],
            license=lic,license_url=meta['license'],title=meta.get('title',''),
            original_url='https://freesound.org/people/'+quote(meta['uploader'],safe='')+'/sounds/'+fid+'/',
            sampling_identity='freesound:'+fid,class_contract=CONTRACT,
            split_decision='retained_known_uploader' if group in frozen else 'frozen_uploader_hash')
        buckets[group].append(candidate)
    for rows in buckets.values():rows.sort(key=lambda r:fsd.digest(('alert4bg:'+r['freesound_id']).encode()))
    selected=[]
    for rank in range(config['max_bark_per_uploader']):
        for group in sorted(buckets,key=lambda g:fsd.digest(g.encode())):
            if rank<len(buckets[group]) and len(selected)<config['bark_limit']:selected.append(buckets[group][rank])
    stats['eligible_bark_before_diversity_cap']=sum(map(len,buckets.values()))
    stats['selected_bark']=len(selected);stats['selected_bark_uploaders']=len({r['source_group'] for r in selected})
    return legacy,selected,rejected,dict(stats),{g:sorted(v) for g,v in sorted(frozen.items())}


def acquire(row,entry,root,caches,config,revisions):
    # acquire_one never writes when the exact existing member is present.
    directory=next((p for p in caches if (p/'audio'/(row['freesound_id']+'.wav')).is_file()),root)
    result,errors=fsd.acquire_one(row,entry,directory,config,revisions)
    result['source_reuse']='existing_read_only_verified' if directory!=root else 'new_bounded_download'
    return result,errors


def audit_counts(rows):
    result={}
    for purpose in ('train','validation'):
        result[purpose]={}
        for c,label in enumerate(LABELS):
            part=[r for r in rows if r['purpose']==purpose and r['target_index']==c]
            result[purpose][label]=dict(files=len(part),original_recordings=len({r['sampling_identity'] for r in part}),
                source_groups=len({r['source_group'] for r in part}),known_uploaders=len({r['uploader_group'] for r in part if r.get('uploader_group')}))
    return result


def verify_dataset(root=DATA_ROOT):
    audit=json.loads((root/'audit.json').read_bytes());rows=read_rows(root/'manifest.jsonl')
    for name,key in [('manifest.jsonl','manifest_sha256'),('selection-plan.json','selection_plan_sha256'),('attributions.json','attributions_sha256')]:
        if digest_file(root/name)!=audit[key]:raise ValueError('published evidence drift '+name)
    if digest_file(CONFIG_FILE)!=audit['config_sha256'] or digest_file(CLASS_FILE)!=audit['class_map_sha256']:raise ValueError('configuration drift')
    validate_rows(rows);partition_records(rows)
    for row in rows:
        if digest_file(row['path'])!=row['sha256']:raise ValueError('source drift '+row['recording_id'])
        if row['source_id']=='fsd50k':fsd.verify_member(Path(row['path']).read_bytes(),row['official_archive_member'])
    if audit_counts(rows)!=audit['counts']:raise ValueError('count drift')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--verify-only',action='store_true');parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args();config=json.loads(CONFIG_FILE.read_bytes())
    if Path(config['dataset_root']).resolve()!=DATA_ROOT.resolve():raise ValueError('unauthorized write root')
    if args.verify_only or (DATA_ROOT/'audit.json').exists():print(json.dumps(verify_dataset()),flush=True);return
    fsd_config=json.loads((ROOT/'config/soundjie_fsd50k_sources.json').read_bytes())
    cache=Path(fsd_config['dataset_root']);old_path=Path(config['existing_manifest'])
    # Missing official metadata is an error, never an implicit write into the old dataset.
    for name in list(fsd_config['metadata_md5'])+['official-dev-tail.bin','official-dev-central.bin']:
        if not (cache/name).is_file():raise FileNotFoundError(cache/name)
    gt,metadata,metadata_evidence=fsd.load_metadata(cache,fsd_config)
    with zipfile.ZipFile(cache/'FSD50K.metadata.zip') as z:metadata.update(json.loads(z.read('FSD50K.metadata/eval_clips_info_FSD50K.json')))
    entries,directory=fsd.load_directory(cache,fsd_config)
    existing=read_rows(old_path);historical=read_rows(fsd_config['frozen_manifest'])
    seen={r['recording_id'] for r in historical};historical += [r for r in existing if r['recording_id'] not in seen]
    legacy,selected,rejected,stats,frozen=plan_sources(existing,historical,gt,metadata,config,fsd_config)
    old_plan=json.loads((old_path.parent/'selection-plan.json').read_bytes())
    plan=dict(schema=2,contract=CONTRACT,config=config,config_sha256=digest_file(CONFIG_FILE),class_map_sha256=digest_file(CLASS_FILE),
        upstream_manifests={str(p):digest_file(p) for p in (old_path,Path(fsd_config['frozen_manifest']))},
        metadata=metadata_evidence,official_directory=directory,candidates=selected,
        retained_existing_ids=[r['recording_id'] for r in legacy],rejected_existing=rejected,
        bark_selection=stats,frozen_uploaders=frozen,mirror_revisions=old_plan['revisions'],
        split_seed=fsd_config['split_seed'],validation_fraction=fsd_config['validation_fraction'])
    DATA_ROOT.mkdir(parents=True,exist_ok=True);(DATA_ROOT/'audio').mkdir(exist_ok=True)
    fsd.save_verified(DATA_ROOT/'selection-plan.json',fsd.json_bytes(plan))
    print(json.dumps(dict(stage='frozen_plan',retained_existing=len(legacy),excluded_existing=dict(Counter(r['reason'] for r in rejected)),**stats)),flush=True)
    if args.plan_only:return
    for row in legacy:
        if digest_file(row['path'])!=row['sha256']:raise ValueError('existing source drift')
        if row['source_id']=='fsd50k':fsd.verify_member(Path(row['path']).read_bytes(),entries[row['freesound_id']])
    acquired=[];failures=[];fallback=[]
    with ThreadPoolExecutor(max_workers=min(8,max(1,args.workers))) as pool:
        jobs={pool.submit(acquire,r,entries[r['freesound_id']],DATA_ROOT,[old_path.parent,cache],fsd_config,plan['mirror_revisions']):r for r in selected}
        for i,future in enumerate(as_completed(jobs),1):
            try:
                row,errors=future.result();acquired.append(row);fallback.extend(errors)
            except Exception as error:failures.append(dict(freesound_id=jobs[future]['freesound_id'],error=str(error)))
            if i%10==0 or i==len(jobs):print(json.dumps(dict(stage='download',done=i,total=len(jobs),ok=len(acquired),failed=len(failures))),flush=True)
    # Metadata-only D5 exclusion: never open a historical nontraining source.
    _,hashes=fsd.exclusions(historical);seen_pcm={r['quality']['pcm_sha256'] for r in legacy if r.get('quality')}
    clean=[];duplicates=[]
    for row in sorted(acquired,key=lambda r:r['recording_id']):
        if row['sha256'] in hashes or row['quality']['pcm_sha256'] in seen_pcm:duplicates.append(row['recording_id']);continue
        hashes.add(row['sha256']);seen_pcm.add(row['quality']['pcm_sha256']);clean.append(row)
    combined=sorted(legacy+clean,key=lambda r:r['recording_id']);validate_rows(combined);partition_records(combined)
    partition_records(historical+clean)
    manifest=b''.join((json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n').encode() for r in combined)
    fsd.save_verified(DATA_ROOT/'manifest.jsonl',manifest)
    fsd.save_verified(DATA_ROOT/'failures.json',fsd.json_bytes(dict(failures=failures,fallback_errors=fallback,duplicates_excluded=duplicates)))
    attribution=[{k:r.get(k) for k in ('recording_id','original_url','license','license_url','author','title','sha256','path')} for r in combined]
    fsd.save_verified(DATA_ROOT/'attributions.json',fsd.json_bytes(attribution))
    audit=dict(schema=2,contract=CONTRACT,status='verified_sources_not_yet_trained',
        manifest_sha256=fsd.digest(manifest),selection_plan_sha256=digest_file(DATA_ROOT/'selection-plan.json'),
        attributions_sha256=digest_file(DATA_ROOT/'attributions.json'),config_sha256=digest_file(CONFIG_FILE),class_map_sha256=digest_file(CLASS_FILE),
        counts=audit_counts(combined),source_files=len(combined),existing_source_files=len(legacy),new_bark_source_files=len(clean),
        subtypes=dict(Counter(r['purpose']+'/'+r['negative_subtype'] for r in combined)),
        licenses=dict(Counter(r['license'] for r in combined)),bark_selection=stats,
        rejected_existing=dict(Counter(r['reason'] for r in rejected)),failures=len(failures),duplicates_excluded=duplicates,
        historical_unknown_uploader_records=sum(known_uploader(r,metadata) is None for r in historical),
        d5_audio_read=False,d5_features_read=False,development_replay_used=False,competition_use=fsd_config['competition_use'],
        limitations=['weak clip labels; energy windows are not human onset annotations',
                    'historic uploader metadata unavailable for some excluded sources; known uploader separation enforced',
                    'LibriSpeech chapter utterances are not independent original recordings',
                    'user horn recordings share one authorized development session; no independent acceptance'])
    fsd.save_verified(DATA_ROOT/'audit.json',fsd.json_bytes(audit))
    verify_dataset();print(json.dumps(dict(stage='dataset_verified',**audit)),flush=True)


if __name__=='__main__':main()
