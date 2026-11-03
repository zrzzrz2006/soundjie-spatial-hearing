"""Bounded fine-label dataset. Source titles are weak subtype labels, not audition."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.parse import quote
import zipfile
from scripts.host.acquire_soundjie_fsd50k import (parse_central_directory, parse_ground_truth,
    acquire_one, license_name, uploader_group)

def labels_for(labels,title):
    labels=set(labels);s=title.lower()
    ambulance='ambulance' in s
    other=bool(re.search(r'police|fire.?truck|fire.?engine',s))
    if 'Siren' in labels:
        if ambulance and not other: return [1,0,0],[1,0,0]
        if other and not ambulance: return [0,0,0],[1,0,0]
        return None
    if 'Traffic_noise_and_roadway_noise' in labels and not ambulance:
        return [0,1,0],[1,1,0]
    return None

def split_for(group):
    return 'validation' if int(hashlib.sha256(('traffic-v1:'+group).encode()).hexdigest()[:8],16)%5==0 else 'train'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fsd-root',type=Path,default=Path(r'F:\MLK\DPU_ZYNQ7000\datasets\soundjie_fsd50k_20261025'))
    p.add_argument('--old-manifest',type=Path,default=Path('artifacts/expanded-data-20261025a/manifest.jsonl'))
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True);(a.output/'audio').mkdir()
    cfg=json.loads((a.fsd_root/'acquisition-config.json').read_text())
    existing=[json.loads(l) for l in (a.fsd_root/'manifest.jsonl').read_text().splitlines()]
    records=[]
    for r in existing:
        lab=labels_for(r['original_labels'],r['title'])
        if lab and r['target_index']==1:
            q=dict(r,labels=lab[0],label_mask=lab[1],purpose=split_for(r['source_group']),
                   label_basis='official_siren_plus_uploader_title_weak_subtype',task='traffic_fine_v1')
            records.append(q)
    with zipfile.ZipFile(a.fsd_root/'FSD50K.metadata.zip') as z:
        meta=json.loads(z.read('FSD50K.metadata/dev_clips_info_FSD50K.json'))
    with zipfile.ZipFile(a.fsd_root/'FSD50K.ground_truth.zip') as z:
        rows=parse_ground_truth(z.read('FSD50K.ground_truth/dev.csv').decode())
    entries=parse_central_directory((a.fsd_root/'official-dev-central.bin').read_bytes())
    plan=[];authors=set()
    for row in sorted(rows,key=lambda r:int(r['fname'])):
        m=meta[row['fname']];lic=license_name(m['license'],cfg)
        lab=labels_for(row['labels'],m['title'])
        if not lab or not lab[0][1] or not lic or m['uploader'] in authors: continue
        authors.add(m['uploader']);group=uploader_group(m['uploader'])
        plan.append(dict(freesound_id=row['fname'],author=m['uploader'],title=m['title'],
            source_group=group,purpose=split_for(group),labels=lab[0],label_mask=lab[1],
            target_id='traffic_noise',target_index=1,original_labels=row['labels'],original_mids=row['mids'],
            original_url='https://freesound.org/people/'+quote(m['uploader'])+'/sounds/'+row['fname']+'/',
            license=lic,license_url=m['license'],label_basis='official_weak_clip_traffic_label',task='traffic_fine_v1'))
        if len(plan)>=30: break
    (a.output/'selection.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding='utf-8')
    revisions={'Fhrozen/FSD50k':'ccf1acaff12f3f4a4c10052dddafbb6d22152c9b'}
    errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(acquire_one,r,entries[r['freesound_id']],a.output,cfg,revisions):r for r in plan}
        for future in as_completed(futures):
            try:
                r,_=future.result();records.append(r);print('TRAFFIC_DOWNLOADED',r['freesound_id'],flush=True)
            except Exception as e:errors.append({'id':futures[future]['freesound_id'],'error':str(e)})
    counts={6:0,7:0,0:0,3:0,4:0}
    for r in map(json.loads,a.old_manifest.read_text().splitlines()):
        k=r['target_index']
        if k not in counts or counts[k]>=20 or r['domain']!='direct': continue
        counts[k]+=1
        labels=[0,0,1 if k==6 else 0]
        records.append(dict(r,labels=labels,label_mask=[1,1,1],purpose=split_for(r['source_group']),
            label_basis='existing_clean_speech_or_nonroad_background',task='traffic_fine_v1'))
    with (a.output/'manifest.jsonl').open('x',encoding='utf-8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    (a.output/'errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    summary={'records':len(records),'traffic':sum(r['labels'][1] for r in records),
        'ambulance':sum(r['labels'][0] for r in records),'download_failures':len(errors),
        'purpose_counts':{s:sum(r['purpose']==s for r in records) for s in ('train','validation')},
        'independent_unseen_acceptance':False,'label_limit':'weak clip labels and uploader-title subtypes, not human audition'}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    if summary['traffic']<12: raise RuntimeError('too few traffic sources')

if __name__=='__main__': main()
