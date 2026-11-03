"""Read-only audit of published sources, reloaded checkpoint and serialized export.

--publish writes a new verification.json under the authorized artifact root.
No D5 audio/features, device commands or playback are used.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from scripts.host import acquire_soundjie_fsd50k as fsd
from scripts.host.acquire_soundjie_alert6 import read_rows
from scripts.host.acquire_soundjie_alert4bg import verify_dataset,known_uploader,freeze_groups
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.soundjie_alert4bg import ROOT,CONTRACT,LABELS,CLASS_FILE,CONFIG_FILE,DATA_ROOT,OUTPUT_ROOT,make_model,classify_fsd,choose_purpose,metrics,apply_thresholds
from scripts.host.train_soundjie_alert4bg import load_cache,write_json,implementation_hashes
from scripts.host.train_soundjie_alert6 import predict,softmax


def fused_logits(weights,x):
    values=torch.from_numpy(x)
    with torch.no_grad():
        for c in range(1,5):
            values=torch.nn.functional.conv2d(values,torch.from_numpy(weights[f'conv{c}.weight']),torch.from_numpy(weights[f'conv{c}.bias']),padding=1)
            values=torch.nn.functional.max_pool2d(torch.relu(values),2)
        return torch.nn.functional.linear(values.flatten(1),torch.from_numpy(weights['fc8.weight']),torch.from_numpy(weights['fc8.bias'])).numpy()


def verify():
    torch.set_num_threads(4);dataset=verify_dataset();rows=read_rows(DATA_ROOT/'manifest.jsonl')
    plan=json.loads((DATA_ROOT/'selection-plan.json').read_bytes())
    for path,sha in plan['upstream_manifests'].items():
        if digest_file(path)!=sha:raise ValueError('upstream manifest drift')
    fsd_config=json.loads((ROOT/'config/soundjie_fsd50k_sources.json').read_bytes());cache=Path(fsd_config['dataset_root'])
    for name in fsd_config['metadata_md5']:
        if not (cache/name).is_file():raise FileNotFoundError(cache/name)
    gt,metadata,metadata_evidence=fsd.load_metadata(cache,fsd_config);gt={r['fname']:r for r in gt}
    with zipfile.ZipFile(cache/'FSD50K.metadata.zip') as z:metadata.update(json.loads(z.read('FSD50K.metadata/eval_clips_info_FSD50K.json')))
    if metadata_evidence!=plan['metadata']:raise ValueError('metadata drift')
    existing=read_rows(json.loads(CONFIG_FILE.read_bytes())['existing_manifest'])
    historical=read_rows(fsd_config['frozen_manifest']);ids={r['recording_id'] for r in historical}
    historical += [r for r in existing if r['recording_id'] not in ids]
    frozen=freeze_groups(historical,metadata);historical_by_id={r['recording_id']:r for r in historical}
    forbidden_hashes={r['sha256'] for r in historical if r['purpose'] not in ('train','validation')}
    for row in rows:
        if row['sha256'] in forbidden_hashes:raise ValueError('heldout hash reused')
        old=historical_by_id.get(row['recording_id'])
        if old and (old['purpose']!=row['purpose'] or old['sha256']!=row['sha256']):raise ValueError('frozen source or split changed')
        if row['source_id']=='fsd50k':
            fid=row['freesound_id'];official=gt[fid];meta=metadata[fid]
            if row['original_labels']!=official['labels'] or row['original_mids']!=official['mids']:raise ValueError('official label drift')
            if classify_fsd(official['labels'])!=(row['target_index'],row['negative_subtype']):raise ValueError('semantic drift')
            if row['author']!=meta['uploader'] or row['license']!=fsd.license_name(meta['license'],fsd_config):raise ValueError('official attribution drift')
            group=known_uploader(row,metadata)
            if row['source_group']!=group or row['uploader_group']!=group:raise ValueError('uploader identity drift')
            if choose_purpose(group,frozen,fsd.purpose_for(group,fsd_config))!=row['purpose']:raise ValueError('frozen uploader changed')
    intersections={}
    for field in ('recording_id','sampling_identity','source_group','uploader_group','sha256'):
        sets=[{r[field] for r in rows if r['purpose']==p and r.get(field)} for p in ('train','validation')]
        intersections[field]=len(sets[0]&sets[1])
        if intersections[field]:raise ValueError('source separation failure '+field)
    run=json.loads((OUTPUT_ROOT/'training/result.json').read_bytes())
    x,y,purpose,provenance,audit=load_cache(run['features_directory'])
    if run['actual_torch_threads']!=4 or run['config']['training']['epochs']!=45:raise ValueError('training protocol drift')
    if run['implementation_sha256']!=implementation_hashes():raise ValueError('implementation drift since training')
    if audit['implementation_sha256']!=run['implementation_sha256']:raise ValueError('implementation changed after feature preparation')
    history=json.loads((OUTPUT_ROOT/'training/history.json').read_bytes())
    if [r['epoch'] for r in history]!=list(range(46)):raise ValueError('incomplete training history')
    selected=run['artifacts']['selected'];directory=Path(selected['directory'])
    checkpoint=directory/'model.pt'
    if digest_file(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('checkpoint drift')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ckpt['labels']!=list(LABELS) or ckpt['contract']!=CONTRACT or ckpt['class_map_sha256']!=digest_file(CLASS_FILE):raise ValueError('checkpoint class order drift')
    model=make_model();model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
    mask=purpose=='validation';vx=x[mask];vy=y[mask];val_rows=[r for r in provenance if r['purpose']=='validation']
    logits=predict(model,vx);scores=softmax(logits);thresholds=selected['metrics']['thresholds'];pred=apply_thresholds(scores,thresholds)
    for name,values in [('raw',logits.argmax(1)),('thresholded',pred)]:
        if metrics(vy,values,val_rows)!=selected['metrics'][name]:raise ValueError('reloaded metrics drift')
    export=OUTPUT_ROOT/'export';manifest=json.loads((export/'export-manifest.json').read_bytes())
    if manifest['contract']!=CONTRACT or manifest['checkpoint_sha256']!=selected['checkpoint_sha256']:raise ValueError('export binding drift')
    for name,sha in manifest['files'].items():
        if digest_file(export/name)!=sha:raise ValueError('export file drift '+name)
    with np.load(export/'fused-weights.npz',allow_pickle=False) as data:weights={k:data[k] for k in data.files}
    if weights['fc8.weight'].shape!=(5,2304) or weights['fc8.bias'].shape!=(5,):raise ValueError('serialized output shape drift')
    fused=np.concatenate([fused_logits(weights,vx[i:i+64]) for i in range(0,len(vx),64)])
    error=float(np.max(np.abs(fused-logits)))
    if error>1e-4 or not np.array_equal(fused.argmax(1),logits.argmax(1)):raise ValueError('serialized fused parity failed')
    cal=np.load(export/'calibration-train.npy',allow_pickle=False);cal_rows=manifest['calibration_sources']
    if cal.shape!=(160,1,64,96) or len(cal_rows)!=160 or len({r['sampling_identity'] for r in cal_rows})!=160:raise ValueError('calibration shape/identity drift')
    for feature,row in zip(cal,cal_rows):
        if row['purpose']!='train' or row['derivation'] is not None or hashlib.sha256(feature.tobytes()).hexdigest()!=row['feature_sha256']:raise ValueError('calibration provenance drift')
    subgroup={}
    for subtype in sorted({r['negative_subtype'] for r in val_rows}):
        m=np.array([r['negative_subtype']==subtype for r in val_rows])
        subgroup[subtype]=dict(source_files=int(m.sum()),raw_recall=float(np.mean(logits.argmax(1)[m]==vy[m])),thresholded_recall=float(np.mean(pred[m]==vy[m])))
    return dict(schema=2,contract=CONTRACT,status='verified_fp32_candidate_and_export',
        dataset=dataset,source_intersections=intersections,window_counts=audit['window_counts'],
        skipped_augmentations=audit['skipped_augmentations'],features_directory=run['features_directory'],
        manifest_sha256=dataset['manifest_sha256'],cache_sha256=audit['cache_sha256'],
        checkpoint_path=str(checkpoint),checkpoint_sha256=selected['checkpoint_sha256'],
        selected_epoch=selected['epoch'],epochs_trained=45,actual_threads=4,
        epoch0=run['epoch0_random_baseline'],metrics=selected['metrics'],validation_subtypes=subgroup,
        serialized_fused_max_abs_error=error,serialized_fused_top1_identical=True,
        export_manifest_path=str(export/'export-manifest.json'),export_manifest_sha256=digest_file(export/'export-manifest.json'),
        export_hashes=manifest['files'],calibration_counts=[32]*5,
        feature_source_counts=Counter(r['purpose'] for r in rows),
        d5_audio_read=False,d5_features_read=False,live_replay_used=False,
        Caffe_verified=False,INT8_verified=False,DPU_verified=False,board_deployed=False,independent_acceptance=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    report=verify()
    if args.publish:write_json(OUTPUT_ROOT/'verification.json',report)
    print(json.dumps(dict(status=report['status'],selected_epoch=report['selected_epoch'],checkpoint_sha256=report['checkpoint_sha256'],
        source_intersections=report['source_intersections'],serialized_fused_max_abs_error=report['serialized_fused_max_abs_error'],metrics=report['metrics'])),flush=True)


if __name__=='__main__':main()
