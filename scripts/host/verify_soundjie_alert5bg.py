"""Reload selected checkpoint and serialized six-logit weights; verify the full source chain."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.host.acquire_soundjie_alert6 import read_rows
from scripts.host.acquire_soundjie_alert5bg import verify_dataset
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.soundjie_alert5bg import CONTRACT,LABELS,CLASS_FILE,DATA_ROOT,OUTPUT_ROOT,make_model,metrics,apply_thresholds,noise_eligible
from scripts.host.train_soundjie_alert5bg import load_cache,write_json,implementation_hashes
from scripts.host.train_soundjie_alert6 import predict,softmax
from scripts.host.verify_soundjie_alert4bg import fused_logits


def verify():
    torch.set_num_threads(4); dataset=verify_dataset();sources=read_rows(DATA_ROOT/'manifest.jsonl')
    intersections={}
    for field in ('recording_id','sampling_identity','source_group','uploader_group','sha256'):
        sets=[{r[field] for r in sources if r['purpose']==p and r.get(field)} for p in ('train','validation')]
        intersections[field]=len(sets[0]&sets[1])
        if intersections[field]:raise ValueError('source separation failure '+field)
    run=json.loads((OUTPUT_ROOT/'training/result.json').read_bytes())
    x,y,purpose,rows,audit=load_cache(run['features_directory'])
    if run['actual_torch_threads']!=4 or run['config']['training']['epochs']!=45:raise ValueError('training protocol drift')
    if run['implementation_sha256']!=implementation_hashes() or audit['implementation_sha256']!=run['implementation_sha256']:
        raise ValueError('implementation drift')
    history=json.loads((OUTPUT_ROOT/'training/history.json').read_bytes())
    if [r['epoch'] for r in history]!=list(range(46)):raise ValueError('incomplete training')
    selected=run['artifacts']['selected'];directory=Path(selected['directory']);checkpoint=directory/'model.pt'
    if digest_file(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('checkpoint drift')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ckpt['labels']!=list(LABELS) or ckpt['contract']!=CONTRACT or ckpt['class_map_sha256']!=digest_file(CLASS_FILE):raise ValueError('checkpoint class drift')
    if ckpt['initialization']!='random_all_layers':raise ValueError('initialization differs')
    model=make_model();model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
    mask=purpose=='validation';vx=x[mask];vy=y[mask];val=[r for r in rows if r['purpose']=='validation']
    z=predict(model,vx);thresholds=selected['metrics']['thresholds'];pred=apply_thresholds(softmax(z),thresholds)
    for name,values in [('raw',z.argmax(1)),('thresholded',pred)]:
        if metrics(vy,values,val)!=selected['metrics'][name]:raise ValueError('reloaded metrics differ')
    with np.load(directory/'validation-logits.npz',allow_pickle=False) as recorded:
        if not np.array_equal(z,recorded['logits']) or not np.array_equal(vy,recorded['y']):raise ValueError('reloaded logits differ')
    export=OUTPUT_ROOT/'export';manifest=json.loads((export/'export-manifest.json').read_bytes())
    if manifest['contract']!=CONTRACT or manifest['checkpoint_sha256']!=selected['checkpoint_sha256']:raise ValueError('export binding differs')
    for name,sha in manifest['files'].items():
        if digest_file(export/name)!=sha:raise ValueError('export file differs '+name)
    with np.load(export/'fused-weights.npz',allow_pickle=False) as data:weights={k:data[k] for k in data.files}
    if weights['fc8.weight'].shape!=(6,2304) or weights['fc8.bias'].shape!=(6,):raise ValueError('serialized physical shape differs')
    fused=np.concatenate([fused_logits(weights,vx[i:i+64]) for i in range(0,len(vx),64)])
    error=float(np.max(np.abs(fused-z)))
    if error>1e-4 or not np.array_equal(fused.argmax(1),z.argmax(1)):raise ValueError('serialized fused parity failed')
    cal=np.load(export/'calibration-train.npy',allow_pickle=False);cal_rows=manifest['calibration_sources']
    if cal.shape!=(192,1,64,96) or len(cal_rows)!=192 or len({r['sampling_identity'] for r in cal_rows})!=192:raise ValueError('calibration identity differs')
    if Counter(r['target_index'] for r in cal_rows)!={c:32 for c in range(6)}:raise ValueError('calibration balance differs')
    for feature,row in zip(cal,cal_rows):
        if row['purpose']!='train' or row['derivation'] is not None or row not in rows or hashlib.sha256(feature.tobytes()).hexdigest()!=row['feature_sha256']:
            raise ValueError('calibration provenance differs')
    allowed={r['recording_id'] for r in sources if noise_eligible(r)}
    if set(audit['noise_bank_recording_ids'])!=allowed:raise ValueError('noise bank differs')
    subgroups={}
    for subset in ('explicit_speech','inherited_vocal_weak_label'):
        m=np.array([r.get('speech_evidence')==subset for r in val])
        subgroups[subset]=dict(support=int(m.sum()),raw_correct=int(np.sum(z.argmax(1)[m]==4)),thresholded_correct=int(np.sum(pred[m]==4)),
            raw_recall=float(np.mean(z.argmax(1)[m]==4)),thresholded_recall=float(np.mean(pred[m]==4)))
    return dict(schema=3,contract=CONTRACT,status='verified_fp32_candidate_and_export',dataset=dataset,
        source_intersections=intersections,window_counts=audit['window_counts'],skipped_augmentations=audit['skipped_augmentations'],
        manifest_sha256=dataset['manifest_sha256'],cache_sha256=audit['cache_sha256'],
        checkpoint_path=str(checkpoint),checkpoint_sha256=selected['checkpoint_sha256'],selected_epoch=selected['epoch'],
        epochs_trained=45,actual_threads=4,epoch0=run['epoch0_random_baseline'],metrics=selected['metrics'],speech_subgroups=subgroups,
        exact_checkpoint_reload=True,serialized_fused_max_abs_error=error,serialized_fused_top1_identical=True,
        export_manifest_path=str(export/'export-manifest.json'),export_manifest_sha256=digest_file(export/'export-manifest.json'),
        export_hashes=manifest['files'],calibration_counts=[32]*6,noise_bank_sources=len(allowed),
        d5_audio_read=False,d5_features_read=False,live_replay_used=False,board_deployed=False,independent_acceptance=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--publish',action='store_true');a=p.parse_args()
    report=verify()
    if a.publish:write_json(OUTPUT_ROOT/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
