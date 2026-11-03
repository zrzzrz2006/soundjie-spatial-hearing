"""Paired digital/acoustic host diagnostics, never an independent DPU acceptance."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.evaluate_soundjie_alert6 import BASELINE,BASELINE_SHA
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.prepare_soundjie_d3 import compute_feature
from scripts.host.soundjie_alert6 import ROOT,LABELS,make_model
from scripts.host.train_soundjie_alert6 import predict,softmax,apply_thresholds,write_json
from scripts.host.train_soundjie_d3 import SoundjieCNN,load_model_config

RUN=Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_alert6/20261031b')
CANDIDATES=[('seed-20261031','38655f978f29e9e9ecdd3d0255d41cf4cf5796b5535ed03b59353b692c34d0f8'),
            ('seed-20261101','12153514fe8dadfa2816b89bd816117942c7b1eddc01ffab74c0be397a6299dd')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alignment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    aligned=json.loads(a.alignment.read_text(encoding='utf-8'))
    if digest_file(a.alignment.parent/'capture-stereo.wav')!=aligned['recording_sha256']:
        raise ValueError('raw capture drift')
    frontend=load_frontend_config(ROOT/'config/logmel_frontend_v1.json')
    features=[];rows=[]
    events=[e for e in aligned['events'] if not e['quarantined'] and e['reception']['passed']]
    if not events:raise ValueError('no reception-verified events')
    for event in events:
        for domain,key,sha_key in [('digital','reference_wav','reference_sha256'),
                                   ('acoustic','captured_wav','captured_sha256')]:
            if digest_file(event[key])!=event[sha_key]:raise ValueError('excerpt drift')
            signal,rate=sf.read(event[key],dtype='float64')
            if rate!=16000 or signal.ndim!=1 or not 39500<=len(signal)<=40500:
                raise ValueError('paired excerpt contract failed')
            # Four equally spaced 1s windows; tail anchor accommodates measured clock drift.
            starts=np.rint(np.linspace(0,len(signal)-16000,4)).astype(int)
            for index,start in enumerate(starts):
                samples=signal[start:start+16000]
                features.append(compute_feature(samples,frontend))
                rows.append(dict(event_index=event['event_index'],domain=domain,window_index=index,
                    start_sample=int(start),recording_id=event['recording_id'],purpose=event['purpose'],
                    source_group=event['source_group'],target_index=event['target_index'],
                    negative_subtype=event['negative_subtype'],rms_dbfs=float(20*np.log10(np.sqrt(np.mean(samples**2)))),
                    scores={}))
    x=np.stack(features);models=[]
    if digest_file(BASELINE)!=BASELINE_SHA:raise ValueError('baseline drift')
    model=SoundjieCNN(load_model_config(ROOT/'config/soundjie_d3_model.json'))
    model.load_state_dict(torch.load(BASELINE,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    models.append(('baseline8',BASELINE_SHA,model,None))
    for name,sha in CANDIDATES:
        path=RUN/'candidates'/name/'soundjie_alert6_float32.pt'
        if digest_file(path)!=sha:raise ValueError('candidate drift')
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
        if ckpt['labels']!=list(LABELS):raise ValueError('candidate class order drift')
        model=make_model();model.load_state_dict(ckpt['state_dict'],strict=True)
        thresholds=json.loads((path.parent/'metrics.json').read_text(encoding='utf-8'))['thresholds']
        models.append((name,sha,model,thresholds))
    for name,sha,model,thresholds in models:
        scores=softmax(predict(model,x));raw=scores.argmax(1);mapped=np.minimum(raw,5)
        gated=apply_thresholds(scores,thresholds) if thresholds is not None else mapped
        for i,row in enumerate(rows):
            row['scores'][name]=dict(values=scores[i].tolist(),raw_top1=int(raw[i]),
                mapped_top1=int(mapped[i]),thresholded_top1=int(gated[i]))
    paired=[]
    for event in events:
        entry={k:event[k] for k in ['event_index','recording_id','purpose','target_index','negative_subtype','captured_rms_dbfs','reference_rms_dbfs']}
        entry['models']={}
        for name,sha,model,thresholds in models:
            entry['models'][name]={}
            for domain in ['digital','acoustic']:
                part=[r for r in rows if r['event_index']==event['event_index'] and r['domain']==domain]
                # Mean-score clip summary is descriptive; source labels are weak, not per-window truth.
                score=np.mean([r['scores'][name]['values'] for r in part],axis=0)
                entry['models'][name][domain]=dict(clip_mean_top1=min(int(score.argmax()),5),
                    target_score_mean=float(score[event['target_index']]) if name!='baseline8' or event['target_index']<5 else None,
                    raw_window_top1=[r['scores'][name]['mapped_top1'] for r in part],
                    thresholded_window_top1=[r['scores'][name]['thresholded_top1'] for r in part])
        paired.append(entry)
    summary={}
    for name,sha,model,thresholds in models:
        summary[name]={}
        for purpose in ['train','validation']:
            subset=[r for r in paired if r['purpose']==purpose]
            summary[name][purpose]={domain:dict(clips=len(subset),
                clip_mean_matches_weak_label=sum(r['models'][name][domain]['clip_mean_top1']==r['target_index'] for r in subset),
                background_clips=sum(r['target_index']==5 for r in subset),
                background_clip_alerts=sum(r['target_index']==5 and r['models'][name][domain]['clip_mean_top1']!=5 for r in subset))
                for domain in ['digital','acoustic']}
    report=dict(schema=1,alignment_sha256=digest_file(a.alignment),recording_sha256=aligned['recording_sha256'],
        frontend_sha256=frontend.source_sha256,model_hashes={name:sha for name,sha,_,_ in models},
        independent_acceptance=False,board_inference=False,new_training=False,
        interpretation='Host FP32 paired replay diagnostics; source clip labels are weak. No per-window accuracy claim. '
          'Validation clips stay validation, never adaptation training. Same-room replay is not new independent D5. '
          'Baseline raw top1 5/6/7 maps to background without runner-up promotion. Existing validation thresholds are frozen.',
        excluded_events=[dict(event_index=e['event_index'],recording_id=e['recording_id'],reception=e['reception'])
                         for e in aligned['events'] if e['quarantined']],
        summary=summary,paired_clips=paired,windows=rows)
    write_json(a.output,report)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
