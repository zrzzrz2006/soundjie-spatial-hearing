"""Verify the frozen paired siren merge and assess already-captured replay."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.host.ablate_soundjie_glass import checked_inputs,verify_pins,selection_key,state_sha
from scripts.host.assess_soundjie_glass_ablation import replay_inputs
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.diagnose_soundjie_alert6_replay import CANDIDATES
from scripts.host.export_soundjie_siren_merge import export
from scripts.host.soundjie_alert6 import LABELS,make_model,balanced_source_indices
from scripts.host.soundjie_siren_merge import (
    ROOT,LABELS5,OUTPUT_ROOT,committed_protocol,paired_models,map_targets,probabilities,
    metrics,apply_thresholds,calibrate_thresholds,promotion_decision,
)
from scripts.host.train_soundjie_alert6 import predict,write_json


def replay_summary(clips):
    result={}
    for purpose in ['train','validation']:
        subset=[r for r in clips if r['purpose']==purpose];result[purpose]={}
        for domain in ['digital','acoustic']:
            target=[int(map_targets(np.array([r['target_index']]))[0]) for r in subset]
            result[purpose][domain]=dict(clips=len(subset),
                matches=sum(r[domain]['top1']==y for r,y in zip(subset,target)),
                positive_clips=sum(y!=4 for y in target),
                any_correct_thresholded=sum(y!=4 and y in r[domain]['thresholded_windows'] for r,y in zip(subset,target)),
                background_clips=sum(y==4 for y in target),
                background_false_alerts=sum(y==4 and r[domain]['top1']!=4 for r,y in zip(subset,target)),
                background_any_thresholded_alert=sum(y==4 and any(v!=4 for v in r[domain]['thresholded_windows']) for r,y in zip(subset,target)))
    return result


def evaluate_replay(model,rx,windows,events,thresholds):
    logits=predict(model,rx);scores=probabilities(logits);clips=[]
    for e in events:
        row={k:e[k] for k in ['event_index','recording_id','purpose','target_index','negative_subtype']}
        for domain in ['digital','acoustic']:
            ix=[i for i,w in enumerate(windows) if w['event_index']==e['event_index'] and w['domain']==domain]
            values=scores[ix]
            row[domain]=dict(top1=int(values.mean(0).argmax()),mean_scores=values.mean(0).tolist(),
                raw_windows=values.argmax(1).tolist(),thresholded_windows=apply_thresholds(values,thresholds).tolist())
        clips.append(row)
    return dict(summary=replay_summary(clips),clips=clips)


def verify_history(directory,result,ty,ids,vy,val_rows):
    history=[json.loads(l) for l in (directory/'progress.jsonl').read_text().splitlines()];cfg=result['config'];seed=result['seed']
    if [r['epoch'] for r in history]!=list(range(1,cfg['epochs']+1)):raise ValueError('incomplete epoch history')
    for row in history:
        ix=balanced_source_indices(ty,ids,cfg['per_original_class_epoch'],seed+row['epoch']-1)
        if hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest()!=row['sampling_sha256']:raise ValueError('sampling drift')
        if row['original_class_counts']!=[cfg['per_original_class_epoch']]*6:raise ValueError('unequal exposure')
        path=directory/'epoch-logits'/('%02d.npz'%row['epoch']);verify_pins({str(path):row['logits_sha256']})
        with np.load(path,allow_pickle=False) as data:
            np.testing.assert_array_equal(data['y_original'],vy)
            for name in ['control6','merged5']:
                actual=metrics(vy,probabilities(data[name]).argmax(1),val_rows)
                if actual!=row['arms'][name]['metrics']:raise ValueError('per-epoch metric mismatch')
    for name in ['control6','merged5']:
        best=max([r['arms'][name] for r in history],key=selection_key)
        if best['epoch']!=result['arms'][name]['selected']['epoch']:raise ValueError('incorrect selected checkpoint')
    return dict(epochs_checked=len(history),sampling_and_metrics_verified=True,selection_recomputed=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.resolve().relative_to(OUTPUT_ROOT.resolve())
    if a.output.exists():raise FileExistsError(a.output)
    cfg=committed_protocol();x,y,purpose,source_rows,audit=checked_inputs(cfg)
    va=purpose=='validation';tr=purpose=='train';vx,vy=x[va],y[va]
    val_rows=[r for r in source_rows if r['purpose']=='validation']
    ids=np.array([r['sampling_identity'] for r in source_rows if r['purpose']=='train'])
    rx,windows,events=replay_inputs(cfg);assessed=[];pairs=[];proof=[];pins={};exports={}
    a.output.mkdir(parents=True,exist_ok=False)
    for seed in cfg['seeds']:
        directory=OUTPUT_ROOT/('seed-'+str(seed));path=directory/'result.json';pins[str(path)]=digest_file(path)
        result=json.loads(path.read_text(encoding='utf-8'))
        if result['seed']!=seed or result['config']!=cfg:raise ValueError('training configuration drift')
        verify_pins(result['files']);proof.append(dict(seed=seed,**verify_history(directory,result,y[tr],ids,vy,val_rows)))
        models=paired_models(seed);torch.set_num_threads(cfg['threads']);arms={};comparison=dict(seed=seed)
        for name,model in zip(['control6','merged5'],models):
            if state_sha(model.state_dict())!=result['initial_model_sha256'][name]:raise ValueError('initial state drift')
            arms[name]={}
            for kind in ['selected','final']:
                entry=result['arms'][name][kind];folder=Path(entry['directory']);checkpoint=folder/'model.pt'
                verify_pins({str(checkpoint):entry['checkpoint_sha256'],str(folder/'validation-logits.npz'):entry['logits_sha256']})
                ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True);expected=LABELS if name=='control6' else LABELS5
                if ckpt['labels']!=list(expected) or ckpt['config']!=cfg or ckpt['seed']!=seed:raise ValueError('checkpoint contract mismatch')
                model.load_state_dict(ckpt['state_dict'],strict=True);logits=predict(model,vx)
                with np.load(folder/'validation-logits.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['y_original'],vy);np.testing.assert_allclose(logits,saved['logits'],atol=1e-6,rtol=1e-6)
                    error=float(np.max(np.abs(logits-saved['logits'])))
                scores=probabilities(logits);m=metrics(vy,scores.argmax(1),val_rows)
                if m!=entry['metrics']['raw']:raise ValueError('saved metrics inconsistent with reloaded weights')
                thresholds=entry['metrics']['thresholds']
                if thresholds!=calibrate_thresholds(vy,scores,cfg['thresholds']):raise ValueError('threshold calibration mismatch')
                replay=evaluate_replay(model,rx,windows,events,thresholds)
                arms[name][kind]=dict(epoch=entry['epoch'],metrics=entry['metrics'],replay=replay,
                    checkpoint_sha256=entry['checkpoint_sha256'],fresh_forward_max_error=error)
                if kind=='selected':
                    comparison[name]=dict(metrics=m,replay=replay['summary'])
                    if name=='merged5':exports[str(seed)]=export(model,x,y,purpose,source_rows,entry['checkpoint_sha256'],a.output/('export-seed-'+str(seed)))
        assessed.append(dict(seed=seed,arms=arms));pairs.append(comparison)
    # Same-task relabel/sum diagnostic on the two previously trained parent models.
    historical={}
    for name,sha in CANDIDATES:
        checkpoint=Path(cfg['features']).parent/'candidates'/name/'soundjie_alert6_float32.pt';verify_pins({str(checkpoint):sha})
        model=make_model();ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if ckpt['labels']!=list(LABELS):raise ValueError('historical output order drift')
        model.load_state_dict(ckpt['state_dict'],strict=True);logits=predict(model,vx);scores=probabilities(logits)
        thresholds=calibrate_thresholds(vy,scores,cfg['thresholds'])
        historical[name]=dict(checkpoint_sha256=sha,label_only=metrics(vy,map_targets(logits.argmax(1)),val_rows),
            probability_merge=metrics(vy,scores.argmax(1),val_rows),thresholds=thresholds,
            thresholded=metrics(vy,apply_thresholds(scores,thresholds),val_rows),
            replay=evaluate_replay(model,rx,windows,events,thresholds),new_training=False)
    decision=promotion_decision(pairs,cfg['promotion_guards'])
    report=dict(schema=1,protocol=cfg,results=assessed,historical=historical,decision=decision,verification=proof,
        training_result_hashes=pins,exports=exports,replay_events=len(events),replay_windows=len(windows),
        classification_semantics='Broad siren includes old alarm and siren; glass retained; voice remains background.',
        scope='Previously exposed development validation and weak-label playback clips, not independent D5. No new board run. All thresholds fit on original validation only; replay used after selection.',
        board_deployed=False,Caffe=False,INT8=False,DPU=False)
    write_json(a.output/'report.json',report)
    print(json.dumps(dict(decision=decision,report=str(a.output/'report.json'))),flush=True)


if __name__=='__main__':main()
