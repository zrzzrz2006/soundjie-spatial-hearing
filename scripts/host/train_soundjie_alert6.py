"""Reproducible six-output training; host-only until separate DPU acceptance."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scripts.host.acquire_soundjie_alert6 import read_rows
from scripts.host.acquire_soundjie_fsd50k import save_verified, json_bytes
from scripts.host.augment_soundjie_d4 import augment_waveform
from scripts.host.build_soundjie_d4_manifest import digest_file, partition_records
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.prepare_soundjie_d3 import decode_audio, resample_mono_16k, make_windows, WindowPolicy, Window, compute_feature
from scripts.host.soundjie_alert6 import LABELS, ROOT, validate_rows, make_model, metrics, balanced_source_indices, sampling_identity
from scripts.host.train_soundjie_d3 import _set_deterministic


def write_json(path,value):save_verified(Path(path),json_bytes(value))


def source_windows(samples,row,seed,min_relative_event_rms=.15):
    if row['purpose'] not in ('train','validation'):raise ValueError('training cannot read heldout purpose')
    if not 0<min_relative_event_rms<=1:raise ValueError('invalid relative event energy gate')
    center=next(make_windows(samples, WindowPolicy('energy_center',seed)))
    result=[center]
    if row['purpose']=='validation' or len(samples)<=16000:return result
    if row['target_index']==5:
        starts=[int(np.random.default_rng(seed).integers(0,len(samples)-16000+1))]
    else:
        starts=[max(0,min(len(samples)-16000,center.start_sample+offset)) for offset in (-3200,3200)]
    seen={center.start_sample}
    min_rms=1e-6
    if row['target_index']!=5:
        min_rms=max(min_rms,float(np.sqrt(np.mean(center.samples**2)))*min_relative_event_rms)
    for start in starts:
        segment=samples[start:start+16000]
        if start not in seen and np.ptp(segment)>1e-7 and np.sqrt(np.mean(segment**2))>=min_rms:
            seen.add(start);result.append(Window(start,segment.copy()))
    return result


def prepare(manifest, config, output):
    manifest=Path(manifest);output=Path(output)
    if output.exists():raise FileExistsError(output)
    source_audit=json.loads((manifest.parent/'audit.json').read_text(encoding='utf-8'))
    if digest_file(manifest)!=source_audit['manifest_sha256']:raise ValueError('manifest drift')
    rows=read_rows(manifest);validate_rows(rows);partition_records(rows)
    frontend=load_frontend_config(ROOT/'config/logmel_frontend_v1.json')
    windows=[];pcm_splits={}
    for i,row in enumerate(rows):
        if digest_file(row['path'])!=row['sha256']:raise ValueError('source drift '+row['recording_id'])
        audio=resample_mono_16k(decode_audio(Path(row['path'])))
        pcm_hash=hashlib.sha256(np.clip(np.rint(audio*32768),-32768,32767).astype('<i2').tobytes()).hexdigest()
        if pcm_splits.get(pcm_hash,row['purpose'])!=row['purpose']:raise ValueError('decoded PCM crosses partitions')
        pcm_splits[pcm_hash]=row['purpose']
        windows.append(source_windows(audio,row,config['data_seed']+i*7919,config['min_relative_event_rms']))
    backgrounds=[ws[0].samples for row,ws in zip(rows,windows) if row['target_index']==5 and row['purpose']=='train']
    arrays=[];y=[];purposes=[];provenance=[]
    for i,(row,ws) in enumerate(zip(rows,windows)):
        for j,window in enumerate(ws):
            variants=[(window.samples,None)]
            if row['purpose']=='train':
                for k in range(config['augmentations']):
                    variants.append(augment_waveform(window.samples, config['data_seed']+i*101+j*100000+k,
                                                     'train',backgrounds[(i+j+k)%len(backgrounds)]))
            for signal,recipe in variants:
                pcm=np.clip(np.rint(signal*32768),-32768,32767).astype('<i2')
                feature=compute_feature(pcm.astype(np.float64)/32768.,frontend)
                arrays.append(feature);y.append(row['target_index']);purposes.append(row['purpose'])
                provenance.append(dict(recording_id=row['recording_id'],source_group=row['source_group'],
                    sampling_identity=sampling_identity(row),
                    parent_sha256=row['sha256'],purpose=row['purpose'],target_index=row['target_index'],
                    negative_subtype=row['negative_subtype'],start_sample_16k=window.start_sample,
                    derivation=recipe,feature_sha256=hashlib.sha256(feature.tobytes()).hexdigest()))
        if i%50==0:print(json.dumps(dict(stage='features',sources=i+1,total=len(rows),windows=len(y))),flush=True)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'features.npz').open('xb') as f:
        np.savez(f,x=np.stack(arrays),y=np.asarray(y,dtype=np.int64),purpose=np.asarray(purposes))
    write_json(output/'provenance.json',provenance)
    audit=dict(schema=1,manifest_sha256=digest_file(manifest),source_audit=source_audit,
        cache_sha256=digest_file(output/'features.npz'),provenance_sha256=digest_file(output/'provenance.json'),
        class_map_sha256=digest_file(ROOT/'config/soundjie_alert6_classes.json'),
        frontend_sha256=frontend.source_sha256,config=config,window_counts=dict(Counter(purposes)),
        script_sha256=digest_file(Path(__file__)),validation='one_unaugmented_energy_window_per_source',
        unique_original_recordings=len({sampling_identity(r) for r in rows}),
        unique_original_recordings_per_class={name:len({sampling_identity(r) for r in rows if r['target_id']==name}) for name in LABELS},
        d5_holdout_audio_read=False)
    write_json(output/'audit.json',audit)
    print(json.dumps(dict(stage='features_done',counts=audit['window_counts'],sha256=audit['cache_sha256'])),flush=True)


def load_cache(directory):
    directory=Path(directory);audit=json.loads((directory/'audit.json').read_text(encoding='utf-8'))
    for name,key in [('features.npz','cache_sha256'),('provenance.json','provenance_sha256')]:
        if digest_file(directory/name)!=audit[key]:raise ValueError('cache/provenance drift')
    if digest_file(ROOT/'config/soundjie_alert6_classes.json')!=audit['class_map_sha256']:
        raise ValueError('class map drift')
    with np.load(directory/'features.npz',allow_pickle=False) as data:
        x,y,purpose=data['x'],data['y'],data['purpose']
    provenance=json.loads((directory/'provenance.json').read_text(encoding='utf-8'))
    if x.shape!=(len(y),1,64,96) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('invalid feature shape')
    if len(provenance)!=len(y) or set(purpose.tolist())!={'train','validation'}:raise ValueError('invalid provenance/purposes')
    identities={}
    for i,row in enumerate(provenance):
        if row['purpose']!=purpose[i] or row['target_index']!=int(y[i]) or hashlib.sha256(x[i].tobytes()).hexdigest()!=row['feature_sha256']:
            raise ValueError('feature provenance mismatch')
        for field in ('source_group','parent_sha256','recording_id','sampling_identity'):
            key=(field,row[field])
            if identities.get(key,row['purpose'])!=row['purpose']:raise ValueError('cache split leakage')
            identities[key]=row['purpose']
        if row['purpose']=='validation' and row['derivation'] is not None:raise ValueError('augmented validation')
    return x,y,purpose,provenance,audit


def predict(model,x,batch=64):
    model.eval()
    with torch.no_grad():return np.concatenate([model(torch.from_numpy(x[i:i+batch])).numpy() for i in range(0,len(x),batch)])


def softmax(logits):
    v=np.exp(logits-logits.max(1,keepdims=True));return v/v.sum(1,keepdims=True)


def apply_thresholds(scores,thresholds):
    scores=np.asarray(scores)
    if scores.ndim!=2 or scores.shape[1]!=6 or len(thresholds)!=5 or not np.isfinite(scores).all():
        raise ValueError('six-output score contract required')
    pred=scores.argmax(1);result=pred.copy()
    for c in range(5):result[(pred==c)&(scores[:,c]<thresholds[c])]=5
    return result


def calibrate_thresholds(y,scores):
    thresholds=[];pred=scores.argmax(1)
    for c in range(5):
        options=[]
        for t in np.arange(.4,1.0,.025):
            alert=(pred==c)&(scores[:,c]>=t)
            tp=np.sum(alert&(y==c));fp=np.sum(alert&(y!=c));fn=np.sum(~alert&(y==c))
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
            neg_fpr=np.mean(alert[y==5]) if np.any(y==5) else 1.
            options.append((neg_fpr<=.02,f1,-neg_fpr,-t,t))
        thresholds.append(float(max(options)[-1]))
    return thresholds


def negative_breakdown(y,pred,provenance):
    result={}
    for subtype in sorted({r['negative_subtype'] for r in provenance if r['target_index']==5}):
        mask=np.array([r['target_index']==5 and r['negative_subtype']==subtype for r in provenance])
        result[subtype]=dict(support=int(mask.sum()),false_alerts=int(np.sum(pred[mask]!=5)),
                            false_alert_rate=float(np.mean(pred[mask]!=5)))
    return result


def train(features, config, output):
    x,y,purpose,provenance,audit=load_cache(features)
    if audit['config']!=config:raise ValueError('training configuration drift')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=config['training'];torch.set_num_threads(cfg['threads'])
    tx,ty=x[purpose=='train'],y[purpose=='train'];vx,vy=x[purpose=='validation'],y[purpose=='validation']
    train_sources=np.array([r['sampling_identity'] for r in provenance if r['purpose']=='train'])
    val_rows=[r for r in provenance if r['purpose']=='validation']
    results=[]
    for seed in config['seeds']:
        _set_deterministic(seed);model=make_model()
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        criterion=torch.nn.CrossEntropyLoss()
        directory=output/('seed-'+str(seed));directory.mkdir()
        history=[];best=None;best_score=-1.;best_epoch=0
        for epoch in range(cfg['epochs']):
            started=time.monotonic();model.train()
            indices=balanced_source_indices(ty,train_sources,cfg['per_class_epoch'],seed+epoch)
            total_loss=0.
            for offset in range(0,len(indices),cfg['batch_size']):
                batch=indices[offset:offset+cfg['batch_size']];optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(torch.from_numpy(tx[batch])),torch.from_numpy(ty[batch]))
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();optimizer.step();total_loss+=float(loss.detach())*len(batch)
            logits=predict(model,vx);m=metrics(vy,logits.argmax(1))
            progress=dict(seed=seed,epoch=epoch+1,loss=total_loss/len(indices),
                          macro_recall=m['macro_recall'],negative_false_alert_rate=m['negative_false_alert_rate'],
                          seconds=time.monotonic()-started)
            history.append(progress)
            if m['macro_recall']>best_score:
                best_score=m['macro_recall'];best=copy.deepcopy(model.state_dict());best_epoch=epoch+1
            with (directory/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(progress)+'\n')
            print(json.dumps(progress),flush=True)
        model.load_state_dict(best);logits=predict(model,vx);scores=softmax(logits)
        thresholds=calibrate_thresholds(vy,scores);rejected=apply_thresholds(scores,thresholds)
        report=dict(raw=metrics(vy,logits.argmax(1)),thresholded=metrics(vy,rejected),
                    negative_subtypes_raw=negative_breakdown(vy,logits.argmax(1),val_rows),
                    negative_subtypes_thresholded=negative_breakdown(vy,rejected,val_rows),
                    thresholds=thresholds,thresholds_status='validation_calibration_only_not_board_or_independent_test',
                    best_epoch=best_epoch,board_inference='not_run',quantization='not_run')
        checkpoint=directory/'soundjie_alert6_float32.pt'
        with checkpoint.open('xb') as f:torch.save(dict(schema=1,state_dict=best,labels=list(LABELS),
                                                       seed=seed,upstream=audit),f)
        with (directory/'validation-logits.npz').open('xb') as f:np.savez(f,logits=logits,y=vy)
        write_json(directory/'metrics.json',report)
        write_json(directory/'training-manifest.json',dict(seed=seed,best_epoch=best_epoch,training=cfg,
            upstream=audit,checkpoint_sha256=digest_file(checkpoint),metrics=report,
            torch_version=torch.__version__,numpy_version=np.__version__,training_script_sha256=digest_file(Path(__file__))))
        results.append(dict(seed=seed,directory=str(directory),macro_recall=best_score,
                            checkpoint_sha256=digest_file(checkpoint)))
    selected=max(results,key=lambda r:r['macro_recall'])
    write_json(output/'selection.json',dict(candidates=results,selected=selected,
        criterion='validation_macro_recall_only',status='trained_host_float32_not_board_deployed'))
    print(json.dumps(dict(stage='training_complete',selected=selected)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','train'])
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'config/soundjie_alert6_training.json')
    args=parser.parse_args();config=json.loads(args.config.read_text(encoding='utf-8'))
    (prepare if args.mode=='prepare' else train)(args.input,config,args.output)


if __name__=='__main__':main()
