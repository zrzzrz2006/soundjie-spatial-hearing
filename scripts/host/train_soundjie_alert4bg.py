"""Controlled CPU training and cache creation for the actual alert4bg-v2 CNN."""
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
from scripts.host.acquire_soundjie_fsd50k import save_verified,json_bytes
from scripts.host.acquire_soundjie_alert4bg import verify_dataset
from scripts.host.augment_soundjie_d4 import augment_waveform
from scripts.host.build_soundjie_d4_manifest import digest_file,partition_records
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.prepare_soundjie_d3 import decode_audio,resample_mono_16k,compute_feature
from scripts.host.train_soundjie_alert6 import source_windows as six_windows,predict,softmax
from scripts.host.train_soundjie_d3 import _set_deterministic
from scripts.host.soundjie_alert4bg import (
    ROOT,LABELS,CONTRACT,CONFIG_FILE,CLASS_FILE,DATA_ROOT,OUTPUT_ROOT,
    validate_rows,make_model,metrics,balanced_source_indices,calibrate_thresholds,apply_thresholds,
)


def write_json(path,value):save_verified(Path(path),json_bytes(value))


def checked_output(path):
    path=Path(path);path.resolve().relative_to(OUTPUT_ROOT.resolve())
    if path.resolve()==OUTPUT_ROOT.resolve():raise ValueError('a new child output directory is required')
    if path.exists():raise FileExistsError(path)
    return path


def source_windows(samples,row,seed,min_relative_event_rms=.15):
    # The unchanged helper treats index 5 as background. This is only a local
    # windowing adapter, never a label or feature-cache remap.
    return six_windows(samples,dict(row,target_index=5 if row['target_index']==4 else row['target_index']),seed,min_relative_event_rms)


def implementation_hashes():
    files=[Path(__file__),ROOT/'scripts/host/soundjie_alert4bg.py',ROOT/'scripts/host/acquire_soundjie_alert4bg.py',
           ROOT/'scripts/host/train_soundjie_alert6.py',ROOT/'scripts/host/soundjie_alert6.py',
           ROOT/'scripts/host/train_soundjie_d3.py',ROOT/'scripts/host/prepare_soundjie_d3.py',
           ROOT/'scripts/host/logmel_reference.py',ROOT/'scripts/host/augment_soundjie_d4.py']
    return {str(p.relative_to(ROOT)):digest_file(p) for p in files}


def quantized_feature(signal,frontend,derivation,purpose):
    if purpose not in ('train','validation') or (derivation is not None and purpose!='train'):
        raise ValueError('only train may have augmented features')
    signal=np.asarray(signal)
    if signal.shape!=(16000,) or not np.isfinite(signal).all():raise ValueError('invalid waveform')
    pcm=np.clip(np.rint(signal*32768),-32768,32767).astype('<i2')
    if derivation is not None and int(pcm.max())==int(pcm.min()):return None
    return compute_feature(pcm.astype(np.float64)/32768.,frontend)


def prepare(config,output):
    output=checked_output(output);source_audit=verify_dataset()
    rows=read_rows(DATA_ROOT/'manifest.jsonl');validate_rows(rows);partition_records(rows)
    frontend=load_frontend_config(ROOT/'config/logmel_frontend_v1.json')
    windows=[];pcm_splits={}
    for i,row in enumerate(rows):
        # Purpose is checked before any source audio is opened.
        if row['purpose'] not in ('train','validation'):raise ValueError('forbidden audio purpose')
        if digest_file(row['path'])!=row['sha256']:raise ValueError('source drift '+row['recording_id'])
        audio=resample_mono_16k(decode_audio(Path(row['path'])))
        pcm_hash=hashlib.sha256(np.clip(np.rint(audio*32768),-32768,32767).astype('<i2').tobytes()).hexdigest()
        if pcm_splits.get(pcm_hash,row['purpose'])!=row['purpose']:raise ValueError('decoded PCM crosses partitions')
        pcm_splits[pcm_hash]=row['purpose'];row['pcm_sha256_16k']=pcm_hash
        windows.append(source_windows(audio,row,config['data_seed']+i*7919,config['min_relative_event_rms']))
    validate_rows(rows)
    backgrounds=[(r,ws[0].samples) for r,ws in zip(rows,windows) if r['target_index']==4 and r['purpose']=='train']
    arrays=[];targets=[];purposes=[];provenance=[];skipped=[]
    output.mkdir(parents=True,exist_ok=False)
    for i,(row,ws) in enumerate(zip(rows,windows)):
        for j,window in enumerate(ws):
            variants=[(window.samples,None)]
            if row['purpose']=='train':
                for k in range(config['augmentations']):
                    bg,noise=backgrounds[(i+j+k)%len(backgrounds)]
                    signal,recipe=augment_waveform(window.samples,config['data_seed']+i*101+j*100000+k,'train',noise)
                    recipe.update(background_recording_id=bg['recording_id'],background_purpose='train',background_source_sha256=bg['sha256'])
                    variants.append((signal,recipe))
            for signal,recipe in variants:
                feature=quantized_feature(signal,frontend,recipe,row['purpose'])
                if feature is None:
                    skipped.append(dict(recording_id=row['recording_id'],purpose=row['purpose'],
                        start_sample_16k=window.start_sample,derivation=recipe,
                        reason='augmented_train_window_quantized_to_constant_pcm'))
                    continue
                arrays.append(feature);targets.append(row['target_index']);purposes.append(row['purpose'])
                provenance.append(dict(recording_id=row['recording_id'],source_group=row['source_group'],
                    uploader_group=row.get('uploader_group'),sampling_identity=row['sampling_identity'],
                    parent_sha256=row['sha256'],pcm_sha256_16k=row['pcm_sha256_16k'],purpose=row['purpose'],
                    target_index=row['target_index'],negative_subtype=row['negative_subtype'],
                    start_sample_16k=window.start_sample,derivation=recipe,
                    feature_sha256=hashlib.sha256(feature.tobytes()).hexdigest()))
        if (i+1)%50==0 or i+1==len(rows):print(json.dumps(dict(stage='features',sources=i+1,total=len(rows),windows=len(targets))),flush=True)
    x=np.stack(arrays);y=np.asarray(targets,dtype=np.int64);purpose=np.asarray(purposes)
    validate_cache_arrays(x,y,purpose,provenance)
    with (output/'features.npz').open('xb') as f:np.savez(f,x=x,y=y,purpose=purpose)
    write_json(output/'provenance.json',provenance)
    write_json(output/'skipped-augmentations.json',skipped)
    audit=dict(schema=2,contract=CONTRACT,source_audit=source_audit,config=config,
        manifest_path=str(DATA_ROOT/'manifest.jsonl'),manifest_sha256=digest_file(DATA_ROOT/'manifest.jsonl'),
        cache_sha256=digest_file(output/'features.npz'),provenance_sha256=digest_file(output/'provenance.json'),
        class_map_sha256=digest_file(CLASS_FILE),frontend_sha256=frontend.source_sha256,
        implementation_sha256=implementation_hashes(),window_counts=dict(Counter(purposes)),
        source_counts=source_audit['counts'],validation='one_unaugmented_energy_center_window_per_file',
        skipped_augmentations=len(skipped),skipped_augmentations_sha256=digest_file(output/'skipped-augmentations.json'),
        training_holdout_audio_read=False,d5_features_read=False,live_replay_used=False)
    write_json(output/'audit.json',audit)
    print(json.dumps(dict(stage='features_done',window_counts=audit['window_counts'],cache_sha256=audit['cache_sha256'])),flush=True)


def validate_cache_arrays(x,y,purpose,provenance):
    if x.shape!=(len(y),1,64,96) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('invalid 64x96 features')
    if y.ndim!=1 or not np.issubdtype(y.dtype,np.integer) or np.any((y<0)|(y>4)):raise ValueError('invalid targets')
    if len(provenance)!=len(y) or purpose.shape!=y.shape or set(purpose.tolist())!={'train','validation'}:raise ValueError('invalid cache purposes')
    identities={};validation_ids=set()
    for i,row in enumerate(provenance):
        if row['purpose']!=purpose[i] or row['target_index']!=int(y[i]) or hashlib.sha256(x[i].tobytes()).hexdigest()!=row['feature_sha256']:
            raise ValueError('feature provenance mismatch')
        for field in ('recording_id','source_group','parent_sha256','sampling_identity','uploader_group','pcm_sha256_16k'):
            value=row.get(field)
            if value:
                key=(field,value)
                if identities.get(key,row['purpose'])!=row['purpose']:raise ValueError('cache identity crosses purposes')
                identities[key]=row['purpose']
        if row['purpose']=='validation':
            if row['derivation'] is not None or row['recording_id'] in validation_ids:raise ValueError('augmented or repeated validation')
            validation_ids.add(row['recording_id'])
        elif row['derivation'] and row['derivation'].get('background_purpose')!='train':
            raise ValueError('augmentation background is not train')


def load_cache(directory):
    directory=Path(directory);directory.resolve().relative_to(OUTPUT_ROOT.resolve())
    audit=json.loads((directory/'audit.json').read_bytes())
    if audit['contract']!=CONTRACT or digest_file(CLASS_FILE)!=audit['class_map_sha256']:raise ValueError('cache class contract drift')
    if digest_file(ROOT/'config/logmel_frontend_v1.json')!=audit['frontend_sha256']:raise ValueError('frontend drift')
    for name,key in [('features.npz','cache_sha256'),('provenance.json','provenance_sha256'),('skipped-augmentations.json','skipped_augmentations_sha256')]:
        if digest_file(directory/name)!=audit[key]:raise ValueError('cache drift '+name)
    with np.load(directory/'features.npz',allow_pickle=False) as data:x,y,purpose=data['x'],data['y'],data['purpose']
    provenance=json.loads((directory/'provenance.json').read_bytes());validate_cache_arrays(x,y,purpose,provenance)
    return x,y,purpose,provenance,audit


def train(features,config,output):
    output=checked_output(output);x,y,purpose,rows,audit=load_cache(features)
    if audit['config']!=config:raise ValueError('configuration drift')
    cfg=config['training'];_set_deterministic(config['seed'])
    # The old seed helper sets one thread, so apply the authorized limit afterward.
    torch.set_num_threads(cfg['threads']);torch.set_num_interop_threads(1)
    model=make_model();optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    tr,va=purpose=='train',purpose=='validation';tx,ty=x[tr],y[tr];vx,vy=x[va],y[va]
    ids=np.asarray([r['sampling_identity'] for r in rows if r['purpose']=='train'])
    val_rows=[r for r in rows if r['purpose']=='validation']
    output.mkdir(parents=True,exist_ok=False);(output/'epochs').mkdir()
    write_json(output/'protocol.json',config);write_json(output/'validation-sources.json',val_rows)
    best=None;best_key=None;best_epoch=None;history=[]
    for epoch in range(cfg['epochs']+1):
        started=time.monotonic();total_loss=0.;indices=None
        if epoch:
            model.train();indices=balanced_source_indices(ty,ids,cfg['per_class_epoch'],config['seed']+epoch-1)
            for offset in range(0,len(indices),cfg['batch_size']):
                batch=indices[offset:offset+cfg['batch_size']];optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(tx[batch])),torch.from_numpy(ty[batch]))
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();optimizer.step();total_loss+=float(loss.detach())*len(batch)
        logits=predict(model,vx);m=metrics(vy,logits.argmax(1),val_rows)
        progress=dict(epoch=epoch,loss=total_loss/len(indices) if indices is not None else None,
            metrics=m,seconds=time.monotonic()-started,
            sampling_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest() if indices is not None else None)
        write_json(output/'epochs'/('%02d.json'%epoch),progress);history.append(progress)
        key=(m['macro_recall'],-m['negative_false_alert_rate'],-epoch)
        if epoch and (best_key is None or key>best_key):best=copy.deepcopy(model.state_dict());best_key=key;best_epoch=epoch
        print(json.dumps(dict(stage='training',epoch=epoch,loss=progress['loss'],macro_recall=m['macro_recall'],dog_recall=m['dog_recall'],bg_fpr=m['negative_false_alert_rate'],seconds=progress['seconds'])),flush=True)
    artifacts={};final=copy.deepcopy(model.state_dict())
    for name,state,epoch in [('selected',best,best_epoch),('final',final,cfg['epochs'])]:
        directory=output/name;directory.mkdir();model.load_state_dict(state)
        logits=predict(model,vx);scores=softmax(logits);thresholds=calibrate_thresholds(vy,scores,config['thresholds'])
        report=dict(raw=metrics(vy,logits.argmax(1),val_rows),thresholded=metrics(vy,apply_thresholds(scores,thresholds),val_rows),
            thresholds=thresholds,thresholds_fit_on_same_validation=True,independent_acceptance=False,
            quantization=False,board_deployed=False)
        with (directory/'model.pt').open('xb') as f:torch.save(dict(schema=2,contract=CONTRACT,state_dict=state,labels=list(LABELS),
            seed=config['seed'],epoch=epoch,config=config,cache_sha256=audit['cache_sha256'],class_map_sha256=digest_file(CLASS_FILE),
            initialization='random_all_layers'),f)
        with (directory/'validation-logits.npz').open('xb') as f:np.savez(f,logits=logits,y=vy)
        write_json(directory/'metrics.json',report)
        artifacts[name]=dict(directory=str(directory),epoch=epoch,checkpoint_sha256=digest_file(directory/'model.pt'),
            logits_sha256=digest_file(directory/'validation-logits.npz'),metrics=report)
    write_json(output/'history.json',history)
    result=dict(schema=2,contract=CONTRACT,seed=config['seed'],config=config,config_sha256=digest_file(CONFIG_FILE),
        actual_torch_threads=torch.get_num_threads(),torch_version=str(torch.__version__),numpy_version=np.__version__,
        features_directory=str(features),cache_sha256=audit['cache_sha256'],source_counts=audit['source_counts'],
        epoch0_random_baseline=history[0]['metrics'],artifacts=artifacts,implementation_sha256=implementation_hashes(),
        selection_used_replay=False,d5_audio_read=False,d5_features_read=False,independent_acceptance=False)
    write_json(output/'result.json',result)
    print(json.dumps(dict(stage='trained',selected_epoch=best_epoch,checkpoint_sha256=artifacts['selected']['checkpoint_sha256'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','train'])
    parser.add_argument('--features',type=Path,default=OUTPUT_ROOT/'features')
    args=parser.parse_args();config=json.loads(CONFIG_FILE.read_bytes())
    if Path(config['artifact_root']).resolve()!=OUTPUT_ROOT.resolve():raise ValueError('unauthorized artifact root')
    if args.stage=='prepare':prepare(config,args.features)
    else:train(args.features,config,OUTPUT_ROOT/'training')


if __name__=='__main__':main()
