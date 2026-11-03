"""Guarded real-replay fine-tuning; keeps original validation and board untouched."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as nnf

from scripts.host.acquire_soundjie_alert6 import read_rows
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.prepare_soundjie_d3 import compute_feature
from scripts.host.soundjie_alert6 import ROOT,LABELS,make_model,metrics,balanced_source_indices,sampling_identity
from scripts.host.train_soundjie_alert6 import load_cache,source_windows,predict,write_json,softmax,calibrate_thresholds,apply_thresholds,negative_breakdown
from scripts.host.train_soundjie_d3 import _set_deterministic

ORIGINAL=Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_alert6/20261031b')
PARENTS=Path('F:/MLK/DPU_ZYNQ7000/datasets/soundjie_alert6_20261031/manifest.jsonl')
CAPTURE=Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_alert6_replay/20261031a/capture-20261031b')
INITIAL=ORIGINAL/'candidates/seed-20261031/soundjie_alert6_float32.pt'


def checked(path,sha):
    if digest_file(path)!=sha:raise ValueError('SHA mismatch: '+str(path))


def guard_replay_rows(rows,parents):
    indexed={r['recording_id']:r for r in parents}
    validation_groups={r['source_group'] for r in parents if r['purpose']!='train'}
    seen=set()
    for row in rows:
        if row['recording_id'] in seen:raise ValueError('duplicate replay identity')
        seen.add(row['recording_id']);parent=indexed.get(row['parent_recording_id'])
        if parent is None or parent['purpose']!='train' or row['purpose']!='train':
            raise ValueError('original parent must be train')
        if row.get('adaptation_allowed') is not True or row.get('reception',{}).get('passed') is not True:
            raise ValueError('unreceived/unapproved replay')
        if row.get('domain')!='acoustic_replay':raise ValueError('unexpected domain')
        if row['source_group'] in validation_groups:raise ValueError('validation group in adaptation')
        for field in ['source_group','target_id','target_index']:
            if row[field]!=parent[field]:raise ValueError('parent identity/label changed')
        if row['parent_sha256']!=parent['sha256'] or row['source_recording_id']!=sampling_identity(parent):
            raise ValueError('parent SHA/recording identity changed')


def replay_provenance(row,start,gain):
    return dict(recording_id=row['recording_id'],parent_recording_id=row['parent_recording_id'],
        sampling_identity=row['source_recording_id'],source_group=row['source_group'],purpose='train',
        target_index=row['target_index'],negative_subtype=row['negative_subtype'],
        parent_sha256=row['parent_sha256'],replay_sha256=row['sha256'],start_sample=start,gain_db=gain)


def guard_cache_membership(provenance,approved):
    indexed={r['recording_id']:r for r in approved}
    for entry in provenance:
        row=indexed.get(entry['recording_id'])
        if row is None or entry['purpose']!='train':raise ValueError('unapproved cache source')
        for field in ['parent_recording_id','parent_sha256','source_group','target_index','negative_subtype']:
            if entry[field]!=row[field]:raise ValueError('cache lineage mismatch: '+field)
        if entry['sampling_identity']!=row['source_recording_id'] or entry['replay_sha256']!=row['sha256']:
            raise ValueError('cache recording/hash mismatch')


def verify_receipt_bindings(receipt,config,audit,cache_sha,provenance_sha,audit_sha):
    if receipt['config']!=config:raise ValueError('receipt configuration drift')
    for key,actual in [('cache_sha256',cache_sha),('provenance_sha256',provenance_sha),('audit_sha256',audit_sha)]:
        if receipt[key]!=actual:raise ValueError('committed receipt mismatch: '+key)
    if audit['cache_sha256']!=cache_sha or audit['provenance_sha256']!=provenance_sha:
        raise ValueError('feature audit hash drift')


def load_committed_receipt(path):
    if path is None:raise ValueError('training requires a committed external cache receipt')
    path=Path(path).resolve();allowed=(ROOT/'evidence/soundjie-d4').resolve()
    if not path.is_relative_to(allowed):raise ValueError('cache receipt must be in repository evidence')
    relative=path.relative_to(ROOT).as_posix()
    result=subprocess.run(['git','show','HEAD:'+relative],cwd=ROOT,check=True,capture_output=True)
    if result.stdout!=path.read_bytes():raise ValueError('receipt must exactly match committed Git HEAD')
    return json.loads(result.stdout)


def approved_rows(config):
    checked(PARENTS,config['original_manifest_sha256'])
    checked(CAPTURE/'adaptation-train-subset.jsonl',config['replay_manifest_sha256'])
    rows=read_rows(CAPTURE/'adaptation-train-subset.jsonl');guard_replay_rows(rows,read_rows(PARENTS))
    return rows


def publish_receipt(config,features,output):
    features=Path(features);audit=json.loads((features/'audit.json').read_text(encoding='utf-8'))
    if audit['config']!=config:raise ValueError('config mismatch')
    for name,key in [('features.npz','cache_sha256'),('provenance.json','provenance_sha256')]:checked(features/name,audit[key])
    provenance=json.loads((features/'provenance.json').read_text(encoding='utf-8'))
    guard_cache_membership(provenance,approved_rows(config))
    write_json(output,dict(schema=1,config=config,feature_directory=str(features.resolve()),
        cache_sha256=audit['cache_sha256'],provenance_sha256=audit['provenance_sha256'],
        audit_sha256=digest_file(features/'audit.json'),purpose='train_only_verified_capture',sources=13,
        provenance_contract='parent_recording_id/parent_sha256 refer to original; replay_sha256 refers to captured excerpt'))
    print(json.dumps(dict(stage='receipt_prepared_commit_before_training',path=str(output))))


def domain_batches(original_y,original_ids,replay_y,replay_ids,per_class,seed):
    a=balanced_source_indices(original_y,original_ids,per_class,seed)
    b=balanced_source_indices(replay_y,replay_ids,per_class,seed+1000000)
    if len(a)!=len(b) or len(a)%16:raise ValueError('domain sizes must form full 16+16 batches')
    return [(a[i:i+16],b[i:i+16]) for i in range(0,len(a),16)]


def fixed_bn_train(model):
    model.train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()


def mixed_loss(student,teacher,original_x,original_y,replay_x,replay_y,temperature,distill_weight,replay_fraction=.5):
    if not 0<replay_fraction<1:raise ValueError('replay CE fraction must be between zero and one')
    values=student(torch.cat([original_x,replay_x]));n=len(original_y)
    ce=(1-replay_fraction)*nnf.cross_entropy(values[:n],original_y)+replay_fraction*nnf.cross_entropy(values[n:],replay_y)
    with torch.no_grad():target=torch.softmax(teacher(original_x)/temperature,dim=1)
    kl=nnf.kl_div(torch.log_softmax(values[:n]/temperature,dim=1),target,reduction='batchmean')
    return ce+distill_weight*temperature*temperature*kl


def eligible(m,parent):
    return (m['macro_recall']>=parent['macro_recall']-.02
        and m['negative_false_alert_rate']<=parent['negative_false_alert_rate']+.02
        and all(m['per_class'][name]['recall']>=parent['per_class'][name]['recall']-.10 for name in LABELS[:5]))


def select_epoch(history,parent):
    rows=[r for r in history if eligible(r['metrics'],parent)]
    if not rows:raise ValueError('missing eligible parent baseline')
    return max(rows,key=lambda r:(r['metrics']['macro_recall'],-r['metrics']['negative_false_alert_rate'],-r['epoch']))


def prepare(config,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    checked(PARENTS,config['original_manifest_sha256'])
    manifest=CAPTURE/'adaptation-train-subset.jsonl';checked(manifest,config['replay_manifest_sha256'])
    alignment_path=CAPTURE/'alignment-report.json';checked(alignment_path,config['alignment_sha256'])
    checked(CAPTURE/'capture-stereo.wav',config['raw_capture_sha256'])
    checked(INITIAL,config['initial_checkpoint_sha256'])
    checked(ORIGINAL/'features/features.npz',config['original_cache_sha256'])
    parents=read_rows(PARENTS);rows=read_rows(manifest);guard_replay_rows(rows,parents)
    aligned=json.loads(alignment_path.read_text(encoding='utf-8'))
    accepted={e['recording_id']:e for e in aligned['events'] if e['adaptation_allowed'] and not e['quarantined']}
    frontend=load_frontend_config(ROOT/'config/logmel_frontend_v1.json');arrays=[];labels=[];provenance=[]
    for row in rows:
        event=accepted.get(row['parent_recording_id'])
        if event is None or event['captured_sha256']!=row['sha256'] or event['captured_wav']!=row['path']:
            raise ValueError('replay row not in accepted capture evidence')
        checked(row['path'],row['sha256']);audio,rate=sf.read(row['path'],dtype='float64')
        if rate!=16000 or audio.ndim!=1:raise ValueError('unexpected replay excerpt encoding')
        windows=source_windows(audio,row,config['seed'],config['minimum_relative_event_rms'])
        for window in windows:
            for gain in [0]+config['gain_augment_db']:
                signal=np.clip(np.rint(window.samples*10**(gain/20)*32768),-32768,32767)/32768.
                arrays.append(compute_feature(signal,frontend));labels.append(row['target_index'])
                provenance.append(replay_provenance(row,window.start_sample,gain))
    if len(rows)!=13 or set(labels)!=set(range(6)):raise ValueError('expected 13 sources covering all six outputs')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'features.npz').open('xb') as f:np.savez(f,x=np.stack(arrays),y=np.array(labels,dtype=np.int64))
    write_json(output/'provenance.json',provenance)
    write_json(output/'audit.json',dict(schema=1,config=config,sources=13,windows=len(labels),
        cache_sha256=digest_file(output/'features.npz'),provenance_sha256=digest_file(output/'provenance.json'),
        source_manifest_sha256=digest_file(manifest),frontend_sha256=frontend.source_sha256,
        independent_sources_added=0,validation_read_as_training=False,script_sha256=digest_file(Path(__file__))))
    print(json.dumps(dict(stage='adaptation_features_ready',sources=len(rows),windows=len(labels))),flush=True)


def train(config,features,output,receipt_path):
    features=Path(features);output=Path(output)
    if output.exists():raise FileExistsError(output)
    audit=json.loads((features/'audit.json').read_text(encoding='utf-8'))
    if audit['config']!=config:raise ValueError('replay config changed')
    receipt=load_committed_receipt(receipt_path)
    if Path(receipt['feature_directory']).resolve()!=features.resolve():raise ValueError('receipt feature path mismatch')
    verify_receipt_bindings(receipt,config,audit,digest_file(features/'features.npz'),
        digest_file(features/'provenance.json'),digest_file(features/'audit.json'))
    for name,key in [('features.npz','cache_sha256'),('provenance.json','provenance_sha256')]:checked(features/name,audit[key])
    checked(INITIAL,config['initial_checkpoint_sha256'])
    checked(ORIGINAL/'features/features.npz',config['original_cache_sha256'])
    x,y,purposes,provenance,original_audit=load_cache(ORIGINAL/'features')
    replay_rows=json.loads((features/'provenance.json').read_text(encoding='utf-8'))
    guard_cache_membership(replay_rows,approved_rows(config))
    if any(r['purpose']!='train' for r in replay_rows):raise ValueError('nontrain replay cache')
    with np.load(features/'features.npz',allow_pickle=False) as data:rx,ry=data['x'],data['y']
    if rx.shape!=(len(ry),1,64,96) or rx.dtype!=np.float32 or len(replay_rows)!=len(ry):raise ValueError('replay tensor contract')
    if any(r['target_index']!=int(t) for r,t in zip(replay_rows,ry)):raise ValueError('replay label contract')
    train_mask=purposes=='train';tx,ty=x[train_mask],y[train_mask];vx,vy=x[~train_mask],y[~train_mask]
    original_ids=np.array([r['sampling_identity'] for r in provenance if r['purpose']=='train'])
    replay_ids=np.array([r['sampling_identity'] for r in replay_rows]);val_rows=[r for r in provenance if r['purpose']=='validation']
    _set_deterministic(config['seed']);torch.set_num_threads(config['threads'])
    initial=torch.load(INITIAL,map_location='cpu',weights_only=True)
    if initial['labels']!=list(LABELS) or initial['upstream']!=original_audit:raise ValueError('initial model contract drift')
    teacher=make_model();teacher.load_state_dict(initial['state_dict'],strict=True);teacher.eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    student=make_model();student.load_state_dict(initial['state_dict'],strict=True)
    initial_bn={k:v.clone() for k,v in student.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k}
    parent=metrics(vy,predict(teacher,vx).argmax(1));history=[dict(epoch=0,metrics=parent)]
    best=copy.deepcopy(student.state_dict());best_epoch=0
    optimizer=torch.optim.Adam(student.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    output.mkdir(parents=True,exist_ok=False);write_json(output/'frozen-config.json',config)
    write_json(output/'parent-validation.json',parent)
    for epoch in range(1,config['epochs']+1):
        start=time.monotonic();fixed_bn_train(student);loss_sum=0.
        batches=domain_batches(ty,original_ids,ry,replay_ids,config['per_class_per_domain'],config['seed']+epoch)
        for oi,ri in batches:
            optimizer.zero_grad(set_to_none=True)
            loss=mixed_loss(student,teacher,torch.from_numpy(tx[oi]),torch.from_numpy(ty[oi]),
                torch.from_numpy(rx[ri]),torch.from_numpy(ry[ri]),config['temperature'],config['distill_weight'],
                config.get('replay_loss_fraction',.5))
            if not torch.isfinite(loss):raise ValueError('nonfinite adaptation loss')
            loss.backward();optimizer.step();loss_sum+=float(loss.detach())
        for key,original in initial_bn.items():
            if not torch.equal(student.state_dict()[key],original):raise ValueError('BN statistics changed')
        m=metrics(vy,predict(student,vx).argmax(1));record=dict(epoch=epoch,metrics=m,
            loss=loss_sum/len(batches),seconds=time.monotonic()-start,eligible=eligible(m,parent))
        history.append(record)
        if select_epoch(history,parent)['epoch']==epoch:best=copy.deepcopy(student.state_dict());best_epoch=epoch
        with (output/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(dict(epoch=epoch,loss=record['loss'],macro=m['macro_recall'],false_alert=m['negative_false_alert_rate'],
                              eligible=record['eligible'],selected_epoch=best_epoch,seconds=record['seconds'])),flush=True)
    final=copy.deepcopy(student.state_dict());parent_thresholds=json.loads((INITIAL.parent/'metrics.json').read_text())['thresholds']
    artifacts={}
    for name,state,epoch in [('selected',best,best_epoch),('final',final,config['epochs'])]:
        directory=output/name;directory.mkdir();student.load_state_dict(state,strict=True)
        logits=predict(student,vx);scores=softmax(logits);thresholds=calibrate_thresholds(vy,scores)
        raw=scores.argmax(1);fixed=apply_thresholds(scores,parent_thresholds);fitted=apply_thresholds(scores,thresholds)
        checkpoint=directory/'soundjie_alert6_float32.pt'
        with checkpoint.open('xb') as f:torch.save(dict(schema=1,state_dict=state,labels=list(LABELS),seed=config['seed'],epoch=epoch,
            upstream=dict(original=original_audit,replay=audit,initial_sha256=config['initial_checkpoint_sha256'])),f)
        report=dict(raw=metrics(vy,raw),fixed_parent_thresholds=metrics(vy,fixed),refit_thresholds=metrics(vy,fitted),
            negative_raw=negative_breakdown(vy,raw,val_rows),negative_fixed=negative_breakdown(vy,fixed,val_rows),
            negative_refit=negative_breakdown(vy,fitted,val_rows),thresholds=thresholds,parent_thresholds=parent_thresholds,
            thresholds_status='same_original_validation_calibration_not_independent',epoch=epoch)
        write_json(directory/'metrics.json',report)
        with (directory/'validation-logits.npz').open('xb') as f:np.savez(f,logits=logits,y=vy)
        artifacts[name]=dict(directory=str(directory),epoch=epoch,checkpoint_sha256=digest_file(checkpoint),metrics=report)
    write_json(output/'selection.json',dict(schema=1,artifacts=artifacts,selected_epoch=best_epoch,
        parent_retained=best_epoch==0,selection=config['selection'],history=history,config=config,
        bn_statistics_unchanged=True,teacher_frozen=True,real_validation_used_for_selection=False,
        committed_cache_receipt_sha256=digest_file(receipt_path),
        board_deployed=False,independent_acceptance=False,script_sha256=digest_file(Path(__file__))))
    print(json.dumps(dict(stage='adaptation_training_complete',selected_epoch=best_epoch,parent_retained=best_epoch==0)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','receipt','train'])
    p.add_argument('--config',type=Path,default=ROOT/'config/soundjie_alert6_adaptation.json')
    p.add_argument('--features',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receipt',type=Path)
    a=p.parse_args();config=json.loads(a.config.read_text(encoding='utf-8'))
    if a.mode=='prepare':prepare(config,a.output)
    elif a.mode=='receipt':publish_receipt(config,a.features,a.output)
    else:train(config,a.features,a.output,a.receipt)


if __name__=='__main__':main()
