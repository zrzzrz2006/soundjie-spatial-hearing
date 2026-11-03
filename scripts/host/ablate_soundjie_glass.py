"""Paired host experiment: retain glass versus learn it as background."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.soundjie_alert6 import ROOT,LABELS,make_model,balanced_source_indices,metrics
from scripts.host.train_soundjie_alert6 import load_cache,predict,write_json
from scripts.host.train_soundjie_d3 import _set_deterministic

KEEP=(0,1,2,4,5)
LABELS5=tuple(LABELS[i] for i in KEEP)
PROTOCOL=ROOT/'config/soundjie_glass_ablation.json'
OUTPUT_ROOT=Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_glass_ablation/20261031a')


def map_without_glass(values):
    values=np.asarray(values)
    if not np.issubdtype(values.dtype,np.integer) or np.any((values<0)|(values>5)):
        raise ValueError('expected original integer labels 0..5')
    return np.array([0,1,2,4,3,4],dtype=np.int64)[values]


def common_metrics(y,pred,outputs):
    y=np.asarray(y);pred=np.asarray(pred)
    if y.ndim!=1 or pred.shape!=y.shape or outputs not in (5,6):raise ValueError('invalid predictions')
    mapped_y=map_without_glass(y)
    if not np.issubdtype(pred.dtype,np.integer) or np.any((pred<0)|(pred>=outputs)):
        raise ValueError('invalid output indices')
    p=map_without_glass(pred) if outputs==6 else pred
    common=y!=3;glass=~common
    cm=np.zeros((5,5),dtype=np.int64)
    np.add.at(cm,(mapped_y[common],p[common]),1)
    per={}
    for c,label in enumerate(LABELS5):
        n=int(cm[c].sum());per[label]=dict(support=n,recall=float(cm[c,c]/n) if n else 0.)
    background=y==5
    return dict(support=int(common.sum()),accuracy=float(np.mean(mapped_y[common]==p[common])),
        macro_recall=float(np.mean([v['recall'] for v in per.values()])),
        negative_false_alert_rate=float(np.mean(p[background]!=4)) if background.any() else None,
        per_class=per,confusion_matrix=cm.tolist(),
        glass_as_negative=dict(support=int(glass.sum()),
            false_alert_rate=float(np.mean(p[glass]!=4)) if glass.any() else None,
            output_counts=np.bincount(p[glass],minlength=5).tolist()))


def paired_models(seed):
    _set_deterministic(seed)
    six=make_model();five=copy.deepcopy(six)
    five.fc8=torch.nn.Linear(six.fc8.in_features,5)
    with torch.no_grad():
        five.fc8.weight.copy_(six.fc8.weight[list(KEEP)])
        five.fc8.bias.copy_(six.fc8.bias[list(KEEP)])
    return six,five


def verify_pins(pins):
    for path,expected in pins.items():
        if digest_file(path)!=expected:raise ValueError('pinned input drift: '+str(path))


def committed_protocol():
    raw=PROTOCOL.read_bytes()
    committed=subprocess.check_output(['git','show','HEAD:'+PROTOCOL.relative_to(ROOT).as_posix()],cwd=ROOT)
    if raw!=committed:raise ValueError('protocol must match committed HEAD before training')
    return json.loads(raw)


def checked_inputs(config):
    verify_pins(config['file_pins'])
    result=load_cache(config['features'])
    _,y,purpose,_,audit=result
    if int((purpose=='train').sum())!=12252 or int((purpose=='validation').sum())!=356:
        raise ValueError('unexpected source cache size')
    if int(((purpose=='validation')&(y==3)).sum())!=35:raise ValueError('glass validation split drift')
    if audit['manifest_sha256']!=config['file_pins']['F:/MLK/DPU_ZYNQ7000/datasets/soundjie_alert6_20261031/manifest.jsonl']:
        raise ValueError('cache bound to different manifest')
    return result


def selection_key(row):
    m=row['metrics']
    return (m['macro_recall'],-m['negative_false_alert_rate'],-row['epoch'])


def state_sha(state,keys=None):
    h=hashlib.sha256()
    for key in sorted(state if keys is None else keys):
        h.update(key.encode());h.update(state[key].cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_pair(seed,output):
    config=committed_protocol()
    if seed not in config['seeds']:raise ValueError('unfrozen seed')
    output=Path(output);output.resolve().relative_to(OUTPUT_ROOT.resolve())
    if output.exists():raise FileExistsError(output)
    x,y,purpose,rows,audit=checked_inputs(config)
    train=purpose=='train';val=purpose=='validation'
    tx,ty=x[train],y[train];vx,vy=x[val],y[val]
    train_ids=np.array([r['sampling_identity'] for r in rows if r['purpose']=='train'])
    val_rows=[r for r in rows if r['purpose']=='validation']
    six,five=paired_models(seed);torch.set_num_threads(config['threads'])
    models={'retain6':six,'drop5':five}
    initial={name:state_sha(model.state_dict()) for name,model in models.items()}
    shared_keys=[k for k in six.state_dict() if not k.startswith('fc8.')]
    trunk=[state_sha(m.state_dict(),shared_keys) for m in models.values()]
    if trunk[0]!=trunk[1]:raise ValueError('unpaired initialization')
    targets={'retain6':ty,'drop5':map_without_glass(ty)}
    optimizers={n:torch.optim.Adam(m.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']) for n,m in models.items()}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'protocol.json',config)
    write_json(output/'validation-sources.json',val_rows)
    best={};best_states={};history=[]
    for epoch in range(1,config['epochs']+1):
        started=time.monotonic()
        # One index array is consumed by both arms, with no validation positions.
        indices=balanced_source_indices(ty,train_ids,config['per_original_class_epoch'],seed+epoch-1)
        sample_sha=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()
        epoch_row=dict(epoch=epoch,sampling_sha256=sample_sha,original_class_counts=np.bincount(ty[indices],minlength=6).tolist(),arms={})
        for name,model in models.items():
            model.train();optimizer=optimizers[name];loss_total=0.
            for offset in range(0,len(indices),config['batch_size']):
                b=indices[offset:offset+config['batch_size']];optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(tx[b])),torch.from_numpy(targets[name][b]))
                if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                loss.backward();optimizer.step();loss_total+=float(loss.detach())*len(b)
            logits=predict(model,vx)
            m=common_metrics(vy,logits.argmax(1),logits.shape[1])
            choice=dict(epoch=epoch,metrics=m)
            if name not in best or selection_key(choice)>selection_key(best[name]):
                best[name]=choice;best_states[name]=copy.deepcopy(model.state_dict())
            epoch_row['arms'][name]=dict(loss=loss_total/len(indices),**choice)
        epoch_row['seconds']=time.monotonic()-started;history.append(epoch_row)
        with (output/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(epoch_row)+'\n')
        print(json.dumps(dict(seed=seed,epoch=epoch,seconds=epoch_row['seconds'],
            arms={n:dict(macro=v['metrics']['macro_recall'],bg_fpr=v['metrics']['negative_false_alert_rate']) for n,v in epoch_row['arms'].items()})),flush=True)
    artifacts={}
    for name,model in models.items():
        artifacts[name]={};final_state=copy.deepcopy(model.state_dict())
        for kind,state,ep in [('selected',best_states[name],best[name]['epoch']),('final',final_state,config['epochs'])]:
            folder=output/name/kind;folder.mkdir(parents=True,exist_ok=False)
            model.load_state_dict(state);logits=predict(model,vx)
            m=common_metrics(vy,logits.argmax(1),logits.shape[1])
            m['negative_subtypes']={}
            for subtype in sorted({r['negative_subtype'] for r in val_rows if r['target_index']==5}):
                mask=np.array([r['target_index']==5 and r['negative_subtype']==subtype for r in val_rows])
                pred=map_without_glass(logits.argmax(1)) if name=='retain6' else logits.argmax(1)
                m['negative_subtypes'][subtype]=dict(support=int(mask.sum()),false_alerts=int(np.sum(pred[mask]!=4)))
            if name=='retain6':m['original_six_task']=metrics(vy,logits.argmax(1))
            labels=LABELS if name=='retain6' else LABELS5
            path=folder/'model.pt'
            with path.open('xb') as f:torch.save(dict(state_dict=state,labels=list(labels),seed=seed,epoch=ep,arm=name,config=config,upstream=audit),f)
            with (folder/'validation-logits.npz').open('xb') as f:np.savez(f,logits=logits,y_original=vy)
            write_json(folder/'metrics.json',m)
            artifacts[name][kind]=dict(directory=str(folder),epoch=ep,metrics=m,checkpoint_sha256=digest_file(path),
                logits_sha256=digest_file(folder/'validation-logits.npz'))
    result=dict(schema=1,seed=seed,config=config,config_sha256=digest_file(PROTOCOL),
        initial_model_sha256=initial,shared_initial_trunk_sha256=trunk[0],
        training_script_sha256=digest_file(Path(__file__)),actual_torch_threads=torch.get_num_threads(),
        train_windows=len(ty),validation_windows=len(vy),original_train_class_counts=np.bincount(ty,minlength=6).tolist(),
        arms=artifacts,selection_used_replay=False,board_test=False,independent_acceptance=False,
        files={str(output/n):digest_file(output/n) for n in ['progress.jsonl','protocol.json','validation-sources.json']})
    write_json(output/'result.json',result)
    print(json.dumps(dict(seed=seed,complete=True,epochs={n:v['selected']['epoch'] for n,v in artifacts.items()})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();train_pair(a.seed,a.output)


if __name__=='__main__':main()
