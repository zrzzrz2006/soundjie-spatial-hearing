"""Frozen paired six-output control / five-output merged-siren training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from scripts.host.ablate_soundjie_glass import checked_inputs,state_sha,selection_key
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.soundjie_alert6 import LABELS,balanced_source_indices
from scripts.host.soundjie_siren_merge import (
    LABELS5,PROTOCOL,OUTPUT_ROOT,committed_protocol,paired_models,map_targets,
    probabilities,metrics,calibrate_thresholds,apply_thresholds,
)
from scripts.host.train_soundjie_alert6 import predict,write_json


def train(seed,output):
    config=committed_protocol();output=Path(output);output.resolve().relative_to(OUTPUT_ROOT.resolve())
    if seed not in config['seeds']:raise ValueError('unfrozen seed')
    if output.exists():raise FileExistsError(output)
    x,y,purpose,rows,audit=checked_inputs(config)
    tr=purpose=='train';va=purpose=='validation';tx,ty=x[tr],y[tr];vx,vy=x[va],y[va]
    ids=np.array([r['sampling_identity'] for r in rows if r['purpose']=='train'])
    val_rows=[r for r in rows if r['purpose']=='validation']
    six,five=paired_models(seed);torch.set_num_threads(config['threads']);models={'control6':six,'merged5':five}
    shared=[k for k in six.state_dict() if not k.startswith('fc8.')]
    trunk=[state_sha(m.state_dict(),shared) for m in models.values()]
    if trunk[0]!=trunk[1]:raise ValueError('unpaired trunk')
    initial={name:state_sha(m.state_dict()) for name,m in models.items()}
    targets={'control6':ty,'merged5':map_targets(ty)}
    optimizers={n:torch.optim.Adam(m.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay']) for n,m in models.items()}
    output.mkdir(parents=True,exist_ok=False);(output/'epoch-logits').mkdir()
    write_json(output/'protocol.json',config);write_json(output/'validation-sources.json',val_rows)
    best={};states={}
    for epoch in range(1,config['epochs']+1):
        started=time.monotonic();indices=balanced_source_indices(ty,ids,config['per_original_class_epoch'],seed+epoch-1)
        row=dict(epoch=epoch,sampling_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                 original_class_counts=np.bincount(ty[indices],minlength=6).tolist(),arms={})
        logits_record={}
        for name,model in models.items():
            model.train();opt=optimizers[name];total=0.
            for offset in range(0,len(indices),config['batch_size']):
                batch=indices[offset:offset+config['batch_size']];opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(tx[batch])),torch.from_numpy(targets[name][batch]))
                if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
                loss.backward();opt.step();total+=float(loss.detach())*len(batch)
            logits=predict(model,vx);m=metrics(vy,probabilities(logits).argmax(1),val_rows)
            entry=dict(epoch=epoch,metrics=m);logits_record[name]=logits
            if name not in best or selection_key(entry)>selection_key(best[name]):
                best[name]=entry;states[name]=copy.deepcopy(model.state_dict())
            row['arms'][name]=dict(loss=total/len(indices),**entry)
        path=output/'epoch-logits'/('%02d.npz'%epoch)
        with path.open('xb') as f:np.savez(f,y_original=vy,**logits_record)
        row['logits_sha256']=digest_file(path);row['seconds']=time.monotonic()-started
        with (output/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(seed=seed,epoch=epoch,seconds=row['seconds'],
            arms={n:dict(macro=v['metrics']['macro_recall'],bg_fpr=v['metrics']['negative_false_alert_rate']) for n,v in row['arms'].items()})),flush=True)
    artifacts={}
    for name,model in models.items():
        artifacts[name]={};final=copy.deepcopy(model.state_dict())
        for kind,state,ep in [('selected',states[name],best[name]['epoch']),('final',final,config['epochs'])]:
            folder=output/name/kind;folder.mkdir(parents=True,exist_ok=False)
            model.load_state_dict(state);logits=predict(model,vx);scores=probabilities(logits)
            thresholds=calibrate_thresholds(vy,scores,config['thresholds'])
            report=dict(raw=metrics(vy,scores.argmax(1),val_rows),
                thresholded=metrics(vy,apply_thresholds(scores,thresholds),val_rows),thresholds=thresholds,
                thresholds_status='original_validation_calibration_only_not_independent',board_test=False)
            if name=='control6':report['label_only']=metrics(vy,map_targets(logits.argmax(1)),val_rows)
            labels=LABELS if name=='control6' else LABELS5
            with (folder/'model.pt').open('xb') as f:torch.save(dict(state_dict=state,labels=list(labels),seed=seed,epoch=ep,arm=name,config=config,upstream=audit),f)
            with (folder/'validation-logits.npz').open('xb') as f:np.savez(f,logits=logits,y_original=vy)
            write_json(folder/'metrics.json',report)
            artifacts[name][kind]=dict(directory=str(folder),epoch=ep,metrics=report,
                checkpoint_sha256=digest_file(folder/'model.pt'),logits_sha256=digest_file(folder/'validation-logits.npz'))
    result=dict(schema=1,seed=seed,config=config,config_sha256=digest_file(PROTOCOL),
        initial_model_sha256=initial,shared_initial_trunk_sha256=trunk[0],arms=artifacts,
        original_train_counts=np.bincount(ty,minlength=6).tolist(),merged_train_counts=np.bincount(map_targets(ty),minlength=5).tolist(),
        training_script_sha256=digest_file(Path(__file__)),actual_torch_threads=torch.get_num_threads(),
        selection_used_replay=False,independent_acceptance=False,board_test=False,
        files={str(output/n):digest_file(output/n) for n in ['protocol.json','progress.jsonl','validation-sources.json']})
    write_json(output/'result.json',result)
    print(json.dumps(dict(seed=seed,complete=True,epochs={n:v['selected']['epoch'] for n,v in artifacts.items()})),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();train(a.seed,a.output)


if __name__=='__main__':main()
