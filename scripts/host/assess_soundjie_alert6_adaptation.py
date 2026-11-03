"""Post-selection acoustic diagnostics and numerical export checks only."""
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scripts.host.adapt_soundjie_alert6 import CAPTURE,ORIGINAL,checked
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.evaluate_soundjie_alert6 import fold_and_check
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.prepare_soundjie_d3 import compute_feature
from scripts.host.soundjie_alert6 import ROOT,LABELS,make_model
from scripts.host.train_soundjie_alert6 import write_json,predict,softmax,apply_thresholds


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists():raise FileExistsError(a.output)
    selection=json.loads((a.training/'selection.json').read_text(encoding='utf-8'));config=selection['config']
    checked(CAPTURE/'alignment-report.json',config['alignment_sha256'])
    aligned=json.loads((CAPTURE/'alignment-report.json').read_text(encoding='utf-8'))
    previous_path=CAPTURE/'paired-host-diagnostics.json'
    receipt=json.loads((ROOT/'evidence/soundjie-d4/alert6-acoustic-results-20261031.json').read_text(encoding='utf-8'))
    checked(previous_path,receipt['files'][str(previous_path)])
    previous=json.loads(previous_path.read_text(encoding='utf-8'))
    if previous['model_hashes']['seed-20261031']!=config['initial_checkpoint_sha256']:raise ValueError('parent comparison mismatch')
    events={r['event_index']:r for r in aligned['events'] if not r['quarantined']}
    rows=[r for r in previous['windows'] if r['domain']=='acoustic'];arrays=[]
    frontend=load_frontend_config(ROOT/'config/logmel_frontend_v1.json');audio={}
    for index,event in events.items():
        checked(event['captured_wav'],event['captured_sha256']);signal,rate=sf.read(event['captured_wav'])
        if rate!=16000 or signal.ndim!=1:raise ValueError('capture excerpt contract')
        audio[index]=signal
    for row in rows:
        start=row['start_sample'];arrays.append(compute_feature(audio[row['event_index']][start:start+16000],frontend))
    x=np.stack(arrays);outputs={};exports={};a.output.mkdir(parents=True,exist_ok=False)
    parent_scores=np.array([r['scores']['seed-20261031']['values'] for r in rows])
    parent_thresholds=json.loads((ORIGINAL/'candidates/seed-20261031/metrics.json').read_text())['thresholds']
    outputs['parent']=dict(scores=parent_scores,fixed=apply_thresholds(parent_scores,parent_thresholds),
                           fitted=apply_thresholds(parent_scores,parent_thresholds))
    for name,artifact in selection['artifacts'].items():
        path=Path(artifact['directory'])/'soundjie_alert6_float32.pt';checked(path,artifact['checkpoint_sha256'])
        model=make_model();checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['labels']!=list(LABELS):raise ValueError('six output order drift')
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval();scores=softmax(predict(model,x))
        fitted=artifact['metrics']['thresholds']
        outputs[name]=dict(scores=scores,fixed=apply_thresholds(scores,parent_thresholds),fitted=apply_thresholds(scores,fitted))
        arrays_fused,parity=fold_and_check(model,x[:12])
        folder=a.output/name;folder.mkdir()
        with (folder/'fused-weights.npz').open('xb') as f:np.savez(f,**arrays_fused)
        with (folder/'parity-inputs.npy').open('xb') as f:np.save(f,x[:12],allow_pickle=False)
        with (folder/'pytorch-logits.npy').open('xb') as f:np.save(f,predict(model,x[:12]),allow_pickle=False)
        exports[name]=dict(checkpoint_sha256=artifact['checkpoint_sha256'],folding=parity,
            files={str(p):digest_file(p) for p in folder.iterdir()},caffe_parity=False,int8=False,DPU=False)
    clips=[]
    for index,event in events.items():
        positions=[i for i,r in enumerate(rows) if r['event_index']==index]
        result={k:event[k] for k in ['event_index','recording_id','purpose','target_index','negative_subtype']}
        result['models']={}
        for name,value in outputs.items():
            score=value['scores'][positions];raw=score.argmax(1)
            result['models'][name]=dict(clip_mean_top1=int(score.mean(0).argmax()),
                target_score_mean=float(score.mean(0)[event['target_index']]),raw_window_top1=raw.tolist(),
                fixed_window_top1=value['fixed'][positions].tolist(),refit_window_top1=value['fitted'][positions].tolist())
        clips.append(result)
    summary={}
    for name in outputs:
        summary[name]={}
        for purpose in ['train','validation']:
            subset=[c for c in clips if c['purpose']==purpose]
            summary[name][purpose]=dict(clips=len(subset),
                clip_mean_matches_weak_label=sum(c['models'][name]['clip_mean_top1']==c['target_index'] for c in subset),
                background_clips=sum(c['target_index']==5 for c in subset),
                background_clip_alerts=sum(c['target_index']==5 and c['models'][name]['clip_mean_top1']!=5 for c in subset),
                positive_clips_with_any_correct_fixed_alert=sum(c['target_index']<5 and c['target_index'] in c['models'][name]['fixed_window_top1'] for c in subset),
                positive_clips=sum(c['target_index']<5 for c in subset))
    report=dict(schema=1,selection_sha256=digest_file(a.training/'selection.json'),summary=summary,clips=clips,exports=exports,
        original_validation={k:v['metrics'] for k,v in selection['artifacts'].items()},
        selected_epoch=selection['selected_epoch'],parent_retained=selection['parent_retained'],
        selected_using_acoustic_results=False,independent_acceptance=False,new_DPU_inference=False,
        warning='Known weak-label replay; train replay metrics are fitted data. Seven validation clips are previously exposed sources, not D5. No threshold tuning on replay; final epoch is descriptive, not replacement for selected checkpoint.')
    write_json(a.output/'report.json',report);print(json.dumps(dict(selected_epoch=selection['selected_epoch'],summary=summary)),flush=True)


if __name__=='__main__':main()
