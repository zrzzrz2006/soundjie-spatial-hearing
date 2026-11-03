"""Post-selection glass ablation assessment; never called by training."""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from scripts.host.ablate_soundjie_glass import (
    ROOT,LABELS,LABELS5,OUTPUT_ROOT,committed_protocol,verify_pins,paired_models,
    map_without_glass,common_metrics,
)
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.prepare_soundjie_d3 import compute_feature
from scripts.host.train_soundjie_alert6 import predict,softmax,write_json


def replay_summary(clips):
    def one(rows,domain):
        return dict(clips=len(rows),matches=sum(r[domain]['mapped_top1']==int(map_without_glass(np.array([r['target_index']]))[0]) for r in rows),
            background_clips=sum(r['target_index']==5 for r in rows),
            background_false_alerts=sum(r['target_index']==5 and r[domain]['mapped_top1']!=4 for r in rows))
    out={};glass={}
    for purpose in ['train','validation']:
        out[purpose]={d:one([r for r in clips if r['purpose']==purpose and r['target_index']!=3],d) for d in ['digital','acoustic']}
        glass[purpose]={d:one([r for r in clips if r['purpose']==purpose and r['target_index']==3],d) for d in ['digital','acoustic']}
    out['glass']=glass
    return out


def promotion_decision(rows,guards):
    changes=[]
    for row in rows:
        six,five=row['retain6'],row['drop5'];sm,fm=six['metrics'],five['metrics']
        changes.append(dict(seed=row['seed'],macro_gain=fm['macro_recall']-sm['macro_recall'],
            background_fpr_change=fm['negative_false_alert_rate']-sm['negative_false_alert_rate'],
            per_class_recall_change={k:fm['per_class'][k]['recall']-sm['per_class'][k]['recall'] for k in LABELS5[:4]},
            acoustic_validation_matches_change=five['replay']['validation']['acoustic']['matches']-six['replay']['validation']['acoustic']['matches']))
    if not changes:raise ValueError('no paired seeds')
    # Rate differences such as 14/20 - 16/20 are slightly below -0.1 in binary.
    eps=1e-12
    checks=dict(each_seed_macro_improves=all(r['macro_gain']>eps for r in changes),
        mean_macro_gain_at_least_target=float(np.mean([r['macro_gain'] for r in changes]))>=guards['mean_macro_gain']-eps,
        original_background_not_regressed=all(r['background_fpr_change']<=guards['max_background_fpr_increase']+eps for r in changes),
        positive_recalls_guard=all(min(r['per_class_recall_change'].values())>=-guards['max_positive_class_recall_decrease']-eps for r in changes),
        no_replay_validation_regression=all(r['acoustic_validation_matches_change']>=0 for r in changes),
        replay_validation_gain=any(r['acoustic_validation_matches_change']>=guards['at_least_one_seed_replay_gain'] for r in changes))
    return dict(eligible_for_deployment_preparation=all(checks.values()),checks=checks,paired_changes=changes,
        mean_macro_gain=float(np.mean([r['macro_gain'] for r in changes])),
        board_deployment=False,independent_acceptance=False)


def replay_inputs(protocol):
    alignment=Path(protocol['replay_alignment'])
    verify_pins({str(alignment):protocol['replay_alignment_sha256']})
    aligned=json.loads(alignment.read_text(encoding='utf-8'))
    verify_pins({str(alignment.parent/'capture-stereo.wav'):aligned['recording_sha256']})
    frontend=load_frontend_config(ROOT/'config/logmel_frontend_v1.json')
    # Frontend hash must match the pinned original feature-cache audit.
    cache_audit=json.loads((Path(protocol['features'])/'audit.json').read_text(encoding='utf-8'))
    if frontend.source_sha256!=cache_audit['frontend_sha256']:raise ValueError('frontend drift')
    events=[e for e in aligned['events'] if not e['quarantined'] and e['reception']['passed']]
    arrays=[];rows=[]
    for e in events:
        if e['purpose'] not in ('train','validation'):raise ValueError('unexpected replay purpose')
        for domain,key,sha in [('digital','reference_wav','reference_sha256'),('acoustic','captured_wav','captured_sha256')]:
            verify_pins({e[key]:e[sha]});signal,rate=sf.read(e[key],dtype='float64')
            if rate!=16000 or signal.ndim!=1 or not 39500<=len(signal)<=40500:raise ValueError('replay format drift')
            for start in np.rint(np.linspace(0,len(signal)-16000,4)).astype(int):
                arrays.append(compute_feature(signal[start:start+16000],frontend))
                rows.append(dict(event_index=e['event_index'],domain=domain,start_sample=int(start)))
    return np.stack(arrays),rows,events


def evaluate_replay(model,x,windows,events):
    scores=softmax(predict(model,x));outputs=scores.shape[1];clips=[]
    for e in events:
        row={k:e[k] for k in ['event_index','recording_id','purpose','target_index','negative_subtype']}
        for domain in ['digital','acoustic']:
            mask=[i for i,w in enumerate(windows) if w['event_index']==e['event_index'] and w['domain']==domain]
            values=scores[mask];raw=int(values.mean(0).argmax())
            mapped=int(map_without_glass(np.array([raw]))[0]) if outputs==6 else raw
            preds=values.argmax(1);mapped_windows=map_without_glass(preds) if outputs==6 else preds
            row[domain]=dict(raw_top1=raw,mapped_top1=mapped,window_top1=mapped_windows.tolist(),
                             mean_scores=values.mean(0).tolist())
        clips.append(row)
    return dict(summary=replay_summary(clips),clips=clips)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=OUTPUT_ROOT)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.resolve().relative_to(OUTPUT_ROOT.resolve())
    if a.output.exists():raise FileExistsError(a.output)
    protocol=committed_protocol();verify_pins(protocol['file_pins'])
    x,windows,events=replay_inputs(protocol)
    all_results=[];comparison=[];source_files={}
    for seed in protocol['seeds']:
        path=a.root/('seed-'+str(seed))/'result.json';source_files[str(path)]=digest_file(path)
        result=json.loads(path.read_text(encoding='utf-8'))
        if result['seed']!=seed or result['config']!=protocol:raise ValueError('training protocol drift')
        verify_pins(result['files'])
        pairs=paired_models(seed);torch.set_num_threads(protocol['threads'])
        pair_row=dict(seed=seed);assessed={}
        for name,model in zip(['retain6','drop5'],pairs):
            assessed[name]={}
            for kind in ['selected','final']:
                artifact=result['arms'][name][kind];folder=Path(artifact['directory']);checkpoint=folder/'model.pt'
                verify_pins({str(checkpoint):artifact['checkpoint_sha256'],str(folder/'validation-logits.npz'):artifact['logits_sha256']})
                state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                labels=LABELS if name=='retain6' else LABELS5
                if state['labels']!=list(labels) or state['config']!=protocol or state['seed']!=seed:
                    raise ValueError('checkpoint contract drift')
                model.load_state_dict(state['state_dict'],strict=True);model.eval()
                replay=evaluate_replay(model,x,windows,events)
                with np.load(folder/'validation-logits.npz',allow_pickle=False) as data:
                    original=common_metrics(data['y_original'],data['logits'].argmax(1),len(labels))
                if any(original[k]!=artifact['metrics'][k] for k in original):raise ValueError('metrics mismatch')
                assessed[name][kind]=dict(epoch=artifact['epoch'],metrics=artifact['metrics'],replay=replay,
                    checkpoint_sha256=artifact['checkpoint_sha256'],path=str(checkpoint))
                if kind=='selected':pair_row[name]=dict(metrics=artifact['metrics'],replay=replay['summary'])
            # The final and selected artifacts are both explicit; never substitute final.
        comparison.append(pair_row);all_results.append(dict(seed=seed,arms=assessed))
    decision=promotion_decision(comparison,protocol['promotion_guards'])
    original_existing={}
    for seed in protocol['seeds']:
        directory=Path(protocol['features']).parent/'candidates'/('seed-'+str(seed))
        report=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        raw=np.array(report['raw']['confusion_matrix']);threshold=np.array(report['thresholded']['confusion_matrix'])
        original_existing[str(seed)]=dict(raw_positive_others_to_glass=int(raw[[0,1,2,4],3].sum()),
            raw_other_positive_support=int(raw[[0,1,2,4]].sum()),raw_background_to_glass=int(raw[5,3]),
            thresholded_background_to_glass=int(threshold[5,3]),
            raw_siren_to_background=int(raw[1,5]),raw_horn_to_background=int(raw[2,5]),
            metrics_file_sha256=digest_file(directory/'metrics.json'))
    report=dict(schema=1,protocol=protocol,training_results_sha256=source_files,results=all_results,decision=decision,
        existing_six_model_direct_confusion=original_existing,
        replay_events=len(events),replay_windows=len(windows),independent_acceptance=False,board_test=False,
        limitations=['Previously exposed development validation; two seeds do not establish population-level causality.',
            'Glass becomes background in drop5, changing its semantic target and effective negative weight; data/exposure otherwise paired.',
            'No-glass comparison excludes35 glass from321 common windows; old193 background denominator unchanged.',
            'Clip replay labels are weak; train examples are fitted data. Raw argmax diagnostics are not thresholded UI metrics.',
            'No DPU/INT8 or physical replay of new weights; enrolled-video subtitles and board files unchanged.'])
    a.output.mkdir(parents=True,exist_ok=False);write_json(a.output/'report.json',report)
    print(json.dumps(dict(decision=decision,output=str(a.output/'report.json'))),flush=True)


if __name__=='__main__':main()
