"""Five-class broad-siren contract and paired experiment metrics."""
import copy
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from scripts.host.soundjie_alert6 import ROOT,LABELS,make_model
from scripts.host.train_soundjie_d3 import _set_deterministic
from scripts.host.train_soundjie_alert6 import softmax

LABELS5=('siren','car_horn','glass_breaking','door_knock_bell','background_unknown')
MAPPING=np.array([0,0,1,2,3,4],dtype=np.int64)
PROTOCOL=ROOT/'config/soundjie_siren_merge.json'
OUTPUT_ROOT=Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_siren_merge/20261031a')


def map_targets(values):
    values=np.asarray(values)
    if not np.issubdtype(values.dtype,np.integer) or np.any((values<0)|(values>=6)):
        raise ValueError('original six integer targets required')
    return MAPPING[values]


def validate_scores(scores,outputs):
    scores=np.asarray(scores)
    if scores.ndim!=2 or scores.shape[1]!=outputs or not np.isfinite(scores).all() or np.any(scores<0) or not np.allclose(scores.sum(1),1,rtol=1e-6,atol=1e-6):
        raise ValueError('normalized finite class probabilities required')
    return scores


def merge_probabilities(scores):
    scores=validate_scores(scores,6)
    return np.column_stack((scores[:,0]+scores[:,1],scores[:,2:]))


def probabilities(logits):
    scores=softmax(logits)
    return merge_probabilities(scores) if scores.shape[1]==6 else validate_scores(scores,5)


def metrics(y_original,pred,rows=None):
    y=np.asarray(y_original);p=np.asarray(pred);target=map_targets(y)
    if y.ndim!=1 or not len(y) or p.shape!=y.shape or not np.issubdtype(p.dtype,np.integer) or np.any((p<0)|(p>=5)):
        raise ValueError('invalid five-class predictions')
    cm=np.zeros((5,5),dtype=np.int64);np.add.at(cm,(target,p),1);per={}
    for c,label in enumerate(LABELS5):
        n=int(cm[c].sum());pos=int(cm[:,c].sum());tp=int(cm[c,c])
        per[label]=dict(support=n,correct=tp,recall=tp/n if n else 0.,precision=tp/pos if pos else 0.)
    bg=y==5
    result=dict(support=len(y),accuracy=float(np.mean(p==target)),
        macro_recall=float(np.mean([v['recall'] for v in per.values()])),
        negative_false_alert_rate=float(np.mean(p[bg]!=4)) if bg.any() else None,
        per_class=per,confusion_matrix=cm.tolist(),original_family={},negative_subtypes={})
    for index,label in enumerate(LABELS):
        mask=y==index
        result['original_family'][label]=dict(support=int(mask.sum()),correct=int(np.sum(p[mask]==target[mask])),
            recall=float(np.mean(p[mask]==target[mask])) if mask.any() else 0.)
    if rows is not None:
        if len(rows)!=len(y) or any(r['target_index']!=int(a) for r,a in zip(rows,y)):raise ValueError('source-label mismatch')
        for name in sorted({r['negative_subtype'] for r in rows if r['target_index']==5}):
            mask=np.array([r['target_index']==5 and r['negative_subtype']==name for r in rows])
            result['negative_subtypes'][name]=dict(support=int(mask.sum()),false_alerts=int(np.sum(p[mask]!=4)),false_alert_rate=float(np.mean(p[mask]!=4)))
    return result


def paired_models(seed):
    _set_deterministic(seed);six=make_model();five=copy.deepcopy(six)
    five.fc8=torch.nn.Linear(six.fc8.in_features,5)
    with torch.no_grad():
        five.fc8.weight.copy_(six.fc8.weight[[0,2,3,4,5]])
        five.fc8.bias.copy_(six.fc8.bias[[0,2,3,4,5]])
    return six,five


def committed_protocol():
    raw=PROTOCOL.read_bytes()
    pinned=subprocess.check_output(['git','show','HEAD:'+PROTOCOL.relative_to(ROOT).as_posix()],cwd=ROOT)
    if raw!=pinned:raise ValueError('protocol must match Git HEAD')
    classes=json.loads((ROOT/'config/soundjie_siren5_classes.json').read_text(encoding='utf-8'))
    if classes['labels']!=list(LABELS5) or classes['source_six_mapping']!=MAPPING.tolist():raise ValueError('five-class contract drift')
    return json.loads(raw)


def apply_thresholds(scores,thresholds):
    scores=validate_scores(scores,5);t=np.asarray(thresholds)
    if t.shape!=(4,) or not np.isfinite(t).all() or np.any((t<0)|(t>1)):raise ValueError('four alert thresholds required')
    top=scores.argmax(1);out=top.copy()
    for c in range(4):out[(top==c)&(scores[:,c]<t[c])]=4
    return out


def calibrate_thresholds(y_original,scores,config):
    scores=validate_scores(scores,5);target=map_targets(y_original);top=scores.argmax(1)
    if not np.any(target==4):raise ValueError('background needed for calibration')
    thresholds=[]
    for c in range(4):
        options=[]
        for t in np.arange(config['grid_start'],config['grid_stop_exclusive'],config['grid_step']):
            alert=(top==c)&(scores[:,c]>=t);tp=np.sum(alert&(target==c));fp=np.sum(alert&(target!=c));fn=np.sum(~alert&(target==c))
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;fpr=float(np.mean(alert[target==4]))
            options.append((fpr<=config['per_class_background_fpr_limit']+1e-12,f1,-fpr,-t,t))
        thresholds.append(float(max(options)[-1]))
    return thresholds


def promotion_decision(rows,guards):
    if not rows:raise ValueError('empty paired comparison')
    changes=[]
    for row in rows:
        a,b=row['control6'],row['merged5'];ma,mb=a['metrics'],b['metrics']
        changes.append(dict(seed=row['seed'],macro_gain=mb['macro_recall']-ma['macro_recall'],
            background_fpr_change=mb['negative_false_alert_rate']-ma['negative_false_alert_rate'],
            positive_recall_changes={k:mb['per_class'][k]['recall']-ma['per_class'][k]['recall'] for k in LABELS5[:4]},
            original_siren_recall_change=mb['original_family']['emergency_siren']['recall']-ma['original_family']['emergency_siren']['recall'],
            voice_extra_false_alerts={k:mb['negative_subtypes'][k]['false_alerts']-ma['negative_subtypes'][k]['false_alerts'] for k in ['ordinary_speech','crying_screaming']},
            replay_validation_change=b['replay']['validation']['acoustic']['matches']-a['replay']['validation']['acoustic']['matches']))
    eps=1e-12;mean=float(np.mean([r['macro_gain'] for r in changes]))
    checks=dict(macro_each=all(r['macro_gain']>=guards['minimum_each_macro_gain']-eps for r in changes),
        macro_mean=mean>=guards['minimum_mean_macro_gain']-eps,
        background=all(r['background_fpr_change']<=guards['max_background_fpr_increase']+eps for r in changes),
        positive_classes=all(min(r['positive_recall_changes'].values())>=-guards['max_positive_recall_decrease']-eps for r in changes),
        siren=all(min(r['positive_recall_changes']['siren'],r['original_siren_recall_change'])>=-guards['max_siren_recall_decrease']-eps for r in changes),
        voice=all(max(r['voice_extra_false_alerts'].values())<=guards['max_voice_extra_false_alerts'] for r in changes),
        replay=all(r['replay_validation_change']>=0 for r in changes))
    return dict(eligible_for_deployment_preparation=all(checks.values()),checks=checks,paired_changes=changes,
        mean_macro_gain=mean,board_deployed=False,independent_acceptance=False)
