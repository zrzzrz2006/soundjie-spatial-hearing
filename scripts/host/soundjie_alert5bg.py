"""Six physical logits; ordinary_speech is a trained positive, background index 5."""
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('siren','car_horn','doorbell','dog_bark','speech','background_unknown')
CONTRACT = 'soundjie-alert5bg-v3'
CLASS_FILE = ROOT/'config/soundjie_alert5bg_classes.json'
CONFIG_FILE = ROOT/'config/soundjie_alert5bg_training.json'
DATA_ROOT = Path('F:/MLK/DPU_ZYNQ7000/datasets/soundjie_alert5bg_20261101')
OUTPUT_ROOT = Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_alert5bg/20261101a')


def map_existing(row):
    if row.get('class_contract') != 'soundjie-alert4bg-v2':
        raise ValueError('verified upstream alert4bg contract required')
    c = row['target_index']
    if type(c) is not int or not 0 <= c < 5:
        raise ValueError('invalid upstream target')
    if row['target_id'] != ('siren','car_horn','doorbell','dog_bark','background_unknown')[c]:
        raise ValueError('upstream label order differs')
    if row['negative_subtype'] == 'ordinary_speech':
        if c != 4: raise ValueError('ordinary speech upstream class differs')
        c = 4
    elif c == 4: c = 5
    explicit = row['source_id'] == 'librispeech_dev_clean' or 'Speech' in row.get('original_labels', [])
    return dict(row, upstream_target_index=row['target_index'], upstream_target_id=row['target_id'],
                upstream_class_contract=row['class_contract'], target_index=c, target_id=LABELS[c],
                class_contract=CONTRACT, speech_evidence=('explicit_speech' if explicit else 'inherited_vocal_weak_label') if c==4 else None)


def noise_eligible(row):
    return (row['purpose']=='train' and row['target_index']==5
            and row['negative_subtype'] not in ('ordinary_speech','crying_screaming'))


def validate_rows(rows, require_all=True):
    seen=set();mappings=defaultdict(dict);counts=defaultdict(set)
    for row in rows:
        index,purpose=row['target_index'],row['purpose']
        if type(index) is not int or not 0<=index<6 or row['target_id']!=LABELS[index]:
            raise ValueError('alert4bg six-class contract violated')
        if row.get('class_contract')!=CONTRACT:raise ValueError('source contract mismatch')
        if index==4 and row['negative_subtype']!='ordinary_speech':raise ValueError('nonordinary speech mislabeled')
        if index==5 and row['negative_subtype']=='ordinary_speech':raise ValueError('ordinary speech remains background')
        if purpose not in ('train','validation'):raise ValueError('unauthorized purpose')
        if row['recording_id'] in seen:raise ValueError('duplicate recording identity')
        seen.add(row['recording_id'])
        for field in ('source_group','sha256','sampling_identity','uploader_group','pcm_sha256_16k'):
            value=row.get(field)
            if not value and field in ('source_group','sha256'):raise ValueError('missing '+field)
            if value:
                if mappings[field].get(value,purpose)!=purpose:raise ValueError(field+' crosses purposes')
                mappings[field][value]=purpose
        counts[purpose].add(index)
    if require_all and any(counts[p]!=set(range(6)) for p in ('train','validation')):
        raise ValueError('all six classes required in both partitions')


def balanced_source_indices(y, identities, per_class, seed):
    y,identities=np.asarray(y),np.asarray(identities)
    if y.ndim!=1 or len(y)!=len(identities) or set(y.tolist())!=set(range(6)) or per_class<1:
        raise ValueError('invalid six-class sampling arrays')
    rng=np.random.default_rng(seed);result=[]
    for c in range(6):
        buckets=defaultdict(list)
        for i in np.flatnonzero(y==c):buckets[str(identities[i])].append(int(i))
        for key in rng.choice(sorted(buckets),per_class):result.append(rng.choice(buckets[key]))
    return rng.permutation(np.asarray(result,dtype=np.int64))


def output_contract():
    config=json.loads(CLASS_FILE.read_text(encoding='utf-8'))
    if config['labels']!=list(LABELS) or config['contract']!=CONTRACT:raise ValueError('class map drift')
    return dict(contract=CONTRACT,physical_output_count=6,logical_output_count=6,
                physical_labels=list(LABELS),display_labels=list(LABELS),background_index=5,
                probability_merge_groups=[[i] for i in range(6)],output_tensor_name='fc8')


def make_model():
    from scripts.host.train_soundjie_d3 import SoundjieCNN,load_model_config,audit_model_operators
    output_contract()
    config=replace(load_model_config(ROOT/'config/soundjie_d3_model.json'),num_classes=6,
                   class_map_sha256=hashlib.sha256(CLASS_FILE.read_bytes()).hexdigest())
    model=SoundjieCNN(config);audit_model_operators(model)
    return model


def apply_thresholds(scores, thresholds):
    scores=np.asarray(scores);thresholds=np.asarray(thresholds)
    if scores.ndim!=2 or scores.shape[1]!=6 or not np.isfinite(scores).all() or np.any(scores<0) or not np.allclose(scores.sum(1),1,atol=1e-6):
        raise ValueError('six normalized finite probabilities required')
    if thresholds.shape!=(5,) or not np.isfinite(thresholds).all() or np.any((thresholds<0)|(thresholds>1)):
        raise ValueError('five valid alert thresholds required')
    top=scores.argmax(1);pred=top.copy()
    for c in range(5):pred[(top==c)&(scores[:,c]<thresholds[c])]=5
    return pred


def metrics(y,pred,rows=None):
    y,pred=np.asarray(y),np.asarray(pred)
    if y.ndim!=1 or pred.shape!=y.shape or not len(y) or not np.issubdtype(y.dtype,np.integer) or not np.issubdtype(pred.dtype,np.integer) or np.any((y<0)|(y>=6)|(pred<0)|(pred>=6)):
        raise ValueError('invalid six-class predictions')
    cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(y,pred),1);per={}
    for c,label in enumerate(LABELS):
        support=int(cm[c].sum());positive=int(cm[:,c].sum());tp=int(cm[c,c])
        per[label]=dict(support=support,correct=tp,recall=tp/support if support else 0.,
                        precision=tp/positive if positive else 0.)
    bg=y==5
    result=dict(accuracy=float(np.mean(y==pred)),macro_recall=float(np.mean([r['recall'] for r in per.values()])),
                negative_false_alert_rate=float(np.mean(pred[bg]!=5)) if bg.any() else None,
                dog_recall=per['dog_bark']['recall'],per_class=per,confusion_matrix=cm.tolist())
    if rows is not None:
        if len(rows)!=len(y) or any(r['target_index']!=int(t) for r,t in zip(rows,y)):raise ValueError('metric provenance mismatch')
        result['negative_subtypes']={}
        for subtype in sorted({r['negative_subtype'] for r in rows if r['target_index']==5}):
            mask=np.array([r['target_index']==5 and r['negative_subtype']==subtype for r in rows])
            result['negative_subtypes'][subtype]=dict(support=int(mask.sum()),false_alerts=int(np.sum(pred[mask]!=5)),false_alert_rate=float(np.mean(pred[mask]!=5)))
        for field in ('sampling_identity','source_group'):
            per_group=[]
            for c in range(6):
                buckets=defaultdict(list)
                for i,r in enumerate(rows):
                    if y[i]==c:buckets[r[field]].append(bool(pred[i]==c))
                per_group.append(float(np.mean([np.mean(v) for v in buckets.values()])) if buckets else 0.)
            result[field+'_macro_recall']=float(np.mean(per_group))
    return result


def calibrate_thresholds(y,scores,config):
    y=np.asarray(y);apply_thresholds(scores,[0.]*5)
    if not np.any(y==5):raise ValueError('background required for calibration')
    top=scores.argmax(1);thresholds=[]
    for c in range(5):
        options=[]
        # Include 1.0 so a class can fail closed when no lower grid point meets the bound.
        for t in list(np.arange(config['grid_start'],config['grid_stop_exclusive'],config['grid_step']))+[1.0]:
            alert=(top==c)&(scores[:,c]>=t);tp=np.sum(alert&(y==c));fp=np.sum(alert&(y!=c));fn=np.sum(~alert&(y==c))
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;fpr=float(np.mean(alert[y==5]))
            options.append((fpr<=config['per_class_background_fpr_limit']+1e-12,f1,-fpr,-t,t))
        thresholds.append(float(max(options)[-1]))
    return thresholds
