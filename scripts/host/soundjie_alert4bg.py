"""Five physical outputs with source-aware semantics, never a glass-to-dog rename."""
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.host.soundjie_alert6 import classify_fsd as classify_six, sampling_identity

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('siren','car_horn','doorbell','dog_bark','background_unknown')
CONTRACT = 'soundjie-alert4bg-v2'
CLASS_FILE = ROOT/'config/soundjie_alert4bg_classes.json'
CONFIG_FILE = ROOT/'config/soundjie_alert4bg_training.json'
DATA_ROOT = Path('F:/MLK/DPU_ZYNQ7000/datasets/soundjie_alert4bg_20261031')
OUTPUT_ROOT = Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_alert4bg/20261031a')


def classify_fsd(labels):
    s = set(labels)
    if 'Bark' in s:
        return (3,'bark') if s <= {'Bark','Dog','Animal','Domestic_animals_and_pets'} else None
    choice = classify_six(labels)
    if choice is None:return None
    old, subtype = choice
    index = {0:0,1:0,2:1,3:4,5:4}.get(old)
    if old == 4:index = 2 if subtype == 'doorbell' else 4
    return index, subtype


def map_existing(row):
    """Original semantic labels, not the old classifier output, determine remapping."""
    source = row['source_id']
    labels = row.get('original_labels', [])
    if source == 'fsd50k':choice = classify_fsd(labels)
    elif source == 'esc50':
        table = {'clock_alarm':(0,'alarm'), 'siren':(0,'siren'), 'car_horn':(1,'horn'),
                 'glass_breaking':(4,'glass'), 'door_wood_knock':(4,'knock'),
                 'crying_baby':(4,'crying_screaming')}
        table.update({k:(4,'ambient') for k in ('sea_waves','crickets','vacuum_cleaner','rain','chirping_birds','washing_machine')})
        choice = table.get(labels[0]) if len(labels)==1 else None
    elif source == 'librispeech_dev_clean':
        choice = (4,'ordinary_speech') if labels == ['speech'] else None
    elif source == 'user_recording' and row.get('semantic_annotation_source') == 'user':
        choice = {'user_horn':(1,'user_horn'), 'road_background':(4,'road_background')}.get(row.get('negative_subtype'))
    else:choice = None
    if choice is None:return None
    index,subtype = choice
    return dict(row, previous_target_index=row['target_index'], previous_target_id=row['target_id'],
                target_index=index, target_id=LABELS[index], negative_subtype=subtype,
                sampling_identity=sampling_identity(row), class_contract=CONTRACT)


def choose_purpose(group, frozen, fallback):
    purposes = frozen.get(group,set())
    if not purposes:return fallback
    if len(purposes)!=1 or not purposes <= {'train','validation'}:return None
    return next(iter(purposes))


def validate_rows(rows, require_all=True):
    seen=set();mappings=defaultdict(dict);counts=defaultdict(set)
    for row in rows:
        index,purpose=row['target_index'],row['purpose']
        if type(index) is not int or not 0<=index<5 or row['target_id']!=LABELS[index]:
            raise ValueError('alert4bg five-class contract violated')
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
    if require_all and any(counts[p]!=set(range(5)) for p in ('train','validation')):
        raise ValueError('all five classes required in both partitions')


def balanced_source_indices(y, identities, per_class, seed):
    y,identities=np.asarray(y),np.asarray(identities)
    if y.ndim!=1 or len(y)!=len(identities) or set(y.tolist())!=set(range(5)) or per_class<1:
        raise ValueError('invalid five-class sampling arrays')
    rng=np.random.default_rng(seed);result=[]
    for c in range(5):
        buckets=defaultdict(list)
        for i in np.flatnonzero(y==c):buckets[str(identities[i])].append(int(i))
        for key in rng.choice(sorted(buckets),per_class):result.append(rng.choice(buckets[key]))
    return rng.permutation(np.asarray(result,dtype=np.int64))


def output_contract():
    config=json.loads(CLASS_FILE.read_text(encoding='utf-8'))
    if config['labels']!=list(LABELS) or config['contract']!=CONTRACT:raise ValueError('class map drift')
    return dict(contract=CONTRACT,physical_output_count=5,logical_output_count=5,
                physical_labels=list(LABELS),display_labels=list(LABELS),background_index=4,
                probability_merge_groups=[[i] for i in range(5)],output_tensor_name='fc8')


def make_model():
    from scripts.host.train_soundjie_d3 import SoundjieCNN,load_model_config,audit_model_operators
    output_contract()
    config=replace(load_model_config(ROOT/'config/soundjie_d3_model.json'),num_classes=5,
                   class_map_sha256=hashlib.sha256(CLASS_FILE.read_bytes()).hexdigest())
    model=SoundjieCNN(config);audit_model_operators(model)
    return model


def apply_thresholds(scores, thresholds):
    scores=np.asarray(scores);thresholds=np.asarray(thresholds)
    if scores.ndim!=2 or scores.shape[1]!=5 or not np.isfinite(scores).all() or np.any(scores<0) or not np.allclose(scores.sum(1),1,atol=1e-6):
        raise ValueError('five normalized finite probabilities required')
    if thresholds.shape!=(4,) or not np.isfinite(thresholds).all() or np.any((thresholds<0)|(thresholds>1)):
        raise ValueError('four valid alert thresholds required')
    top=scores.argmax(1);pred=top.copy()
    for c in range(4):pred[(top==c)&(scores[:,c]<thresholds[c])]=4
    return pred


def metrics(y,pred,rows=None):
    y,pred=np.asarray(y),np.asarray(pred)
    if y.ndim!=1 or pred.shape!=y.shape or not len(y) or not np.issubdtype(y.dtype,np.integer) or not np.issubdtype(pred.dtype,np.integer) or np.any((y<0)|(y>=5)|(pred<0)|(pred>=5)):
        raise ValueError('invalid five-class predictions')
    cm=np.zeros((5,5),dtype=np.int64);np.add.at(cm,(y,pred),1);per={}
    for c,label in enumerate(LABELS):
        support=int(cm[c].sum());positive=int(cm[:,c].sum());tp=int(cm[c,c])
        per[label]=dict(support=support,correct=tp,recall=tp/support if support else 0.,
                        precision=tp/positive if positive else 0.)
    bg=y==4
    result=dict(accuracy=float(np.mean(y==pred)),macro_recall=float(np.mean([r['recall'] for r in per.values()])),
                negative_false_alert_rate=float(np.mean(pred[bg]!=4)) if bg.any() else None,
                dog_recall=per['dog_bark']['recall'],per_class=per,confusion_matrix=cm.tolist())
    if rows is not None:
        if len(rows)!=len(y) or any(r['target_index']!=int(t) for r,t in zip(rows,y)):raise ValueError('metric provenance mismatch')
        result['negative_subtypes']={}
        for subtype in sorted({r['negative_subtype'] for r in rows if r['target_index']==4}):
            mask=np.array([r['target_index']==4 and r['negative_subtype']==subtype for r in rows])
            result['negative_subtypes'][subtype]=dict(support=int(mask.sum()),false_alerts=int(np.sum(pred[mask]!=4)),false_alert_rate=float(np.mean(pred[mask]!=4)))
        for field in ('sampling_identity','source_group'):
            per_group=[]
            for c in range(5):
                buckets=defaultdict(list)
                for i,r in enumerate(rows):
                    if y[i]==c:buckets[r[field]].append(bool(pred[i]==c))
                per_group.append(float(np.mean([np.mean(v) for v in buckets.values()])) if buckets else 0.)
            result[field+'_macro_recall']=float(np.mean(per_group))
    return result


def calibrate_thresholds(y,scores,config):
    y=np.asarray(y);apply_thresholds(scores,[0.]*4)
    if not np.any(y==4):raise ValueError('background required for calibration')
    top=scores.argmax(1);thresholds=[]
    for c in range(4):
        options=[]
        # Include 1.0 so a class can fail closed when no lower grid point meets the bound.
        for t in list(np.arange(config['grid_start'],config['grid_stop_exclusive'],config['grid_step']))+[1.0]:
            alert=(top==c)&(scores[:,c]>=t);tp=np.sum(alert&(y==c));fp=np.sum(alert&(y!=c));fn=np.sum(~alert&(y==c))
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;fpr=float(np.mean(alert[y==4]))
            options.append((fpr<=config['per_class_background_fpr_limit']+1e-12,f1,-fpr,-t,t))
        thresholds.append(float(max(options)[-1]))
    return thresholds
