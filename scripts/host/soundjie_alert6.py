"""Six-output alert contract; legacy eight-class loaders remain unchanged."""
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('alarm_sound','emergency_siren','car_horn','glass_breaking',
          'door_knock_bell','background_unknown')


def map_legacy(row):
    old = row['target_index']
    if type(old) is not int or not 0 <= old < 8:
        raise ValueError('invalid legacy target')
    index = old if old < 5 else 5
    subtype = {5:'crying_screaming',6:'ordinary_speech',7:'ambient'}.get(old, LABELS[index])
    return dict(row, legacy_target_index=old, legacy_target_id=row['target_id'],
                target_index=index, target_id=LABELS[index], negative_subtype=subtype)


def classify_fsd(labels):
    """Conservative clip labels: reject mixed target classes/unknown labels."""
    s = set(labels)
    matches = []
    def add(index, subtype, required, allowed):
        if set(required) <= s and s <= set(allowed):matches.append((index, subtype))
    add(0,'alarm', ['Alarm'], ['Alarm','Clock','Mechanisms','Domestic_sounds_and_home_sounds','Beep_and_bleep','Buzzer'])
    add(1,'siren', ['Siren'], ['Siren','Alarm','Motor_vehicle_(road)','Vehicle','Traffic_noise_and_roadway_noise'])
    add(2,'horn', ['Vehicle_horn_and_car_horn_and_honking'],
        ['Vehicle_horn_and_car_horn_and_honking','Alarm','Car','Bus','Truck',
         'Motor_vehicle_(road)','Vehicle','Traffic_noise_and_roadway_noise','Engine','Motorcycle'])
    add(3,'glass', ['Glass','Shatter'], ['Glass','Shatter','Crushing','Chink_and_clink'])
    add(4,'knock', ['Door','Knock'], ['Door','Knock','Wood','Domestic_sounds_and_home_sounds'])
    add(4,'doorbell', ['Doorbell'], ['Doorbell','Door','Alarm','Bell','Domestic_sounds_and_home_sounds','Clock','Mechanisms'])
    voice = {'Speech','Human_voice','Male_speech_and_man_speaking','Female_speech_and_woman_speaking',
             'Child_speech_and_kid_speaking','Speech_synthesizer','Conversation','Chatter',
             'Human_group_actions','Whispering','Laughter','Giggle','Sigh','Breathing',
             'Crying_and_sobbing','Screaming','Shout','Yell'}
    if s and s <= voice:
        subtype = 'crying_screaming' if s & {'Crying_and_sobbing','Screaming','Shout','Yell'} else 'ordinary_speech'
        matches.append((5, subtype))
    road = {'Traffic_noise_and_roadway_noise','Vehicle','Motor_vehicle_(road)','Car','Bus','Truck',
            'Engine','Motorcycle','Idling','Accelerating_and_revving_and_vroom','Wind','Wind_noise_(microphone)'}
    if s and s <= road:matches.append((5,'road_background'))
    ambient = {'Rain','Raindrop','Water','Wind','Wind_noise_(microphone)','Rustle',
               'Walk_and_footsteps','Mechanisms','Clock','Tick','Tick-tock',
               'Domestic_sounds_and_home_sounds','Running_water','Stream','Frying_(food)',
               'Writing','Computer_keyboard','Typing','Mechanical_fan','Crackle'}
    if s and s <= ambient:matches.append((5,'ambient'))
    return matches[0] if len(matches) == 1 else None


def validate_rows(rows, require_all=True):
    identities=set();groups={};hashes={};counts=defaultdict(set)
    for row in rows:
        index=row['target_index'];purpose=row['purpose']
        if type(index) is not int or not 0 <= index < 6 or row['target_id'] != LABELS[index]:
            raise ValueError('six-class label contract violated')
        if purpose not in ('train','validation'):raise ValueError('unauthorized purpose')
        identity=row['recording_id']
        if identity in identities:raise ValueError('duplicate recording ID')
        identities.add(identity)
        for key, mapping in (('source_group',groups),('sha256',hashes)):
            value=row[key]
            if not value or mapping.get(value,purpose) != purpose:raise ValueError('source crosses purposes')
            mapping[value]=purpose
        counts[purpose].add(index)
    if require_all and any(counts[p] != set(range(6)) for p in ('train','validation')):
        raise ValueError('all six classes required in both partitions')


def balanced_source_indices(y, source_ids, per_class, seed):
    y=np.asarray(y);source_ids=np.asarray(source_ids)
    if len(y) != len(source_ids) or set(y.tolist()) != set(range(6)) or per_class < 1:
        raise ValueError('invalid sampling arrays')
    rng=np.random.default_rng(seed);result=[]
    for c in range(6):
        positions=np.flatnonzero(y==c)
        buckets=defaultdict(list)
        for i in positions:buckets[str(source_ids[i])].append(int(i))
        keys=sorted(buckets)
        for key in rng.choice(keys, per_class):result.append(rng.choice(buckets[key]))
    return rng.permutation(np.asarray(result,dtype=np.int64))


def sampling_identity(row):
    """Original recording, not excerpt ID, uploader ID, or adaptation session."""
    if row.get('source_id')=='esc50':
        group=row['source_group']
        if not group.startswith('esc50:'):raise ValueError('missing ESC original source')
        return 'freesound:'+group.split(':',1)[1]
    if row.get('source_recording_id'):
        return row['source_recording_id']
    if row.get('source_id')=='librispeech_dev_clean':
        # Utterances in one audiobook chapter were cut from the same recording.
        stem=Path(row['path']).stem
        parts=stem.split('-')
        if len(parts)!=3 or not all(p.isdigit() for p in parts):raise ValueError('invalid LibriSpeech identity')
        return 'librispeech-chapter:'+parts[0]+'-'+parts[1]
    return row['recording_id']


def metrics(y, pred):
    y=np.asarray(y);pred=np.asarray(pred)
    if y.ndim != 1 or pred.shape != y.shape or not len(y) or np.any((y<0)|(y>=6)|(pred<0)|(pred>=6)):
        raise ValueError('invalid six-class predictions')
    cm=np.zeros((6,6),dtype=np.int64)
    for a,b in zip(y,pred):cm[int(a),int(b)]+=1
    per={}
    for c,name in enumerate(LABELS):
        support=int(cm[c].sum());p=int(cm[:,c].sum());tp=int(cm[c,c])
        recall=tp/support if support else 0.;precision=tp/p if p else 0.
        per[name]=dict(support=support,recall=recall,precision=precision,
                      f1=2*recall*precision/(recall+precision) if recall+precision else 0.)
    negative=cm[5].sum()
    return dict(accuracy=float(np.mean(y==pred)),macro_recall=float(np.mean([v['recall'] for v in per.values()])),
                negative_false_alert_rate=float((negative-cm[5,5])/negative) if negative else None,
                per_class=per,confusion_matrix=cm.tolist())


def make_model():
    from scripts.host.train_soundjie_d3 import SoundjieCNN,load_model_config,audit_model_operators
    path=ROOT/'config/soundjie_alert6_classes.json'
    config=replace(load_model_config(ROOT/'config/soundjie_d3_model.json'),num_classes=6,
                   class_map_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    model=SoundjieCNN(config)
    audit_model_operators(model)
    return model
