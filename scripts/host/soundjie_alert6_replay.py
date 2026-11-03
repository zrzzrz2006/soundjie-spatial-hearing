"""Frozen playback schedule, guarded serial recovery and acoustic alignment."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import time
import wave

import numpy as np
from scipy.signal import chirp, correlate, resample
import soundfile as sf

from scripts.host.acquire_soundjie_alert6 import read_rows
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.prepare_soundjie_d3 import decode_audio,resample_mono_16k
from scripts.host.soundjie_alert6 import ROOT
from scripts.host.train_soundjie_alert6 import write_json

RATE=16000


def marker():
    result=np.zeros(2*RATE)
    for i,(a,b) in enumerate([(400,2200),(2200,600),(700,2600),(2400,400)]):
        t=np.arange(5600)/RATE
        result[i*8000:i*8000+5600]=.125*np.hanning(5600)*chirp(t,f0=a,f1=b,t1=t[-1],method='linear')
    return result


def crop_reference(samples):
    samples=np.asarray(samples,dtype=np.float64)
    if samples.ndim!=1 or not len(samples) or not np.isfinite(samples).all():raise ValueError('invalid source waveform')
    n=40000
    if len(samples)<n:
        pad=(n-len(samples))//2
        return np.pad(samples,(pad,n-len(samples)-pad)),-pad
    energy=np.concatenate(([0.],np.cumsum(samples*samples)))
    start=int(np.argmax(energy[n:]-energy[:-n]))
    return samples[start:start+n].copy(),start


def verify_schedule(refs):
    seen=set();groups={}
    for r in refs:
        if r['purpose'] not in ('train','validation'):raise ValueError('not eligible for this capture')
        if r['recording_id'] in seen:raise ValueError('duplicate source')
        seen.add(r['recording_id'])
        g=r['source_group']
        if groups.get(g,r['purpose'])!=r['purpose']:raise ValueError('source crosses purposes')
        groups[g]=r['purpose']


def build_waveform(refs):
    verify_schedule(refs)
    end_sync=(8+4*len(refs)+2)*RATE
    output=np.zeros(end_sync+len(marker())+4*RATE)
    output[4*RATE:4*RATE+len(marker())]=marker()
    output[end_sync:end_sync+len(marker())]=marker()
    events=[]
    for i,r in enumerate(refs):
        start=(8+4*i)*RATE
        samples=np.asarray(r['samples'])
        if samples.shape!=(40000,) or not np.isfinite(samples).all():raise ValueError('invalid reference excerpt')
        output[start:start+40000]=samples
        event={k:v for k,v in r.items() if k!='samples'}
        event.update(event_index=i,start_sample=start,end_sample=start+40000,
                     source_kind='known_playback_weak_clip_label',independent_D5=False)
        events.append(event)
    return output,dict(schema=1,rate=RATE,sync_start_sample=4*RATE,sync_end_sample=end_sync,
        sync_samples=len(marker()),events=events,total_samples=len(output),duration_s=len(output)/RATE,
        capture_seconds=int(np.ceil(len(output)/RATE))+20,
        labels_are_playback_schedule_not_confirmed_recording=True)


def write_wav(path,signal):
    pcm=np.clip(np.rint(np.asarray(signal)*32768),-32768,32767).astype('<i2')
    with Path(path).open('xb') as f:
        with wave.open(f,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(RATE);w.writeframes(pcm.tobytes())


def build(output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=Path('F:/MLK/DPU_ZYNQ7000/datasets/soundjie_alert6_20261031/manifest.jsonl')
    audit=json.loads((manifest.parent/'audit.json').read_text(encoding='utf-8'))
    if digest_file(manifest)!=audit['manifest_sha256']:raise ValueError('source manifest drift')
    rows=read_rows(manifest);selected=[];used=set();groups=set()
    def pick(target,purpose,subtype=None,count=1):
        pool=[r for r in rows if r['source_id']=='fsd50k' and r['target_index']==target and r['purpose']==purpose
              and (subtype is None or r['negative_subtype']==subtype)]
        pool.sort(key=lambda r:hashlib.sha256(('acoustic-20261031:'+r['recording_id']).encode()).hexdigest())
        chosen=0
        for r in pool:
            if r['recording_id'] in used or r['source_group'] in groups:continue
            selected.append(r);used.add(r['recording_id']);groups.add(r['source_group']);chosen+=1
            if chosen==count:return
        raise ValueError('not enough disjoint eligible sources')
    for target in range(5):
        pick(target,'train',count=2);pick(target,'validation')
    for subtype in ('ordinary_speech','crying_screaming','road_background'):
        pick(5,'train',subtype);pick(5,'validation',subtype)
    selected.append(next(r for r in rows if r['recording_id']=='user:horn_40'))
    # Freeze order by identity hash instead of grouping every class together.
    selected.sort(key=lambda r:hashlib.sha256(('playback-order:'+r['recording_id']).encode()).hexdigest())
    refs=[]
    for r in selected:
        if digest_file(r['path'])!=r['sha256']:raise ValueError('source drift')
        clip,start=crop_reference(resample_mono_16k(decode_audio(Path(r['path']))))
        peak=float(np.max(np.abs(clip)))
        if peak<=1e-6:raise ValueError('silent reference')
        gain=min(1.,.5/peak)
        clip=clip*gain
        refs.append(dict(r,samples=clip,excerpt_start_sample_16k=start,digital_gain=gain,
                         reference_rms_dbfs=float(20*np.log10(max(1e-12,np.sqrt(np.mean(clip**2)))))))
    waveform,plan=build_waveform(refs)
    output.mkdir(parents=True,exist_ok=False);(output/'references').mkdir()
    for i,r in enumerate(refs):
        p=output/'references'/('%02d.wav'%i);write_wav(p,r['samples'])
        plan['events'][i]['reference_wav']=str(p);plan['events'][i]['reference_sha256']=digest_file(p)
    write_wav(output/'playback.wav',waveform)
    plan.update(playback_sha256=digest_file(output/'playback.wav'),source_manifest_sha256=digest_file(manifest),
                protocol='five_alerts_plus_nonalert_speech_cry_road; no D5 sources; no board inference claim')
    write_json(output/'schedule.json',plan)
    print(json.dumps(dict(stage='built',duration_s=plan['duration_s'],capture_seconds=plan['capture_seconds'],
                         events=len(refs),train_events=sum(r['purpose']=='train' for r in refs),
                         validation_events=sum(r['purpose']=='validation' for r in refs),sha256=plan['playback_sha256'])))


def play(directory):
    import winsound
    directory=Path(directory);plan=json.loads((directory/'schedule.json').read_text(encoding='utf-8'))
    path=directory/'playback.wav'
    if digest_file(path)!=plan['playback_sha256']:raise ValueError('playback checksum drift')
    started=time.time();perf=time.perf_counter()
    print(json.dumps(dict(stage='playback_started',unix_time=started,duration_s=plan['duration_s'])),flush=True)
    # Synchronous is the default (zero flag); named SND_SYNC is absent on Python 3.13.
    winsound.PlaySound(str(path),winsound.SND_FILENAME|winsound.SND_NODEFAULT)
    elapsed=time.perf_counter()-perf
    if elapsed<plan['duration_s']-.5:raise ValueError('playback ended early')
    write_json(directory/'playback-run.json',dict(started_unix=started,elapsed_seconds=elapsed,
        sha256=plan['playback_sha256'],api='winsound default Windows output; physical reception requires microphone verification'))
    print(json.dumps(dict(stage='playback_finished',elapsed_seconds=elapsed)),flush=True)


def align_capture(recording,reference,search_samples):
    x=np.asarray(recording,dtype=np.float64)[:search_samples];r=np.asarray(reference,dtype=np.float64)
    r=r-r.mean();n=len(r)
    if len(x)<n or np.sum(r*r)<=1e-15:raise ValueError('insufficient sync data')
    x=x-x.mean();corr=correlate(x,r,mode='valid',method='fft')
    cumulative=np.concatenate(([0.],np.cumsum(x*x)))
    denom=np.sqrt(np.maximum((cumulative[n:]-cumulative[:-n])*np.sum(r*r),1e-25))
    scores=np.abs(corr)/denom;index=int(np.argmax(scores));score=float(scores[index])
    if score<.18:raise ValueError('sync correlation too weak: %.4f'%score)
    return dict(sample=index,normalized_score=score,polarity=1 if corr[index]>=0 else -1)


def recover(log,sha,output):
    raw=Path(log).read_bytes().replace(b'\r',b'')
    begin=b'\nSJ_ALERT6_CAPTURE_BEGIN\n';end=b'\nSJ_ALERT6_CAPTURE_END\n'
    if raw.count(begin)!=1:raise ValueError('capture marker missing or ambiguous')
    payload=raw.split(begin,1)[1].split(end,1)
    if len(payload)!=2:raise ValueError('serial transfer incomplete')
    audio=gzip.decompress(base64.b64decode(b''.join(payload[0].split()),validate=True))
    if hashlib.sha256(audio).hexdigest()!=sha:raise ValueError('recording SHA mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    with (output/'capture-stereo.wav').open('xb') as f:f.write(audio)
    print(json.dumps(dict(stage='recovered',bytes=len(audio),sha256=sha)))


def validate_native_capture(path,expected_frames):
    info=sf.info(path)
    if (info.format!='WAV' or info.subtype!='PCM_16' or info.endian not in ('FILE','LITTLE')
            or info.samplerate!=RATE or info.channels!=2 or info.frames!=expected_frames):
        raise ValueError('capture must be native little-endian PCM16 WAV, stereo 16k, exact duration')


def reception_check(samples,reference):
    """Conservative content check, not event truth or a recognition confidence."""
    samples=np.asarray(samples,dtype=np.float64);reference=np.asarray(reference,dtype=np.float64)
    if samples.ndim!=1 or reference.ndim!=1 or min(len(samples),len(reference))<16000:
        raise ValueError('insufficient event content')
    # Correct only the tiny measured clock-length difference for this diagnostic.
    x=resample(samples,len(reference));x-=x.mean()
    r=reference[3200:-3200].copy();r-=r.mean()
    energy=float(np.dot(r,r));rms=float(np.sqrt(np.mean(samples**2)))
    score=0.
    if energy>1e-10 and rms>1e-5:
        corr=correlate(x,r,mode='valid',method='fft');n=len(r)
        accum=np.concatenate(([0.],np.cumsum(x*x)))
        denom=np.sqrt(np.maximum((accum[n:]-accum[:-n])*energy,1e-25))
        score=float(np.max(np.abs(corr)/denom))
    clipped=float(np.mean(np.abs(samples)>=32767/32768.))
    passed=score>=.18 and rms>1e-5 and clipped<=.001
    return dict(passed=passed,normalized_content_score=score,minimum_score=.18,
                rms=rms,clipped_fraction=clipped,
                meaning='waveform reception evidence only; weak source label is not manually verified')


def inspect(directory,recording):
    directory=Path(directory);recording=Path(recording)
    plan=json.loads((directory/'schedule.json').read_text(encoding='utf-8'))
    verify_schedule(plan['events'])
    validate_native_capture(recording,plan['capture_seconds']*RATE)
    stereo,rate=sf.read(recording,dtype='int16',always_2d=True)
    if rate!=RATE or stereo.shape[1]!=2 or len(stereo)!=plan['capture_seconds']*RATE:
        raise ValueError('capture format or exact duration mismatch')
    summed=stereo[:,0].astype(np.int32)+stereo[:,1].astype(np.int32)
    # Match C++ division toward zero, including negative odd sums.
    mono=(np.sign(summed)*(np.abs(summed)//2)).astype(np.float64)/32768.
    first=align_capture(mono,marker(),30*RATE)
    end_search=max(0,len(mono)-35*RATE)
    last=align_capture(mono[end_search:],marker(),len(mono)-end_search);last['sample']+=end_search
    ratio=(last['sample']-first['sample'])/(plan['sync_end_sample']-plan['sync_start_sample'])
    if abs(ratio-1)>.005:raise ValueError('sync drift too large or wrong end marker')
    report=dict(schema=1,recording_sha256=digest_file(recording),sample_rate=RATE,channels=2,
        duration_s=len(mono)/RATE,start_sync=first,end_sync=last,clock_ratio=ratio,
        channel_rms=(np.sqrt(np.mean((stereo.astype(float)/32768.)**2,axis=0))).tolist(),
        channel_correlation=float(np.corrcoef(stereo.T)[0,1]),
        clipped_fraction=float(np.mean(np.abs(stereo.astype(np.int32))>=32767)),
        equal_mix_rms=float(np.sqrt(np.mean(mono**2))),events=[],
        independent_D5=False,labels='source clip weak labels; gaps/markers not positive training data',
        use_policy='diagnostic excerpts only; any training importer must require adaptation_allowed; unreceived events quarantined')
    excerpts=recording.parent/'aligned';excerpts.mkdir(exist_ok=False)
    for event in plan['events']:
        start=int(round(first['sample']+(event['start_sample']-plan['sync_start_sample'])*ratio))
        end=int(round(first['sample']+(event['end_sample']-plan['sync_start_sample'])*ratio))
        if start<0 or end>len(mono) or end<=start:raise ValueError('aligned event outside recording')
        samples=mono[start:end];path=excerpts/('%02d.wav'%event['event_index']);write_wav(path,samples)
        if digest_file(event['reference_wav'])!=event['reference_sha256']:raise ValueError('reference drift')
        reference,reference_rate=sf.read(event['reference_wav'],dtype='float64')
        if reference_rate!=RATE or reference.ndim!=1:raise ValueError('reference format drift')
        reception=reception_check(samples,reference)
        eligible=reception['passed'] and report['clipped_fraction']<=.001
        report['events'].append(dict(event,capture_start_sample=start,capture_end_sample=end,
            captured_wav=str(path),captured_sha256=digest_file(path),
            reception=reception,quarantined=not eligible,
            adaptation_allowed=eligible and event['purpose']=='train',
            captured_rms_dbfs=float(20*np.log10(max(1e-12,np.sqrt(np.mean(samples**2)))))))
    report['reception_verified_events']=sum(not e['quarantined'] for e in report['events'])
    report['quarantined_events']=sum(e['quarantined'] for e in report['events'])
    write_json(recording.parent/'alignment-report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='events'}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['build','play','recover','inspect'])
    p.add_argument('--directory',type=Path);p.add_argument('--recording',type=Path)
    p.add_argument('--log',type=Path);p.add_argument('--sha256');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.mode=='build':build(a.directory)
    elif a.mode=='play':play(a.directory)
    elif a.mode=='recover':recover(a.log,a.sha256,a.output)
    else:inspect(a.directory,a.recording)


if __name__=='__main__':main()
