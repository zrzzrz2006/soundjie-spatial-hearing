"""Recover captured PCM with overlap/gap checks; report known-clip replay evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def reconstruct(rows, directory):
    captures=[]
    for row in rows:
        name=row.get('capture_file')
        if not name:
            continue
        end=row.get('window_end_sample')
        if not re.fullmatch(r'capture-[0-9]+\.s16',name) or type(end) is not int or not 16000<=end<=16000*600:
            raise ValueError('invalid capture identity or sample range')
        path=Path(directory)/name
        # A deliberately selected export can omit unrelated capture files.
        if not path.exists():
            continue
        b=path.read_bytes()
        if len(b)!=32000:
            raise ValueError('capture must contain one second of mono s16')
        captures.append((end,np.frombuffer(b,dtype='<i2')))
    if not captures:
        raise ValueError('no retained captures')
    maximum=max(end for end,_ in captures)
    pcm=np.zeros(maximum,dtype=np.int16); covered=np.zeros(maximum,dtype=bool)
    for end,x in sorted(captures,key=lambda item:item[0]):
        start=end-16000; old=covered[start:end]
        if np.any(pcm[start:end][old]!=x[old]):
            raise ValueError('overlapping capture samples disagree')
        pcm[start:end]=x; covered[start:end]=True
    return pcm,covered


def covered_slice(pcm,covered,start,end):
    if not 0<=start<end<=len(pcm) or len(covered)!=len(pcm) or not covered[start:end].all():
        raise ValueError('requested interval has missing capture samples')
    return pcm[start:end].copy()


def matched_origin(rows):
    offsets=[r['window_end_sample']/16000-r['tv_position_seconds'] for r in rows if r.get('tv_matched')]
    if len(offsets)<8 or not np.isfinite(offsets).all():
        raise ValueError('insufficient finite audio matches for alignment')
    origin=float(np.median(offsets))
    if origin<0 or max(abs(x-origin) for x in offsets)>.05:
        raise ValueError('reference positions are not a consistent audio alignment')
    return origin


def main():
    p=argparse.ArgumentParser()
    for key in ('session','canonical','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--evaluation-role', choices=('diagnostic','calibration','acceptance'), default='diagnostic')
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    rows=[json.loads(s) for s in (a.session/'inference/timeline.jsonl').read_text(encoding='utf-8').splitlines()]
    pcm,covered=reconstruct(rows,a.session/'inference'); origin=matched_origin(rows)
    start=round(origin*16000)
    clip=covered_slice(pcm,covered,start,start+480000)
    context=covered_slice(pcm,covered,start-32000,start+512000)
    clean=np.frombuffer(a.canonical.read_bytes(),dtype='<i2')
    if len(clean)!=480000:
        raise ValueError('canonical must be exactly 30 seconds')
    active=[r for r in rows if r['dpu_ok'] and origin+1.5<=r['window_end_sample']/16000<=origin+30]
    def level(x):
        return float(20*np.log10(max(1e-12,np.sqrt(np.mean((x.astype(float)/32768)**2)))))
    report=dict(scope='physical replay of known clip; not unseen-source accuracy',evaluation_role=a.evaluation_role,
        origin_seconds=origin,context_start_seconds=origin-2,continuous_clip_samples=len(clip),
        timeline_sha256=hashlib.sha256((a.session/'inference/timeline.jsonl').read_bytes()).hexdigest(),
        clip_sha256=hashlib.sha256(clip.astype('<i2').tobytes()).hexdigest(),
        canonical_dbfs=level(clean),captured_dbfs=level(clip),clipped_samples=int(np.sum(abs(clip.astype(np.int32))>=32767)),
        active_windows=len(active),matched_windows=sum(r['tv_matched'] for r in active),
        ambulance_windows=sum(r['fine_valid'] and r['ambulance_score']>=.65 for r in active),
        traffic_windows=sum(r['fine_valid'] and r['road_score']>=.65 for r in active),
        both_windows=sum(r['fine_valid'] and min(r['ambulance_score'],r['road_score'])>=.65 for r in active),
        cues=sorted({r['caption_cue'] for r in active if r['caption_cue']}),
        positions=[dict(sequence=r['sequence'],reference_seconds=r['window_end_sample']/16000-origin,matched=r['tv_matched'],score=r['tv_score'],ambulance=r['ambulance_score'],traffic=r['road_score'],speech=r['speech_score']) for r in active])
    a.output.mkdir(parents=True)
    (a.output/'clip.s16le').write_bytes(clip.astype('<i2').tobytes())
    (a.output/'context.s16le').write_bytes(context.astype('<i2').tobytes())
    (a.output/'reconstructed.s16le').write_bytes(pcm.astype('<i2').tobytes())
    np.save(a.output/'coverage.npy',covered)
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='positions'},ensure_ascii=False))


if __name__=='__main__':
    main()
