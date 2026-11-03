#!/usr/bin/env python3
"""Finite read-only session observer (Python 3.5 compatible); creates one report."""
import argparse
import json
import os
import time


def observe(session, seconds, limit):
    names={'runtime':'inference/timeline.jsonl','doa':'state/doa.jsonl','ui':'logs/ui.jsonl'}
    stats={k:dict(rotations=0,max_bytes=0,missing=0,malformed=0,unique_rows=0) for k in names}
    identities={}; seen={k:set() for k in names}; runtime=[]
    ui_bad=0; background_score_violations=0; doa_valid=0
    started=time.monotonic(); exit_status=None
    while time.monotonic()-started<seconds:
        for role,name in names.items():
            path=os.path.join(session,name)
            try:
                with open(path,'rb') as f:
                    info=os.fstat(f.fileno()); raw=f.read(limit+1)
            except OSError:
                stats[role]['missing']+=1;continue
            identity=(info.st_dev,info.st_ino)
            if role in identities and identities[role]!=identity:stats[role]['rotations']+=1
            identities[role]=identity
            stats[role]['max_bytes']=max(stats[role]['max_bytes'],info.st_size,len(raw))
            # A concurrently appended final partial row is not a complete record.
            for line in raw.split(b'\n')[:-1]:
                try:row=json.loads(line.decode('utf-8'))
                except (ValueError,UnicodeDecodeError):stats[role]['malformed']+=1;continue
                key=row.get('sequence' if role=='runtime' else 'frame' if role=='ui' else 'monotonic_ms')
                if key in seen[role]:continue
                seen[role].add(key);stats[role]['unique_rows']+=1
                if role=='runtime':runtime.append(row)
                if role=='doa' and row.get('valid'):doa_valid+=1
                if role=='ui':
                    if not all(row.get(k) for k in ('usb_ok','dpu_ok','framebuffer_ok','classification_fresh')):ui_bad+=1
                    if row.get('legacy_class_id')==4 and (row.get('alert_active') or row.get('main_score_visible') or row.get('main_display_confidence') is not None or row.get('legacy_class_scores')):
                        background_score_violations+=1
        status=os.path.join(session,'status.json')
        if os.path.isfile(status):
            with open(status) as f:exit_status=json.load(f)
            break
        time.sleep(.1)
    steps=[(b['window_end_sample']-a['window_end_sample'])/16.0 for a,b in zip(runtime,runtime[1:])
           if a.get('usb_ok') and b.get('usb_ok') and b['sequence']==a['sequence']+1]
    return dict(schema=1,scope='instrumented live log-rotation check; not acoustic accuracy or physical latency',
        seconds=time.monotonic()-started,max_bytes=limit,files=stats,ui_unhealthy=ui_bad,
        background_score_violations=background_score_violations,doa_valid_samples=doa_valid,
        runtime_rows=runtime,consecutive_step_ms=steps,session_exit_status=exit_status,
        size_bound_ok=all(v['max_bytes']<=limit for v in stats.values()),
        all_rotated=all(v['rotations']>0 for v in stats.values()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',required=True);p.add_argument('--seconds',type=int,default=60)
    p.add_argument('--max-bytes',type=int,default=4096);p.add_argument('--output',required=True)
    a=p.parse_args()
    if not 1<=a.seconds<=120 or not 4096<=a.max_bytes<=65536:p.error('bounded observer duration/limit required')
    if not os.path.realpath(a.session).startswith('/run/') or not os.path.realpath(a.output).startswith('/run/'):
        p.error('session and output must be in /run')
    with open(a.output,'x') as output:
        json.dump(observe(a.session,a.seconds,a.max_bytes),output,ensure_ascii=True,sort_keys=True)
        output.write('\n')
    print('BOUNDED_OBSERVER_COMPLETE='+a.output)
