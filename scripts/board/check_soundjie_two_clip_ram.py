#!/usr/bin/env python3
"""Real DPU PCM probes and digital two-clip replay; Python 3.5, RAM only."""
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=OrderedDict)

def main():
    bundle=Path(sys.argv[1]);checks=Path(sys.argv[2]);out=Path(sys.argv[3])
    if out.exists() or out.is_symlink() or out.parent.resolve()!=Path('/run') or not out.name.startswith('sj6-'):
        raise ValueError('new RAM output required')
    out.mkdir(mode=0o700)
    env=os.environ.copy()
    env['SOUNDJIE_BUNDLE_DIR']=str(bundle)
    env['LD_LIBRARY_PATH']=str(bundle/'dnndk/pkgs/lib')+':'+str(bundle/'audio/lib')
    env['ALSA_CONFIG_PATH']=str(bundle/'audio/share/alsa/alsa.conf')
    def run(args,timeout=30):
        start=time.monotonic()
        result=subprocess.run(list(map(str,args)),env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
        return result,time.monotonic()-start
    report=OrderedDict(scope='selected validation smoke and digital replay; not independent acoustic acceptance',probes=[],replays=[])
    manifest=read_json(checks/'probe/manifest.json')
    for p in manifest['records']:
        pcm=checks/'probe'/p['pcm_file']
        if hashlib.sha256(pcm.read_bytes()).hexdigest()!=p['pcm_sha256']:
            raise ValueError('PCM checksum differs')
        result,seconds=run([bundle/'runtime/soundjie_d4_classifier',pcm])
        (out/(p['label']+'.stderr')).write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError('DPU probe failed '+p['label']+': '+result.stderr.decode('utf-8','replace')[-1000:])
        fields=result.stdout.decode('ascii').split()
        if len(fields)!=9:
            raise ValueError('not six physical scores')
        predicted=int(fields[0]);scores=list(map(float,fields[3:]));inference=float(fields[2])
        if not all(math.isfinite(x) and 0<=x<=1 for x in scores) or abs(sum(scores)-1)>1e-6:
            raise ValueError('invalid scores')
        record=dict(label=p['label'],class_index=p['class_index'],raw_id=predicted,scores=scores,
                    inference_ms=inference,wall_ms=seconds*1000,
                    fixed_score_max_abs_error=max(abs(a-b) for a,b in zip(scores,p['expected_fixed_neuron_scores'])))
        report['probes'].append(record)
        if predicted!=p['expected_fixed_neuron_raw_id']:
            raise ValueError('DPU class differs on selected probe '+p['label'])
    (out/'probes.json').write_text(json.dumps(report['probes'],indent=2),encoding='utf-8')
    for name in ('dialogue','dialogue_horn','ambulance'):
        mono=(checks/'replay'/(name+'.mono')).read_bytes()
        pcm=out/(name+'.stereo')
        with pcm.open('xb') as stream:
            stream.write(b''.join(mono[i:i+2]*2 for i in range(0,len(mono),2)))
        session=out/(name+'-replay')
        result,seconds=run([bundle/'runtime/soundjie_d4_runtime','--input',pcm,
            '--output-dir',session,'--current',out/(name+'-current.json'),
            '--window-samples','16000','--hop-samples','6400','--cadence-ms','400',
            '--classifier-command',bundle/'runtime/soundjie_d4_classifier',
            '--class-contract','soundjie-alert5bg-v3','--alert-thresholds',
            (bundle/'config/alert-thresholds.txt').read_text().strip(),
            '--tv-bank',bundle/'tv/bank.tsv','--traffic-model',bundle/'tv/traffic.bin',
            '--tv-dialogue-horn-start','4.8','--tv-dialogue-horn-end','6.8'],timeout=90)
        (out/(name+'-runtime.stderr')).write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError('runtime failed '+name+': '+result.stderr.decode('utf-8','replace')[-1000:])
        rows=[json.loads(x,object_pairs_hook=OrderedDict) for x in (session/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
        # The native runtime emits one terminal unavailable record at EOF;
        # session_state belongs to the supervisor's stop record, not this schema.
        if len(rows)<2 or rows[-1]['class_id']!=-1 or rows[-1]['tv_matched']:
            raise ValueError('missing clean runtime EOF state')
        live=rows[:-1]
        if any(not s['dpu_ok'] or len(s['class_scores'])!=6 for s in live):
            raise ValueError('unhealthy real inference row')
        matched=[s for s in live if s['tv_matched']]
        expected='ambulance_traffic' if name=='ambulance' else 'dialogue'
        if not matched or any(s['tv_scene_profile']!=expected for s in matched):
            raise ValueError('wrong or absent clip identification '+name)
        if any(s['fine_valid'] or s['road_score']!=0 for s in matched if expected=='dialogue'):
            raise ValueError('dialogue leaked traffic score')
        if name=='dialogue_horn':
            selected=[s for s in matched if 4.8<=s['tv_position_seconds']<6.8]
            if not selected:
                raise ValueError('opening horn interval was never identified')
        else:
            expected_cue='open_map' if name=='ambulance' else None
            selected=[s for s in matched if s['caption_text'] and
                      (expected_cue is None or s['caption_cue']==expected_cue)]
            if not selected:
                raise ValueError('no subtitle recovered during '+name)
        snapshot=selected[-1]
        # Genuine replay scores, with framebuffer availability supplied for this
        # offline display check. No fabricated DoA or inference output.
        snapshot['framebuffer_ok']=True
        snap=out/(name+'-ui.json');snap.write_text(json.dumps(snapshot,ensure_ascii=False),encoding='utf-8')
        ui_result,_=run([bundle/'runtime/soundjie_hdmi_ui','--snapshot',snap,
            '--atlas',bundle/'assets/hdmi/soundjie_glyph_atlas.bin','--class-contract','soundjie-alert5bg-v3',
            '--width','1280','--height','720','--offscreen',out/(name+'.ppm'),
            '--report',out/(name+'-ui-report.json')])
        if ui_result.returncode:
            raise RuntimeError('UI render failed '+ui_result.stderr.decode('utf-8','replace'))
        ui=read_json(out/(name+'-ui-report.json'))
        if ui['display_class_count']!=6 or ui['layout']!='unified_live_caption_v3':
            raise ValueError('wrong HDMI layout')
        if name=='dialogue_horn' and ui['display_label']!='车辆鸣笛':
            raise ValueError('opening horn heading missing')
        if name=='dialogue' and ui['display_label']!='普通说话声':
            raise ValueError('dialogue heading differs')
        report['replays'].append(dict(name=name,windows=len(live),matched=len(matched),
            clip_ids=sorted(set(s['tv_clip_id'] for s in matched)),
            captions=sorted(set((s['caption_speaker'],s['caption_text']) for s in matched if s['caption_text']),key=str),
            max_match_ms=max(s['clip_match_ms'] for s in live),
            mean_match_ms=sum(s['clip_match_ms'] for s in live)/len(live),
            max_runtime_ms=max(s['end_to_end_ms'] for s in live),wall_ms=seconds*1000,
            ui_display_label=ui['display_label'],ui_sha256=hashlib.sha256((out/(name+'.ppm')).read_bytes()).hexdigest()))
    report['result']='DPU_AND_DIGITAL_REPLAY_OK'
    report['live_acoustic_replay_tested']=False
    (out/'result.json').write_text(json.dumps(report,ensure_ascii=True,indent=2),encoding='utf-8')
    print('SJ6_CHECK_RESULT '+json.dumps(report,ensure_ascii=True))

if __name__=='__main__':
    main()
