"""Reproducible digital replay checks. They are not acoustic/board validation."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import wave
import numpy as np
def lin(p):return '/mnt/'+str(p)[0].lower()+str(p)[2:].replace('\\','/')
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--negative-dir',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2];binary=a.output/'known_clip_cli'
    subprocess.run(['wsl','-d','Ubuntu','--','g++','-std=c++11','-O2','-Wall','-Wextra','-Werror','-I'+lin(root/'src'),*[lin(root/'src'/f) for f in ('fft_radix2.cpp','known_clip_matcher.cpp','known_clip_cli.cpp')],'-o',lin(binary)],check=True,capture_output=True)
    ref=a.output/'reference.features'
    subprocess.run(['wsl','-d','Ubuntu','--',lin(binary),'--make-reference',lin(a.reference_dir/'reference.s16le'),'--output',lin(ref)],check=True,capture_output=True)
    x=np.frombuffer((a.reference_dir/'reference.s16le').read_bytes(),dtype='<i2').astype(float)/32768;rng=np.random.default_rng(9001)
    room=x.copy();room[480:]+=.3*x[:-480];room[1120:]+=.17*x[:-1120]
    cases={'direct':x,'gain_025':x*.25,'noise_20db':x+rng.normal(0,np.sqrt(np.mean(x*x))*.1,len(x)),
        'room':room,'phase_101':x[101:],'phase_219':x[219:],'seek':x[160000:],
        'pause':np.r_[x[:240000],np.zeros(48000),x[240000:]],'unknown_noise':rng.normal(0,.1,160000),'reversed':x[::-1]}
    if a.negative_dir:
        for wav in sorted(a.negative_dir.glob('*.wav')):
            with wave.open(str(wav),'rb') as f:
                if(f.getframerate(),f.getnchannels(),f.getsampwidth())!=(16000,1,2):raise ValueError('negative format')
                cases['negative_'+wav.stem]=np.frombuffer(f.readframes(f.getnframes()),'<i2')/32768
    summary=[]
    for name,z in cases.items():
        raw=a.output/(name+'.s16le');raw.write_bytes(np.clip(np.rint(z*32768),-32768,32767).astype('<i2').tobytes());start=time.monotonic()
        r=subprocess.run(['wsl','-d','Ubuntu','--',lin(binary),'--reference',lin(ref),'--cues',lin(a.reference_dir/'cues.tsv'),'--pcm',lin(raw)],check=True,capture_output=True)
        (a.output/(name+'.jsonl')).write_bytes(r.stdout);rows=[json.loads(l) for l in r.stdout.splitlines()];m=[v for v in rows if v['matched']]
        summary.append(dict(case=name,windows=len(rows),matched=len(m),cues=sorted({v['cue_id'] for v in m if v['cue_id']}),elapsed_seconds=time.monotonic()-start,
            silence_false_matches=sum(v['matched'] for v in rows if name=='pause' and 15.4<v['input_seconds']<=18)))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for r in summary:
        if r['case'] in ('direct','gain_025','noise_20db','room','phase_101','phase_219','seek') and len(r['cues'])!=3:raise RuntimeError('missing cue: '+r['case'])
        if (r['case'].startswith('negative_') or r['case'] in ('unknown_noise','reversed')) and r['matched']:raise RuntimeError('negative matched: '+r['case'])
        if r['silence_false_matches']:raise RuntimeError('silence matched')
if __name__=='__main__':main()
