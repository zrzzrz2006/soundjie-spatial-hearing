"""Compare optimized full-search matching with the recorded pre-optimization source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scripts.host.build_soundjie_dual_ui_patch import ROOT,lin

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    old=subprocess.check_output(['git','show','9637bfe:src/known_clip_matcher.cpp'],cwd=ROOT)
    baseline=out/'old.cpp';baseline.write_bytes(old)
    for name,source in [('old',baseline),('new',ROOT/'src/known_clip_matcher.cpp')]:
        subprocess.run(['wsl','-d','Ubuntu','--','g++','-O2','-std=c++11','-I'+lin(ROOT/'src'),lin(source),
            lin(ROOT/'src/fft_radix2.cpp'),lin(ROOT/'src/known_clip_cli.cpp'),'-o',lin(out/name)],check=True,capture_output=True)
    clip=ROOT/'artifacts/tv-ambulance-20261030a';ref=clip/'acoustic-calibration-v1/reference.features';cues=clip/'reference-v1/cues.tsv'
    pcm=np.fromfile(clip/'reference-v1/reference.s16le',dtype='<i2')
    rng=np.random.default_rng(9001)
    cases={'digital':pcm,'gain':np.rint(pcm*.4).astype('<i2'),'shift':pcm[101:],
           'noise':rng.integers(-8000,8000,size=pcm.size,dtype=np.int16),'silence':np.zeros_like(pcm)}
    results=[]
    for name,x in cases.items():
        raw=out/(name+'.s16');raw.write_bytes(x.astype('<i2').tobytes());rows=[]
        for variant in ('old','new'):
            r=subprocess.run(['wsl','-d','Ubuntu','--',lin(out/variant),'--reference',lin(ref),'--cues',lin(cues),'--pcm',lin(raw)],check=True,capture_output=True)
            (out/(name+'-'+variant+'.jsonl')).write_bytes(r.stdout)
            rows.append([json.loads(line) for line in r.stdout.splitlines()])
        assert len(rows[0])==len(rows[1])
        differences={key:max(abs(x[key]-y[key]) for x,y in zip(*rows)) for key in ('score','margin','position_seconds')}
        decisions=all(all(x[k]==y[k] for k in ('matched','cue_id','speaker','text')) for x,y in zip(*rows))
        assert decisions and differences['score']<1e-5 and differences['margin']<1e-5 and differences['position_seconds']==0,(name,differences,decisions)
        results.append(dict(case=name,windows=len(rows[0]),decisions_equal=decisions,differences=differences))
    report=dict(baseline_commit='9637bfe',baseline_sha256=hashlib.sha256(old).hexdigest(),
        optimized_sha256=hashlib.sha256((ROOT/'src/known_clip_matcher.cpp').read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest(),results=results,
        scope='numerical/decision regression on development clip plus transformations, not independent accuracy')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
