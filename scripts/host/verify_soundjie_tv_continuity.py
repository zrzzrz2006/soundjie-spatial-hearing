"""Replay known-source and rejection regressions; no claim of new physical acceptance."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np


def linux(path):
    path=path.resolve()
    return '/mnt/'+str(path)[0].lower()+str(path)[2:].replace('\\','/')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); area=args.artifact_root.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2];binary=out/'known_clip_cli'
    subprocess.run(['wsl','-d','Ubuntu','--','g++','-std=c++11','-O2','-Wall','-Wextra','-Werror',
        '-I'+linux(root/'src'),*[linux(root/'src'/name) for name in ('fft_radix2.cpp','known_clip_matcher.cpp','known_clip_cli.cpp')],
        '-o',linux(binary)],check=True,capture_output=True)
    previous=json.loads((area/'acoustic-reference-verification-v1/summary.json').read_text(encoding='utf-8'))
    cases={r['case']:area/('acoustic-reference-verification-v1' if r['case'] in ('room_before','room_after') else 'digital-checks-v3')/(r['case']+'.s16le') for r in previous['cases']}
    cases['captured_first']=area/'acoustic-analysis-v1/context.s16le'
    cases['captured_second']=area/'acoustic-analysis-v2/context.s16le'
    context=np.frombuffer(cases['captured_second'].read_bytes(),'<i2')
    clip=np.frombuffer((area/'acoustic-analysis-v2/clip.s16le').read_bytes(),'<i2')
    extra={'room2_before':context[:32000],'room2_after':context[-32000:],
           'captured_pause':np.r_[clip[:240000],np.tile(context[:16000],3),clip[240000:]]}
    for name,pcm in extra.items():
        cases[name]=out/(name+'.s16le');cases[name].write_bytes(pcm.astype('<i2').tobytes())
    jobs=[]
    for name,path in cases.items():
        if not path.is_file():raise FileNotFoundError(path)
        jobs.append(dict(name=name,pcm=linux(path),output=linux(out/(name+'.jsonl'))))
    (out/'jobs.json').write_text(json.dumps(jobs),encoding='utf-8')
    worker="""import json,subprocess,sys
from pathlib import Path
for j in json.loads(Path(sys.argv[1]).read_text()):
    r=subprocess.run([sys.argv[2],'--reference',sys.argv[3],'--cues',sys.argv[4],'--pcm',j['pcm']],check=True,capture_output=True)
    with open(j['output'],'xb') as f:f.write(r.stdout)
    print('REPLAY '+j['name'],flush=True)
"""
    subprocess.run(['wsl','-d','Ubuntu','--','python3','-c',worker,linux(out/'jobs.json'),linux(binary),
        linux(area/'acoustic-calibration-v1/reference.features'),linux(area/'reference-v1/cues.tsv')],check=True)
    summary=[]
    negative_names={'unknown_noise','reversed','room_before','room_after','room2_before','room2_after'}
    for name in cases:
        rows=[json.loads(s) for s in (out/(name+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        matches=[r for r in rows if r['matched']]
        cues=sorted({r['cue_id'] for r in matches if r['cue_id']})
        negative=name.startswith('negative_') or name in negative_names
        pause_false=sum(r['matched'] for r in rows if name in ('pause','captured_pause') and 16.6<=r['input_seconds']<=18)
        good=(not matches) if negative else len(cues)==3 and pause_false==0
        report=dict(case=name,windows=len(rows),matched=len(matches),cues=cues,pause_false_matches=pause_false,passed=good)
        if name in ('captured_first','captured_second'):
            active=[r for r in rows if 5<=r['input_seconds']<=32]
            alignment=max([abs(r['position_seconds']-(r['input_seconds']-2)) for r in matches] or [999])
            coverage=sum(r['matched'] for r in active)/len(active)
            outside=sum(r['matched'] for r in rows if r['input_seconds']<2 or r['input_seconds']>=33.5)
            report.update(coverage_after_3s=coverage,max_alignment_error=alignment,outside_matches=outside)
            report['passed']=good and coverage>=.95 and alignment<.08 and outside==0
        summary.append(report)
    report=dict(passed=all(r['passed'] for r in summary),cases=summary,
                scope='development regressions on two prior recordings; fresh physical replay still required')
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('continuity regression failed')


if __name__=='__main__':main()
