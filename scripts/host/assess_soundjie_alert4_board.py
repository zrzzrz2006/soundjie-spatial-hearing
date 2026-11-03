"""Check real-board five-class probabilities against frozen development fixtures."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import numpy as np
from scripts.host.assess_soundjie_siren5_board import read_archive


def digest(data):
    return hashlib.sha256(data).hexdigest()


def assess(package, manifest, results):
    pins=json.loads(manifest.read_text(encoding='utf-8'))
    if digest(package.read_bytes())!=pins['archive_sha256']:
        raise ValueError('package hash mismatch')
    files=read_archive(package)
    for name,sha in pins['payload_sha256'].items():
        if digest(files['sj-alert4/'+name])!=sha:
            raise ValueError('payload mismatch '+name)
    refs=json.loads(files['sj-alert4/probe/reference.json'])['records']
    if len(refs)!=10 or pins['contract']!='soundjie-alert4bg-v2':
        raise ValueError('incorrect probe contract')
    received=read_archive(results)
    data={}
    for name,raw in received.items():
        key=PurePosixPath(name).name
        if key in data:raise ValueError('duplicate probe result')
        data[key]=raw
    expected={PurePosixPath(r['file']).name+s for r in refs for s in ('.txt','.stderr')}
    if set(data)!=expected:raise ValueError('probe set mismatch')
    thresholds=pins['thresholds']
    def decide(scores):
        top=int(np.argmax(scores))
        return 4 if top<4 and scores[top]<thresholds[top] else top
    records=[]
    for ref in refs:
        name=PurePosixPath(ref['file']).name
        values=data[name+'.txt'].decode('ascii').split()
        if len(values)!=8:raise ValueError('expected five probabilities')
        top=int(values[0]);numbers=np.asarray([float(v) for v in values[1:]])
        confidence,ms=numbers[:2];actual=numbers[2:]
        if (not np.isfinite(numbers).all() or not 0<=top<5 or ms<=0
                or np.any(actual<0) or np.any(actual>1) or abs(actual.sum()-1)>1e-6
                or top!=int(actual.argmax()) or abs(confidence-actual[top])>1e-8):
            raise ValueError('invalid physical DPU protocol')
        z=np.asarray(ref['fixed_logits'],dtype=np.float64)
        if z.shape!=(5,) or not np.isfinite(z).all():raise ValueError('invalid fixed reference')
        fixed=np.exp(z-z.max());fixed/=fixed.sum()
        records.append(dict(file=name,source=ref['source'],actual=actual.tolist(),fixed=fixed.tolist(),
            physical_top1=top,fixed_top1=int(fixed.argmax()),physical_decision=decide(actual),
            fixed_decision=decide(fixed),inference_ms=float(ms),
            max_probability_error=float(np.abs(actual-fixed).max()),
            stderr=data[name+'.stderr'].decode('utf-8',errors='replace')))
    ms=np.asarray([r['inference_ms'] for r in records])
    return dict(schema=1,contract=pins['contract'],model_sha256=pins['linked_model_sha256'],
        package_sha256=pins['archive_sha256'],results_sha256=digest(results.read_bytes()),
        count=len(records),protocol_valid=True,top1_agreement=sum(r['physical_top1']==r['fixed_top1'] for r in records),
        decision_agreement=sum(r['physical_decision']==r['fixed_decision'] for r in records),
        max_probability_error=max(r['max_probability_error'] for r in records),
        dpu_ms=dict(min=float(ms.min()),median=float(np.median(ms)),p95=float(np.percentile(ms,95)),max=float(ms.max())),
        scope='ten development PCM conversion checks, not independent acoustic accuracy',
        independent_acceptance=False,records=records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','manifest','results','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=assess(a.package,a.manifest,a.results)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
