"""Pinned physical-six/display-five export. Build subcommand runs in legacy Caffe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from scripts.host import audit_soundjie_d3_model as audit

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT='7b547d1e55173617b251009cb65386db5719aacba684e353380bdfd32a7a73d1'
POLICY='151020e782a93315e326f99881e0bd1f2ffbe9a2b85d7e6889413e075914ec23'
EXPORT='60e0fd79498d9263030f1a379550c6081cb785b1afc6199043fa51c37cc7d4f0'
THRESHOLDS=np.array([.4,.6500000000000002,.4,.4])
SHAPES={}
for layer,inc,outc in [('conv1',1,16),('conv2',16,32),('conv3',32,64),('conv4',64,96)]:
    SHAPES[layer+'.weight']=(outc,inc,3,3);SHAPES[layer+'.bias']=(outc,)
SHAPES.update({'fc8.weight':(6,2304),'fc8.bias':(6,)})
sha_bytes=lambda b:hashlib.sha256(b).hexdigest()
sha=audit.sha256_file


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,sort_keys=True)
def save(path,value):
    with Path(path).open('xb') as f: np.save(f,value,allow_pickle=False)


def check_files(root,pins):
    root=Path(root).resolve()
    for name,digest in pins.items():
        path=root/name
        if Path(name).name!=name or path.resolve().parent!=root: raise ValueError('unsafe artifact name')
        if sha(path)!=digest: raise ValueError('artifact hash drift: '+name)


def check_weights(arrays):
    if set(arrays)!=set(SHAPES): raise ValueError('weight names differ')
    for k,s in SHAPES.items():
        if arrays[k].shape!=s or arrays[k].dtype!=np.float32 or not np.isfinite(arrays[k]).all():
            raise ValueError('weight shape/type/value differs: '+k)


def check_calibration(x,rows):
    if x.shape!=(192,1,64,96) or x.dtype!=np.float32 or not np.isfinite(x).all() or len(rows)!=192:
        raise ValueError('calibration shape/type/value differs')
    if len({r['sampling_identity'] for r in rows})!=192: raise ValueError('calibration recordings overlap')
    labels=[]
    for tensor,r in zip(x,rows):
        if r['purpose']!='train' or r['derivation'] is not None or sha_bytes(tensor.tobytes())!=r['feature_sha256']:
            raise ValueError('calibration must match original train sources')
        labels.append(r['target_index'])
    if sorted(labels)!=[i//32 for i in range(192)]: raise ValueError('calibration class balance differs')


def parity(reference,candidate):
    if reference.shape!=(12,6) or candidate.shape!=reference.shape or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError('six-output parity shape/value differs')
    error=float(np.max(np.abs(reference-candidate)));matches=int(np.sum(reference.argmax(1)==candidate.argmax(1)))
    if error>1e-4 or matches!=12: raise ValueError('Caffe parity failed')
    return dict(max_absolute_error=error,top1_matches=matches,samples=12)


def five_scores(logits):
    z=np.asarray(logits,dtype=np.float64)
    if z.ndim!=2 or z.shape[1]!=6 or not np.isfinite(z).all(): raise ValueError('finite six logits required')
    p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
    return np.column_stack((p[:,0]+p[:,1],p[:,2:]))


def decide(scores):
    ids=scores.argmax(1);out=ids.copy()
    for i,t in enumerate(THRESHOLDS):out[(ids==i)&(scores[:,i]<t)]=4
    return out


def stage(handoff,output):
    """Windows/PyTorch preparation; the container has no dependency on torch."""
    from scripts.host.ablate_soundjie_glass import checked_inputs
    from scripts.host.assess_soundjie_glass_ablation import replay_inputs
    from scripts.host.soundjie_alert6 import make_model
    from scripts.host.train_soundjie_alert6 import predict
    import torch
    if output.exists():raise FileExistsError(output)
    policy=read(handoff/'candidate-policy.json')
    if sha(handoff/'candidate-policy.json')!=POLICY or policy['checkpoint_sha256']!=CHECKPOINT: raise ValueError('policy drift')
    ckpt=Path(policy['checkpoint'])
    if sha(ckpt)!=CHECKPOINT:raise ValueError('checkpoint drift')
    exported=handoff/'physical6-display5';m=read(exported/'export-manifest.json')
    if sha(exported/'export-manifest.json')!=EXPORT:raise ValueError('export manifest drift')
    pins={str(p).replace('\\','/').split('/')[-1]:s for p,s in m['files'].items()}
    if len(pins)!=len(m['files']) or m['checkpoint_sha256']!=CHECKPOINT or m['output_shape']!=[1,6]:raise ValueError('export contract drift')
    check_files(exported,pins)
    with np.load(exported/'fused-weights.npz',allow_pickle=False) as f:check_weights({k:f[k] for k in f.files})
    check_calibration(np.load(exported/'calibration-train.npy',allow_pickle=False),m['calibration_sources'])
    config=read(ROOT/'config/soundjie_siren_merge.json')
    x,y,purpose,rows,_=checked_inputs(config)
    val=np.flatnonzero(purpose=='validation')
    if len(val)!=356:raise ValueError('validation length drift')
    model=make_model();torch.set_num_threads(4)
    model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    rx,rwindows,events=replay_inputs(config)
    output.mkdir(parents=True,exist_ok=False)
    for name in pins:shutil.copyfile(str(exported/name),str(output/name))
    shutil.copyfile(str(exported/'export-manifest.json'),str(output/'export-manifest.json'))
    shutil.copyfile(str(handoff/'candidate-policy.json'),str(output/'candidate-policy.json'))
    for name,value in [('validation-inputs.npy',x[val]),('validation-targets.npy',y[val]),
                       ('validation-float.npy',predict(model,x[val])),('replay-inputs.npy',rx),('replay-float.npy',predict(model,rx))]:
        save(output/name,value)
    write(output/'validation-sources.json',[rows[i] for i in val])
    write(output/'replay-sources.json',dict(windows=rwindows,events=events))
    write(output/'input-manifest.json',dict(schema=1,checkpoint_sha256=CHECKPOINT,policy_sha256=POLICY,
        artifacts={p.name:sha(p) for p in sorted(output.iterdir())},independent_acceptance=False))


def forward(net,inputs):
    out=[]
    for feature in inputs:
        net.blobs['data'].data[...]=feature[None,...];net.forward()
        out.append(np.asarray(net.blobs['fc8'].data[0]).reshape(-1).copy())
    z=np.stack(out).astype(np.float32)
    if z.shape!=(len(inputs),6) or not np.isfinite(z).all():raise ValueError('Caffe output shape/value differs')
    return z


def build(input_dir,output,compiler):
    import caffe
    import h5py
    from scripts.host.quantize_soundjie_d4 import require_arm32
    manifest=read(input_dir/'input-manifest.json');check_files(input_dir,manifest['artifacts'])
    if manifest['checkpoint_sha256']!=CHECKPOINT or sha(input_dir/'candidate-policy.json')!=POLICY or sha(input_dir/'export-manifest.json')!=EXPORT:raise ValueError('candidate drift')
    meta=read(input_dir/'export-manifest.json');cal=np.load(input_dir/'calibration-train.npy',allow_pickle=False)
    check_calibration(cal,meta['calibration_sources'])
    output.mkdir(parents=True,exist_ok=False);commands=[]
    env=dict(os.environ);env['LD_LIBRARY_PATH']=str(compiler.parent.parent/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    def run(argv,name):
        commands.append(argv)
        with (output/name).open('xb') as f:subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    run([str(compiler),'--version'],'compiler-version.txt')
    version=(output/'compiler-version.txt').read_text()
    if 'dnnc version v2.05' not in version or 'DPU Target : v1.4.0' not in version:raise ValueError('compiler mismatch')
    caffe.set_mode_cpu();net=caffe.Net(str(input_dir/'deploy.prototxt'),caffe.TEST)
    with np.load(input_dir/'fused-weights.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    check_weights(arrays)
    if set(net.params)!=set(k.split('.')[0] for k in SHAPES):raise ValueError('Caffe layers differ')
    for layer in net.params:
        if len(net.params[layer])!=2:raise ValueError('Caffe parameter blobs differ')
        for blob,suffix in zip(net.params[layer],('weight','bias')):
            w=arrays[layer+'.'+suffix]
            if blob.data.shape!=w.shape:raise ValueError('Caffe weight shape differs')
            blob.data[...]=w
    z=forward(net,np.load(input_dir/'parity-inputs.npy',allow_pickle=False))
    pr=parity(np.load(input_dir/'pytorch-logits.npy',allow_pickle=False),z)
    save(output/'caffe-parity-logits.npy',z);write(output/'caffe-parity.json',pr)
    model=output/'float.caffemodel';net.save(str(model))
    # Preserve real calibration labels; never use validation for quantizer input.
    hdf=output/'calibration.h5'
    with h5py.File(str(hdf),'w') as f:
        f.create_dataset('data',data=cal)
        f.create_dataset('label',data=np.array([r['target_index'] for r in meta['calibration_sources']],np.float32).reshape(-1,1))
    listing=output/'hdf5-list.txt';listing.write_text(str(hdf)+'\n')
    quantproto=output/'quantize.prototxt';quantproto.write_text(audit.make_quantizer_prototxt((input_dir/'deploy.prototxt').read_text(),listing))
    run(['vai_q_caffe','quantize','-model',str(quantproto),'-weights',str(model),'-output_dir',str(output/'quantized'),'-calib_iter','192'],'quantize.log')
    runtime=output/'quantized/runtime-deploy.prototxt';runtime.write_text(audit.make_runtime_prototxt((output/'quantized/deploy.prototxt').read_text()))
    quant=caffe.Net(str(runtime),str(output/'quantized/deploy.caffemodel'),caffe.TEST)
    for name in ('parity','validation','replay'):
        save(output/(name+'-quantized.npy'),forward(quant,np.load(input_dir/(name+'-inputs.npy'),allow_pickle=False)))
    (output/'compiled').mkdir()
    run([str(compiler),'--parser','caffe','--prototxt',str(runtime),'--caffemodel',str(output/'quantized/deploy.caffemodel'),
         '--dpu','1152FA','--cpu_arch','arm32','--abi','1','--mode','normal','--output_dir',str(output/'compiled'),'--net_name','soundjie_d3'],'compile.log')
    elfs=list((output/'compiled').glob('dpu_*.elf'))
    if len(elfs)!=1:raise ValueError('expected one ELF')
    raw=elfs[0].read_bytes();require_arm32(raw)
    contract=audit.load_compiler_contract(ROOT/'config/soundjie_d3_compile.json');contract['output_count']=6
    log=(output/'compile.log').read_text();evidence=audit.extract_compile_evidence(contract,log,raw,log,runtime.read_text())
    audit.validate_compile_evidence(contract,evidence)
    write(output/'commands.json',commands)
    write(output/'compiled-manifest.json',dict(schema=1,checkpoint_sha256=CHECKPOINT,policy_sha256=POLICY,
        input_manifest_sha256=sha(input_dir/'input-manifest.json'),compiler_sha256=sha(compiler),compile_evidence=evidence,
        caffe_parity=pr,board_tested=False,independent_acceptance=False,
        artifacts={p.relative_to(output).as_posix():sha(p) for p in sorted(output.rglob('*')) if p.is_file()}))


def fixed_eval(input_dir,compiled,output):
    """Evaluate learned FixedNeuron activations as well as weights; never recalibrate."""
    import caffe
    m=read(input_dir/'input-manifest.json');check_files(input_dir,m['artifacts'])
    cm=read(compiled/'compiled-manifest.json')
    if cm['input_manifest_sha256']!=sha(input_dir/'input-manifest.json'):raise ValueError('compiled/input mismatch')
    for name,digest in cm['artifacts'].items():
        if sha(compiled/name)!=digest:raise ValueError('compiled artifact drift')
    output.mkdir(parents=True,exist_ok=False)
    text=(compiled/'quantized/quantize_train_test.prototxt').read_text()
    if text.count('type: "FixedNeuron"')!=6 or 'type: "ConvolutionFixed"' not in text:
        raise ValueError('fixed activation network missing')
    proto=output/'fixed-evaluation.prototxt';proto.write_text(audit.make_runtime_prototxt(text))
    caffe.set_mode_cpu();net=caffe.Net(str(proto),str(compiled/'quantized/quantize_train_test.caffemodel'),caffe.TEST)
    def params():return {name:[b.data.copy() for b in blobs] for name,blobs in net.params.items()}
    before=params()
    for name in ('parity','validation','replay'):
        save(output/(name+'-fixed.npy'),forward(net,np.load(input_dir/(name+'-inputs.npy'),allow_pickle=False)))
    after=params()
    if any(not np.array_equal(a,b) for name in before for a,b in zip(before[name],after[name])):
        raise ValueError('TEST changed learned parameters/quantization positions')
    write(output/'fixed-evaluation.json',dict(scope='Caffe TEST FixedNeuron simulation; not physical DPU',
        input_manifest_sha256=sha(input_dir/'input-manifest.json'),compiled_manifest_sha256=sha(compiled/'compiled-manifest.json'),
        learned_parameters_unchanged=True,artifacts={p.name:sha(p) for p in sorted(output.iterdir())},board_tested=False))


def assess(input_dir,compiled,fixed,output):
    from scripts.host.soundjie_siren_merge import metrics
    check_files(input_dir,read(input_dir/'input-manifest.json')['artifacts'])
    cm=read(compiled/'compiled-manifest.json');fm=read(fixed/'fixed-evaluation.json')
    for name,h in cm['artifacts'].items():
        if sha(compiled/name)!=h:raise ValueError('compiled output drift')
    check_files(fixed,fm['artifacts'])
    if fm['compiled_manifest_sha256']!=sha(compiled/'compiled-manifest.json') or fm['input_manifest_sha256']!=sha(input_dir/'input-manifest.json'):
        raise ValueError('fixed evaluation input chain drift')
    y=np.load(input_dir/'validation-targets.npy',allow_pickle=False);rows=read(input_dir/'validation-sources.json')
    sources=read(input_dir/'replay-sources.json');mapping=[0,0,1,2,3,4];modes={}
    for name,vl,rl in [('pytorch',input_dir/'validation-float.npy',input_dir/'replay-float.npy'),
                       ('caffe_quantized_weights',compiled/'validation-quantized.npy',compiled/'replay-quantized.npy'),
                       ('caffe_fixed_activations',fixed/'validation-fixed.npy',fixed/'replay-fixed.npy')]:
        p=five_scores(np.load(vl,allow_pickle=False));rp=five_scores(np.load(rl,allow_pickle=False));clips=[]
        for e in sources['events']:
            row={k:e[k] for k in ('event_index','recording_id','purpose','target_index')}
            for domain in ('digital','acoustic'):
                idx=[i for i,w in enumerate(sources['windows']) if w['event_index']==e['event_index'] and w['domain']==domain]
                q=rp[idx];pred=decide(q);target=mapping[e['target_index']]
                row[domain]=dict(mean_top1=int(q.mean(0).argmax()),thresholded_window_ids=pred.tolist(),
                    clip_match=bool(q.mean(0).argmax()==target),any_correct=bool(np.any(pred==target)))
            clips.append(row)
        val=[r for r in clips if r['purpose']=='validation']
        modes[name]=dict(raw=metrics(y,p.argmax(1),rows),thresholded=metrics(y,decide(p),rows),replay=clips,
                        validation_acoustic=dict(clips=len(val),matches=sum(r['acoustic']['clip_match'] for r in val),
                        positive_clips=sum(r['target_index']!=5 for r in val),
                        positive_any_correct=sum(r['target_index']!=5 and r['acoustic']['any_correct'] for r in val)))
    a=modes['pytorch']['thresholded'];b=modes['caffe_fixed_activations']['thresholded']
    result=dict(schema=1,checkpoint_sha256=CHECKPOINT,policy_sha256=POLICY,
        compiled_manifest_sha256=sha(compiled/'compiled-manifest.json'),fixed_manifest_sha256=sha(fixed/'fixed-evaluation.json'),
        elf_sha256=sha(compiled/'compiled/dpu_soundjie_d3.elf'),thresholds=THRESHOLDS.tolist(),results=modes,
        fixed_minus_float=dict(macro_recall=b['macro_recall']-a['macro_recall'],background_fpr=b['negative_false_alert_rate']-a['negative_false_alert_rate']),
        scope='development conversion/replay check; not independent D5 or physical DPU',board_tested=False,
        known_limits=['glass replay miss persists','crying/screaming false positives persist','thresholds fitted on original validation'])
    write(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['stage','build','fixed-eval','assess'])
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--compiler',type=Path)
    p.add_argument('--compiled',type=Path)
    p.add_argument('--fixed',type=Path)
    a=p.parse_args()
    if a.command=='stage':stage(a.input,a.output)
    elif a.command=='build':build(a.input,a.output,a.compiler)
    elif a.command=='fixed-eval':fixed_eval(a.input,a.compiled,a.output)
    else:assess(a.input,a.compiled,a.fixed,a.output)
if __name__=='__main__':main()
