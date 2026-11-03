"""Compare the selected six-class model and export DPU-preparation inputs only."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import numpy as np
import torch

from scripts.host.acquire_soundjie_alert6 import read_rows
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.soundjie_alert6 import LABELS,ROOT,make_model,metrics
from scripts.host.train_soundjie_alert6 import load_cache,predict,write_json,softmax,apply_thresholds,negative_breakdown
from scripts.host.train_soundjie_d3 import SoundjieCNN,load_model_config
from scripts.host.train_soundjie_d4 import select_records
from scripts.host.prepare_soundjie_d3 import decode_audio,resample_mono_16k,compute_feature,make_windows,WindowPolicy
from scripts.host.logmel_reference import load_frontend_config

BASELINE=Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d4/candidates-20261021a/20261023/soundjie_d4_float32.pt')
BASELINE_SHA='20fb0d907fec2a7a125302c8367342a56b4b631a629ff5c271e0d2d04f656bdb'


def fold_and_check(model,x):
    model.eval();folded=copy.deepcopy(model).eval();arrays={}
    for index in range(1,5):
        name='conv'+str(index);bn='bn'+str(index)
        conv=torch.nn.utils.fuse_conv_bn_eval(getattr(folded,name),getattr(folded,bn))
        setattr(folded,name,conv);setattr(folded,bn,torch.nn.Identity())
        arrays[name+'.weight']=conv.weight.detach().numpy().copy()
        arrays[name+'.bias']=conv.bias.detach().numpy().copy()
    arrays['fc8.weight']=folded.fc8.weight.detach().numpy().copy()
    arrays['fc8.bias']=folded.fc8.bias.detach().numpy().copy()
    reference=predict(model,x);candidate=predict(folded,x)
    error=float(np.max(np.abs(reference-candidate)))
    if error>1e-4 or not np.array_equal(reference.argmax(1),candidate.argmax(1)):
        raise ValueError('BN folding parity failed')
    if arrays['fc8.weight'].shape!=(6,2304):raise ValueError('six-output export required')
    return arrays,dict(max_absolute_error=error,top1_identical=True,scope='PyTorch BN-folding only; not Caffe/INT8/DPU parity')


def historical_checks(model,baseline):
    """Previously exposed validation/demo diagnostics, never selection or new acceptance."""
    cache=ROOT/'artifacts/rebuilt-features-20261025a'
    audit=json.loads((cache/'audit.json').read_text(encoding='utf-8'))
    if digest_file(cache/'features.npz')!=audit['cache_sha256']:raise ValueError('historical cache drift')
    with np.load(cache/'features.npz',allow_pickle=False) as data:
        mask=data['purpose']=='validation';x=data['x'][mask];y=np.minimum(data['y'][mask],5)
    frozen=dict(baseline=metrics(y,np.minimum(predict(baseline,x).argmax(1),5)),
                candidate=metrics(y,predict(model,x).argmax(1)),source_count=len(y),
                old_cache_sha256=audit['cache_sha256'],purpose='previously_exposed_validation_not_independent_test')
    frontend=load_frontend_config(ROOT/'config/logmel_frontend_v1.json')
    inputs=json.loads((ROOT/'artifacts/window-diagnostic-inputs-20261025.json').read_text(encoding='utf-8'))
    diagnostics=[]
    for row in inputs:
        if digest_file(row['path'])!=row['sha256']:raise ValueError('historical replay source drift')
        signal=resample_mono_16k(decode_audio(Path(row['path'])))
        if len(signal)<16000:
            windows=[next(make_windows(signal,WindowPolicy('energy_center',1))).samples];starts=[0]
        else:
            starts=list(range(0,len(signal)-16000+1,8000));windows=[signal[s:s+16000] for s in starts]
        features=np.stack([compute_feature(np.clip(np.rint(w*32768),-32768,32767)/32768.,frontend) for w in windows])
        old_pred=np.minimum(predict(baseline,features).argmax(1),5)
        scores=softmax(predict(model,features));pred=scores.argmax(1)
        diagnostics.append(dict(source=row,clip_label_only=min(row['expected_index'],5),
            window_labels_are_not_ground_truth=True,window_count=len(starts),
            baseline_distribution=dict(Counter(LABELS[int(i)] for i in old_pred)),
            candidate_distribution=dict(Counter(LABELS[int(i)] for i in pred)),
            windows=[dict(start_s=s/16000.,scores=p.tolist(),candidate=LABELS[int(c)],baseline=LABELS[int(b)])
                     for s,p,c,b in zip(starts,scores,pred,old_pred)]))
    return dict(frozen_validation=frozen,historical_demo_diagnostics=diagnostics,
                used_for_model_selection=False,used_for_threshold_calibration=False,
                fresh_board_inference=False,independent_acceptance=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('features','candidates','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    x,y,purpose,provenance,audit=load_cache(args.features)
    selection=json.loads((args.candidates/'selection.json').read_text(encoding='utf-8'))
    selected=selection['selected'];directory=Path(selected['directory'])
    checkpoint=directory/'soundjie_alert6_float32.pt'
    if digest_file(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('selected checkpoint drift')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ckpt['labels']!=list(LABELS) or ckpt['upstream']!=audit:raise ValueError('checkpoint contract drift')
    model=make_model();model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
    val=purpose=='validation';vx=x[val];vy=y[val];val_rows=[r for r in provenance if r['purpose']=='validation']
    logits=predict(model,vx)
    if digest_file(BASELINE)!=BASELINE_SHA:raise ValueError('onboard baseline checkpoint drift')
    old=torch.load(BASELINE,map_location='cpu',weights_only=True)
    baseline=SoundjieCNN(load_model_config(ROOT/'config/soundjie_d3_model.json'))
    baseline.load_state_dict(old['state_dict'],strict=True);baseline.eval()
    baseline_logits=predict(baseline,vx)
    old_raw=baseline_logits.argmax(1);old_mapped=np.minimum(old_raw,5)
    new_pred=logits.argmax(1)
    selected_metrics=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    thresholded=apply_thresholds(softmax(logits),selected_metrics['thresholds'])
    comparison=dict(schema=1,validation_only=True,source_count=len(vy),
        baseline_sha256=BASELINE_SHA,candidate_sha256=selected['checkpoint_sha256'],
        baseline_mapping='old argmax 5/6/7 -> background; never promote second choice',
        baseline=metrics(vy,old_mapped),candidate=metrics(vy,new_pred),candidate_thresholded=metrics(vy,thresholded),
        baseline_negative_subtypes=negative_breakdown(vy,old_mapped,val_rows),
        candidate_negative_subtypes=negative_breakdown(vy,new_pred,val_rows),
        candidate_thresholded_negative_subtypes=negative_breakdown(vy,thresholded,val_rows),
        thresholds=selected_metrics['thresholds'],thresholds_fit_on_same_validation=True,
        independent_acceptance=False,physical_replay_test='not_run',DPU_quantization='not_run')
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'validation-comparison.json',comparison)
    # Two distinct validation recordings per class for a later Caffe parity check.
    indices=[]
    for c in range(6):
        groups=set()
        for i in np.flatnonzero(vy==c):
            group=val_rows[i]['source_group']
            if group in groups:continue
            groups.add(group);indices.append(int(i))
            if len(groups)==2:break
        if len(groups)!=2:raise ValueError('insufficient independent validation groups')
    arrays,parity=fold_and_check(model,vx[indices])
    with (args.output/'fused-weights.npz').open('xb') as f:np.savez(f,**arrays)
    for name,value in [('parity-inputs.npy',vx[indices]),('pytorch-logits.npy',logits[indices])]:
        with (args.output/name).open('xb') as f:np.save(f,value,allow_pickle=False)
    deploy=(ROOT/'models/soundjie_d3/deploy.prototxt').read_text(encoding='utf-8')
    if deploy.count('num_output: 8')!=1:raise ValueError('unexpected legacy deploy layout')
    (args.output/'deploy.prototxt').write_text(deploy.replace('num_output: 8','num_output: 6'),encoding='utf-8')
    (args.output/'labels.txt').write_text('\n'.join(LABELS)+'\n',encoding='utf-8')
    # Calibration uses original train windows only, balanced by class and recording.
    calibration=[]
    for c in range(6):
        used=set()
        for i,row in enumerate(provenance):
            if row['purpose']=='train' and row['target_index']==c and row['derivation'] is None and row['recording_id'] not in used:
                calibration.append(i);used.add(row['recording_id'])
                if len(used)>=32:break
    with (args.output/'calibration-train.npy').open('xb') as f:np.save(f,x[calibration],allow_pickle=False)
    write_json(args.output/'export-manifest.json',dict(schema=1,contract='soundjie-alert6-v1',
        output_shape=[1,6],output_tensor_name='fc8',tensor_name_note='legacy name only; six output values',
        checkpoint_sha256=selected['checkpoint_sha256'],frontend_sha256=audit['frontend_sha256'],
        class_map_sha256=audit['class_map_sha256'],folding_parity=parity,
        parity_sources=[val_rows[i] for i in indices],calibration_sources=[provenance[i] for i in calibration],
        artifacts={p.name:digest_file(p) for p in sorted(args.output.iterdir()) if p.is_file()},
        status='host_weights_and_export_inputs_ready',caffe_parity='not_run',quantization='not_run',
        DPU_compilation='not_run',board_inference='not_run',
        deployment_requires=['Caffe export/parity with six outputs','INT8 calibration and DPU compile',
                             'classifier/runtime/state/HDMI six-class protocol support','physical replay acceptance']))
    old_rows=read_rows('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d4/data-audit-20261021b/manifest.jsonl')
    old_config=old['upstream']['config']
    original=select_records(old_rows,old_config['max_sources_per_class'],old_config['data_seed'])
    write_json(args.output/'source-count-comparison.json',dict(
        baseline_original=dict(Counter(r['purpose']+'/'+r['target_id'] for r in original)),
        new=audit['source_audit']['counts'],new_subtypes=audit['source_audit']['subtypes'],
        note='counts are source files; augmented/adjacent windows are reported separately'))
    write_json(args.output/'historical-checks.json',historical_checks(model,baseline))
    print(json.dumps(dict(stage='evaluation_export_done',baseline_macro=comparison['baseline']['macro_recall'],
        candidate_macro=comparison['candidate']['macro_recall'],
        baseline_negative_false_alert=comparison['baseline']['negative_false_alert_rate'],
        candidate_negative_false_alert=comparison['candidate']['negative_false_alert_rate'],
        output=str(args.output))),flush=True)


if __name__=='__main__':main()
