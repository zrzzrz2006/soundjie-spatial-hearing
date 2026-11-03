"""Five-output host export preparation; no Caffe, INT8 or DPU execution."""
import copy
from pathlib import Path
import numpy as np
import torch
from scripts.host.soundjie_alert6 import ROOT,LABELS
from scripts.host.soundjie_siren_merge import LABELS5,map_targets
from scripts.host.train_soundjie_alert6 import predict,write_json
from scripts.host.build_soundjie_d4_manifest import digest_file


def calibration_indices(rows):
    result=[];used=set()
    for c in range(6):
        selected=[]
        for i,r in enumerate(rows):
            if r['purpose']=='train' and r['target_index']==c and r['derivation'] is None and r['sampling_identity'] not in used:
                selected.append(i);used.add(r['sampling_identity'])
                if len(selected)==32:break
        if len(selected)!=32:raise ValueError('not enough distinct train originals for calibration')
        result.extend(selected)
    return result


def fold_and_check(model,x):
    model.eval();folded=copy.deepcopy(model).eval();arrays={}
    for i in range(1,5):
        conv='conv'+str(i);bn='bn'+str(i)
        fused=torch.nn.utils.fuse_conv_bn_eval(getattr(folded,conv),getattr(folded,bn))
        setattr(folded,conv,fused);setattr(folded,bn,torch.nn.Identity())
        arrays[conv+'.weight']=fused.weight.detach().numpy().copy();arrays[conv+'.bias']=fused.bias.detach().numpy().copy()
    arrays['fc8.weight']=folded.fc8.weight.detach().numpy().copy();arrays['fc8.bias']=folded.fc8.bias.detach().numpy().copy()
    if arrays['fc8.weight'].shape!=(5,2304):raise ValueError('five-output shape required')
    before=predict(model,x);after=predict(folded,x);error=float(np.max(np.abs(before-after)))
    if error>1e-4 or not np.array_equal(before.argmax(1),after.argmax(1)):raise ValueError('folding parity failed')
    return arrays,dict(max_absolute_error=error,top1_identical=True,scope='PyTorch BN folding only')


def output_contract(physical_outputs):
    if physical_outputs not in (5,6):raise ValueError('only five merged or six source outputs supported')
    return dict(physical_output_count=physical_outputs,
        physical_labels=list(LABELS5 if physical_outputs==5 else LABELS),display_labels=list(LABELS5),
        probability_merge_groups=[[i] for i in range(5)] if physical_outputs==5 else [[0,1],[2],[3],[4],[5]],
        merge_stage='after_softmax_never_sum_logits',display_contract='soundjie-siren5-v1')


def export(model,x,y,purpose,rows,checkpoint_sha,output,physical_outputs=5):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    contract=output_contract(physical_outputs)
    if model.fc8.out_features!=physical_outputs:raise ValueError('physical model/output count mismatch')
    parity=[]
    for c in range(6):
        part=np.flatnonzero((purpose=='validation')&(y==c))[:2]
        if len(part)!=2:raise ValueError('insufficient parity inputs')
        parity.extend(part.tolist())
    calibration=calibration_indices(rows)
    if physical_outputs==5:arrays,fold=fold_and_check(model,x[parity])
    else:
        from scripts.host.evaluate_soundjie_alert6 import fold_and_check as fold_six
        arrays,fold=fold_six(model,x[parity])
    output.mkdir(parents=True,exist_ok=False)
    with (output/'fused-weights.npz').open('xb') as f:np.savez(f,**arrays)
    for name,value in [('parity-inputs.npy',x[parity]),('pytorch-logits.npy',predict(model,x[parity])),('calibration-train.npy',x[calibration])]:
        with (output/name).open('xb') as f:np.save(f,value,allow_pickle=False)
    proto=(ROOT/'models/soundjie_d3/deploy.prototxt').read_text(encoding='utf-8')
    if proto.count('num_output: 8')!=1:raise ValueError('unexpected deploy layout')
    with (output/'deploy.prototxt').open('x',encoding='utf-8',newline='\n') as f:f.write(proto.replace('num_output: 8','num_output: '+str(physical_outputs)))
    with (output/'labels.txt').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(contract['physical_labels'])+'\n')
    model_contract='soundjie-siren5-v1' if physical_outputs==5 else 'soundjie-alert6-v1'
    class_file='soundjie_siren5_classes.json' if physical_outputs==5 else 'soundjie_alert6_classes.json'
    manifest=dict(schema=1,contract=model_contract,checkpoint_sha256=checkpoint_sha,output_shape=[1,physical_outputs],
        output_contract=contract,display_class_map_sha256=digest_file(ROOT/'config/soundjie_siren5_classes.json'),
        output_tensor_name='fc8',class_map_sha256=digest_file(ROOT/'config'/class_file),
        folding=fold,parity_sources=[rows[i] for i in parity],calibration_sources=[rows[i] for i in calibration],
        calibration_merged_counts=np.bincount(map_targets(y[calibration]),minlength=5).tolist(),
        files={str(p):digest_file(p) for p in output.iterdir()},
        caffe_parity=False,int8_quantization=False,DPU_compilation=False,board_deployed=False)
    write_json(output/'export-manifest.json',manifest)
    return manifest
