"""Verified FP32 host export for a separate Caffe/DNNDK/ARM deployment stage."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.export_soundjie_siren_merge import fold_and_check
from scripts.host.soundjie_alert4bg import ROOT,LABELS,CONTRACT,CLASS_FILE,OUTPUT_ROOT,make_model,output_contract,metrics,apply_thresholds
from scripts.host.train_soundjie_alert6 import predict,softmax
from scripts.host.train_soundjie_alert4bg import checked_output,load_cache,write_json


def calibration_indices(rows,per_class=32):
    result=[];used=set()
    for c in range(5):
        chosen=[]
        for i,row in enumerate(rows):
            if row['purpose']=='train' and row['target_index']==c and row['derivation'] is None and row['sampling_identity'] not in used:
                chosen.append(i);used.add(row['sampling_identity'])
                if len(chosen)==per_class:break
        if len(chosen)!=per_class:raise ValueError('insufficient distinct train originals for calibration')
        result.extend(chosen)
    return result


def export():
    torch.set_num_threads(4);output=checked_output(OUTPUT_ROOT/'export')
    result=json.loads((OUTPUT_ROOT/'training/result.json').read_bytes());selection=result['artifacts']['selected']
    directory=Path(selection['directory']);checkpoint=directory/'model.pt'
    if digest_file(checkpoint)!=selection['checkpoint_sha256']:raise ValueError('checkpoint drift')
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ckpt['contract']!=CONTRACT or ckpt['labels']!=list(LABELS) or ckpt['class_map_sha256']!=digest_file(CLASS_FILE):raise ValueError('checkpoint contract drift')
    model=make_model();model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
    x,y,purpose,rows,audit=load_cache(result['features_directory'])
    if audit['cache_sha256']!=ckpt['cache_sha256']:raise ValueError('training feature cache drift')
    va=purpose=='validation';vx=x[va];vy=y[va];val_rows=[r for r in rows if r['purpose']=='validation']
    logits=predict(model,vx)
    with np.load(directory/'validation-logits.npz',allow_pickle=False) as recorded:
        if not np.array_equal(vy,recorded['y']) or not np.allclose(logits,recorded['logits'],rtol=0,atol=1e-6):raise ValueError('checkpoint reload parity failed')
    thresholds=selection['metrics']['thresholds']
    if metrics(vy,apply_thresholds(softmax(logits),thresholds),val_rows)!=selection['metrics']['thresholded']:
        raise ValueError('reloaded candidate metric drift')
    calibration=calibration_indices(rows);parity=[]
    for c in range(5):
        groups=set()
        for i,row in enumerate(rows):
            if row['purpose']=='validation' and row['target_index']==c and row['source_group'] not in groups:
                parity.append(i);groups.add(row['source_group'])
                if len(groups)==2:break
        if len(groups)!=2:raise ValueError('two distinct validation groups required per class')
    arrays,fold=fold_and_check(model,vx)  # Check all validation inputs, not only ten parity fixtures.
    output.mkdir(parents=True,exist_ok=False)
    with (output/'fused-weights.npz').open('xb') as f:np.savez(f,**arrays)
    for name,values in [('parity-inputs.npy',x[parity]),('pytorch-logits.npy',predict(model,x[parity])),('calibration-train.npy',x[calibration])]:
        with (output/name).open('xb') as f:np.save(f,values,allow_pickle=False)
    proto=(ROOT/'models/soundjie_d3/deploy.prototxt').read_text(encoding='utf-8')
    if proto.count('num_output: 8')!=1:raise ValueError('unexpected deploy architecture')
    with (output/'deploy.prototxt').open('x',encoding='utf-8',newline='\n') as f:f.write(proto.replace('num_output: 8','num_output: 5'))
    with (output/'labels.txt').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(LABELS)+'\n')
    write_json(output/'thresholds.json',dict(schema=2,contract=CONTRACT,labels=list(LABELS),background_index=4,
        alert_thresholds=thresholds,decision='softmax argmax; below winning alert threshold -> background; never promote runner-up',
        calibrated_on='same_source_disjoint_development_validation_used_for_selection',independent_acceptance=False,
        post_int8_recalibration_required=True,checkpoint_sha256=selection['checkpoint_sha256']))
    manifest=dict(schema=2,contract=CONTRACT,checkpoint_path=str(checkpoint),checkpoint_sha256=selection['checkpoint_sha256'],
        output_shape=[1,5],output_tensor_name='fc8',output_contract=output_contract(),
        class_map_sha256=digest_file(CLASS_FILE),frontend_sha256=audit['frontend_sha256'],
        cache_sha256=audit['cache_sha256'],source_manifest_sha256=audit['manifest_sha256'],
        folding=fold,checkpoint_reload_parity=True,validation_metrics=selection['metrics'],
        parity_sources=[rows[i] for i in parity],calibration_sources=[rows[i] for i in calibration],
        calibration_counts=np.bincount(y[calibration],minlength=5).tolist(),calibration_train_only=True,
        files={p.name:digest_file(p) for p in sorted(output.iterdir())},
        export_script_sha256=digest_file(Path(__file__)),fold_helper_sha256=digest_file(ROOT/'scripts/host/export_soundjie_siren_merge.py'),
        caffe_parity=False,int8_quantization=False,DPU_compilation=False,board_deployed=False,
        status='verified_fp32_export_ready_for_separate_DNNDK_ARM_stage')
    write_json(output/'export-manifest.json',manifest)
    print(json.dumps(dict(stage='export_verified',directory=str(output),folding=fold,
        checkpoint_sha256=selection['checkpoint_sha256'],export_manifest_sha256=digest_file(output/'export-manifest.json'))),flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args();export()


if __name__=='__main__':main()
