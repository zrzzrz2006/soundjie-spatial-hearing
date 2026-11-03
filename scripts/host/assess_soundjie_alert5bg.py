"""Publish validation-calibrated recognition thresholds and pinned INT8 handoff.

Ordinary speech is a non-alert foreground class. Keep all six raw probabilities;
this report distinguishes background-to-alert from background-to-speech mistakes.
"""
import json
from pathlib import Path
import numpy as np
from scripts.host.soundjie_alert5bg import CONTRACT,LABELS,OUTPUT_ROOT,apply_thresholds,calibrate_thresholds,metrics
from scripts.host.deploy_soundjie_alert5bg import (
    staged_inputs,check_compiled,check_files,require_hash,probabilities,read,write,sha)


def operating_metrics(y,p,rows,thresholds):
    pred=apply_thresholds(p,thresholds);result=metrics(y,pred,rows);bg=y==5
    result['background_false_foreground_count']=int(np.sum(pred[bg]!=5))
    result['background_false_foreground_rate']=float(np.mean(pred[bg]!=5))
    result['background_false_alert_count']=int(np.sum(pred[bg]<4))
    result['background_false_alert_rate']=float(np.mean(pred[bg]<4))
    result['background_false_speech_count']=int(np.sum(pred[bg]==4))
    result['background_false_speech_rate']=float(np.mean(pred[bg]==4))
    result['speech_is_non_alert']=True
    result['legacy_negative_false_alert_rate_means_any_foreground']=True
    result['speech_subgroups']={}
    for subset in ('explicit_speech','inherited_vocal_weak_label'):
        m=np.array([r.get('speech_evidence')==subset for r in rows])
        n=int(m.sum());correct=int(np.sum(pred[m]==4))
        result['speech_subgroups'][subset]=dict(support=n,correct=correct,recall=correct/n if n else None)
    return result


def recognition_thresholds(y,p,config):
    # A 100% bound makes F1 primary, followed by lower background error and lower
    # threshold. This is an explicit alternate operating policy, not a relabel.
    cfg=dict(config,per_class_background_fpr_limit=1.0)
    exact=calibrate_thresholds(y,p,cfg)
    rounded=[round(t,3) for t in exact]
    if not np.array_equal(apply_thresholds(p,exact),apply_thresholds(p,rounded)):
        raise ValueError('readable threshold rounding changes validation decisions')
    return rounded


def publish(run):
    run=Path(run);root=run/'input';build=run/'container-output/build';fixed=run/'container-output/fixed'
    input_meta,export=staged_inputs(root);compiled=check_compiled(root,build)
    fm=read(fixed/'fixed-evaluation.json')
    if fm['status']!='fixed_evaluation_complete' or fm['learned_parameters_unchanged'] is not True:
        raise ValueError('incomplete FixedNeuron evidence')
    require_hash(root/'input-manifest.json',fm['input_manifest_sha256'])
    require_hash(build/'compiled-manifest.json',fm['compiled_manifest_sha256'])
    check_files(fixed,fm['artifacts'],{'validation-fixed.npy','parity-fixed.npy','fixed-evaluation.prototxt'})
    y=np.load(root/'validation-targets.npy',allow_pickle=False);rows=read(root/'validation-sources.json')
    logits=np.load(fixed/'validation-fixed.npy',allow_pickle=False);p=probabilities(logits)
    cfg=read(root/'training-config.json')['thresholds'];recommended=recognition_thresholds(y,p,cfg)
    strict=calibrate_thresholds(y,p,cfg)
    candidates={
        'recognition_f1_recommended':recommended,
        'strict_per_class_bg_2pct_coarse_grid':strict,
        'uniform_0p5':[.5]*5,
        'uniform_0p6':[.6]*5,
        'uniform_0p7':[.7]*5,
        'uniform_0p8':[.8]*5,
    }
    profiles={name:dict(thresholds=t,metrics=operating_metrics(y,p,rows,t)) for name,t in candidates.items()}
    scores=[]
    pred=apply_thresholds(p,recommended)
    for i,row in enumerate(rows):
        scores.append(dict(recording_id=row['recording_id'],purpose=row['purpose'],source_group=row['source_group'],
            sampling_identity=row['sampling_identity'],true_index=int(y[i]),true_label=LABELS[y[i]],
            negative_subtype=row['negative_subtype'],speech_evidence=row.get('speech_evidence'),
            logits=logits[i].tolist(),probabilities=p[i].tolist(),raw_prediction=int(p[i].argmax()),
            recommended_prediction=int(pred[i]),feature_sha256=row['feature_sha256']))
    write(run/'validation-scores-fixed.json',dict(contract=CONTRACT,labels=list(LABELS),
        numerical_semantics='double softmax, matching classifier C++ exp/sum precision',
        logits_sha256=sha(fixed/'validation-fixed.npy'),rows=scores))
    assessment=dict(schema=3,contract=CONTRACT,labels=list(LABELS),recommended_profile='recognition_f1_recommended',
        speech_is_non_alert=True,alert_indices=[0,1,2,3],speech_index=4,background_index=5,
        input_shape=['N',1,64,96],output_node='fc8',kernel='soundjie_alert5bg',output_count=6,
        elf_path=str(build/'compiled/dpu_soundjie_alert5bg.elf'),elf_sha256=compiled['elf_sha256'],
        checkpoint_sha256=export['checkpoint_sha256'],input_manifest_sha256=sha(root/'input-manifest.json'),
        compiled_manifest_sha256=sha(build/'compiled-manifest.json'),fixed_manifest_sha256=sha(fixed/'fixed-evaluation.json'),
        scores_path=str(run/'validation-scores-fixed.json'),scores_sha256=sha(run/'validation-scores-fixed.json'),
        profiles=profiles,raw=operating_metrics(y,p,rows,[0.]*5),caffe_parity=compiled['caffe_parity'],
        calibration_samples=192,calibration_train_only=True,
        threshold_selection='per-class validation F1 on fixed 0.025 grid, then lower background error then lower threshold; no 2% restriction for recognition profile',
        threshold_policy_authorization='User prioritizes usable demonstration recognition, rejects almost-zero detections, and identifies speech as non-alert',
        training_validation_source_disjoint=True,selection_and_thresholds_reuse_validation=True,
        independent_acceptance=False,board_tested=False,d5_accessed=False,live_replay_used=False,
        implementation_sha256=sha(Path(__file__)))
    write(run/'assessment-recognition.json',assessment)
    handoff=dict(schema=3,contract=CONTRACT,labels=list(LABELS),physical_output_count=6,logical_output_count=6,
        speech_index=4,background_index=5,alert_indices=[0,1,2,3],speech_is_non_alert=True,
        foreground_thresholds=recommended,alert_thresholds=recommended,
        alert_thresholds_field_is_legacy_cli_name=True,
        runtime_argument='--alert-thresholds '+','.join(format(t,'.3f') for t in recommended),
        decision='double softmax argmax; reject below the winning foreground threshold to background; no runner-up promotion; speech is non-alert',
        preserve_all_six_raw_outputs=True,profile='recognition_f1_recommended',
        metrics=profiles['recognition_f1_recommended']['metrics'],
        elf_path=assessment['elf_path'],elf_sha256=assessment['elf_sha256'],kernel='soundjie_alert5bg',
        input_node='conv1',output_node='fc8',checkpoint_sha256=export['checkpoint_sha256'],
        assessment_path=str(run/'assessment-recognition.json'),assessment_sha256=sha(run/'assessment-recognition.json'),
        scores_path=assessment['scores_path'],scores_sha256=assessment['scores_sha256'],
        board_tested=False,independent_acceptance=False)
    write(run/'thresholds-fixed.json',handoff)
    return handoff


if __name__=='__main__':
    result=publish(OUTPUT_ROOT/'compile-20261101a')
    print(json.dumps(result,ensure_ascii=False),flush=True)
