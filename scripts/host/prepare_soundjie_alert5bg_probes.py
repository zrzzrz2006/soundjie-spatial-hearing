"""Recover six exact validation PCM windows for a labeled board sanity probe.

This tool never accesses the board. These deliberately easy validation examples
are calibration/integration smoke probes, not unseen acoustic acceptance.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scripts.host.acquire_soundjie_alert6 import read_rows
from scripts.host.soundjie_alert5bg import ROOT,OUTPUT_ROOT,DATA_ROOT,LABELS,CONTRACT,CLASS_FILE,make_model,apply_thresholds
from scripts.host.train_soundjie_alert5bg import source_windows,write_json
from scripts.host.train_soundjie_alert6 import predict
from scripts.host.prepare_soundjie_d3 import decode_audio,resample_mono_16k,compute_feature
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.deploy_soundjie_alert5bg import staged_inputs,check_compiled,check_files,require_hash,sha,read,probabilities

PROBE_ROOT=Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_two_clips/20261101a/model-probes')


def select_indices(rows,scores,thresholds):
    if any(r['purpose']!='validation' or r['derivation'] is not None for r in rows):
        raise ValueError('only original validation windows may become probes')
    if scores.shape!=(len(rows),6):raise ValueError('six scores required per source')
    pred=apply_thresholds(scores,thresholds);selected=[]
    for c in range(6):
        candidates=[i for i,r in enumerate(rows) if r['target_index']==c and pred[i]==c
                    and (c!=4 or r.get('speech_evidence')=='explicit_speech')]
        if not candidates:raise ValueError('no correctly recognized probe for '+LABELS[c])
        candidates.sort(key=lambda i:(-float(scores[i,c]),rows[i]['recording_id']))
        selected.append(candidates[0])
    if len(set(selected))!=6:raise ValueError('probe sources repeat')
    return selected


def prepare():
    if PROBE_ROOT.exists():raise FileExistsError(PROBE_ROOT)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    compiled_root=OUTPUT_ROOT/'compile-20261101a';staged=compiled_root/'input'
    _,export=staged_inputs(staged);compiled=check_compiled(staged,compiled_root/'container-output/build')
    handoff=read(compiled_root/'thresholds-fixed.json')
    if handoff['elf_sha256']!=compiled['elf_sha256']:raise ValueError('handoff ELF differs')
    require_hash(Path(handoff['assessment_path']),handoff['assessment_sha256'])
    fixed_root=compiled_root/'container-output/fixed';fixed_meta=read(fixed_root/'fixed-evaluation.json')
    require_hash(staged/'input-manifest.json',fixed_meta['input_manifest_sha256'])
    require_hash(compiled_root/'container-output/build/compiled-manifest.json',fixed_meta['compiled_manifest_sha256'])
    check_files(fixed_root,fixed_meta['artifacts'],{'validation-fixed.npy'})
    rows=read(staged/'validation-sources.json');cached=np.load(staged/'validation-inputs.npy',allow_pickle=False)
    fixed_logits=np.load(fixed_root/'validation-fixed.npy',allow_pickle=False);scores=probabilities(fixed_logits)
    ix=select_indices(rows,scores,handoff['foreground_thresholds'])
    source_list=read_rows(DATA_ROOT/'manifest.jsonl');sources={r['recording_id']:(i,r) for i,r in enumerate(source_list)}
    cfg=read(staged/'training-config.json');frontend=load_frontend_config(ROOT/'config/logmel_frontend_v1.json')
    require_hash(ROOT/'config/logmel_frontend_v1.json',export['frontend_sha256'])
    require_hash(DATA_ROOT/'manifest.jsonl',export['source_manifest_sha256'])
    require_hash(CLASS_FILE,export['class_map_sha256'])
    checkpoint=Path(export['checkpoint_path']);require_hash(checkpoint,export['checkpoint_sha256'])
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ckpt['contract']!=CONTRACT or ckpt['labels']!=list(LABELS):raise ValueError('checkpoint contract differs')
    model=make_model();model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
    records=[];pcm_values=[];features=[]
    for c,i in enumerate(ix):
        row=rows[i];order,source=sources[row['recording_id']]
        if source['purpose']!='validation' or source['target_index']!=c:raise ValueError('source purpose/target differs')
        require_hash(Path(source['path']),source['sha256'])
        audio=resample_mono_16k(decode_audio(Path(source['path'])))
        full_pcm=np.clip(np.rint(audio*32768),-32768,32767).astype('<i2')
        if hashlib.sha256(full_pcm.tobytes()).hexdigest()!=row['pcm_sha256_16k']:raise ValueError('decoded source PCM drift')
        windows=source_windows(audio,source,cfg['data_seed']+order*7919,cfg['min_relative_event_rms'])
        if len(windows)!=1 or windows[0].start_sample!=row['start_sample_16k']:raise ValueError('validation window offset differs')
        pcm=np.clip(np.rint(windows[0].samples*32768),-32768,32767).astype('<i2')
        if pcm.shape!=(16000,) or pcm.nbytes!=32000:raise ValueError('probe must be exactly one second mono S16LE')
        feature=compute_feature(pcm.astype(np.float64)/32768.,frontend)
        if (not np.array_equal(feature,cached[i])
                or hashlib.sha256(feature.tobytes()).hexdigest()!=row['feature_sha256']):
            raise ValueError('recovered PCM frontend is not bit-exact to the original validation feature')
        pcm_values.append(pcm);features.append(feature)
        records.append(dict(class_index=c,label=LABELS[c],pcm_file=LABELS[c]+'.s16le',
            pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),pcm_bytes=32000,samples=16000,sample_rate=16000,channels=1,format='signed_16_bit_little_endian',
            recording_id=source['recording_id'],source_id=source['source_id'],source_group=source['source_group'],
            sampling_identity=source['sampling_identity'],uploader_group=source.get('uploader_group'),
            source_audio_path=source['path'],source_audio_sha256=source['sha256'],original_labels=source['original_labels'],
            original_url=source.get('original_url'),license=source.get('license'),
            source_purpose='validation',derivation=None,negative_subtype=source['negative_subtype'],speech_evidence=source.get('speech_evidence'),
            validation_row_index=i,start_sample_16k=row['start_sample_16k'],start_seconds=row['start_sample_16k']/16000.,
            source_length_samples_16k=len(audio),source_decoded_pcm_sha256=row['pcm_sha256_16k'],
            frontend_feature_file=LABELS[c]+'-feature.npy',frontend_feature_sha256=row['feature_sha256'],feature_bit_exact_to_training_validation_cache=True,
            expected_fixed_neuron_logits=fixed_logits[i].tolist(),expected_fixed_neuron_scores=scores[i].tolist(),
            expected_fixed_neuron_raw_id=int(scores[i].argmax()),expected_thresholded_id=int(apply_thresholds(scores[i:i+1],handoff['foreground_thresholds'])[0]),
            expected_fixed_is_host_simulation_not_board_measurement=True))
    z=predict(model,np.stack(features));p=probabilities(z)
    stored=np.load(staged/'validation-float.npy',allow_pickle=False)[ix]
    err=float(np.max(np.abs(z-stored)))
    if err>1e-4 or not np.array_equal(z.argmax(1),stored.argmax(1)):raise ValueError('recomputed selected-model logits differ')
    if not np.array_equal(z.argmax(1),np.arange(6)):raise ValueError('FP32 smoke inputs are not all correctly classified')
    for i,row in enumerate(records):
        row.update(expected_recomputed_fp32_logits=z[i].tolist(),expected_recomputed_fp32_scores=p[i].tolist(),
                   expected_fp32_raw_id=int(z[i].argmax()))
    # Only publish after all source, waveform, feature and model checks succeed.
    PROBE_ROOT.mkdir(parents=True,exist_ok=False)
    for row,pcm,feature in zip(records,pcm_values,features):
        with (PROBE_ROOT/row['pcm_file']).open('xb') as f:f.write(pcm.tobytes())
        with (PROBE_ROOT/row['frontend_feature_file']).open('xb') as f:np.save(f,feature,allow_pickle=False)
        row['frontend_feature_npy_sha256']=sha(PROBE_ROOT/row['frontend_feature_file'])
    manifest=dict(schema=3,contract=CONTRACT,labels=list(LABELS),status='six_exact_validation_pcm_smoke_probes',
        selection='highest correctly recognized Caffe FixedNeuron confidence per class; speech requires explicit Speech/LibriSpeech evidence',
        source_split='validation',selected_for_easy_smoke_testing=True,independent_acceptance=False,used_for_new_training=False,
        board_accessed=False,board_measured=False,all_features_bit_exact=True,fp32_recomputed_from_written_pcm=True,
        recomputed_fp32_vs_stored_max_abs_error=err,frontend_sha256=frontend.source_sha256,
        checkpoint_path=str(checkpoint),checkpoint_sha256=export['checkpoint_sha256'],elf_path=handoff['elf_path'],elf_sha256=handoff['elf_sha256'],
        kernel='soundjie_alert5bg',input_node='conv1',output_node='fc8',output_count=6,foreground_thresholds=handoff['foreground_thresholds'],speech_is_non_alert=True,
        source_manifest_sha256=export['source_manifest_sha256'],input_manifest_sha256=sha(staged/'input-manifest.json'),
        fixed_logits_sha256=sha(fixed_root/'validation-fixed.npy'),fixed_evaluation_sha256=sha(fixed_root/'fixed-evaluation.json'),
        script_sha256=sha(Path(__file__)),records=records)
    write_json(PROBE_ROOT/'manifest.json',manifest)
    print(json.dumps(dict(status=manifest['status'],directory=str(PROBE_ROOT),manifest_sha256=sha(PROBE_ROOT/'manifest.json'),
        recomputed_fp32_max_abs_error=err,probes=[dict(label=r['label'],recording_id=r['recording_id'],confidence=r['expected_fixed_neuron_scores'][r['class_index']],sha256=r['pcm_sha256']) for r in records]),indent=2),flush=True)


if __name__=='__main__':prepare()
