"""D4 replay candidates: reuse D3 CNN and validated frontend, isolate purposes."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scripts.host.augment_soundjie_d4 import augment_waveform, balanced_indices, candidate_gate
from scripts.host.build_soundjie_d4_manifest import partition_records, digest_file
from scripts.host.prepare_soundjie_d3 import decode_audio, resample_mono_16k, make_windows, WindowPolicy, Window, compute_feature
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.train_soundjie_d3 import (SoundjieCNN, load_model_config, audit_model_operators,
    evaluate_logits, _set_deterministic, save_checkpoint_create_only, _report_to_dict)

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)


def select_records(rows, max_per_class, seed):
    if type(max_per_class) is not int or max_per_class <= 0:
        raise ValueError('invalid source cap')
    selected = []
    for purpose in ('train', 'validation'):
        for target in range(8):
            pool = [r for r in rows if r['purpose'] == purpose and r['target_index'] == target]
            if not pool:
                raise ValueError('missing class in ' + purpose)
            pool.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['recording_id']}".encode()).hexdigest())
            selected.extend(pool[:max_per_class])
    return selected


def temporal_windows(samples, row, seed, offsets=(), min_relative_rms=.15):
    if row['purpose'] not in ('train', 'validation'):
        raise ValueError('training preparation accepts train/validation only')
    if not 0 < min_relative_rms <= 1 or any(type(v) is not int for v in offsets):
        raise ValueError('invalid temporal window policy')
    policy = WindowPolicy('energy_center' if row['target_index'] < 6 else 'seeded', seed)
    central = next(make_windows(samples, policy))
    windows = [central]
    if row['purpose'] != 'train' or row['target_index'] >= 6 or len(samples) <= 16000:
        return windows
    seen = {central.start_sample}
    threshold = max(1e-6, float(np.sqrt(np.mean(central.samples**2)))*min_relative_rms)
    for offset in offsets:
        start = max(0, min(len(samples)-16000, central.start_sample+offset))
        signal = np.asarray(samples[start:start+16000], dtype=np.float64)
        if start in seen:
            continue
        seen.add(start)
        if np.ptp(signal) <= 1e-7 or np.sqrt(np.mean(signal**2)) < threshold:
            continue
        windows.append(Window(start, signal.copy()))
    return windows


def prepare(manifest, config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = Path(manifest)
    audit = json.loads((manifest.parent / 'audit.json').read_text(encoding='utf-8'))
    if audit['manifest_sha256'] != digest_file(manifest):
        raise ValueError('input manifest hash mismatch')
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines()]
    partition_records(rows)
    chosen = select_records(rows, config['max_sources_per_class'], config['data_seed'])
    labels = (ROOT / 'models/soundjie_d3/labels.txt').read_text(encoding='utf-8').splitlines()
    frontend = load_frontend_config(ROOT / 'config/logmel_frontend_v1.json')
    windows = []
    for i, row in enumerate(chosen):
        if digest_file(row['path']) != row['sha256'] or labels[row['target_index']] != row['target_id']:
            raise ValueError('source hash/label drift: ' + row['recording_id'])
        samples = resample_mono_16k(decode_audio(Path(row['path'])))
        temporal = config.get('temporal_windows', {})
        windows.append(temporal_windows(samples, row, config['data_seed'] + i * 7919,
                                       temporal.get('offsets_samples', []), temporal.get('min_relative_rms', .15)))
    backgrounds = [ws[0].samples for r, ws in zip(chosen, windows) if r['purpose'] == 'train' and r['target_index'] == 7]
    features, targets, purposes, provenance = [], [], [], []
    expanded = [(row, w, i, j) for i, (row, ws) in enumerate(zip(chosen, windows)) for j, w in enumerate(ws)]
    for row, window, i, window_index in expanded:
        variants = [(window.samples, None)]
        if row['purpose'] == 'train':
            for j in range(config['augmentations_per_train_source']):
                variants.append(augment_waveform(window.samples, config['data_seed'] + i*101 + j + window_index*100000,
                                                 'train', backgrounds[(i+j) % len(backgrounds)]))
        for signal, recipe in variants:
            # Quantize before feature extraction to model the board's S16 capture path.
            signal = np.clip(np.rint(np.asarray(signal)*32768), -32768, 32767).astype('<i2')
            feature = compute_feature(signal.astype(np.float64)/32768., frontend)
            features.append(feature)
            targets.append(row['target_index'])
            purposes.append(row['purpose'])
            provenance.append(dict(recording_id=row['recording_id'], source_group=row['source_group'],
                parent_sha256=row['sha256'], purpose=row['purpose'], target_index=row['target_index'],
                domain='simulated_replay' if recipe else 'direct', derivation=recipe,
                start_sample_16k=window.start_sample, pcm_sha256=hashlib.sha256(signal.tobytes()).hexdigest(),
                feature_sha256=hashlib.sha256(feature.tobytes()).hexdigest()))
        if i % 50 == 0 and window_index == 0:
            print(f'PREPARED_SOURCES={i+1}/{len(chosen)}', flush=True)
    output.mkdir(parents=True, exist_ok=False)
    cache = output / 'features.npz'
    with cache.open('xb') as stream:
        np.savez_compressed(stream, x=np.stack(features), y=np.asarray(targets, dtype=np.int64),
                            purpose=np.asarray(purposes))
    write_json(output / 'provenance.json', provenance)
    report = dict(schema=1, cache_sha256=digest_file(cache), manifest_sha256=digest_file(manifest),
                  frontend_sha256=frontend.source_sha256, class_map_sha256=digest_file(ROOT/'config/soundjie_d3_classes.json'),
                  provenance_sha256=digest_file(output/'provenance.json'), config=config,
                  counts=dict(Counter(purposes)), calibration_status='pending_labelled_real_replay_audit')
    write_json(output/'audit.json', report)
    print(json.dumps(report), flush=True)


def predict(model, x, batch_size=64):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+batch_size])).numpy()
                               for i in range(0, len(x), batch_size)])


def train_candidate(train_x, train_y, val_x, val_y, config, seed, output, upstream):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for x, y in ((train_x, train_y), (val_x, val_y)):
        if x.shape != (len(y), 1, 64, 96) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError('invalid feature arrays')
        if set(y.tolist()) != set(range(8)):
            raise ValueError('eight classes required')
    torch.set_num_threads(config['threads'])
    _set_deterministic(seed)
    model_config = load_model_config(ROOT / 'config/soundjie_d3_model.json')
    model = SoundjieCNN(model_config)
    audit_model_operators(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    output.mkdir(parents=True, exist_ok=False)
    history, best, best_score = [], None, -1.
    for epoch in range(config['epochs']):
        model.train()
        indices = balanced_indices(train_y, config['per_class_epoch'], seed+epoch)
        loss_sum = 0.
        for start in range(0, len(indices), config['batch_size']):
            batch = indices[start:start+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(torch.from_numpy(train_x[batch])), torch.from_numpy(train_y[batch]))
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(batch)
        logits = predict(model, val_x)
        gate = candidate_gate(val_y, logits.argmax(1))
        row = dict(epoch=epoch+1, train_loss=loss_sum/len(indices), **gate)
        history.append(row)
        if gate['macro_recall'] > best_score:
            best_score, best = gate['macro_recall'], copy.deepcopy(model.state_dict())
        with (output/'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row)+'\n')
        print(f"seed={seed} epoch={epoch+1} macro_recall={gate['macro_recall']:.3f} coverage={gate['predicted_class_count']}", flush=True)
    if best is None:
        raise ValueError('no completed training epoch')
    model.load_state_dict(best)
    logits = predict(model, val_x)
    labels = (ROOT/'models/soundjie_d3/labels.txt').read_text(encoding='utf-8').splitlines()
    gate = candidate_gate(val_y, logits.argmax(1))
    metrics = dict(validation=_report_to_dict(evaluate_logits(logits, val_y, labels)), gate=gate,
                   calibration_status='pending_labelled_real_replay_audit', board_status='not_tested')
    checkpoint = output/'soundjie_d4_float32.pt'
    save_checkpoint_create_only(checkpoint, dict(schema=1, state_dict=best, labels=labels, seed=seed, upstream=upstream))
    write_json(output/'metrics.json', metrics)
    write_json(output/'training-history.json', history)
    write_json(output/'training-manifest.json', dict(schema=1, seed=seed, training=config, upstream=upstream,
               checkpoint=dict(path=checkpoint.name, sha256=digest_file(checkpoint)), metrics=metrics,
               torch_version=torch.__version__, numpy_version=np.__version__))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'train'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'config/soundjie_d4_training.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.mode == 'prepare':
        prepare(args.input, config, args.output)
        return
    audit = json.loads((args.input/'audit.json').read_text(encoding='utf-8'))
    if digest_file(args.input/'features.npz') != audit['cache_sha256']:
        raise ValueError('feature cache hash drift')
    if digest_file(args.input/'provenance.json') != audit['provenance_sha256']:
        raise ValueError('provenance hash drift')
    if audit['config'] != config:
        raise ValueError('preparation configuration drift')
    with np.load(args.input/'features.npz', allow_pickle=False) as data:
        x, y, purpose = data['x'], data['y'], data['purpose']
    if set(purpose.tolist()) != {'train', 'validation'}:
        raise ValueError('cache contains unauthorized purpose')
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in config['seeds']:
        metrics = train_candidate(x[purpose=='train'], y[purpose=='train'],
                                  x[purpose=='validation'], y[purpose=='validation'],
                                  config['training'], seed, args.output/str(seed), audit)
        results.append(dict(seed=seed, **metrics))
    write_json(args.output/'candidate-summary.json', results)


if __name__ == '__main__':
    main()
