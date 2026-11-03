"""Prepare hash-bound D4 weights and validation-only Caffe parity inputs.

This does NOT certify Caffe parity, INT8 quantization, or board inference.
Run as python -m scripts.host.export_soundjie_d4.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.guest import export_soundjie_d3_caffe as legacy

ROOT = Path(__file__).resolve().parents[2]


def check_hash(path, expected):
    if legacy.sha256_file(path) != expected:
        raise ValueError('SHA drift: ' + str(path))


def select_parity(x, y, purpose, provenance):
    n = len(provenance)
    if (x.shape != (n, 1, 64, 96) or x.dtype != np.float32 or
            y.shape != (n,) or y.dtype != np.int64 or purpose.shape != (n,) or
            not np.isfinite(x).all() or np.any((y < 0) | (y > 7))):
        raise ValueError('feature tensor contract drift')
    if set(purpose.tolist()) != {'train', 'validation'}:
        raise ValueError('unauthorized purpose')
    seen = {key: {} for key in ('source_group', 'parent_sha256', 'recording_id')}
    for i, row in enumerate(provenance):
        if (row['purpose'] != purpose[i] or row['target_index'] != int(y[i]) or
                row['feature_sha256'] != hashlib.sha256(x[i].tobytes()).hexdigest()):
            raise ValueError('provenance drift')
        for key, mapping in seen.items():
            identity = row[key]
            if not identity or mapping.get(identity, purpose[i]) != purpose[i]:
                raise ValueError('source leakage')
            mapping[identity] = purpose[i]
    selected = []
    used_groups = set()
    for target in range(8):
        candidates = []
        for i in np.flatnonzero((y == target) & (purpose == 'validation')):
            group = provenance[int(i)]['source_group']
            if group not in used_groups:
                candidates.append(int(i))
                used_groups.add(group)
                if len(candidates) == 2:
                    break
        if len(candidates) != 2:
            raise ValueError('two distinct validation sources required per class')
        selected.extend(candidates)
    return selected


def prepare(candidate, features, output):
    import torch
    from scripts.host import train_soundjie_d3 as training

    candidate, features, output = map(Path, (candidate, features, output))
    if output.exists():
        raise ValueError('output directory exists')
    manifest_path = candidate / 'training-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    checkpoint_path = candidate / 'soundjie_d4_float32.pt'
    check_hash(checkpoint_path, manifest['checkpoint']['sha256'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    labels_path = ROOT / 'models/soundjie_d3/labels.txt'
    legacy.validate_checkpoint_contract(checkpoint, manifest,
        labels_path.read_text(encoding='utf-8').splitlines())
    upstream = checkpoint['upstream']
    audit = json.loads((features / 'audit.json').read_text(encoding='utf-8'))
    if upstream != audit:
        raise ValueError('feature audit differs from training')
    for filename, key in [('features.npz', 'cache_sha256'),
                          ('provenance.json', 'provenance_sha256')]:
        check_hash(features / filename, upstream[key])
    config_path = ROOT / 'config/soundjie_d3_model.json'
    config = training.load_model_config(config_path)
    if (config.frontend_sha256 != upstream['frontend_sha256'] or
            config.class_map_sha256 != upstream['class_map_sha256']):
        raise ValueError('frontend/class map contract drift')
    with np.load(features / 'features.npz', allow_pickle=False) as data:
        x, y, purpose = data['x'], data['y'], data['purpose']
    provenance = json.loads((features / 'provenance.json').read_text(encoding='utf-8'))
    indices = select_parity(x, y, purpose, provenance)
    arrays = legacy.fuse_state_dict(checkpoint['state_dict'])
    torch.set_num_threads(4)
    model = training.SoundjieCNN(config)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x[indices].copy())).numpy()
    if logits.shape != (16, 8) or not np.isfinite(logits).all():
        raise ValueError('invalid reference logits')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'fused-weights.npz').open('xb') as stream:
        np.savez(stream, **arrays)
    legacy._save_npy(output / 'parity-inputs.npy', x[indices])
    legacy._save_npy(output / 'pytorch-logits.npy', logits)
    deploy = ROOT / 'models/soundjie_d3/deploy.prototxt'
    for source, name in [(deploy, 'deploy.prototxt'),
                         (ROOT / 'scripts/guest/export_soundjie_d3_caffe.py', 'export_soundjie_d3_caffe.py')]:
        with (output / name).open('xb') as stream:
            stream.write(source.read_bytes())
    report = dict(schema=1, status='host_export_only', caffe_parity='not_run',
        quantization='not_run', board_inference='not_run', independent_acceptance=False,
        checkpoint_sha256=legacy.sha256_file(checkpoint_path),
        training_manifest_sha256=legacy.sha256_file(manifest_path),
        model_config_sha256=legacy.sha256_file(config_path),
        labels_sha256=legacy.sha256_file(labels_path), upstream=upstream,
        parity_purpose='validation', parity_indices=indices,
        parity_sources=[provenance[i] for i in indices],
        artifacts={p.name: legacy.sha256_file(p) for p in sorted(output.iterdir())})
    legacy._write_json(output / 'fused-manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.candidate, args.features, args.output)
    print(json.dumps(dict(status=report['status'], output=str(args.output),
                         checkpoint_sha256=report['checkpoint_sha256'])))


if __name__ == '__main__':
    main()
