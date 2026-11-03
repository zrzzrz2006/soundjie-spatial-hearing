"""Offline diagnostic on hash-pinned historical board recordings; not a new board run."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np
import torch
from scripts.host.train_soundjie_d4 import ROOT, predict, write_json
from scripts.host.train_soundjie_d3 import SoundjieCNN, load_model_config
from scripts.host.build_soundjie_d4_manifest import digest_file

ARCHIVE_SHA = 'a7869f93e2d432c71eceb511278196b3b703fd4f74f928b17f15c152d84fdd68'
CASES = [('fire', 0), ('glass', 3), ('door', 4), ('distress', 5)]


def evaluate(checkpoint_path, training_manifest, archive_path):
    manifest = json.loads(Path(training_manifest).read_text(encoding='utf-8'))
    if digest_file(checkpoint_path) != manifest['checkpoint']['sha256']:
        raise ValueError('checkpoint hash mismatch')
    if digest_file(archive_path) != ARCHIVE_SHA:
        raise ValueError('historical board archive hash mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    labels = (ROOT/'models/soundjie_d3/labels.txt').read_text(encoding='utf-8').splitlines()
    if checkpoint['labels'] != labels or checkpoint['upstream'] != manifest['upstream']:
        raise ValueError('checkpoint contract mismatch')
    model = SoundjieCNN(load_model_config(ROOT/'config/soundjie_d3_model.json'))
    model.load_state_dict(checkpoint['state_dict'])
    torch.set_num_threads(2)
    reports = []
    with tarfile.open(archive_path, 'r:gz') as archive:
        for name, target in CASES:
            for index in range(2):
                prefix = f'sjd3l1-live-{name}-{index:02d}'
                session = json.load(archive.extractfile(prefix+'/session-result.json'))
                if session['expected_class'] != labels[target] or session['LIVE_STATUS'] != 'pass':
                    raise ValueError('historical label/session mismatch')
                tensors, hashes = [], []
                for rank in (1, 2, 3):
                    raw = archive.extractfile(prefix+f'/window-{rank}/logmel.f32le').read()
                    tensor = np.frombuffer(raw, dtype='<f4').reshape(1, 64, 96).copy()
                    if not np.isfinite(tensor).all():
                        raise ValueError('nonfinite historical feature')
                    tensors.append(tensor)
                    hashes.append(hashlib.sha256(raw).hexdigest())
                logits = predict(model, np.stack(tensors))
                predicted = int(logits.mean(axis=0).argmax())
                reports.append(dict(case=prefix, expected_index=target, predicted_index=predicted,
                                    predicted_label=labels[predicted], matched=predicted == target,
                                    window_predictions=logits.argmax(1).tolist(),
                                    logits=logits.tolist(), feature_sha256=hashes))
    return dict(schema=1, mode='host_float32_on_historical_board_features',
                checkpoint_sha256=digest_file(checkpoint_path), historical_archive_sha256=ARCHIVE_SHA,
                matched=sum(r['matched'] for r in reports), total=len(reports), cases=reports,
                new_board_inference=False, independent_acceptance=False,
                note='Diagnostic only. Existing recordings previously examined in D3; no new DPU calls.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-manifest', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = evaluate(args.checkpoint, args.training_manifest, args.archive)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
