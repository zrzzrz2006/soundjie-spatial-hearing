"""Read-only host diagnostics; never labels phone audio as board-captured PCM."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def sequential_windows(audio, hop=4000):
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or hop < 1:
        raise ValueError('invalid audio/window hop')
    if len(audio) < 16000:
        return [(0, np.pad(audio, (0, 16000-len(audio))))]
    starts = list(range(0, len(audio)-16000+1, hop))
    if starts[-1] != len(audio)-16000:
        starts.append(len(audio)-16000)
    return [(start, audio[start:start+16000].copy()) for start in starts]


def score_row(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (8,) or not np.isfinite(logits).all():
        raise ValueError('eight finite logits required')
    scores = np.exp(logits-logits.max())
    scores /= scores.sum()
    order = np.argsort(-scores, kind='stable')
    return dict(predicted_index=int(order[0]), scores=scores.tolist(),
                confidence=float(scores[order[0]]), top2_indices=order[:2].tolist(),
                top2_margin=float(scores[order[0]]-scores[order[1]]))


def reject_summary(scores, targets, minimum, margin):
    scores, targets = np.asarray(scores), np.asarray(targets)
    if scores.ndim != 2 or len(scores) != len(targets) or not len(targets):
        raise ValueError('invalid validation arrays')
    ordered = np.sort(scores, axis=1)
    accepted = (ordered[:, -1] >= minimum) & (ordered[:, -1]-ordered[:, -2] >= margin)
    correct = (scores.argmax(1) == targets) & accepted
    return dict(minimum_score=minimum, minimum_margin=margin, total=len(targets),
                accepted=int(accepted.sum()), correct_accepted=int(correct.sum()),
                coverage=float(accepted.mean()), correct_over_all=float(correct.mean()),
                accuracy_among_accepted=float(correct.sum()/accepted.sum()) if accepted.any() else None)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_board_session(path):
    base = Path(path).resolve()
    if (base/'timeline.jsonl').is_file():
        return base
    if (base/'inference/timeline.jsonl').is_file():
        resolved = (base/'inference').resolve()
        if resolved.parent == base:
            return resolved
    raise ValueError('session must contain timeline.jsonl or inference/timeline.jsonl')


def main():
    import torch
    from scripts.host.train_soundjie_d3 import SoundjieCNN, load_model_config
    from scripts.host.train_soundjie_d4 import predict, ROOT
    from scripts.host.prepare_soundjie_d3 import decode_audio, resample_mono_16k
    from scripts.host.logmel_reference import load_frontend_config, compute_logmel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path)
    parser.add_argument('--board-session', type=Path)
    parser.add_argument('--validation-features', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.input_manifest and not args.board_session:
        parser.error('input-manifest or board-session required')
    manifest = json.loads((args.candidate/'training-manifest.json').read_text(encoding='utf-8'))
    checkpoint_path = args.candidate/manifest['checkpoint']['path']
    if sha(checkpoint_path) != manifest['checkpoint']['sha256']:
        raise ValueError('checkpoint hash mismatch')
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    labels = (ROOT/'models/soundjie_d3/labels.txt').read_text().splitlines()
    if ck['labels'] != labels or ck['upstream'] != manifest['upstream']:
        raise ValueError('checkpoint provenance mismatch')
    torch.set_num_threads(2)
    model = SoundjieCNN(load_model_config(ROOT/'config/soundjie_d3_model.json'))
    model.load_state_dict(ck['state_dict'])
    frontend = load_frontend_config(ROOT/'config/logmel_frontend_v1.json')
    inputs = []
    if args.input_manifest:
        inputs = json.loads(args.input_manifest.read_text(encoding='utf-8'))
    if args.board_session:
        base = resolve_board_session(args.board_session)
        for row in map(json.loads, (base/'timeline.jsonl').read_text(encoding='utf-8').splitlines()):
            if not row.get('capture_file'):
                continue
            path = (base/row['capture_file']).resolve()
            if path.parent != base or path.suffix != '.s16' or path.stat().st_size != 32000:
                raise ValueError('invalid captured window path/size')
            inputs.append(dict(path=str(path), sha256=sha(path), kind='retained_board_window',
                               sequence=row['sequence'], board_scores=row.get('class_scores', []),
                               expected_index=None, window_end_sample=row.get('window_end_sample')))
    results = []
    for row in inputs:
        path = Path(row['path'])
        if sha(path) != row['sha256']:
            raise ValueError('audio hash drift: ' + str(path))
        if path.suffix == '.s16':
            audio = np.frombuffer(path.read_bytes(), dtype='<i2').astype(np.float64)/32768
        else:
            audio = resample_mono_16k(decode_audio(path))
        windows, features = [], []
        for start, signal in sequential_windows(audio):
            pcm = np.clip(np.rint(signal*32768), -32768, 32767).astype('<i2')
            feature = compute_logmel(pcm.astype(np.float64)/32768, frontend).astype('<f4').reshape(1,64,96)
            features.append(feature)
            windows.append(dict(start_sample=start, pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                                feature_sha256=hashlib.sha256(feature.tobytes()).hexdigest()))
        logits = predict(model, np.stack(features))
        for window, output in zip(windows, logits):
            window.update(score_row(output))
        record = dict(source=row, duration_seconds=len(audio)/16000, windows=windows,
                      prediction_counts=dict(Counter(str(w['predicted_index']) for w in windows)))
        if row.get('board_scores'):
            scores = np.asarray(row['board_scores'], dtype=float)
            if scores.shape != (8,) or not np.isfinite(scores).all() or np.any(scores < 0) or not np.isclose(scores.sum(), 1):
                raise ValueError('invalid board scores')
            record['float_board_top1_equal'] = int(scores.argmax()) == windows[0]['predicted_index']
            record['max_probability_difference'] = float(np.abs(scores-windows[0]['scores']).max())
        results.append(record)
    thresholds = []
    if args.validation_features:
        folder = args.validation_features
        audit = json.loads((folder/'audit.json').read_text(encoding='utf-8'))
        if sha(folder/'features.npz') != audit['cache_sha256']:
            raise ValueError('validation cache hash mismatch')
        with np.load(folder/'features.npz', allow_pickle=False) as cache:
            val = cache['purpose'] == 'validation'
            scores = np.array([score_row(x)['scores'] for x in predict(model, cache['x'][val])])
            for minimum in (0., .5, .6, .7, .8, .9):
                for margin in (0., .1, .2, .3):
                    thresholds.append(reject_summary(scores, cache['y'][val], minimum, margin))
    report = dict(schema=1, checkpoint_sha256=sha(checkpoint_path), frontend_sha256=frontend.source_sha256,
                  independent_acceptance=False, new_board_inference=False, model_changed=False,
                  note='Previously exposed diagnostics; no window ground-truth inferred from whole recording label. '
                       'Softmax is not calibrated accuracy. Threshold grid is validation only and NOT deployed.',
                  records=results, validation_rejection_grid=thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(records=len(results), windows=sum(len(r['windows']) for r in results),
                         report=str(args.output))))


if __name__ == '__main__':
    main()
