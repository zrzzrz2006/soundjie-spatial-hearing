"""Deterministic waveform replay simulation and balanced candidate gates."""
import hashlib
import json
import numpy as np
from scipy.signal import butter, resample_poly, sosfilt


def zero_shift(values, shift):
    out = np.zeros_like(values)
    if shift == 0:
        return values.copy()
    if abs(shift) < len(values):
        if shift > 0:
            out[shift:] = values[:-shift]
        else:
            out[:shift] = values[-shift:]
    return out


def augment_waveform(samples, seed, purpose, background=None):
    if purpose != 'train':
        raise ValueError('waveform augmentation is train-only')
    x = np.asarray(samples, dtype=np.float64)
    if x.shape != (16000,) or not np.isfinite(x).all():
        raise ValueError('expected finite 16000-sample waveform')
    rng = np.random.default_rng(seed)
    recipe = dict(schema=1, seed=int(seed), gain_db=float(rng.uniform(-18, 3)),
                  highpass_hz=float(rng.uniform(40, 160)), lowpass_hz=float(rng.uniform(3000, 7400)),
                  shift_samples=int(rng.integers(-800, 801)), resample_numerator=int(rng.integers(995, 1006)),
                  resample_denominator=1000, compressor_drive=float(rng.uniform(1, 2.5)),
                  clip_level=float(rng.uniform(.8, 1)), taps=[])
    filtered = sosfilt(butter(2, [recipe['highpass_hz'], recipe['lowpass_hz']],
                              btype='bandpass', fs=16000, output='sos'), x)
    wet = filtered.copy()
    for delay in (160, 480, 1120):
        gain = float(rng.uniform(.03, .22))
        recipe['taps'].append([delay, gain])
        wet += gain * zero_shift(filtered, delay)
    wet = resample_poly(wet, recipe['resample_numerator'], 1000)
    wet = np.pad(wet, (0, max(0, 16000-len(wet))))[:16000]
    wet = zero_shift(wet, recipe['shift_samples']) * 10**(recipe['gain_db']/20)
    if background is not None:
        noise = np.asarray(background, dtype=np.float64)
        if noise.shape != x.shape or not np.isfinite(noise).all():
            raise ValueError('invalid background waveform')
        recipe['background_pcm_sha256'] = hashlib.sha256(noise.astype('<f8').tobytes()).hexdigest()
        recipe['snr_db'] = float(rng.uniform(12, 30))
        scale = np.sqrt(np.mean(wet**2)) / max(np.sqrt(np.mean(noise**2)), 1e-12)
        wet += noise * scale * 10**(-recipe['snr_db']/20)
    drive = recipe['compressor_drive']
    wet = np.tanh(drive * wet) / drive
    wet = np.clip(wet, -recipe['clip_level'], recipe['clip_level'])
    recipe['recipe_sha256'] = hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return wet, recipe


def balanced_indices(targets, per_class, seed):
    y = np.asarray(targets)
    if y.ndim != 1 or not np.issubdtype(y.dtype, np.integer) or set(y.tolist()) != set(range(8)):
        raise ValueError('all eight class IDs must be represented')
    if type(per_class) is not int or per_class <= 0:
        raise ValueError('per_class must be positive')
    rng = np.random.default_rng(seed)
    chunks = [rng.choice(np.flatnonzero(y == c), per_class,
                         replace=np.count_nonzero(y == c) < per_class) for c in range(8)]
    return rng.permutation(np.concatenate(chunks))


def candidate_gate(targets, predictions):
    y, p = np.asarray(targets), np.asarray(predictions)
    if y.ndim != 1 or y.shape != p.shape or not len(y):
        raise ValueError('invalid evaluation shape')
    if not np.issubdtype(y.dtype, np.integer) or not np.issubdtype(p.dtype, np.integer):
        raise ValueError('class IDs must be integers')
    if ((y < 0) | (y > 7) | (p < 0) | (p > 7)).any():
        raise ValueError('class IDs out of range')
    recalls = [float((p[y == c] == c).mean()) if np.any(y == c) else 0. for c in range(8)]
    support = np.bincount(y, minlength=8).tolist()
    coverage = len(np.unique(p))
    focus = min(recalls[c] for c in (0, 3, 4, 5))
    # Balanced macro prediction rate avoids a large speech subset dominating this gate.
    background_rate = float(np.mean([np.mean(p[y == c] == 7) if np.any(y == c) else 1. for c in range(8)]))
    return dict(passed=bool(min(support) > 0 and coverage == 8 and focus >= .4 and background_rate <= .35),
                support=support, per_class_recall=recalls, macro_recall=float(np.mean(recalls)),
                predicted_class_count=coverage, focus_min_recall=focus,
                balanced_background_prediction_rate=background_rate)
