"""Recover a hash-verified diagnostic WAV from serial and report tonal peaks.

Diagnostic only: no classifier score or calibrated sound-pressure estimate.
"""
import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.log.read_bytes().replace(b'\r', b'')
    payload = raw.split(b'\nSJ_TONE_BEGIN\n', 1)[1].split(b'\nSJ_TONE_END\n', 1)
    if len(payload) != 2:
        raise ValueError('serial transfer not complete')
    wav = gzip.decompress(base64.b64decode(b''.join(payload[0].split()), validate=True))
    digest = hashlib.sha256(wav).hexdigest()
    if digest != args.sha256:
        raise ValueError('recording checksum mismatch')
    samples, rate = sf.read(io.BytesIO(wav), always_2d=True)
    if rate != 16000 or samples.shape[1] != 2:
        raise ValueError('unexpected recording format')
    mono = samples.mean(axis=1)
    freqs, times, power = spectrogram(mono, fs=rate, nperseg=1600, noverlap=800)
    band = (freqs >= 400) & (freqs <= 2200)
    indices = np.argmax(power[band], axis=0)
    peaks = freqs[band][indices]
    band_power = power[band]
    prominence = band_power.max(axis=0) / np.maximum(np.median(band_power, axis=0), 1e-20)
    targets = [700, 1300, 900, 1700, 1100, 1500, 700, 1700]
    segments = []
    for t, f, ratio in zip(times, peaks, prominence):
        nearest = min(set(targets), key=lambda target: abs(target-f))
        if abs(nearest-f) > 30 or ratio < 30:
            continue
        if segments and segments[-1]['frequency_hz'] == nearest and t-segments[-1]['end_s'] < .11:
            segments[-1]['end_s'] = round(float(t), 3)
        else:
            segments.append(dict(frequency_hz=nearest, start_s=round(float(t), 3), end_s=round(float(t), 3)))
    segments = [s for s in segments if s['end_s']-s['start_s'] >= .2]
    report = dict(sha256=digest, sample_rate=rate, channels=2, duration_s=len(mono)/rate,
                  rms=float(np.sqrt(np.mean(mono**2))), expected_sequence_hz=targets,
                  detected_tonal_segments=segments,
                  note='Heuristic tonal detection; inspect spectrogram and user audible confirmation.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'capture.wav').write_bytes(wav)
    (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    visible = freqs <= 2500
    db = 10*np.log10(np.maximum(power[visible], 1e-14))
    v = np.clip((db+100)/65, 0, 1)[::-1]
    rgb = np.stack([np.clip(v*3,0,1), np.clip(v*3-1,0,1), np.clip(v*3-2,0,1)],axis=-1)
    panel = Image.fromarray((rgb*255).astype('uint8')).resize((1000,350))
    canvas = Image.new('RGB',(1120,450),'white')
    canvas.paste(panel,(80,40))
    draw = ImageDraw.Draw(canvas)
    draw.text((80,10),'XVF3800 real capture: time / frequency (not classification)',fill='black')
    for sec in range(0,26,5):
        draw.text((80+sec*40,400),str(sec)+'s',fill='black')
    for hz in range(0,2501,500):
        draw.text((5,380-hz/2500*340),str(hz)+'Hz',fill='black')
    canvas.save(args.output/'spectrogram.png')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
