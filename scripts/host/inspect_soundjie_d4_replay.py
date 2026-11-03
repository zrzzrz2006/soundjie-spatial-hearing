"""Read-only input diagnostics; emits a new spectrogram and parity report."""
import argparse
import io
import json
from pathlib import Path
import tarfile
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram
from PIL import Image, ImageDraw
from scripts.host.evaluate_soundjie_d4_replay import ARCHIVE_SHA, CASES
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.prepare_soundjie_d3 import compute_feature
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.train_soundjie_d4 import ROOT, write_json


def panel(samples):
    _, _, power = spectrogram(samples, fs=16000, nperseg=400, noverlap=240, nfft=512)
    value = np.clip((10*np.log10(np.maximum(power, 1e-12))+100)/70, 0, 1)
    value = np.flipud(value)
    rgb = np.stack([np.clip(3*value, 0, 1), np.clip(3*value-1, 0, 1), np.clip(3*value-2, 0, 1)], axis=-1)
    return Image.fromarray((rgb*255).astype('uint8')).resize((540, 160))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--demo-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if digest_file(args.archive) != ARCHIVE_SHA:
        raise ValueError('historical archive hash drift')
    labels = (ROOT/'models/soundjie_d3/labels.txt').read_text().splitlines()
    frontend = load_frontend_config(ROOT/'config/logmel_frontend_v1.json')
    canvas = Image.new('RGB', (1120, 8*195+30), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), 'LEFT: digital playback file   RIGHT: six-second XVF3800 capture. Common spectral color scale.', fill='black')
    reports = []
    with tarfile.open(args.archive, 'r:gz') as archive:
        for name, target in CASES:
            for index in range(2):
                base = f'sjd3l1-live-{name}-{index:02d}'
                raw = archive.extractfile(base+'/capture.wav').read()
                samples, rate = sf.read(io.BytesIO(raw), always_2d=True, dtype='float64')
                if rate != 16000 or samples.shape[1] != 2:
                    raise ValueError('capture format drift')
                mono = samples.mean(1)
                session = json.load(archive.extractfile(base+'/session-result.json'))
                errors = []
                for w in session['windows']:
                    offset = w['start_sample']
                    feature = compute_feature(mono[offset:offset+16000], frontend)
                    prior = np.frombuffer(archive.extractfile(base+f"/window-{w['rank']}/logmel.f32le").read(), dtype='<f4').reshape(1,64,96)
                    errors.append(dict(max_abs=float(np.abs(feature-prior).max()), rmse=float(np.sqrt(np.mean((feature-prior)**2)))))
                demo = args.demo_dir/f'{target:02d}-{labels[target]}-demo-{index:02d}.wav'
                digital, demo_rate = sf.read(demo, always_2d=True, dtype='float64')
                if demo_rate != 16000:
                    raise ValueError('demo format drift')
                row = len(reports)
                draw.text((10, row*195+30), base+' | digital '+str(len(digital)/16000)+'s; recorded 6s', fill='black')
                canvas.paste(panel(digital.mean(1)), (10, row*195+50))
                canvas.paste(panel(mono), (570, row*195+50))
                reports.append(dict(case=base, frontend_errors=errors, rms=float(np.sqrt(np.mean(mono**2))),
                                    demo_sha256=digest_file(demo)))
    args.output.mkdir(parents=True, exist_ok=False)
    canvas.save(args.output/'spectrograms.png')
    write_json(args.output/'frontend-parity.json', reports)
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    main()
