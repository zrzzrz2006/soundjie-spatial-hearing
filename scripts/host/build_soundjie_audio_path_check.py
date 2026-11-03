"""Create a modest-level coded tone WAV for a microphone-path check, not AI training."""
import argparse
import hashlib
import json
from pathlib import Path
import wave
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    rate = 16000
    samples = np.zeros(12*rate, dtype=np.float64)
    pattern = [700, 1300, 900, 1700, 1100, 1500, 700, 1700]
    segments = []
    for index, frequency in enumerate(pattern):
        start = (index+2)*rate
        count = 9600
        envelope = np.ones(count)
        ramp = np.linspace(0, 1, 160)
        envelope[:160], envelope[-160:] = ramp, ramp[::-1]
        samples[start:start+count] = .125 * envelope * np.sin(2*np.pi*frequency*np.arange(count)/rate)
        segments.append(dict(start_seconds=index+2, duration_seconds=.6, frequency_hz=frequency))
    pcm = np.rint(samples*32767).astype('<i2')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path = args.output_dir/'audio-path-check.wav'
    with path.open('xb') as stream:
        with wave.open(stream, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(pcm.tobytes())
    report = dict(purpose='audio_path_diagnostic_only', duration_seconds=12, sample_rate_hz=rate,
                  peak_dbfs=float(20*np.log10(.125)), segments=segments,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  note='Use comfortable speaker volume. Confirm audible tones, then compare recorded frequency sequence; no class accuracy claim.')
    with (args.output_dir/'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
