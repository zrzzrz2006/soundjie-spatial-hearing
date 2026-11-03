import importlib
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def load_prepare():
    sys.path.insert(0, str(REPO))
    try:
        return importlib.import_module("scripts.host.prepare_soundjie_d3")
    finally:
        sys.path.pop(0)


def pcm_wav(path, samples, rate, bits, channels=1):
    values = np.asarray(samples).reshape(-1)
    if bits == 8:
        payload = np.clip(np.rint(values * 128.0 + 128.0), 0, 255).astype(np.uint8).tobytes()
    elif bits == 16:
        payload = np.clip(np.rint(values * 32768.0), -32768, 32767).astype("<i2").tobytes()
    elif bits == 24:
        ints = np.clip(np.rint(values * 8388608.0), -8388608, 8388607).astype(np.int32)
        payload = b"".join(int(value).to_bytes(3, "little", signed=True) for value in ints)
    elif bits == 32:
        payload = np.clip(np.rint(values * 2147483648.0), -2147483648, 2147483647).astype("<i4").tobytes()
    else:
        raise AssertionError(bits)
    block_align = channels * (bits // 8)
    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * block_align, block_align, bits)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    path.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)


class SoundjieD3AudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepare = load_prepare()

    def test_decodes_8_16_24_32_bit_mono_and_stereo(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mono = np.linspace(-0.75, 0.75, 64, dtype=np.float64)
            for bits in (8, 16, 24, 32):
                path = root / f"mono-{bits}.wav"
                pcm_wav(path, mono, 16000, bits)
                audio = self.prepare.decode_audio(path)
                self.assertEqual(audio.sample_rate, 16000)
                self.assertEqual(audio.samples.shape, (64, 1))
                self.assertTrue(np.isfinite(audio.samples).all())
            stereo = np.column_stack((mono, -mono)).reshape(-1)
            path = root / "stereo.wav"
            pcm_wav(path, stereo, 16000, 16, channels=2)
            audio = self.prepare.decode_audio(path)
            self.assertEqual(audio.samples.shape, (64, 2))
            mixed = self.prepare.resample_mono_16k(audio)
            self.assertLess(float(np.max(np.abs(mixed))), 1e-4)

    def test_resamples_8_16_44100_48000_to_exact_one_second(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rate in (8000, 16000, 44100, 48000):
                time = np.arange(rate, dtype=np.float64) / rate
                path = root / f"rate-{rate}.wav"
                pcm_wav(path, 0.25 * np.sin(2 * np.pi * 440 * time), rate, 16)
                output = self.prepare.resample_mono_16k(self.prepare.decode_audio(path))
                self.assertEqual(output.shape, (16000,))
                self.assertTrue(np.isfinite(output).all())

    def test_rejects_empty_truncated_nonfinite_and_silent_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "empty.wav"
            empty.write_bytes(b"")
            with self.assertRaises(self.prepare.AudioPreparationError):
                self.prepare.decode_audio(empty)
            truncated = root / "truncated.wav"
            truncated.write_bytes(b"RIFF\xff\xff\xff\xffWAVE")
            with self.assertRaises(self.prepare.AudioPreparationError):
                self.prepare.decode_audio(truncated)
            for bad in (np.nan, np.inf):
                audio = self.prepare.AudioBuffer(16000, np.array([[0.0], [bad]]))
                with self.assertRaises(self.prepare.AudioPreparationError):
                    self.prepare.resample_mono_16k(audio)
            with self.assertRaisesRegex(self.prepare.AudioPreparationError, "silent|constant"):
                list(self.prepare.make_windows(np.zeros(32000), self.prepare.WindowPolicy("energy_center", 7)))

    def test_short_and_long_windows_are_exact_and_deterministic(self):
        short = np.linspace(-0.2, 0.2, 8000, dtype=np.float64)
        first = list(self.prepare.make_windows(short, self.prepare.WindowPolicy("seeded", 11)))
        second = list(self.prepare.make_windows(short, self.prepare.WindowPolicy("seeded", 11)))
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].samples.shape, (16000,))
        self.assertTrue(np.array_equal(first[0].samples, second[0].samples))
        long = np.zeros(48000, dtype=np.float64)
        long[25000:27000] = 0.8
        event = list(self.prepare.make_windows(long, self.prepare.WindowPolicy("energy_center", 11)))
        seeded = list(self.prepare.make_windows(long + 1e-4 * np.arange(48000), self.prepare.WindowPolicy("seeded", 11)))
        self.assertEqual(event[0].samples.shape, (16000,))
        self.assertEqual(seeded[0].samples.shape, (16000,))
        self.assertGreater(event[0].start_sample, 10000)


if __name__ == "__main__":
    unittest.main()
