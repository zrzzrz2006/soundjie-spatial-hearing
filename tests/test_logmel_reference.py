import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "logmel_frontend_v1.json"
MODULE = ROOT / "scripts" / "host" / "logmel_reference.py"
BUILDER = ROOT / "scripts" / "host" / "build_logmel_golden.py"
EXPECTED = ROOT / "tests" / "fixtures" / "logmel_expected_v1.json"


def load_reference():
    spec = importlib.util.spec_from_file_location("logmel_reference", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk(name: bytes, payload: bytes) -> bytes:
    return name + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) & 1 else b"")


def pcm_wav(samples, *, channels=1, rate=16000, bits=16, format_tag=1,
            prefix_chunks=(), suffix_chunks=()):
    bytes_per_sample = bits // 8
    block_align = channels * bytes_per_sample
    fmt = struct.pack(
        "<HHIIHH", format_tag, channels, rate, rate * block_align, block_align, bits)
    if bits == 16:
        audio = struct.pack("<{}h".format(len(samples)), *samples)
    elif bits == 8:
        audio = bytes(samples)
    else:
        audio = bytes((len(samples) * channels * bytes_per_sample))
    body = b"WAVE" + b"".join(prefix_chunks) + chunk(b"fmt ", fmt) + chunk(b"data", audio)
    body += b"".join(suffix_chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LogMelReferenceTests(unittest.TestCase):
    def setUp(self):
        self.reference = load_reference()
        self.config = self.reference.load_frontend_config(CONFIG)

    def write(self, root: Path, name: str, data: bytes) -> Path:
        path = root / name
        path.write_bytes(data)
        return path

    def test_reads_mono_and_stereo_with_exact_scaling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mono = self.write(root, "mono.wav", pcm_wav([-32768, 0, 32767]))
            values, metadata = self.reference.read_pcm_window(mono, 0, 3)
            np.testing.assert_array_equal(values, np.array([-1.0, 0.0, 32767 / 32768.0]))
            self.assertEqual(values.dtype, np.dtype("float64"))
            self.assertEqual((metadata.channels, metadata.sample_rate_hz,
                              metadata.bits_per_sample, metadata.total_samples),
                             (1, 16000, 16, 3))

            stereo_samples = [-32768, -32768, 32767, 32767, 32767, -32768]
            stereo = self.write(root, "stereo.wav", pcm_wav(stereo_samples, channels=2))
            values, metadata = self.reference.read_pcm_window(stereo, 0, 3)
            np.testing.assert_array_equal(
                values, np.array([-1.0, 32767 / 32768.0, -1 / 65536.0]))
            self.assertEqual(metadata.channels, 2)
            self.assertEqual(metadata.total_samples, 3)

    def test_skips_unknown_chunks_with_odd_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(
                Path(temporary), "padded.wav",
                pcm_wav([123, -456], prefix_chunks=(chunk(b"JUNK", b"abc"),)))
            values, metadata = self.reference.read_pcm_window(path, 0, 2)
            np.testing.assert_array_equal(values, np.array([123 / 32768.0, -456 / 32768.0]))
            self.assertGreater(metadata.data_offset, 44)

    def test_uses_first_valid_fmt_and_data_chunks(self):
        fmt2 = struct.pack("<HHIIHH", 1, 2, 16000, 64000, 4, 16)
        second_data = struct.pack("<2h", 3000, 4000)
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(
                Path(temporary), "duplicates.wav",
                pcm_wav([1000, 2000], suffix_chunks=(chunk(b"fmt ", fmt2),
                                                      chunk(b"data", second_data))))
            values, metadata = self.reference.read_pcm_window(path, 0, 2)
            np.testing.assert_array_equal(values, np.array([1000 / 32768.0, 2000 / 32768.0]))
            self.assertEqual(metadata.channels, 1)

    def test_rejects_wrong_rate_depth_format_channels_and_truncation(self):
        cases = {
            "rate": pcm_wav([0], rate=8000),
            "depth": pcm_wav([0], bits=8),
            "format": pcm_wav([0], format_tag=3),
            "channels": pcm_wav([0, 0, 0], channels=3),
            "truncated": pcm_wav([0, 1])[:-1],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, payload in cases.items():
                path = self.write(root, f"{name}.wav", payload)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.reference.read_pcm_window(path, 0, 1)

    def test_rejects_negative_or_incomplete_window(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write(Path(temporary), "short.wav", pcm_wav([0] * 10))
            with self.assertRaises(ValueError):
                self.reference.read_pcm_window(path, -1, 1)
            with self.assertRaises(ValueError):
                self.reference.read_pcm_window(path, 0, 11)
            with self.assertRaises(ValueError):
                self.reference.read_pcm_window(path, 0, 0)

    def test_synthetic_cases_are_deterministic_and_well_formed(self):
        names = (
            "silence", "impulse", "sine_440", "dual_tone", "noise_seeded",
            "clipping", "stereo_antiphase",
        )
        first = {}
        for name in names:
            samples, metadata = self.reference.synthesize_case(name, self.config)
            first[name] = samples.copy()
            self.assertEqual(samples.dtype, np.dtype("int16"))
            expected = 32000 if name == "stereo_antiphase" else 16000
            self.assertEqual(samples.size, expected)
            self.assertEqual(metadata["name"], name)
            self.assertEqual(metadata["channels"], 2 if name == "stereo_antiphase" else 1)
        for name in names:
            samples, _ = self.reference.synthesize_case(name, self.config)
            np.testing.assert_array_equal(samples, first[name])
        with self.assertRaises(ValueError):
            self.reference.synthesize_case("unknown", self.config)

    def test_inputs_cli_is_create_only_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = self.write(root, "real.wav", pcm_wav([i % 2000 - 1000 for i in range(40000)]))
            a, b = root / "a", root / "b"
            command = [
                sys.executable, str(BUILDER), "inputs", "--config", str(CONFIG),
                "--real-wav", str(real), "--output-dir",
            ]
            first = subprocess.run(command + [str(a)], text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            second = subprocess.run(command + [str(b)], text=True, capture_output=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            repeat = subprocess.run(command + [str(a)], text=True, capture_output=True)
            self.assertNotEqual(repeat.returncode, 0)

            files_a = sorted(path.relative_to(a) for path in a.rglob("*") if path.is_file())
            files_b = sorted(path.relative_to(b) for path in b.rglob("*") if path.is_file())
            self.assertEqual(files_a, files_b)
            self.assertEqual({str(p): sha256(a / p) for p in files_a},
                             {str(p): sha256(b / p) for p in files_b})
            manifest = json.loads((a / "source-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], 1)
            self.assertEqual(manifest["real_wav"]["window_start_samples"], [0, 12000, 24000])
            self.assertEqual(len(manifest["synthetic_cases"]), 7)
            self.assertEqual(manifest["real_wav"]["sha256"], sha256(real))

    def test_periodic_hann_and_frame_count(self):
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
        np.testing.assert_allclose(
            self.reference.periodic_hann(4), expected["periodic_hann_4"], atol=1e-15)
        samples = np.arange(16000, dtype=np.float64)
        frames = self.reference.frame_audio(samples, self.config)
        self.assertEqual(frames.shape, (98, 400))
        np.testing.assert_array_equal(frames[0], samples[:400])
        np.testing.assert_array_equal(frames[-1], samples[15520:15920])

    def test_fft_power_uses_exact_divisor_512(self):
        frames = np.zeros((1, 400), dtype=np.float64)
        frames[0, 0] = 1.0
        power = self.reference.compute_power_spectrum(frames, self.config, apply_window=False)
        self.assertEqual(power.shape, (1, 257))
        np.testing.assert_array_equal(power, np.full((1, 257), 1.0 / 512.0))

    def test_htk_filterbank_has_64_finite_triangles(self):
        expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
        filters = self.reference.build_mel_filterbank(self.config)
        self.assertEqual(filters.shape, (64, 257))
        self.assertEqual(filters.dtype, np.dtype("float64"))
        self.assertTrue(np.isfinite(filters).all())
        self.assertGreaterEqual(float(filters.min()), 0.0)
        self.assertLessEqual(float(filters.max()), 1.0)
        for item in expected["mel_selected_weights"]:
            self.assertAlmostEqual(filters[item["band"], item["bin"]], item["weight"], places=12)

    def test_filterbank_bin_boundary_does_not_divide_by_zero(self):
        config = self.config._replace(mel_min_hz=0.0, mel_max_hz=31.25)
        filters = self.reference.build_mel_filterbank(config)
        self.assertEqual(filters.shape, (64, 257))
        self.assertTrue(np.isfinite(filters).all())
        self.assertGreaterEqual(float(filters.min()), 0.0)

    def test_drops_only_first_and_last_frame_and_uses_mel_major_layout(self):
        bands = np.arange(64, dtype=np.float64)[:, None]
        times = np.arange(98, dtype=np.float64)[None, :] / 100.0
        normalized = -0.9 + bands / 1000.0 + times / 10.0
        logged = (normalized * self.config.profile_std[0] +
                  self.config.profile_mean[0])
        energies = np.exp(logged)
        features = self.reference.log_crop_normalize(energies, self.config)
        self.assertEqual(features.shape, (64, 96))
        self.assertEqual(features.dtype, np.dtype("float32"))
        expected_times = -0.9 + np.arange(1, 97) / 1000.0
        np.testing.assert_allclose(features[0], expected_times, atol=1e-6)
        self.assertAlmostEqual(float(features[1, 0]), -0.898, places=6)
        flattened = features.reshape(-1, order="C")
        self.assertAlmostEqual(float(flattened[95]), -0.804, places=6)
        self.assertAlmostEqual(float(flattened[96]), -0.898, places=6)

    def test_log_floor_and_energy_upper_bound_map_to_minus_one_and_plus_one(self):
        energies = np.empty((64, 98), dtype=np.float64)
        energies[:, 0::2] = 0.0
        energies[:, 1::2] = 400.0
        features = self.reference.log_crop_normalize(energies, self.config)
        np.testing.assert_allclose(features[:, 0::2], 1.0, atol=1e-7)
        np.testing.assert_allclose(features[:, 1::2], -1.0, atol=1e-7)

    def test_normalized_features_stay_in_contract_range(self):
        energies = np.full((64, 98), np.exp(self.config.profile_upper_bound),
                           dtype=np.float64)
        features = self.reference.log_crop_normalize(energies, self.config)
        self.assertLessEqual(float(features.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(features.min()), -1.0 - 1e-6)

        too_high = energies.copy()
        too_high[:, 1:97] = np.exp(
            self.config.profile_upper_bound + self.config.profile_std[0] * 2e-6)
        with self.assertRaises(ValueError):
            self.reference.log_crop_normalize(too_high, self.config)

    def test_quantization_rounds_half_away_from_zero_and_saturates(self):
        features = np.zeros((64, 96), dtype=np.float32)
        features.flat[:7] = [-20.0, -0.1875, -0.0625, 0.0, 0.0625, 0.1875, 20.0]
        quantized = self.reference.quantize_int8(features, 8.0)
        self.assertEqual(quantized.shape, (64, 96))
        self.assertEqual(quantized.dtype, np.dtype("int8"))
        np.testing.assert_array_equal(
            quantized.flat[:7], np.array([-128, -2, -1, 0, 1, 2, 127], dtype=np.int8))

    def test_quantization_rejects_nonfinite_or_nonpositive_scale(self):
        features = np.zeros((64, 96), dtype=np.float32)
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.reference.quantize_int8(features, scale)
        features[2, 3] = np.nan
        with self.assertRaises(ValueError):
            self.reference.quantize_int8(features, 8.0)
        with self.assertRaises(ValueError):
            self.reference.quantize_int8(np.zeros((6144,), dtype=np.float32), 8.0)

    def test_compute_logmel_is_finite_float32_with_fixed_shape(self):
        samples = np.zeros(16000, dtype=np.float64)
        features = self.reference.compute_logmel(samples, self.config)
        self.assertEqual(features.shape, (64, 96))
        self.assertEqual(features.dtype, np.dtype("float32"))
        self.assertTrue(np.isfinite(features).all())

    def test_build_cli_writes_deterministic_features_and_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = self.write(root, "real.wav", pcm_wav([i % 3000 - 1500 for i in range(40000)]))
            a, b = root / "golden-a", root / "golden-b"
            command = [
                sys.executable, str(BUILDER), "build", "--config", str(CONFIG),
                "--real-wav", str(real), "--input-scale", "8.0",
                "--scale-usage", "parity_test_only", "--output-dir",
            ]
            for output in (a, b):
                result = subprocess.run(command + [str(output)], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            paths_a = sorted(path.relative_to(a) for path in a.rglob("*") if path.is_file())
            paths_b = sorted(path.relative_to(b) for path in b.rglob("*") if path.is_file())
            self.assertEqual(paths_a, paths_b)
            self.assertEqual({str(p): sha256(a / p) for p in paths_a},
                             {str(p): sha256(b / p) for p in paths_b})
            float_files = sorted(a.rglob("*.f32le"))
            int8_files = sorted(a.rglob("*.i8"))
            self.assertEqual(len(float_files), 10)
            self.assertEqual(len(int8_files), 10)
            self.assertTrue(all(path.stat().st_size == 24576 for path in float_files))
            self.assertTrue(all(path.stat().st_size == 6144 for path in int8_files))
            manifest = json.loads((a / "golden-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], 2)
            self.assertEqual(manifest["profile_id"], "bounded-log-power-64-v1")
            self.assertEqual(manifest["profile_domain"], "natural_log_mel_power")
            self.assertEqual(manifest["profile_lower_bound"], -23.025850929940457)
            self.assertEqual(manifest["profile_upper_bound"], 5.9914645471079817)
            self.assertEqual(manifest["input_scale"], 8.0)
            self.assertEqual(manifest["scale_usage"], "parity_test_only")
            self.assertEqual(len(manifest["cases"]), 10)
            self.assertNotIn(str(root), json.dumps(manifest))
            for case in manifest["cases"]:
                self.assertEqual(case["float"]["nonfinite"], 0)
                self.assertEqual(len(case["float"]["sha256"]), 64)
                self.assertEqual(len(case["int8"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
