import copy
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "guest" / "build_logmel_cli.sh"
COMPARE = ROOT / "scripts" / "host" / "compare_logmel_outputs.py"
CONFIG = ROOT / "config" / "logmel_frontend_v1.json"
REFERENCE = ROOT / "scripts" / "host" / "logmel_reference.py"


def wsl_path(path: Path) -> str:
    path = Path(path).resolve()
    drive = path.drive[0].lower()
    tail = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def load_reference():
    spec = importlib.util.spec_from_file_location("logmel_reference_cpp_test", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogMelCppBuildTests(unittest.TestCase):
    def test_cpp_rejects_nested_scalar_and_slash_aliases(self):
        case = Path(self.temporary.name) / "nested-alias"
        case.mkdir()
        wav = case / "silence.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 16000)
        valid = json.loads(CONFIG.read_text(encoding="utf-8"))
        candidates = []
        for key in ("schema", "alias/schema"):
            value = copy.deepcopy(valid)
            value["schema"] = {key: 2}
            candidates.append(value)
        for key in ("sample_rate_hz", "alias/sample_rate_hz"):
            value = copy.deepcopy(valid)
            value["audio"]["sample_rate_hz"] = {key: 16000}
            candidates.append(value)
        value = copy.deepcopy(valid)
        value["audio"]["supported_channels"] = {"supported_channels": [1, 2]}
        candidates.append(value)
        for index, value in enumerate(candidates):
            config = case / ("bad-%d.json" % index)
            config.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(index=index):
                result = self.run_case(case, config, wav, "case-%d" % index)
                self.assertNotEqual(result.returncode, 0, result.stdout)

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "logmel_cli"
        result = subprocess.run([
            "wsl.exe", "bash", wsl_path(BUILD), "--mode", "native",
            "--repo-root", wsl_path(ROOT), "--output", wsl_path(cls.output),
        ], text=True, encoding="utf-8", errors="replace", capture_output=True)
        if result.returncode != 0:
            raise AssertionError(f"native C++ build failed\nstdout={result.stdout}\nstderr={result.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_wsl_binary(self, *arguments):
        return subprocess.run(
            ["wsl.exe", wsl_path(self.output), *arguments], text=True,
            encoding="utf-8", errors="replace", capture_output=True)

    def run_case(self, root, config, wav, name, start=0, scale=128,
                 golden_float=None, golden_int8=None):
        arguments = [
            "--config", wsl_path(config), "--wav", wsl_path(wav),
            "--start-sample", str(start), "--input-scale", str(scale),
            "--float-output", wsl_path(root / f"{name}.f32le"),
            "--int8-output", wsl_path(root / f"{name}.i8"),
            "--report", wsl_path(root / f"{name}.txt"),
        ]
        if golden_float is not None:
            arguments += ["--golden-float", wsl_path(golden_float),
                          "--golden-int8", wsl_path(golden_int8)]
        return self.run_wsl_binary(*arguments)

    def test_cpp_units_have_single_responsibilities(self):
        result = subprocess.run(
            ["wsl.exe", "ldd", wsl_path(self.output)], text=True,
            encoding="utf-8", errors="replace", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("n2cube", result.stdout.lower())
        self.assertNotIn("fftw", result.stdout.lower())
        self.assertNotIn("python", result.stdout.lower())

    def test_build_script_has_native_and_arm_modes(self):
        result = subprocess.run(
            ["wsl.exe", "bash", wsl_path(BUILD), "--help"], text=True,
            encoding="utf-8", errors="replace", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mode native|arm", result.stdout)

    def test_cli_requires_all_explicit_paths(self):
        missing = self.run_wsl_binary()
        self.assertEqual(missing.returncode, 64)
        self.assertIn("required option missing", missing.stderr)
        unpaired = self.run_wsl_binary(
            "--config", "/tmp/config", "--wav", "/tmp/input.wav",
            "--start-sample", "0", "--input-scale", "8",
            "--float-output", "/tmp/a", "--int8-output", "/tmp/b",
            "--report", "/tmp/c", "--golden-float", "/tmp/d")
        self.assertEqual(unpaired.returncode, 64)
        self.assertIn("golden files must be provided together", unpaired.stderr)

    def test_cli_never_replaces_racing_destination_or_leaves_partial_set(self):
        case = Path(self.temporary.name) / "publish-race"
        case.mkdir()
        wav = case / "input.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 16000)
        float_output = case / "features.f32le"
        int8_output = case / "tensor.i8"
        report = case / "report.txt"
        link = subprocess.run(
            ["wsl.exe", "ln", "-s", "missing-target", wsl_path(int8_output)],
            text=True, encoding="utf-8", errors="replace", capture_output=True)
        self.assertEqual(link.returncode, 0, link.stderr)

        result = self.run_wsl_binary(
            "--config", wsl_path(ROOT / "config" / "logmel_frontend_v1.json"),
            "--wav", wsl_path(wav), "--start-sample", "0",
            "--input-scale", "8", "--float-output", wsl_path(float_output),
            "--int8-output", wsl_path(int8_output), "--report", wsl_path(report))

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(float_output.exists())
        self.assertFalse(report.exists())
        readlink = subprocess.run(
            ["wsl.exe", "readlink", wsl_path(int8_output)], text=True,
            encoding="utf-8", errors="replace", capture_output=True)
        self.assertEqual(readlink.returncode, 0, readlink.stderr)
        self.assertEqual(readlink.stdout.strip(), "missing-target")

    def test_cli_rejects_text_that_only_contains_valid_json_fragments(self):
        case = Path(self.temporary.name) / "strict-json"
        case.mkdir()
        wav = case / "input.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 16000)
        config = case / "invalid.json"
        valid = (ROOT / "config" / "logmel_frontend_v1.json").read_text(
            encoding="utf-8")
        config.write_text("not-json\n" + valid, encoding="utf-8")
        result = self.run_wsl_binary(
            "--config", wsl_path(config), "--wav", wsl_path(wav),
            "--start-sample", "0", "--input-scale", "8",
            "--float-output", wsl_path(case / "features.f32le"),
            "--int8-output", wsl_path(case / "tensor.i8"),
            "--report", wsl_path(case / "report.txt"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid JSON", result.stderr)

        nested = case / "wrong-level.json"
        nested.write_text('{"wrapper":' + valid + '}', encoding="utf-8")
        result = self.run_wsl_binary(
            "--config", wsl_path(nested), "--wav", wsl_path(wav),
            "--start-sample", "0", "--input-scale", "8",
            "--float-output", wsl_path(case / "nested.f32le"),
            "--int8-output", wsl_path(case / "nested.i8"),
            "--report", wsl_path(case / "nested.txt"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("config structure", result.stderr)

    def test_cpp_rejects_identity_and_single_band_profile_drift(self):
        case = Path(self.temporary.name) / "profile-rejection"
        case.mkdir()
        wav = case / "silence.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(b"\0\0" * 16000)
        valid = json.loads(CONFIG.read_text(encoding="utf-8"))
        identity = copy.deepcopy(valid)
        identity["schema"] = 1
        identity["normalization"] = {
            "id": "identity-64-v1", "mean": [0.0] * 64, "std": [1.0] * 64,
        }
        drift = copy.deepcopy(valid)
        drift["normalization"]["mean"][19] += 1e-12
        for index, candidate in enumerate((identity, drift)):
            config = case / f"bad-{index}.json"
            config.write_text(json.dumps(candidate), encoding="utf-8")
            result = self.run_case(case, config, wav, f"bad-{index}")
            with self.subTest(index=index):
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((case / f"bad-{index}.f32le").exists())

    def test_cpp_rejects_non_json_number_spellings(self):
        case=Path(self.temporary.name)/'number-grammar';case.mkdir();wav=case/'silence.wav'
        with wave.open(str(wav),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*16000)
        for i,token in enumerate(('+2','0x2','02','2.','2e+')):
            config=case/('bad-%d.json'%i)
            config.write_text(CONFIG.read_text(encoding='utf-8').replace('"schema": 2','"schema": '+token),encoding='utf-8')
            r=self.run_case(case,config,wav,'bad-%d'%i)
            with self.subTest(token=token):self.assertNotEqual(r.returncode,0)

    def test_cpp_rejects_escaped_schema_key_aliases(self):
        case=Path(self.temporary.name)/'key-grammar';case.mkdir();wav=case/'silence.wav'
        with wave.open(str(wav),'wb') as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*16000)
        for i,(key,token) in enumerate([('center',r'ce\nter'),('sample_rate_hz',r'sample_ra\te_hz')]):
            config=case/('bad-%d.json'%i)
            config.write_text(CONFIG.read_text(encoding='utf-8').replace('"'+key+'"','"'+token+'"'),encoding='utf-8')
            r=self.run_case(case,config,wav,'bad-%d'%i)
            with self.subTest(key=key):self.assertNotEqual(r.returncode,0)

    def test_cpp_boundary_normalization_matches_python(self):
        case = Path(self.temporary.name) / "boundary-parity"
        case.mkdir()
        reference = load_reference()
        config = reference.load_frontend_config(CONFIG)
        for name in ("silence", "clipping"):
            samples, metadata = reference.synthesize_case(name, config)
            wav = case / f"{name}.wav"
            wav.write_bytes(reference.encode_pcm_wav(samples, metadata["channels"]))
            mono, _ = reference.read_pcm_window(wav, 0, config.window_samples)
            features = reference.compute_logmel(mono, config)
            tensor = reference.quantize_int8(features, 128.0)
            golden_float = case / f"{name}.golden.f32le"
            golden_int8 = case / f"{name}.golden.i8"
            golden_float.write_bytes(features.astype("<f4").tobytes())
            golden_int8.write_bytes(tensor.tobytes())
            result = self.run_case(case, CONFIG, wav, name, golden_float=golden_float,
                                   golden_int8=golden_int8)
            with self.subTest(name=name):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("INT8_EXACT=yes", (case / f"{name}.txt").read_text())

    def test_cpp_real_like_windows_preserve_distinct_int8(self):
        case = Path(self.temporary.name) / "real-like"
        case.mkdir()
        sample_count = 40000
        time = np.arange(sample_count, dtype=np.float64) / 16000.0
        rng = np.random.default_rng(20261012)
        signal = np.empty(sample_count, dtype=np.float64)
        signal[:12000] = 0.35 * np.sin(2 * np.pi * 330 * time[:12000])
        signal[12000:24000] = 0.25 * np.sin(2 * np.pi * 910 * time[12000:24000])
        signal[24000:] = 0.12 * rng.standard_normal(sample_count - 24000)
        pcm = np.clip(np.rint(signal * 32768.0), -32768, 32767).astype("<i2")
        wav = case / "real-like.wav"
        with wave.open(str(wav), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(pcm.tobytes())
        tensors = []
        for index, start in enumerate((0, 12000, 24000)):
            result = self.run_case(case, CONFIG, wav, f"window-{index}", start=start)
            self.assertEqual(result.returncode, 0, result.stderr)
            tensor = (case / f"window-{index}.i8").read_bytes()
            self.assertGreaterEqual(len(set(tensor)), 32)
            tensors.append(tensor)
        self.assertEqual(len({hashlib.sha256(item).hexdigest() for item in tensors}), 3)


class LogMelComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.golden_f32 = self.root / "golden.f32le"
        self.candidate_f32 = self.root / "candidate.f32le"
        self.golden_i8 = self.root / "golden.i8"
        self.candidate_i8 = self.root / "candidate.i8"
        self.report = self.root / "report.txt"
        values = [0.0] * 6144
        values[7], values[511] = 1.25, -2.5
        self.golden_f32.write_bytes(struct.pack("<6144f", *values))
        self.candidate_f32.write_bytes(struct.pack("<6144f", *values))
        tensor = bytearray(6144)
        tensor[7], tensor[511] = 10, 236
        self.golden_i8.write_bytes(tensor)
        self.candidate_i8.write_bytes(tensor)

    def tearDown(self):
        self.temporary.cleanup()

    def run_comparator(self):
        return subprocess.run([
            sys.executable, str(COMPARE),
            "--golden-f32", str(self.golden_f32),
            "--candidate-f32", str(self.candidate_f32),
            "--golden-i8", str(self.golden_i8),
            "--candidate-i8", str(self.candidate_i8),
            "--report", str(self.report),
        ], text=True, capture_output=True)

    def test_comparator_enforces_max_abs_rmse_and_exact_int8(self):
        passed = self.run_comparator()
        self.assertEqual(passed.returncode, 0, passed.stderr)
        report = self.report.read_text(encoding="ascii")
        self.assertIn("PARITY_STATUS=pass", report)
        self.assertIn("FLOAT_COUNT=6144", report)
        self.assertIn("INT8_EXACT=yes", report)

        values = list(struct.unpack("<6144f", self.candidate_f32.read_bytes()))
        values[13] = 2e-4
        self.candidate_f32.write_bytes(struct.pack("<6144f", *values))
        self.report.unlink()
        failed = self.run_comparator()
        self.assertEqual(failed.returncode, 1)
        self.assertIn("PARITY_STATUS=fail", self.report.read_text(encoding="ascii"))

    def test_comparator_reports_first_max_error_and_byte_mismatch(self):
        values = list(struct.unpack("<6144f", self.candidate_f32.read_bytes()))
        values[123] = -0.25
        self.candidate_f32.write_bytes(struct.pack("<6144f", *values))
        tensor = bytearray(self.candidate_i8.read_bytes())
        tensor[321] = 99
        self.candidate_i8.write_bytes(tensor)
        failed = self.run_comparator()
        self.assertEqual(failed.returncode, 1)
        report = self.report.read_text(encoding="ascii")
        self.assertIn("FIRST_MAX_ERROR_INDEX=123", report)
        self.assertIn("FIRST_INT8_MISMATCH_INDEX=321", report)
        self.assertIn("INT8_EXACT=no", report)


if __name__ == "__main__":
    unittest.main()
