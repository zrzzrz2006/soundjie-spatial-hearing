import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "logmel_frontend_v1.json"
MODULE = ROOT / "scripts" / "host" / "logmel_reference.py"


def load_reference():
    spec = importlib.util.spec_from_file_location("logmel_reference", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogMelContractTests(unittest.TestCase):
    def test_config_pins_exact_d2_bounded_profile(self):
        data = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(data["schema"], 2)
        self.assertEqual(data["audio"], {
            "sample_rate_hz": 16000,
            "sample_format": "s16le",
            "supported_channels": [1, 2],
            "window_samples": 16000,
            "stream_hop_samples": 4000,
        })
        self.assertEqual(data["stft"], {
            "frame_length": 400,
            "hop_length": 160,
            "fft_length": 512,
            "window": "periodic_hann",
            "center": False,
            "expected_frames": 98,
            "keep_frame_start": 1,
            "keep_frame_count": 96,
        })
        self.assertEqual(data["power_spectrum"], {
            "formula": "magnitude_squared_div_fft_length",
            "divisor": 512.0,
            "single_sided_doubling": False,
            "window_energy_normalization": False,
        })
        self.assertEqual(data["mel"], {
            "bands": 64,
            "min_hz": 20.0,
            "max_hz": 7600.0,
            "scale": "htk",
            "normalization": "none",
        })
        self.assertEqual(data["log"], {"base": "natural", "floor": 1e-10})
        self.assertEqual(data["tensor"], {
            "layout": "NCHW",
            "shape": [1, 1, 64, 96],
            "order": "mel_major_time_increasing",
            "dtype": "float32-le",
        })
        normalization = data["normalization"]
        self.assertEqual(normalization["id"], "bounded-log-power-64-v1")
        self.assertEqual(normalization["domain"], "natural_log_mel_power")
        self.assertEqual(normalization["lower_bound"], -23.025850929940457)
        self.assertEqual(normalization["upper_bound"], 5.9914645471079817)
        self.assertEqual(normalization["mean"], [-8.5171931914162382] * 64)
        self.assertEqual(normalization["std"], [14.508657738524219] * 64)
        self.assertEqual(data["quantization"], {
            "dtype": "int8",
            "rounding": "half_away_from_zero",
            "clamp": [-128, 127],
            "scale_source": "runtime_dpu_input_tensor",
        })

    def test_rejects_identity_or_perturbed_bounded_profile(self):
        reference = load_reference()
        valid = json.loads(CONFIG.read_text(encoding="utf-8"))
        cases = []
        identity = copy.deepcopy(valid)
        identity["schema"] = 1
        identity["normalization"] = {
            "id": "identity-64-v1", "mean": [0.0] * 64, "std": [1.0] * 64,
        }
        cases.append(identity)
        missing = copy.deepcopy(valid)
        del missing["mel"]
        cases.append(missing)
        short_mean = copy.deepcopy(valid)
        short_mean["normalization"]["mean"] = short_mean["normalization"]["mean"][:63]
        cases.append(short_mean)
        zero_std = copy.deepcopy(valid)
        zero_std["normalization"]["std"][17] = 0.0
        cases.append(zero_std)
        nan_std = copy.deepcopy(valid)
        nan_std["normalization"]["std"][3] = float("nan")
        cases.append(nan_std)
        bad_mel = copy.deepcopy(valid)
        bad_mel["mel"]["max_hz"] = 8001.0
        cases.append(bad_mel)
        wrong_id = copy.deepcopy(valid)
        wrong_id["normalization"]["id"] = "bounded-log-power-64-v2"
        cases.append(wrong_id)
        wrong_domain = copy.deepcopy(valid)
        wrong_domain["normalization"]["domain"] = "log10_mel_power"
        cases.append(wrong_domain)
        wrong_lower = copy.deepcopy(valid)
        wrong_lower["normalization"]["lower_bound"] += 1e-12
        cases.append(wrong_lower)
        wrong_upper = copy.deepcopy(valid)
        wrong_upper["normalization"]["upper_bound"] -= 1e-12
        cases.append(wrong_upper)
        wrong_mean = copy.deepcopy(valid)
        wrong_mean["normalization"]["mean"][31] += 1e-12
        cases.append(wrong_mean)
        wrong_std = copy.deepcopy(valid)
        wrong_std["normalization"]["std"][47] -= 1e-12
        cases.append(wrong_std)
        extra_key = copy.deepcopy(valid)
        extra_key["normalization"]["adaptive"] = True
        cases.append(extra_key)

        with tempfile.TemporaryDirectory() as temporary:
            for index, candidate in enumerate(cases):
                path = Path(temporary) / f"bad-{index}.json"
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(ValueError):
                    reference.load_frontend_config(path)


if __name__ == "__main__":
    unittest.main()
