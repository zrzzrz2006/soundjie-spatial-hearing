import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "board" / "run_soundjie_d3_live_wav.sh"


class LiveWavEntrypointTests(unittest.TestCase):
    def run_script(self, *args):
        return subprocess.run(["C:/Program Files/Git/bin/sh.exe", str(SCRIPT), *map(str, args)],
                              text=True, capture_output=True, check=False)

    def test_wav_path_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            result = self.run_script("--bundle-dir", bundle, "--wav", "../escape.wav",
                                     "--output-dir", root / "out")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("REASON=unsafe_wav_member", result.stdout)
            self.assertIn("DPU_RUN_CALLS=0", result.stdout)

    def test_offline_wav_runs_logmel_and_exactly_one_dpu_call(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("arecord", text)
        self.assertNotIn("xvf3800_record", text)
        self.assertIn("demo-playback.tsv", text)
        self.assertIn("--describe-tensor", text)
        self.assertEqual(text.count('dpu_calls=$((dpu_calls + 1))'), 1)
        self.assertIn("DIRECT_OUTPUT_SHA256", text)

    def test_offline_result_matches_direct_tensor_for_fixed_demo(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("EXPECTED_TENSOR_SHA256", text)
        self.assertIn("direct_output_mismatch", text)
        self.assertIn("PREDICTION_MATCH", text)

    def test_faults_fail_before_dpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            for wav in ("demo/missing.wav", "/absolute.wav"):
                result = self.run_script("--bundle-dir", bundle, "--wav", wav,
                                         "--output-dir", root / ("out-" + str(len(wav))))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("DPU_RUN_CALLS=0", result.stdout)


if __name__ == "__main__":
    unittest.main()
