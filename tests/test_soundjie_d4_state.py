import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "src" / "soundjie_d4_state.cpp",
    ROOT / "src" / "soundjie_d4_state_cli.cpp",
]


def wsl_path(path: Path) -> str:
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


class SoundjieD4StateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temporary.name) / "soundjie_d4_state"
        result = subprocess.run(
            [
                "wsl.exe", "g++", "-std=c++11", "-O2", "-Wall", "-Wextra",
                "-Werror", f"-I{wsl_path(ROOT / 'src')}",
                *(wsl_path(source) for source in SOURCES),
                "-o", wsl_path(cls.binary),
            ],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"state build failed\nstdout={result.stdout}\nstderr={result.stderr}"
            )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_json(self, *args):
        result = subprocess.run(
            ["wsl.exe", wsl_path(self.binary), *map(str, args)],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_circular_mean_handles_359_and_1(self):
        payload = self.run_json("--smooth", "359,1", "--alpha", "0.5")
        self.assertTrue(payload["valid"])
        self.assertLess(min(abs(payload["angle_deg"]), abs(payload["angle_deg"] - 360)), 0.1)

    def test_wraps_negative_and_over_360(self):
        self.assertAlmostEqual(self.run_json("--wrap", "-1")["angle_deg"], 359.0)
        self.assertAlmostEqual(self.run_json("--wrap", "721")["angle_deg"], 1.0)

    def test_invalid_or_stale_doa_clears_angle_and_sector(self):
        invalid = self.run_json(
            "--doa-sequence", "0:90:1:ok,100:90:0:no_speech", "--now", "100"
        )
        self.assertFalse(invalid["doa_valid"])
        self.assertIsNone(invalid["doa_deg"])
        self.assertIsNone(invalid["doa_sector"])
        self.assertEqual(invalid["doa_reason"], "no_speech")

        stale = self.run_json(
            "--doa-sequence", "0:90:1:ok", "--now", "1000", "--stale-ms", "750"
        )
        self.assertFalse(stale["doa_valid"])
        self.assertIsNone(stale["doa_deg"])
        self.assertIsNone(stale["doa_sector"])
        self.assertEqual(stale["doa_reason"], "stale")

    def test_eight_sector_boundaries_are_stable(self):
        cases = {
            "0": "front",
            "22.499": "front",
            "22.5": "front_right",
            "67.5": "right",
            "157.5": "rear",
            "247.5": "left",
            "337.499": "front_left",
            "337.5": "front",
        }
        for angle, expected in cases.items():
            with self.subTest(angle=angle):
                self.assertEqual(self.run_json("--sector", angle)["sector"], expected)

    def test_silence_dbfs_is_finite_floor(self):
        payload = self.run_json("--pcm-s16", "0,0,0,0", "--dbfs-floor", "-96")
        self.assertEqual(payload["rms"], 0.0)
        self.assertEqual(payload["dbfs"], -96.0)
        self.assertTrue(math.isfinite(payload["dbfs"]))
        self.assertEqual(payload["audio_state"], "silence")

    def test_full_scale_counts_clipping(self):
        payload = self.run_json("--pcm-s16", "32767,-32768,100,0")
        self.assertAlmostEqual(payload["peak"], 1.0, places=6)
        self.assertAlmostEqual(payload["clipping_rate"], 0.5, places=6)
        self.assertEqual(payload["audio_state"], "clipping")

    def test_rms_smoothing_has_bounded_attack_and_release(self):
        payload = self.run_json(
            "--rms-sequence", "0,1,0", "--attack", "0.5", "--release", "0.25"
        )
        self.assertEqual(payload["values"], [0.0, 0.5, 0.375])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in payload["values"]))

    def test_utf8_snapshot_is_valid_and_complete(self):
        payload = self.run_json("--snapshot-demo")
        expected = {
            "sequence", "timestamp_ms", "class_id", "label_zh", "confidence", "top3",
            "doa_raw_deg", "doa_deg", "doa_sector", "doa_valid", "doa_reason", "rms",
            "dbfs", "peak", "clipping_rate", "audio_state", "usb_ok", "dpu_ok",
            "framebuffer_ok", "inference_ms", "end_to_end_ms",
            "class_scores", "top2_margin", "capture_file", "window_end_sample",
            "tv_enabled", "tv_matched", "tv_position_seconds", "tv_score",
            "caption_text", "caption_speaker", "caption_cue", "fine_valid",
            "ambulance_score", "road_score", "speech_score",
        }
        self.assertEqual(set(payload), expected)
        self.assertFalse(payload['tv_enabled'])
        self.assertFalse(payload['fine_valid'])
        self.assertEqual(payload['caption_text'], '')
        self.assertEqual(payload['class_scores'], [])
        self.assertIsNone(payload['top2_margin'])
        self.assertIsNone(payload['capture_file'])
        self.assertEqual(payload["label_zh"], "婴儿哭声")
        self.assertEqual(len(payload["top3"]), 3)
        self.assertTrue(payload["doa_valid"])
        self.assertEqual(payload["doa_sector"], "front_right")

    def test_runtime_thresholds_load_from_versioned_tsv(self):
        payload = self.run_json(
            "--config", wsl_path(ROOT / "config" / "soundjie_d4_runtime.tsv")
        )
        self.assertEqual(payload["doa_stale_ms"], 750)
        self.assertEqual(payload["dbfs_floor"], -96.0)
        self.assertEqual(payload["sector_count"], 8)
        self.assertAlmostEqual(payload["clip_threshold"], 0.999)


if __name__ == "__main__":
    unittest.main()
