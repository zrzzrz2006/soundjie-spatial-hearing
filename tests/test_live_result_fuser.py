import csv
import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS_EN = ROOT / "models" / "soundjie_d3" / "labels.txt"
LABELS_ZH = ROOT / "models" / "soundjie_d3" / "display_labels_zh.txt"
SOURCES = [
    ROOT / "src" / "live_result_fuser.cpp",
    ROOT / "src" / "live_result_fuser_cli.cpp",
]
CLASS_IDS = LABELS_EN.read_text(encoding="utf-8").splitlines()


def wsl_path(path: Path) -> str:
    path = Path(path).resolve()
    drive = path.drive[0].lower()
    tail = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def write_predictions(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "rank", "start_sample", "runtime_us", "raw_logits_csv",
            "input_sha256", "output_sha256",
        ])
        for rank, logits in enumerate(rows, 1):
            writer.writerow([
                rank, (rank - 1) * 8000, 600 + rank,
                ",".join(str(value) for value in logits),
                f"input-hash-{rank}", f"output-hash-{rank}",
            ])


class LiveResultFuserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build_root = Path(cls.temporary.name)
        cls.binary = cls.build_root / "live_result_fuser"
        result = subprocess.run(
            [
                "wsl.exe", "g++", "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
                f"-I{wsl_path(ROOT / 'src')}",
                *(wsl_path(source) for source in SOURCES),
                "-o", wsl_path(cls.binary),
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"native fuser build failed\nstdout={result.stdout}\nstderr={result.stderr}"
            )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.case = self.build_root / self._testMethodName
        self.case.mkdir()
        self.predictions = self.case / "window-predictions.tsv"
        self.result_json = self.case / "session-result.json"
        self.display = self.case / "display-status.txt"

    def run_fuser(self, expected=None, predictions=None, labels_zh=None,
                  result_json=None, display=None):
        command = [
            "wsl.exe", wsl_path(self.binary),
            "--predictions", wsl_path(predictions or self.predictions),
            "--labels-en", wsl_path(LABELS_EN),
            "--labels-zh", wsl_path(labels_zh or LABELS_ZH),
            "--output-scale", "0.0625",
            "--session-result", wsl_path(result_json or self.result_json),
            "--display-status", wsl_path(display or self.display),
        ]
        if expected is not None:
            command += ["--expected-class", expected]
        return subprocess.run(
            command, text=True, encoding="utf-8", errors="replace", capture_output=True
        )

    def load_result(self):
        return json.loads(self.result_json.read_text(encoding="utf-8"))

    def test_averages_dequantized_logits_before_softmax(self):
        write_predictions(self.predictions, [
            [8, 0, -8, 4, 0, 0, 0, 0],
            [0, 8, 0, 4, 0, 0, 0, 0],
        ])
        result = self.run_fuser()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = self.load_result()
        self.assertEqual(payload["fused_logits"], [0.25, 0.25, -0.25, 0.25, 0, 0, 0, 0])
        denominator = 3 * math.exp(0.25) + math.exp(-0.25) + 4
        self.assertAlmostEqual(payload["top3"][0]["confidence"], math.exp(0.25) / denominator, places=12)
        self.assertEqual(len(payload["windows"]), 2)
        self.assertEqual(payload["windows"][0]["raw_logits"], [8, 0, -8, 4, 0, 0, 0, 0])

    def test_emits_stable_top3_and_prediction_match(self):
        write_predictions(self.predictions, [[0, 1, 2, 9, 4, 3, -1, -2]])
        result = self.run_fuser(expected="glass_breaking")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = self.load_result()
        self.assertEqual(payload["LIVE_STATUS"], "pass")
        self.assertEqual(payload["PREDICTION_MATCH"], "yes")
        self.assertEqual(payload["top1"]["id"], "glass_breaking")
        self.assertEqual([item["id"] for item in payload["top3"]], [
            "glass_breaking", "door_knock_bell", "distress_voice",
        ])
        self.assertEqual(payload["DPU_RUN_CALLS"], 1)

    def test_prediction_mismatch_is_not_engineering_failure(self):
        write_predictions(self.predictions, [[9, 0, 0, 0, 0, 0, 0, 0]])
        result = self.run_fuser(expected="glass_breaking")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = self.load_result()
        self.assertEqual(payload["LIVE_STATUS"], "pass")
        self.assertEqual(payload["PREDICTION_MATCH"], "no")
        self.assertEqual(payload["expected_class"], "glass_breaking")
        display = self.display.read_text(encoding="utf-8")
        self.assertIn("PREDICTION_MATCH=no\n", display)
        self.assertIn("ERROR=none\n", display)

    def test_rejects_zero_more_than_three_or_non_eight_logits(self):
        invalid_rows = [[], [[0] * 8] * 4, [[0] * 7]]
        for index, rows in enumerate(invalid_rows):
            with self.subTest(index=index):
                predictions = self.case / f"invalid-{index}.tsv"
                output = self.case / f"invalid-{index}.json"
                display = self.case / f"invalid-{index}.txt"
                write_predictions(predictions, rows)
                result = self.run_fuser(
                    predictions=predictions, result_json=output, display=display
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())
                self.assertFalse(display.exists())

    def test_tie_breaks_by_class_index(self):
        write_predictions(self.predictions, [[4] * 8])
        result = self.run_fuser()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = self.load_result()
        self.assertEqual([item["index"] for item in payload["top3"]], [0, 1, 2])
        self.assertEqual(payload["top1"]["id"], CLASS_IDS[0])

    def test_writes_valid_utf8_json_and_keeps_machine_id(self):
        special_labels = self.case / "special-labels-zh.txt"
        labels = LABELS_ZH.read_text(encoding="utf-8").splitlines()
        labels[5] = '求救"人声\\测试'
        special_labels.write_text("\n".join(labels) + "\n", encoding="utf-8")
        write_predictions(self.predictions, [[0, 0, 0, 0, 0, 12, 0, 0]])
        result = self.run_fuser(labels_zh=special_labels)
        self.assertEqual(result.returncode, 0, result.stderr)
        raw = self.result_json.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["top1"]["id"], "distress_voice")
        self.assertEqual(payload["top1"]["label_zh"], '求救"人声\\测试')
        self.assertEqual(payload["windows"][0]["input_sha256"], "input-hash-1")
        self.assertEqual(payload["windows"][0]["output_sha256"], "output-hash-1")

    def test_refuses_existing_outputs(self):
        write_predictions(self.predictions, [[1, 0, 0, 0, 0, 0, 0, 0]])
        self.result_json.write_text("owned-by-user\n", encoding="utf-8")
        result = self.run_fuser()
        self.assertEqual(result.returncode, 3)
        self.assertIn("output already exists", result.stderr)
        self.assertEqual(self.result_json.read_text(encoding="utf-8"), "owned-by-user\n")
        self.assertFalse(self.display.exists())


if __name__ == "__main__":
    unittest.main()
