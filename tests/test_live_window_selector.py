import csv
import math
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "soundjie_live_v1.tsv"
SOURCES = [
    ROOT / "src" / "wav_pcm.cpp",
    ROOT / "src" / "live_window_selector.cpp",
    ROOT / "src" / "live_window_selector_cli.cpp",
]


def wsl_path(path: Path) -> str:
    path = Path(path).resolve()
    drive = path.drive[0].lower()
    tail = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def sine_samples(count, amplitude=0.1, frequency=440.0):
    return [
        int(round(32767.0 * amplitude * math.sin(2.0 * math.pi * frequency * index / 16000.0)))
        for index in range(count)
    ]


def write_wav(path: Path, left, right=None):
    channels = 1 if right is None else 2
    if right is not None and len(left) != len(right):
        raise ValueError("channel lengths differ")
    frames = bytearray()
    for index, sample in enumerate(left):
        frames.extend(struct.pack("<h", sample))
        if right is not None:
            frames.extend(struct.pack("<h", right[index]))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes(frames))


def read_tsv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


class LiveWindowSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.build_root = Path(cls.temporary.name)
        cls.binary = cls.build_root / "live_window_selector"
        command = [
            "wsl.exe", "g++", "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
            f"-I{wsl_path(ROOT / 'src')}",
            *(wsl_path(source) for source in SOURCES),
            "-o", wsl_path(cls.binary),
        ]
        result = subprocess.run(
            command, text=True, encoding="utf-8", errors="replace", capture_output=True
        )
        if result.returncode != 0:
            raise AssertionError(
                f"native selector build failed\nstdout={result.stdout}\nstderr={result.stderr}"
            )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.case = Path(self.build_root) / self._testMethodName
        self.case.mkdir()
        self.wav = self.case / "capture.wav"
        self.metrics = self.case / "window-metrics.tsv"
        self.selected = self.case / "selected-windows.tsv"

    def run_selector(self, wav=None, metrics=None, selected=None):
        return subprocess.run(
            [
                "wsl.exe", wsl_path(self.binary),
                "--wav", wsl_path(wav or self.wav),
                "--config", wsl_path(CONFIG),
                "--metrics", wsl_path(metrics or self.metrics),
                "--selected", wsl_path(selected or self.selected),
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

    def test_enumerates_21_windows_for_six_seconds(self):
        values = {}
        for line in CONFIG.read_text(encoding="utf-8").splitlines():
            key, value = line.split("\t")
            values[key] = value
        self.assertEqual(values, {
            "capture_seconds": "6",
            "window_samples": "16000",
            "hop_samples": "4000",
            "min_rms": "0.005",
            "clip_abs_threshold": "0.99",
            "max_clipping_rate": "0.01",
            "max_windows": "3",
            "min_start_gap_samples": "8000",
        })
        write_wav(self.wav, sine_samples(96000))
        result = self.run_selector()
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = read_tsv(self.metrics)
        self.assertEqual(len(rows), 21)
        self.assertEqual([int(row["start_sample"]) for row in rows], list(range(0, 80001, 4000)))

    def test_rejects_silence_and_excessive_clipping(self):
        write_wav(self.wav, [0] * 96000)
        result = self.run_selector()
        self.assertEqual(result.returncode, 2, result.stderr)
        rows = read_tsv(self.metrics)
        self.assertEqual({row["rejection"] for row in rows}, {"below_min_rms"})
        self.assertEqual(read_tsv(self.selected), [])

        clipped_wav = self.case / "clipped.wav"
        clipped_metrics = self.case / "clipped-metrics.tsv"
        clipped_selected = self.case / "clipped-selected.tsv"
        write_wav(clipped_wav, [32767] * 96000)
        clipped = self.run_selector(clipped_wav, clipped_metrics, clipped_selected)
        self.assertEqual(clipped.returncode, 2, clipped.stderr)
        clipped_rows = read_tsv(clipped_metrics)
        self.assertEqual({row["rejection"] for row in clipped_rows}, {"excessive_clipping"})
        self.assertEqual(read_tsv(clipped_selected), [])

    def test_selects_at_most_three_windows_with_8000_sample_gap(self):
        write_wav(self.wav, sine_samples(96000))
        result = self.run_selector()
        self.assertEqual(result.returncode, 0, result.stderr)
        starts = [int(row["start_sample"]) for row in read_tsv(self.selected)]
        self.assertEqual(len(starts), 3)
        self.assertTrue(all(abs(left - right) >= 8000 for left in starts for right in starts if left != right))

    def test_selects_tail_window_without_off_by_one(self):
        samples = [0] * 80000 + sine_samples(16000, amplitude=0.3)
        write_wav(self.wav, samples)
        result = self.run_selector()
        self.assertEqual(result.returncode, 0, result.stderr)
        selected = read_tsv(self.selected)
        self.assertEqual(int(selected[0]["start_sample"]), 80000)
        metrics = read_tsv(self.metrics)
        self.assertEqual(int(metrics[-1]["start_sample"]), 80000)
        self.assertEqual(metrics[-1]["selected_rank"], "1")

    def test_rejects_stereo_antiphase_after_contract_mix(self):
        left = sine_samples(96000, amplitude=0.2)
        right = [-sample for sample in left]
        write_wav(self.wav, left, right)
        result = self.run_selector()
        self.assertEqual(result.returncode, 2, result.stderr)
        rows = read_tsv(self.metrics)
        self.assertEqual({row["rejection"] for row in rows}, {"below_min_rms"})
        self.assertTrue(all(float(row["rms"]) == 0.0 for row in rows))

    def test_tie_breaks_by_start_sample(self):
        write_wav(self.wav, sine_samples(96000, amplitude=0.2, frequency=400.0))
        result = self.run_selector()
        self.assertEqual(result.returncode, 0, result.stderr)
        selected = read_tsv(self.selected)
        self.assertEqual([int(row["start_sample"]) for row in selected], [0, 8000, 16000])
        self.assertEqual([int(row["rank"]) for row in selected], [1, 2, 3])

    def test_refuses_existing_outputs(self):
        write_wav(self.wav, sine_samples(96000))
        self.metrics.write_text("owned-by-user\n", encoding="utf-8")
        result = self.run_selector()
        self.assertEqual(result.returncode, 3)
        self.assertIn("output already exists", result.stderr)
        self.assertEqual(self.metrics.read_text(encoding="utf-8"), "owned-by-user\n")
        self.assertFalse(self.selected.exists())


if __name__ == "__main__":
    unittest.main()
