import math
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "wav_probe.c"


class WavProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temp.name) / "wav_probe"
        result = subprocess.run(
            ["wsl.exe", "bash", "-lc", f"gcc -std=c99 -O2 -Wall -Wextra -o /tmp/wav_probe_test /mnt/f/MLK/DPU_ZYNQ7000/usb_mic_bringup_v1/src/wav_probe.c && cp /tmp/wav_probe_test /mnt/{cls.binary.drive.rstrip(':').lower()}{cls.binary.as_posix().split(':',1)[1]}"],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        if result.returncode != 0:
            raise AssertionError(result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def make_wav(self, path: Path, silent: bool = False):
        rate = 16000
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            frames = bytearray()
            for index in range(rate // 4):
                value = 0 if silent else int(8000 * math.sin(2 * math.pi * 440 * index / rate))
                frames += struct.pack("<hh", value, value)
            wav.writeframes(frames)

    def run_probe(self, path: Path):
        return subprocess.run(
            [
                "wsl.exe",
                "bash",
                "-lc",
                f"{self.to_wsl(self.binary)} {self.to_wsl(path)}",
            ],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

    @staticmethod
    def to_wsl(path: Path) -> str:
        drive = path.drive.rstrip(":").lower()
        suffix = path.as_posix().split(":", 1)[1]
        return f"/mnt/{drive}{suffix}"

    def test_valid_wav_reports_metadata_and_signal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "valid.wav"
            self.make_wav(path)
            result = self.run_probe(path)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("CHANNELS=2", result.stdout)
            self.assertIn("SAMPLE_RATE=16000", result.stdout)
            self.assertIn("SIGNAL=present", result.stdout)

    def test_truncated_wav_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "truncated.wav"
            self.make_wav(path)
            path.write_bytes(path.read_bytes()[:-101])
            result = self.run_probe(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("truncated", result.stderr.lower())

    def test_silent_wav_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "silent.wav"
            self.make_wav(path, silent=True)
            result = self.run_probe(path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SIGNAL=silent", result.stdout)


if __name__ == "__main__":
    unittest.main()
