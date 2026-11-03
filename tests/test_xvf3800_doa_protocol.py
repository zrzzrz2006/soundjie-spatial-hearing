import json
import math
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "src" / "xvf3800_control_protocol.cpp",
    ROOT / "src" / "xvf3800_doa_cli.cpp",
]
BUILD_SCRIPT = ROOT / "scripts" / "guest" / "build_xvf3800_doa_tool.sh"
LICENSE_NOTE = ROOT / "docs" / "licenses" / "respeaker-xvf3800-source.md"


def wsl_path(path: Path) -> str:
    path = Path(path).resolve()
    drive = path.drive[0].lower()
    tail = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


class Xvf3800DoaProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.binary = cls.root / "xvf3800_doa"
        for source in SOURCES:
            if not source.is_file():
                raise AssertionError(f"missing source: {source}")
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
                f"native DoA build failed\nstdout={result.stdout}\nstderr={result.stderr}"
            )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_cli(self, *args):
        return subprocess.run(
            ["wsl.exe", wsl_path(self.binary), *map(str, args)],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

    def test_bounded_fixture_keeps_latest_complete_sample(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); src=root/'sequence'; out=root/'doa.jsonl'
            src.write_text('005a000100\n'*100)
            r=self.run_cli('--fixture-sequence',wsl_path(src),'--jsonl',wsl_path(out),
                           '--log-max-bytes','4096')
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertLessEqual(out.stat().st_size,4096)
            rows=[json.loads(s) for s in out.read_text().splitlines()]
            self.assertEqual(rows[-1]['monotonic_ms'],9900)
            self.assertTrue(rows[-1]['valid'])

    def decode_doa(self, value):
        result = self.run_cli("--decode-doa-hex", value)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_parses_five_byte_little_endian_doa_response(self):
        payload = self.decode_doa("0028010100")
        self.assertEqual(payload["protocol"], "DOA_VALUE")
        self.assertEqual(payload["status"], 0)
        self.assertEqual(payload["raw_angle_deg"], 296)
        self.assertEqual(payload["angle_deg"], 296)
        self.assertTrue(payload["speech_detected"])
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["reason"], "ok")

    def test_rejects_nonzero_status_short_response_and_angle_over_359(self):
        cases = {
            "0228010100": "status_2",
            "00": "wrong_length",
            "0068010100": "angle_out_of_range",
        }
        for raw, reason in cases.items():
            with self.subTest(raw=raw):
                payload = self.decode_doa(raw)
                self.assertFalse(payload["valid"])
                self.assertIsNone(payload["angle_deg"])
                self.assertEqual(payload["reason"], reason)

    def test_marks_no_speech_invalid(self):
        payload = self.decode_doa("0028010000")
        self.assertEqual(payload["raw_angle_deg"], 296)
        self.assertFalse(payload["speech_detected"])
        self.assertFalse(payload["valid"])
        self.assertIsNone(payload["angle_deg"])
        self.assertEqual(payload["reason"], "no_speech")

    def test_retries_only_documented_retry_status(self):
        retry = self.decode_doa("4028010100")
        unknown = self.decode_doa("4128010100")
        self.assertTrue(retry["retryable"])
        self.assertEqual(retry["reason"], "retry")
        self.assertFalse(unknown["retryable"])
        self.assertEqual(unknown["reason"], "status_65")

    def test_aec_azimuth_parser_rejects_nan_and_wrong_length(self):
        valid = struct.pack("<B4f", 0, 0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
        result = self.run_cli("--decode-aec-hex", valid.hex())
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["valid"])
        for actual, expected in zip(payload["degrees"], [0, 90, 180, 270]):
            self.assertAlmostEqual(actual, expected, places=3)

        invalid = struct.pack("<B4f", 0, 0.0, float("nan"), 1.0, 2.0)
        for raw, reason in ((invalid.hex(), "nonfinite_angle"), ("00", "wrong_length")):
            with self.subTest(reason=reason):
                result = self.run_cli("--decode-aec-hex", raw)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertFalse(payload["valid"])
                self.assertEqual(payload["reason"], reason)

    def test_stale_identical_angle_without_speech_is_not_validated(self):
        fixture = self.root / "no-speech-sequence.txt"
        output = self.root / "no-speech.jsonl"
        fixture.write_text("0028010000\n0028010000\n0028010000\n", encoding="ascii")
        result = self.run_cli(
            "--fixture-sequence", wsl_path(fixture), "--jsonl", wsl_path(output)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = [json.loads(row) for row in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(not row["valid"] for row in rows))
        self.assertTrue(all(row["angle_deg"] is None for row in rows))
        self.assertTrue(all(row["reason"] == "no_speech" for row in rows))

    def test_cli_never_saves_or_clears_configuration(self):
        combined = "\n".join(
            source.read_text(encoding="utf-8") for source in SOURCES
        )
        self.assertNotIn("SAVE_CONFIGURATION", combined)
        self.assertNotIn("CLEAR_CONFIGURATION", combined)
        self.assertNotIn("firmware", combined.lower())
        self.assertTrue(BUILD_SCRIPT.is_file())
        self.assertTrue(LICENSE_NOTE.is_file())
        note = LICENSE_NOTE.read_text(encoding="utf-8")
        self.assertIn("https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY", note)
        self.assertIn("independent implementation", note.lower())

    def test_jsonl_is_create_only(self):
        fixture = self.root / "create-only-sequence.txt"
        output = self.root / "owned.jsonl"
        fixture.write_text("0028010100\n", encoding="ascii")
        output.write_text("owned-by-user\n", encoding="utf-8")
        result = self.run_cli(
            "--fixture-sequence", wsl_path(fixture), "--jsonl", wsl_path(output)
        )
        self.assertEqual(result.returncode, 3)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(output.read_text(encoding="utf-8"), "owned-by-user\n")

    def test_arm_build_report_audits_dynamic_dependencies(self):
        script = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("DT_NEEDED=", script)
        self.assertIn("linux_usbfs_control", script)
        self.assertNotIn("-lusb", script)


if __name__ == "__main__":
    unittest.main()
