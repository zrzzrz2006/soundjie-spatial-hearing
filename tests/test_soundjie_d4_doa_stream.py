import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wsl_path(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


def row(timestamp, angle=90, speech=True):
    raw = bytes([0, angle & 255, angle >> 8, int(speech), 0]).hex()
    return json.dumps(dict(monotonic_ms=timestamp, protocol="DOA_VALUE", raw_hex=raw,
                          status=0, raw_angle_deg=angle,
                          angle_deg=angle if speech else None,
                          speech_detected=speech, valid=speech, retryable=False,
                          reason="ok" if speech else "no_speech"), separators=(",", ":"))


class D4DoaStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="doa-test-", dir=ROOT / ".superpowers")
        cls.binary = Path(cls.temporary.name) / "driver"
        sources = [ROOT / "tests/fixtures/d4_doa_stream_driver.cpp"] + [
            ROOT / "src" / name for name in (
                "soundjie_d4_doa_stream.cpp", "soundjie_d4_state.cpp",
                "xvf3800_control_protocol.cpp")]
        result = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "g++", "-std=c++11",
                                 "-O2", "-Wall", "-Wextra", "-Werror",
                                 "-I" + wsl_path(ROOT / "src"),
                                 *map(wsl_path, sources), "-o", wsl_path(cls.binary)],
                                capture_output=True)
        if result.returncode:
            cls.temporary.cleanup()
            raise AssertionError(result.stderr.decode("utf-8", errors="replace"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_stream(self, commands):
        with tempfile.TemporaryDirectory(dir=self.temporary.name) as folder:
            result = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", wsl_path(self.binary),
                                     wsl_path(Path(folder) / "doa.jsonl")],
                                    input=("\n".join(commands) + "\n").encode(),
                                    capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            return [json.loads(line) for line in result.stdout.decode().splitlines()]

    def assert_invalid(self, state, reason):
        self.assertFalse(state["doa_valid"])
        self.assertIsNone(state["doa_deg"])
        self.assertIsNone(state["doa_sector"])
        self.assertEqual(state["doa_reason"], reason)

    def test_actual_producer_row_and_stale_boundary(self):
        states = self.run_stream(["sample 1000 005a000100", "tick 1000", "tick 1750", "tick 1751"])
        self.assertEqual(states[0]["doa_deg"], 90)
        self.assertEqual(states[0]["doa_sector"], "right")
        self.assertTrue(states[1]["doa_valid"])
        self.assert_invalid(states[2], "stale")

    def test_poll_and_duplicate_do_not_resmooth_or_refresh_sample_age(self):
        states = self.run_stream(["sample 1000 0000000100", "tick 1000",
                                 "sample 1100 005a000100", "tick 1100", "tick 1200",
                                 "sample 1100 005a000100", "tick 1300", "tick 1851"])
        for state in states[1:4]:
            self.assertAlmostEqual(state["doa_deg"], 45, places=4)
        self.assert_invalid(states[4], "stale")

    def test_circular_wrap_and_new_direction_after_long_silence(self):
        states = self.run_stream(["sample 1000 0067010100", "tick 1000",
                                 "sample 1100 0001000100", "tick 1100", "tick 1900",
                                 "sample 2000 00b4000100", "tick 2000"])
        self.assertLess(min(abs(states[1]["doa_deg"]), abs(states[1]["doa_deg"] - 360)), .01)
        self.assert_invalid(states[2], "stale")
        self.assertAlmostEqual(states[3]["doa_deg"], 180)

    def test_no_speech_retry_and_malformed_immediately_clear_direction(self):
        for command, reason in [("sample 1100 005a000000", "no_speech"),
                                ("sample 1100 4000000000", "retry"),
                                ('append {"valid":true}', "malformed_row"),
                                ("pad 10000", "oversized_row")]:
            with self.subTest(command=command):
                states = self.run_stream(["sample 1000 005a000100", "tick 1000", command, "tick 1100"])
                self.assert_invalid(states[-1], reason)

    def test_truncated_write_is_not_consumed_until_newline(self):
        line = row(1100, 180)
        states = self.run_stream(["sample 1000 005a000100", "tick 1000",
                                 "fragment " + line[:55], "tick 1100",
                                 "append " + line[55:], "tick 1200"])
        self.assertEqual(states[1]["doa_deg"], 90)
        self.assertAlmostEqual(states[2]["doa_deg"], 135)

    def test_missing_truncated_or_rotated_log_drops_old_filter(self):
        for command, reason in [("remove", "source_unavailable"), ("truncate", "awaiting_sample"),
                                ("replace " + row(1100, 180), None)]:
            with self.subTest(command=command):
                states = self.run_stream(["sample 1000 005a000100", "tick 1000", command, "tick 1100"])
                if reason:
                    self.assert_invalid(states[-1], reason)
                else:
                    self.assertEqual(states[-1]["doa_deg"], 180)

    def test_future_and_out_of_order_never_become_valid_on_later_poll(self):
        states = self.run_stream(["sample 1000 005a000100", "tick 1000",
                                 "sample 999 00b4000100", "tick 1100",
                                 "sample 2000 00b4000100", "tick 1200", "tick 2000",
                                 "sample 2100 00b4000100", "tick 2100"])
        self.assert_invalid(states[1], "out_of_order")
        self.assert_invalid(states[2], "future_sample")
        self.assert_invalid(states[3], "future_sample")
        self.assertEqual(states[4]["doa_deg"], 180)

    def test_conflicting_duplicate_and_forged_fields_fail_closed(self):
        for bad, reason in [(row(1000, 180), "conflicting_timestamp"),
                             (row(1100).replace('"valid":true', '"valid":false'), "malformed_row"),
                             (row(1100)[:-1] + ',"valid":false}', "malformed_row"),
                             (row(1100).replace('1100', '18446744073709551616'), "malformed_row")]:
            with self.subTest(bad=bad):
                states = self.run_stream(["sample 1000 005a000100", "tick 1000", "append " + bad, "tick 1200"])
                self.assert_invalid(states[-1], reason)

    def test_large_backlog_is_bounded_and_latest_complete_sample_wins(self):
        states = self.run_stream(["pad 2000000", "sample 1000 005a000100", "tick 1000"])
        self.assertTrue(states[-1]["doa_valid"])
        self.assertEqual(states[-1]["doa_deg"], 90)

    def test_clock_is_sampled_after_bounded_read_not_before_it(self):
        states = self.run_stream(["sample 900 0000000100", "clock-race", "tick 1001"])
        self.assertTrue(states[0]["doa_valid"])
        self.assertEqual(states[0]["doa_deg"], 0)
        self.assertTrue(states[1]["doa_valid"])
        self.assertAlmostEqual(states[1]["doa_deg"], 45)


if __name__ == "__main__":
    unittest.main()
