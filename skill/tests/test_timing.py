"""Behavior checks using synthetic reports; no FPGA tool or hardware needed."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
SCRIPT = TESTS.parent / "timing-report-audit/scripts/audit_timing.py"
SPEC = importlib.util.spec_from_file_location("audit_timing", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)

CLEAR = """
1. checking no_clock
--------------------
There are 0 register/latch pins with no clock.
4. checking unconstrained_internal_endpoints
--------------------------------------------
There are 0 pins that are not constrained for maximum delay.
5. checking no_input_delay
--------------------------
There are 0 input ports with no input delay specified.
6. checking no_output_delay
---------------------------
There are 0 ports with no output delay specified.
"""
GOOD = "WNS=0.254\nTNS=0\nWHS=0.080\nTHS=0\n"


def table(columns):
    """Make a realistic fixed-width report, including optional blank cells."""
    widths = [max(len(label), len(value), 9) for label, value in columns]
    return "\n".join([
        "| Design Timing Summary",
        "  ".join(label.rjust(width) for (label, _), width in zip(columns, widths)),
        "  ".join("-" * width for width in widths),
        "  ".join(value.rjust(width) for (_, value), width in zip(columns, widths)),
    ])


class TimingTests(unittest.TestCase):
    def test_positive_summary_is_only_a_report_check(self):
        result = AUDIT.audit_text(CLEAR + table([
            ("WNS(ns)", ".254"), ("TNS(ns)", "0"),
            ("TNS Failing Endpoints", "0"), ("TNS Total Endpoints", "123"),
            ("WHS(ns)", ".08"), ("THS(ns)", "0"),
            ("THS Failing Endpoints", "0"), ("THS Total Endpoints", "123"),
        ]))
        self.assertEqual(result["status"], "REPORT_CHECK_PASS")
        self.assertEqual(result["metrics_ns"], {"WNS": .254, "TNS": 0., "WHS": .08, "THS": 0.})
        self.assertFalse(result["signoff"])

    def test_each_negative_metric_blocks_pass_even_with_success_text(self):
        for key, original in (("WNS", "0.254"), ("TNS", "0"), ("WHS", "0.080"), ("THS", "0")):
            with self.subTest(metric=key):
                report = GOOD.replace(f"{key}={original}", f"{key}=-0.1")
                result = AUDIT.audit_text(CLEAR + report + "All user specified timing constraints are met.\n")
                self.assertEqual(result["status"], "VIOLATIONS")
                self.assertIn(key, result["violations"])

    def test_absent_hold_or_coverage_is_not_pass(self):
        result = AUDIT.audit_text(CLEAR + "WNS=.254\nTNS=0\nRESULT=PASS\n")
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["missing_metrics"], ["WHS", "THS"])
        self.assertIsNone(result["metrics_ns"]["WHS"])
        result = AUDIT.audit_text(GOOD)
        self.assertEqual(result["coverage"], "unknown")
        self.assertEqual(result["status"], "INCOMPLETE")
        partial = CLEAR.replace("maximum delay.", "maximum delay due to constant clock.")
        self.assertEqual(AUDIT.audit_text(partial + GOOD)["coverage"], "unknown")

    def test_unconstrained_and_other_check_findings_block_pass(self):
        reports = [CLEAR.replace("There are 0 pins", "There are 5 pins"),
                   CLEAR + "Unconstrained paths: 3\n",
                   CLEAR + "9. checking loops\n---\nThere is 1 combinational loop.\n"]
        for report in reports:
            with self.subTest(report=report):
                self.assertEqual(AUDIT.audit_text(report + GOOD)["status"], "UNCONSTRAINED")

    def test_blank_hold_cell_does_not_shift_pulse_width_into_hold(self):
        result = AUDIT.audit_text(CLEAR + table([
            ("WNS(ns)", ".254"), ("TNS(ns)", "0"),
            ("WHS(ns)", ""), ("THS(ns)", "NA"),
            ("WPWS(ns)", "1.267"), ("TPWS(ns)", "0"),
        ]))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["missing_metrics"], ["WHS", "THS"])

    def test_invalid_ambiguous_or_empty_reports_never_pass(self):
        for report in ("", "RESULT=PASS", GOOD + "WNS=1\n", GOOD.replace("WNS=0.254", "WNS=NaN"),
                       GOOD.replace("TNS=0", "TNS=1"), GOOD + "WPWS=-.1\n",
                       GOOD.replace("WNS=0.254", "WNS=-1e-999"),
                       GOOD + "Unconstrained paths: unknown\n"):
            with self.subTest(report=report):
                self.assertNotEqual(AUDIT.audit_text(CLEAR + report)["status"], "REPORT_CHECK_PASS")
        empty_design = table([("WNS(ns)", "0"), ("TNS(ns)", "0"),
                              ("TNS Total Endpoints", "0"), ("WHS(ns)", "0"), ("THS(ns)", "0")])
        self.assertEqual(AUDIT.audit_text(CLEAR + empty_design)["status"], "INCOMPLETE")

    def test_cli_arbitrary_path_and_missing_report(self):
        with tempfile.TemporaryDirectory(prefix="timing-", dir=TESTS) as directory:
            report = Path(directory) / "arbitrary report.rpt"
            report.write_text(CLEAR + GOOD, encoding="utf-8")
            command = [sys.executable, "-B", str(SCRIPT), str(report)]
            success = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(json.loads(success.stdout)["status"], "REPORT_CHECK_PASS")
            report.unlink()
            missing = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(missing.returncode, 2)
            self.assertEqual(json.loads(missing.stdout)["status"], "UNREADABLE_REPORT")


if __name__ == "__main__":
    unittest.main()
