import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/host/complete_sd_directory_repair.py"


class SdRepairOrchestrationTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("sd_repair_orchestration", SCRIPT)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_repair_failure_stops_without_retry_or_second_child(self):
        with patch.object(self.mod.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run:
            self.assertEqual(self.mod.main(), 1)
        self.assertEqual(run.call_count, 1)
        args = run.call_args.args[0]
        self.assertIn("--apply-confirmed", args)
        self.assertIn("--resume-second", args)
        self.assertIn("sd-boot-repair-20261024-remaining", args)

    def test_success_opens_second_child_readonly_with_fixed_new_name(self):
        with patch.object(self.mod.subprocess, "run", side_effect=[
            SimpleNamespace(returncode=0), SimpleNamespace(returncode=0),
        ]) as run:
            self.assertEqual(self.mod.main(), 0)
        self.assertEqual(run.call_count, 2)
        second = run.call_args_list[1].args[0]
        self.assertEqual(Path(second[2]).name, "backup_sd_boot_readonly.py")
        self.assertEqual(second[3:], ["--capture-name", "sd-boot-readonly-20261024-final"])

    def test_readonly_capture_failure_cannot_return_success(self):
        with patch.object(self.mod.subprocess, "run", side_effect=[
            SimpleNamespace(returncode=0), SimpleNamespace(returncode=3),
        ]) as run:
            self.assertEqual(self.mod.main(), 3)
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
