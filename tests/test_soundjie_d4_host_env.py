import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "host" / "bootstrap_soundjie_d4_env.ps1"
REQUIREMENTS = ROOT / "config" / "soundjie_d4_host_requirements.txt"


class SoundjieD4HostEnvironmentTests(unittest.TestCase):
    def script_text(self):
        self.assertTrue(SCRIPT.is_file(), f"missing bootstrap script: {SCRIPT}")
        return SCRIPT.read_text(encoding="utf-8")

    def requirements(self):
        self.assertTrue(REQUIREMENTS.is_file(), f"missing requirements: {REQUIREMENTS}")
        rows = []
        for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                rows.append(line)
        return rows

    def test_bootstrap_pins_normal_cpython_not_py_launcher(self):
        text = self.script_text()
        self.assertIn(r"D:\Python3.8\python\python.exe", text)
        self.assertNotIn("py -3", text.lower())
        self.assertNotIn("python3.13t.exe", text.lower())
        self.assertIn("Py_GIL_DISABLED", text)

    def test_venv_path_is_on_f_drive(self):
        text = self.script_text()
        self.assertIn(r"F:\MLK\DPU_ZYNQ7000\envs\soundjie_d4_host", text)
        self.assertIn("GetPathRoot", text)
        self.assertIn("'F:\\'", text)

    def test_requirements_pin_numpy_scipy_soundfile_torch_pyserial(self):
        rows = self.requirements()
        packages = {row.split("==", 1)[0].lower(): row for row in rows}
        self.assertEqual(
            set(packages), {"numpy", "scipy", "soundfile", "torch", "pyserial"}
        )
        self.assertTrue(all("==" in row for row in rows))

    def test_git_safe_directory_is_process_local(self):
        text = self.script_text()
        self.assertIn("GIT_CONFIG_COUNT", text)
        self.assertIn("GIT_CONFIG_KEY_0", text)
        self.assertIn("GIT_CONFIG_VALUE_0", text)
        self.assertNotIn("git config --global", text.lower())

    def test_bootstrap_refuses_non_f_destination(self):
        self.assertTrue(SCRIPT.is_file(), f"missing bootstrap script: {SCRIPT}")
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-VenvPath",
                r"C:\soundjie-d4-forbidden",
                "-ValidateOnly",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("F drive", completed.stdout + completed.stderr)

    def test_default_validate_only_succeeds(self):
        self.assertTrue(SCRIPT.is_file(), f"missing bootstrap script: {SCRIPT}")
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-ValidateOnly",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("D4_ENV_VALIDATION=pass", completed.stdout)


if __name__ == "__main__":
    unittest.main()
