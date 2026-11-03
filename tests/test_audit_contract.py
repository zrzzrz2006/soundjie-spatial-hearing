import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "guest" / "audit_usb_audio_support.sh"
HOST_SCRIPT = ROOT / "scripts" / "host" / "audit_source_tree.ps1"
REQUIRED_KEYS = {
    "USB0_DT_STATUS",
    "USB0_DR_MODE",
    "KERNEL_RELEASE",
    "CONFIG_USB",
    "CONFIG_USB_EHCI_HCD",
    "CONFIG_SND",
    "CONFIG_SND_USB_AUDIO",
    "ARECORD_PRESENT",
    "ACTION",
}


def to_wsl(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    suffix = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


def parse_report(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


class AuditContractTests(unittest.TestCase):
    def run_fixture(self, fixture_name: str) -> dict[str, str]:
        command = (
            f"bash {to_wsl(SCRIPT)} --fixture "
            f"{to_wsl(ROOT / 'tests' / 'fixtures' / fixture_name)} "
            "--expected-kernel 4.14.0-xilinx"
        )
        result = subprocess.run(
            ["wsl.exe", "bash", "-lc", command],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = parse_report(result.stdout)
        self.assertTrue(REQUIRED_KEYS.issubset(report), result.stdout)
        return report

    def test_complete_platform_uses_builtin_support(self) -> None:
        report = self.run_fixture("audit_complete.env")
        self.assertEqual(report["ACTION"], "USE_BUILTIN")

    def test_missing_usb_audio_selects_module_build(self) -> None:
        report = self.run_fixture("audit_missing_driver.env")
        self.assertEqual(report["ACTION"], "BUILD_MODULES_AND_ALSA")

    def test_kernel_mismatch_blocks_packaging(self) -> None:
        report = self.run_fixture("audit_kernel_mismatch.env")
        self.assertEqual(report["ACTION"], "BLOCK_KERNEL_MISMATCH")

    def test_host_audit_has_read_only_vmdk_fallback(self) -> None:
        text = HOST_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("qemu-nbd --read-only", text)
        self.assertIn("mount -o ro,noload", text)
        self.assertIn("AUDIT_TRANSPORT=read-only-vmdk", text)


if __name__ == "__main__":
    unittest.main()
