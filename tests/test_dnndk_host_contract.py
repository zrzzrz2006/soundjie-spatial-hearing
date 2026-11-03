import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts" / "host" / "audit_dnndk_archive.py"
INSTALLER = ROOT / "scripts" / "guest" / "install_dnndk_host.sh"
VERIFIER = ROOT / "scripts" / "guest" / "verify_dnndk_host.sh"
ARCHIVE_NAME = "xilinx_dnndk_v3.1_190809.tar.gz"
TOP = "xilinx_dnndk_v3.1"
TOOLS = ("decent_q", "dnnc", "dlet", "ddump")


def to_wsl(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    suffix = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o755 if name.rsplit("/", 1)[-1] in TOOLS else 0o644
            archive.addfile(info, io.BytesIO(payload))


def valid_members() -> dict[str, bytes]:
    members = {
        f"{TOP}/host_x86/bin/{tool}": b"#!/bin/sh\necho tool-ok\n"
        for tool in TOOLS
    }
    members[f"{TOP}/host_x86/python/caffe/__init__.py"] = b"VERSION = 'fixture'\n"
    return members


def write_contract(path: Path, archive: Path, state: str = "approved") -> None:
    payload = {
        "schema": 1,
        "state": state,
        "archive_basename": ARCHIVE_NAME,
        "approved_sha256": sha256(archive) if state == "approved" else None,
        "allowed_top_levels": [TOP],
        "required_tools": list(TOOLS),
        "host_subtree": "host_x86",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class DnndkHostContractTests(unittest.TestCase):
    def run_audit(
        self, archive: Path, contract: Path, report: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(AUDITOR),
                str(archive),
                "--contract",
                str(contract),
                "--report",
                str(report),
            ],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

    def test_rejects_path_traversal_member(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / ARCHIVE_NAME
            members = valid_members()
            members["../escape"] = b"must-not-escape"
            write_archive(archive, members)
            contract = root / "contract.json"
            write_contract(contract, archive)
            report = root / "report.env"

            result = self.run_audit(archive, contract, report)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsafe_archive_member", result.stderr)
            self.assertFalse((root.parent / "escape").exists())

    def test_rejects_missing_required_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / ARCHIVE_NAME
            members = valid_members()
            del members[f"{TOP}/host_x86/bin/dnnc"]
            write_archive(archive, members)
            contract = root / "contract.json"
            write_contract(contract, archive)

            result = self.run_audit(archive, contract, root / "report.env")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing_required_tool:dnnc", result.stderr)

    def test_rejects_unpinned_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / ARCHIVE_NAME
            write_archive(archive, valid_members())
            contract = root / "contract.json"
            write_contract(contract, archive, state="candidate_only")
            report = root / "report.env"

            result = self.run_audit(archive, contract, report)

            self.assertEqual(result.returncode, 66)
            self.assertIn("candidate_only", result.stderr)
            self.assertIn("RESULT=candidate_only", report.read_text(encoding="utf-8"))

    def test_accepts_exact_pinned_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / ARCHIVE_NAME
            write_archive(archive, valid_members())
            contract = root / "contract.json"
            write_contract(contract, archive)
            report = root / "report.env"

            result = self.run_audit(archive, contract, report)

            self.assertEqual(result.returncode, 0, result.stderr)
            text = report.read_text(encoding="utf-8")
            self.assertIn("RESULT=pass", text)
            self.assertIn(f"ARCHIVE_SHA256={sha256(archive)}", text)
            for tool in TOOLS:
                self.assertIn(f"TOOL_{tool.upper()}=", text)

    def test_installer_refuses_candidate_only_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / ARCHIVE_NAME
            write_archive(archive, valid_members())
            contract = root / "contract.json"
            write_contract(contract, archive, state="candidate_only")
            install_root = root / "installed"
            report = root / "install.env"
            command = (
                f"bash {to_wsl(INSTALLER)} --archive {to_wsl(archive)} "
                f"--contract {to_wsl(contract)} --install-root {to_wsl(install_root)} "
                f"--report {to_wsl(report)}"
            )

            result = subprocess.run(
                ["wsl.exe", "bash", "-lc", command],
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )

            self.assertEqual(result.returncode, 66)
            self.assertFalse(install_root.exists())

    def test_installer_is_create_only_and_verifier_passes_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive = root / ARCHIVE_NAME
            write_archive(archive, valid_members())
            contract = root / "contract.json"
            write_contract(contract, archive)
            install_root = root / "installed"
            install_report = root / "install.env"
            command = (
                f"bash {to_wsl(INSTALLER)} --archive {to_wsl(archive)} "
                f"--contract {to_wsl(contract)} --install-root {to_wsl(install_root)} "
                f"--report {to_wsl(install_report)}"
            )

            first = subprocess.run(
                ["wsl.exe", "bash", "-lc", command],
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            second = subprocess.run(
                ["wsl.exe", "bash", "-lc", command],
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("install_root_exists", second.stderr)
            verify_report = root / "verify.env"
            verify_command = (
                f"bash {to_wsl(VERIFIER)} --root {to_wsl(install_root)} "
                f"--report {to_wsl(verify_report)}"
            )
            verified = subprocess.run(
                ["wsl.exe", "bash", "-lc", verify_command],
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            report_text = verify_report.read_text(encoding="utf-8")
            self.assertIn("CAFFE_PYTHON_STATUS=pass", report_text)
            self.assertIn("RESULT=pass", report_text)

    def test_verifier_fails_when_a_required_tool_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            install_root = root / "installed" / TOP / "host_x86"
            bin_dir = install_root / "bin"
            bin_dir.mkdir(parents=True)
            for tool in ("decent_q", "dnnc", "dlet"):
                path = bin_dir / tool
                path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8", newline="\n")
                path.chmod(0o755)
            caffe = install_root / "python" / "caffe"
            caffe.mkdir(parents=True)
            (caffe / "__init__.py").write_text("VERSION='fixture'\n", encoding="utf-8")
            report = root / "verify.env"
            command = (
                f"bash {to_wsl(VERIFIER)} --root {to_wsl(root / 'installed')} "
                f"--report {to_wsl(report)}"
            )

            result = subprocess.run(
                ["wsl.exe", "bash", "-lc", command],
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            text = report.read_text(encoding="utf-8")
            self.assertIn("DDUMP_STATUS=fail", text)
            self.assertIn("RESULT=fail", text)


if __name__ == "__main__":
    unittest.main()
