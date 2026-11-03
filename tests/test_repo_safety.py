import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositorySafetyTests(unittest.TestCase):
    def git_env(self) -> dict[str, str]:
        env = os.environ.copy()
        local_config = ROOT / ".codex-gitconfig"
        if local_config.is_file():
            env.setdefault("GIT_CONFIG_GLOBAL", str(local_config))
        return env

    def git(self, *args: str) -> subprocess.CompletedProcess[str]:
        git = os.environ.get("GIT_EXE", "git")
        return subprocess.run(
            [git, "-C", str(ROOT), *args],
            check=False,
            text=True,
            capture_output=True,
            env=self.git_env(),
        )

    def test_repository_top_level_is_isolated_project(self) -> None:
        result = self.git("rev-parse", "--show-toplevel")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(Path(result.stdout.strip()).resolve(), ROOT)

    def test_required_platform_constants_are_exact(self) -> None:
        config = ROOT / "config" / "expected-platform.env"
        self.assertTrue(config.is_file(), "expected-platform.env is missing")
        values = dict(
            line.split("=", 1)
            for line in config.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        )
        self.assertEqual(values["TARGET_KERNEL_RELEASE"], "4.14.0-xilinx")
        self.assertEqual(values["TARGET_MACHINE"], "armv7l")
        self.assertEqual(
            values["ARTIFACT_ROOT"],
            r"F:\MLK\DPU_ZYNQ7000\artifacts\xvf3800_usb_ps_capture_v1",
        )

    def test_board_and_guest_shell_scripts_are_forced_to_lf(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("*.sh text eol=lf", attributes)

    def test_generated_and_large_artifact_paths_are_ignored(self) -> None:
        candidates = [
            "build/kernel/module.o",
            "artifacts/rootfs.tar.gz",
            "recordings/live.wav",
            "logs/board.log",
            "vivado/project.runs/impl_1/design.bit",
            "vm/guest.vmdk",
            "installers/tool.tar.gz",
            "release/generated/snd-usb-audio.ko",
        ]
        git = os.environ.get("GIT_EXE", "git")
        result = subprocess.run(
            [git, "-C", str(ROOT), "check-ignore", "--no-index", "-z", "--stdin"],
            input="\0".join(candidates) + "\0",
            check=False,
            text=True,
            capture_output=True,
            env=self.git_env(),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        ignored = {item for item in result.stdout.split("\0") if item}
        self.assertEqual(ignored, set(candidates))

    def test_required_docs_are_present_and_not_ignored(self) -> None:
        docs = [
            "docs/superpowers/specs/2026-10-02-xvf3800-usb-ps-capture-design.md",
            "docs/superpowers/plans/2026-10-02-xvf3800-usb-ps-capture-implementation.md",
        ]
        for relative in docs:
            self.assertTrue((ROOT / relative).is_file(), relative)
            result = self.git("check-ignore", "--quiet", "--no-index", relative)
            self.assertEqual(result.returncode, 1, f"unexpectedly ignored: {relative}")


if __name__ == "__main__":
    unittest.main()
