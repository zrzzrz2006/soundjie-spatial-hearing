import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_SCRIPT = ROOT / "scripts" / "guest" / "build_usb_audio_modules.sh"
ALSA_SCRIPT = ROOT / "scripts" / "guest" / "package_arm_alsa.sh"


def soundjie_d1_external_paths() -> dict[str, Path]:
    sources = json.loads(
        (ROOT / "config" / "sound_wav_bundle_sources.json").read_text(
            encoding="utf-8"
        )
    )
    members = {
        member["name"]: Path(member["source"])
        for member in sources["archive_members"]
    }
    prefix = "external/soundjie-d1/"
    return {
        prefix + "xvf3800-test.wav": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/xvf3800_usb_ps_capture_v1/"
            "live-results/xvf3800-test.wav"
        ),
        prefix + "xvf3800-start-0000000000.wav": members[
            "inputs/xvf3800-start-0000000000.wav"
        ],
        prefix + "xvf3800-start-0000072000.wav": members[
            "inputs/xvf3800-start-0000072000.wav"
        ],
        prefix + "xvf3800-start-0000144000.wav": members[
            "inputs/xvf3800-start-0000144000.wav"
        ],
        prefix + "arm-logmel-cli": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d1/"
            "arm-build-20261011-122220850/logmel_cli"
        ),
        prefix + "sound-shape-runner": members["runtime/sound_shape_runner"],
        prefix + "dpu-sound-shape-probe.elf": members[
            "model/dpu_sound_shape_probe.elf"
        ],
        prefix + "SJD1.TGZ": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d1/"
            "bundle-a-20261011-194813786/SJD1.TGZ"
        ),
        prefix + "SJD1-FINAL-REPORTS.TGZ": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d1/"
            "board-final-20261012-a/SJD1-FINAL-REPORTS.TGZ"
        ),
        "external/soundjie-d1-candidate-v2/arm-logmel-cli": members[
            "frontend/logmel_cli"
        ],
        "external/soundjie-d1-candidate-v2/arm-build-report.txt": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d1/"
            "arm-build-20261012-strict-json-a/build-report.txt"
        ),
        "external/soundjie-d1-candidate-v2/SJD1P.TGZ": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d1/"
            "parity-bundle-20261012-strict-json-a/SJD1P.TGZ"
        ),
        "external/soundjie-d1-candidate-v2/SJD1.TGZ": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d1/"
            "sound-bundle-20261012-strict-json-a/SJD1.TGZ"
        ),
    }


def soundjie_d2_external_paths() -> dict[str, Path]:
    prefix = "external/soundjie-d2/"
    return {
        prefix + "arm-logmel-cli": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d2/"
            "arm-build-20261013-b/logmel_cli"
        ),
        prefix + "SJD2P.TGZ": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d2/"
            "parity-candidate-20261013-a/SJD2P.TGZ"
        ),
        prefix + "SJD2.TGZ": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d2/"
            "dpu-candidate-20261013-a/SJD2.TGZ"
        ),
        prefix + "SJD2-BOARD-EVIDENCE-20261013-A.TGZ": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d2/"
            "board-validation-20261013-a/SJD2-BOARD-EVIDENCE-20261013-A.TGZ"
        ),
        prefix + "SJD2-FINAL-EVIDENCE-20261013-A.TGZ": Path(
            "F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_d2/"
            "board-validation-20261013-a/SJD2-FINAL-EVIDENCE-20261013-A.TGZ"
        ),
    }


def to_wsl(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    suffix = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


def repository_text_sha256(path: Path) -> str:
    """Hash Git's LF-normalized text bytes even in a CRLF checkout."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class BuildBundleDryRunTests(unittest.TestCase):
    def run_script(
        self, script: Path, report: str
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            audit = Path(temp_dir) / "audit.env"
            audit.write_text(report, encoding="utf-8")
            command = f"bash {to_wsl(script)} --audit {to_wsl(audit)} --dry-run"
            return subprocess.run(
                ["wsl.exe", "bash", "-lc", command],
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )

    def test_builtin_support_skips_module_compilation(self) -> None:
        result = self.run_script(
            MODULE_SCRIPT,
            "KERNEL_RELEASE=4.14.0-xilinx\nACTION=USE_BUILTIN\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MODULE_BUILD=skip", result.stdout)

    def test_missing_driver_selects_usb_audio_dependencies(self) -> None:
        result = self.run_script(
            MODULE_SCRIPT,
            "KERNEL_RELEASE=4.14.0-xilinx\nACTION=BUILD_MODULES_AND_ALSA\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MODULE_BUILD=required", result.stdout)
        self.assertIn("snd-hwdep.ko", result.stdout)
        self.assertIn("snd-rawmidi.ko", result.stdout)
        self.assertIn("snd-usb-audio.ko", result.stdout)
        self.assertIn("snd-usbmidi-lib.ko", result.stdout)

    def test_kernel_mismatch_aborts_module_build(self) -> None:
        result = self.run_script(
            MODULE_SCRIPT,
            "KERNEL_RELEASE=4.9.0\nACTION=BLOCK_KERNEL_MISMATCH\n",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BLOCK_KERNEL_MISMATCH", result.stderr)

    def test_missing_arecord_selects_alsa_package(self) -> None:
        result = self.run_script(
            ALSA_SCRIPT,
            "KERNEL_RELEASE=4.14.0-xilinx\nARECORD_PRESENT=no\n"
            "ACTION=BUILD_MODULES_AND_ALSA\n",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ALSA_PACKAGE=required", result.stdout)

    def test_external_artifact_manifest_matches_readback(self) -> None:
        config = (ROOT / "config" / "expected-platform.env").read_text(
            encoding="utf-8"
        )
        artifact_line = next(
            line for line in config.splitlines() if line.startswith("ARTIFACT_ROOT=")
        )
        artifact_root = Path(artifact_line.split("=", 1)[1])
        soundjie_external = soundjie_d1_external_paths()
        soundjie_external.update(soundjie_d2_external_paths())
        manifest = ROOT / "release" / "SHA256SUMS"
        self.assertTrue(manifest.is_file())
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split(maxsplit=1)
            if relative.startswith(("docs/", "config/", "evidence/")):
                path = ROOT / relative
            elif relative.startswith(("external/soundjie-d1", "external/soundjie-d2")):
                self.assertIn(relative, soundjie_external)
                path = soundjie_external[relative]
            else:
                path = artifact_root / relative
            self.assertTrue(path.is_file(), relative)
            if relative.startswith(("docs/", "config/", "evidence/")):
                first = repository_text_sha256(path)
                second = repository_text_sha256(path)
            else:
                first = hashlib.sha256(path.read_bytes()).hexdigest()
                second = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(first, expected, relative)
            self.assertEqual(second, expected, relative)


if __name__ == "__main__":
    unittest.main()
