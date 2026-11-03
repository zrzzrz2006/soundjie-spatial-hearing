import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path(
    r"F:\MLK\DPU_ZYNQ7000\artifacts\xvf3800_usb_ps_capture_v1"
)
DEPLOY = ARTIFACT_ROOT / "deploy" / "xvf3800-usb-ps-capture-v1"


def repository_text_sha256(path: Path) -> str:
    """Hash Git's LF-normalized text bytes even in a CRLF checkout."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class ReleaseBundleTests(unittest.TestCase):
    def test_release_instructions_are_complete(self):
        release_readme = ROOT / "release" / "README.md"
        self.assertTrue(release_readme.is_file())
        text = release_readme.read_text(encoding="utf-8")
        for phrase in [
            "无需 JTAG",
            "给板卡上电",
            "install_usb_audio_bundle.sh",
            "xvf3800_probe.sh",
            "xvf3800_record.sh",
            "wav_probe",
        ]:
            self.assertIn(phrase, text)

    def test_deployment_directory_has_required_payload(self):
        required = [
            "modules/4.14.0-xilinx/snd-usb-audio.ko",
            "modules/4.14.0-xilinx/snd-usbmidi-lib.ko",
            "alsa-runtime/usr/bin/arecord",
            "board-tools/install_usb_audio_bundle.sh",
            "board-tools/xvf3800_probe.sh",
            "board-tools/xvf3800_record.sh",
            "board-tools/xvf3800_diagnostics.sh",
            "wav_probe",
            "README.txt",
            "SHA256SUMS",
        ]
        for relative in required:
            self.assertTrue((DEPLOY / relative).is_file(), relative)

    def test_deployment_manifest_matches_two_fresh_reads(self):
        manifest = DEPLOY / "SHA256SUMS"
        self.assertTrue(manifest.is_file())
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split(maxsplit=1)
            path = DEPLOY / relative
            self.assertTrue(path.is_file(), relative)
            first = hashlib.sha256(path.read_bytes()).hexdigest()
            second = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(first, expected, relative)
            self.assertEqual(second, expected, relative)

    def test_soundjie_d1_release_index_has_complete_provenance(self):
        manifest = ROOT / "release" / "SHA256SUMS"
        entries = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            self.assertRegex(digest, r"^[0-9a-f]{64}$", name)
            self.assertNotIn(name, entries, name)
            entries[name] = digest

        required = {
            "docs/superpowers/specs/2026-10-10-soundjie-d1-logmel-design.md",
            "docs/superpowers/plans/2026-10-10-soundjie-d1-logmel-implementation.md",
            "config/history/logmel_frontend_d1_identity.json",
            "config/sound_wav_bundle_sources.json",
            "external/soundjie-d1/xvf3800-test.wav",
            "external/soundjie-d1/xvf3800-start-0000000000.wav",
            "external/soundjie-d1/xvf3800-start-0000072000.wav",
            "external/soundjie-d1/xvf3800-start-0000144000.wav",
            "external/soundjie-d1/arm-logmel-cli",
            "external/soundjie-d1/sound-shape-runner",
            "external/soundjie-d1/dpu-sound-shape-probe.elf",
            "external/soundjie-d1/SJD1.TGZ",
            "external/soundjie-d1/SJD1-FINAL-REPORTS.TGZ",
        }
        self.assertEqual(set(), required - set(entries))
        self.assertEqual(
            "6e13c3ec0db21463835048ca11e3237003a329ab72eedb9a0737546ddc8a0780",
            entries["config/history/logmel_frontend_d1_identity.json"],
        )

    def test_soundjie_d1_summary_blocks_on_quantization_collapse(self):
        summary = ROOT / "evidence" / "soundjie-d1" / "final-summary.md"
        self.assertTrue(summary.is_file())
        text = summary.read_text(encoding="utf-8")
        statuses = dict(re.findall(r"^(D1_[A-Z_]+)=(\S+)$", text, re.MULTILINE))
        self.assertEqual("PASS", statuses.get("D1_A_GOLDEN"))
        self.assertEqual("FAIL", statuses.get("D1_B_PARITY"))
        self.assertEqual("PASS", statuses.get("D1_C_DPU"))
        self.assertEqual("blocked", statuses.get("D1_OVERALL"))
        self.assertIn("不是正式八类准确率", text)
        self.assertIn("三个 INT8 张量完全相同", text)

    def test_soundjie_d2_release_index_and_board_evidence_are_complete(self):
        manifest = ROOT / "release" / "SHA256SUMS"
        entries = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            self.assertRegex(digest, r"^[0-9a-f]{64}$", name)
            self.assertNotIn(name, entries, name)
            entries[name] = digest

        required = {
            "docs/superpowers/specs/2026-10-12-soundjie-d2-bounded-logmel-design.md",
            "docs/superpowers/plans/2026-10-12-soundjie-d2-bounded-logmel-implementation.md",
            "config/logmel_frontend_v1.json",
            "config/logmel_d2_parity_bundle_sources.json",
            "config/sound_wav_d2_bundle_sources.json",
            "external/soundjie-d2/arm-logmel-cli",
            "external/soundjie-d2/SJD2P.TGZ",
            "external/soundjie-d2/SJD2.TGZ",
            "external/soundjie-d2/SJD2-BOARD-EVIDENCE-20261013-A.TGZ",
            "external/soundjie-d2/SJD2-FINAL-EVIDENCE-20261013-A.TGZ",
            "evidence/soundjie-d2/arm-parity-board-report.txt",
            "evidence/soundjie-d2/dpu-board-report.txt",
            "evidence/soundjie-d2/board-validation.md",
            "evidence/soundjie-d2/task6-fault-injection.sh",
            "evidence/soundjie-d2/final-evidence-manifest.txt",
        }
        self.assertEqual(set(), required - set(entries))
        for name in required:
            if name.startswith("external/"):
                continue
            path = ROOT / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(repository_text_sha256(path), entries[name], name)

        self.assertEqual(
            "182a6c47b25d18f5a4c92a94f4cc89369e8a9854fc9cefe0da037e286c3257c6",
            entries["external/soundjie-d2/SJD2-BOARD-EVIDENCE-20261013-A.TGZ"],
        )
        self.assertEqual(
            "e154558478c922991081ad63d29b40665bdd24977b37693dacac6b7c22323783",
            entries["external/soundjie-d2/SJD2-FINAL-EVIDENCE-20261013-A.TGZ"],
        )

        def read_report(name: str) -> dict[str, str]:
            values = {}
            for line in (ROOT / "evidence" / "soundjie-d2" / name).read_text(
                encoding="utf-8"
            ).splitlines():
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    values[key] = value
            return values

        arm = read_report("arm-parity-board-report.txt")
        self.assertEqual("pass", arm.get("LOGMEL_PARITY_STATUS"))
        self.assertEqual("10", arm.get("CASE_COUNT"))
        self.assertEqual("0", arm.get("PARITY_MAX_ABS"))
        self.assertEqual("0", arm.get("PARITY_MAX_RMSE"))
        self.assertEqual("10", arm.get("INT8_EXACT_CASES"))

        dpu = read_report("dpu-board-report.txt")
        expected_dpu = {
            "DPU_WAV_STATUS": "pass",
            "INPUT_SCALE": "128",
            "INPUT_SIZE": "6144",
            "REAL_WINDOW_COUNT": "3",
            "RUNS_PER_WINDOW": "2",
            "TOTAL_DPU_RUN_CALLS": "6",
            "SCHEDULE_DELTA": "6",
            "INTERRUPT_DELTA": "6",
            "NEW_KERNEL_FAULTS": "0",
            "DISTINCT_INPUT_SHA256": "3",
            "OUTPUT_NONZERO_BYTES_MIN": "8",
            "MISMATCH_ITERATIONS_TOTAL": "0",
            "FAULT_CASE_COUNT": "6",
            "FAULT_SCHEDULE_DELTA": "0",
            "FAULT_INTERRUPT_DELTA": "0",
            "FAULT_NEW_KERNEL_FAULTS": "0",
            "FAULT_INJECTION_STATUS": "pass",
        }
        for key, expected in expected_dpu.items():
            self.assertEqual(expected, dpu.get(key), key)
        input_hashes = {
            dpu.get("INPUT_SHA256_0"),
            dpu.get("INPUT_SHA256_72000"),
            dpu.get("INPUT_SHA256_144000"),
        }
        self.assertNotIn(None, input_hashes)
        self.assertEqual(3, len(input_hashes))

        candidate = (ROOT / "evidence" / "soundjie-d2" / "candidate.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("`board_validated`", candidate)
        self.assertIn("FAULT_INJECTION_STATUS=pass", candidate)

        validation = (ROOT / "evidence" / "soundjie-d2" / "board-validation.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("D2_OVERALL=pass", validation)
        self.assertIn("reboot: System halted", validation)


if __name__ == "__main__":
    unittest.main()
