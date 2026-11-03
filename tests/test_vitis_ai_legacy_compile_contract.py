import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "guest" / "build_sound_shape_vitis_ai.sh"
LAUNCHER = ROOT / "scripts" / "guest" / "run_vitis_ai_compile_container.sh"


class VitisAiLegacyCompileContractTests(unittest.TestCase):
    def test_custom_hwh_dcf_and_arm32_arch_are_mandatory(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for token in (
            "dlet -f", '"target": "dpuv2"', '"cpu_arch": "arm32"',
            '"dcf":', "vai_q_caffe quantize", "vai_c_caffe", "--arch",
            "--net_name sound_shape_probe", "-calib_iter 16",
        ):
            self.assertIn(token, text)

    def test_pre_generated_dcf_is_explicit_and_provenanced(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for token in (
            "--platform-dcf", "PLATFORM_DCF", "sha256sum",
            "platform-dcf.sha256", "source-hwh.sha256",
            "cp \"$PLATFORM_DCF\"",
        ):
            self.assertIn(token, text)

        self.assertIn('[[ -f "$PLATFORM_DCF" ]]', text)
        self.assertIn('if [[ -n "$PLATFORM_DCF" ]]', text)

    def test_container_launcher_fails_closed_when_report_is_missing(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('sound-shape-compile-report.txt', text)
        self.assertIn('grep -q \'RESULT=pass\'', text)
        self.assertIn('[[ ! -s "$report" ]]', text)

    def test_compile_is_create_only_and_audited(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for token in (
            "output directory already exists", "find", "-name '*.dcf'",
            "-name '*.elf'", "CPUKernel", "audit_sound_shape_compile.py",
            "TARGET=1152FA", "CPU=arm32", "FALLBACK=none", "RESULT=pass",
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
