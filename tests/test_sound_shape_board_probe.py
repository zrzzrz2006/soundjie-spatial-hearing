import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "board" / "run_sound_shape_probe.sh"


def to_wsl(path: Path) -> str:
    return f"/mnt/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


class SoundShapeBoardProbeTests(unittest.TestCase):
    def test_probe_pins_platform_model_and_three_run_acceptance(self):
        text = PROBE.read_text(encoding="utf-8")
        for token in (
            "armv7l", "4.14.0-xilinx", "deephi,dpu", "B1152",
            "sound_shape_probe", "--input-node conv1", "--output-node logits",
            "INPUT_NODE=conv1",
            "run_one zero", "run_one pattern", "STRESS_ITERATIONS",
        ):
            self.assertIn(token, text)

    def test_functional_success_and_failure_gates(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle, fake, output = root / "bundle", root / "bin", root / "output"
            for path in (
                bundle / "runtime", bundle / "model", bundle / "dnndk/pkgs/bin",
                bundle / "dnndk/pkgs/lib", fake, root / "dev", root / "lib-target",
            ):
                path.mkdir(parents=True, exist_ok=True)
            for relative in (
                "runtime/dpu-4.14.0-xilinx.ko", "model/dpu_sound_shape_probe.elf",
                "dnndk/pkgs/lib/libn2cube.so", "dnndk/pkgs/lib/libhineon.so",
            ):
                (bundle / relative).write_bytes(b"x")
            (root / "modules").write_text("dpu 1 0 - Live 0x0\n", encoding="utf-8")
            (root / "fdt").write_text("deephi,dpu\n", encoding="utf-8")
            (root / "dev/dpu").write_text("x", encoding="utf-8")
            (root / "dmesg").write_text("boot ok\n", encoding="utf-8")
            (root / "counter").write_text("0\n", encoding="utf-8")

            def exe(path, body):
                path.write_text(textwrap.dedent(body), encoding="utf-8", newline="\n")
                path.chmod(0o755)

            exe(fake / "uname", """#!/bin/sh
                [ "$1" = -m ] && echo armv7l && exit 0
                [ "$1" = -r ] && echo 4.14.0-xilinx && exit 0
                exit 2
            """)
            exe(fake / "modinfo", "#!/bin/sh\necho '4.14.0-xilinx ARMv7'\n")
            exe(fake / "dmesg", "#!/bin/sh\ncat \"$FAKE_DMESG\"\n")
            exe(fake / "mount", "#!/bin/sh\nexit 0\n")
            exe(fake / "umount", "#!/bin/sh\nexit 0\n")
            exe(fake / "insmod", "#!/bin/sh\nexit 0\n")
            exe(bundle / "dnndk/pkgs/bin/dexplorer", """#!/bin/sh
                [ "$1" = -w ] && echo 'DPU Core 0: B1152F' && exit 0
                count=$(cat "$FAKE_COUNTER"); count=$((count+1)); echo "$count" > "$FAKE_COUNTER"
                [ "$count" -eq 1 ] && echo 'Schedule 10 Interrupt 20' || echo 'Schedule 112 Interrupt 122'
            """)
            exe(bundle / "runtime/sound_shape_probe", """#!/bin/sh
                mode=; iterations=; dump=; report=
                while [ "$#" -gt 0 ]; do
                  case "$1" in
                    --input) mode=$2; shift 2;; --iterations) iterations=$2; shift 2;;
                    --dump) dump=$2; shift 2;; --report) report=$2; shift 2;;
                    --kernel|--input-node|--output-node) shift 2;; *) exit 64;;
                  esac
                done
                [ "${FAKE_ALL_ZERO:-0}" = 1 ] && dd if=/dev/zero of="$dump" bs=8 count=1 2>/dev/null || printf '%-8s' "$mode" > "$dump"
                { echo DPU_SOUND_STATUS=pass; echo DPU_RUN_CALLS="$iterations"; echo MISMATCH_ITERATIONS=0; } > "$report"
            """)
            env = {
                "PATH": f"{to_wsl(fake)}:/usr/bin:/bin",
                "DPU_SOUND_FDT": to_wsl(root / "fdt"),
                "DPU_SOUND_PROC_MODULES": to_wsl(root / "modules"),
                "DPU_SOUND_DEV_ROOT": to_wsl(root / "dev"),
                "DPU_SOUND_LIB_TARGET": to_wsl(root / "lib-target"),
                "DPU_SOUND_STRESS_ITERATIONS": "3",
                "FAKE_DMESG": to_wsl(root / "dmesg"),
                "FAKE_COUNTER": to_wsl(root / "counter"),
            }
            assignments = " ".join(f"{key}='{value}'" for key, value in env.items())
            command = (
                f"env {assignments} bash {to_wsl(PROBE)} --bundle-dir {to_wsl(bundle)} "
                f"--output-dir {to_wsl(output)}"
            )
            result = subprocess.run(["wsl.exe", "bash", "-lc", command], text=True, encoding="utf-8",
                                    errors="replace", capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("DPU_SOUND_STATUS=pass", result.stdout)
            report = (output / "board-report.txt").read_text(encoding="utf-8")
            self.assertIn("STRESS_ITERATIONS=3", report)
            self.assertIn("SCHEDULE_DELTA=102", report)

            env["FAKE_ALL_ZERO"] = "1"
            assignments = " ".join(f"{key}='{value}'" for key, value in env.items())
            rejected_output = root / "rejected-output"
            rejected = subprocess.run(
                ["wsl.exe", "bash", "-lc",
                 f"env {assignments} bash {to_wsl(PROBE)} --bundle-dir {to_wsl(bundle)} "
                 f"--output-dir {to_wsl(rejected_output)}"],
                text=True, encoding="utf-8", errors="replace", capture_output=True, check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("REASON=all_zero_output_dump", rejected.stdout)

    def test_probe_requires_nonzero_distinct_repeatable_outputs_and_counters(self):
        text = PROBE.read_text(encoding="utf-8")
        for token in (
            "all_zero_output_dump", "zero_pattern_outputs_identical",
            "pattern_stress_outputs_differ", "schedule_counter_not_advanced",
            "interrupt_counter_not_advanced", "new_kernel_fault",
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
