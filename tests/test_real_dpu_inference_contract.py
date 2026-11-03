import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "board" / "run_real_dpu_inference.sh"


def to_wsl(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    suffix = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


class RealDpuInferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.output = self.root / "output"
        self.fake_bin = self.root / "fake-bin"
        self.proc_modules = self.root / "proc-modules"
        self.fdt = self.root / "fdt"
        self.dev_root = self.root / "dev"
        self.dmesg = self.root / "dmesg"
        self.counter_calls = self.root / "counter-calls"
        self.mount_log = self.root / "mount-log"
        self.mount_target = self.root / "mount-target"
        self.run_index = 0
        self.last_output = self.output
        for path in (
            self.bundle / "runtime", self.bundle / "dnndk/pkgs/bin",
            self.bundle / "dnndk/pkgs/lib", self.bundle / "probe",
            self.fake_bin, self.dev_root, self.mount_target,
        ):
            path.mkdir(parents=True, exist_ok=True)
        (self.bundle / "runtime/dpu-4.14.0-xilinx.ko").write_bytes(b"module")
        (self.bundle / "dnndk/pkgs/lib/libn2cube.so").write_bytes(b"n2")
        (self.bundle / "dnndk/pkgs/lib/libhineon.so").write_bytes(b"hi")
        self.proc_modules.write_text("dpu 1 0 - Live 0x0\n", encoding="utf-8")
        self.fdt.write_text("compatible=deephi,dpu\n", encoding="utf-8")
        (self.dev_root / "dpu").write_text("device", encoding="utf-8")
        self.dmesg.write_text("boot complete\n", encoding="utf-8")
        self._exe(self.fake_bin / "uname", """#!/bin/sh
            [ "$1" = -m ] && echo "${FAKE_ARCH:-armv7l}" && exit 0
            [ "$1" = -r ] && echo "${FAKE_KERNEL:-4.14.0-xilinx}" && exit 0
            exit 2
        """)
        self._exe(self.fake_bin / "modinfo", """#!/bin/sh
            echo "${FAKE_VERMAGIC:-4.14.0-xilinx SMP mod_unload ARMv7}"
        """)
        self._exe(self.fake_bin / "dmesg", """#!/bin/sh
            cat "$FAKE_DMESG"
        """)
        self._exe(self.fake_bin / "mount", """#!/bin/sh
            echo mount >> "$FAKE_MOUNT_LOG"
        """)
        self._exe(self.fake_bin / "umount", """#!/bin/sh
            echo umount >> "$FAKE_MOUNT_LOG"
            [ "${FAKE_UMOUNT_FAIL:-0}" -eq 0 ]
        """)
        self._exe(self.fake_bin / "insmod", "#!/bin/sh\nexit 0\n")
        self._exe(self.bundle / "dnndk/pkgs/bin/dexplorer", """#!/bin/sh
            case ":${LD_LIBRARY_PATH:-}:" in
              *":$FAKE_EXPECTED_DNNDK_LIB:"*) ;;
              *) echo 'missing DNNDK dependency search path' >&2; exit 86;;
            esac
            if [ "$1" = -w ]; then echo 'DPU Core 0: B1152F'; exit 0; fi
            [ "$1" = -s ] || exit 2
            count=0; [ ! -f "$FAKE_COUNTER_CALLS" ] || count=$(cat "$FAKE_COUNTER_CALLS")
            count=$((count + 1)); echo "$count" > "$FAKE_COUNTER_CALLS"
            if [ "$count" -eq 1 ]; then schedule=10; interrupt=20
            else schedule=${FAKE_SCHEDULE_AFTER:-112}; interrupt=${FAKE_INTERRUPT_AFTER:-122}; fi
            echo "DPU Core 0 Schedule Count: $schedule"
            echo "DPU Core 0 Interrupt Count: $interrupt"
        """)
        self._exe(self.bundle / "runtime/dpu_yolo_probe", """#!/bin/sh
            mode=; iterations=; dump=; report=
            while [ "$#" -gt 0 ]; do
              case "$1" in
                --input) mode=$2; shift 2;; --iterations) iterations=$2; shift 2;;
                --dump-dir) dump=$2; shift 2;; --report) report=$2; shift 2;;
                --output-node|--kernel|--input-node) shift 2;; *) exit 64;;
              esac
            done
            [ "${FAKE_RUNNER_FAIL_MODE:-}" != "$mode" ] || exit 9
            mkdir -p "$dump"
            for node in layer81_conv layer93_conv layer105_conv layer117_conv; do
              file="$dump/$node.bin"
              if [ "${FAKE_EMPTY:-0}" -eq 1 ]; then : > "$file"
              elif [ "${FAKE_ALL_ZERO:-0}" -eq 1 ]; then printf '\\000\\000\\000\\000' > "$file"
              elif [ "${FAKE_SAME_INPUT_HASH:-0}" -eq 1 ]; then printf 'same-%s' "$node" > "$file"
              else printf '%s-%s' "$mode" "$node" > "$file"; fi
            done
            mismatch=0; [ "${FAKE_REPEAT_MISMATCH:-0}" -eq 0 ] || mismatch=1
            { echo DPU_RUN_STATUS=pass; echo DPU_RUN_CALLS="$iterations"; echo MISMATCH_ITERATIONS="$mismatch"; } > "$report"
            [ -z "${FAKE_INJECT_DMESG:-}" ] || echo "$FAKE_INJECT_DMESG" >> "$FAKE_DMESG"
        """)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _exe(path: Path, content: str):
        path.write_text(textwrap.dedent(content), encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def run_probe(self, **overrides):
        if not PROBE.is_file():
            self.fail(f"missing probe: {PROBE}")
        self.counter_calls.write_text("0\n", encoding="utf-8")
        self.dmesg.write_text("boot complete\n", encoding="utf-8")
        env = {
            "PATH": f"{to_wsl(self.fake_bin)}:/usr/bin:/bin",
            "DPU_INFERENCE_FDT": to_wsl(self.fdt),
            "DPU_INFERENCE_PROC_MODULES": to_wsl(self.proc_modules),
            "DPU_INFERENCE_DEV_ROOT": to_wsl(self.dev_root),
            "DPU_INFERENCE_LIB_TARGET": to_wsl(self.mount_target),
            "FAKE_DMESG": to_wsl(self.dmesg),
            "FAKE_COUNTER_CALLS": to_wsl(self.counter_calls),
            "FAKE_MOUNT_LOG": to_wsl(self.mount_log),
            "FAKE_EXPECTED_DNNDK_LIB": to_wsl(self.bundle / "dnndk/pkgs/lib"),
            "DPU_INFERENCE_STRESS_ITERATIONS": "3",
        }
        env.update({key: str(value) for key, value in overrides.items()})
        assignments = " ".join(f"{key}='{value}'" for key, value in env.items())
        self.run_index += 1
        self.last_output = self.output / f"run-{self.run_index}"
        command = (
            f"env {assignments} bash {to_wsl(PROBE)} "
            f"--bundle-dir {to_wsl(self.bundle)} --output-dir {to_wsl(self.last_output)} --level yolo"
        )
        return subprocess.run(
            ["wsl.exe", "bash", "-lc", command], check=False, text=True,
            encoding="utf-8", errors="replace", capture_output=True,
        )

    def rejected(self, reason, **env):
        result = self.run_probe(**env)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"REASON={reason}", result.stdout + result.stderr)

    def test_rejects_platform_and_runtime_mismatches(self):
        self.rejected("wrong_architecture", FAKE_ARCH="aarch64")
        self.rejected("wrong_kernel_release", FAKE_KERNEL="5.10")
        self.fdt.write_text("xlnx,zynq-7000\n", encoding="utf-8")
        self.rejected("missing_dpu_device_tree_node")
        self.fdt.write_text("deephi,dpu\n", encoding="utf-8")
        self.rejected("module_vermagic_mismatch", FAKE_VERMAGIC="4.19 ARMv7")

    def test_rejects_failed_run_empty_and_all_zero_output(self):
        self.rejected("zero_inference_failed", FAKE_RUNNER_FAIL_MODE="zero")
        self.rejected("empty_output_dump", FAKE_EMPTY="1")
        self.rejected("all_zero_output_dump", FAKE_ALL_ZERO="1")

    def test_rejects_fake_or_nondeterministic_results(self):
        self.rejected("zero_pattern_outputs_identical", FAKE_SAME_INPUT_HASH="1")
        self.rejected("stress_repeat_mismatch", FAKE_REPEAT_MISMATCH="1")
        self.rejected("schedule_counter_not_advanced", FAKE_SCHEDULE_AFTER="10")
        self.rejected("interrupt_counter_not_advanced", FAKE_INTERRUPT_AFTER="20")

    def test_rejects_kernel_fault_and_failed_unmount(self):
        self.rejected("new_kernel_fault", FAKE_INJECT_DMESG="dpu timeout waiting for irq")
        self.rejected("dexplorer_unmount_failed", FAKE_UMOUNT_FAIL="1")

    def test_success_records_three_runs_counters_and_clean_kernel_interval(self):
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DPU_INFERENCE_STATUS=pass", result.stdout)
        report = (self.last_output / "level-a-board-report.txt").read_text(encoding="utf-8")
        self.assertIn("STRESS_ITERATIONS=3", report)
        self.assertIn("SCHEDULE_DELTA=102", report)
        self.assertIn("INTERRUPT_DELTA=102", report)
        self.assertTrue((self.last_output / "level-a-counter-delta.txt").is_file())
        self.assertTrue((self.last_output / "level-a-kernel-log.txt").is_file())


if __name__ == "__main__":
    unittest.main()
