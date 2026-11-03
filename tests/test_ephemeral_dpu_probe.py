import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "board" / "run_ephemeral_dpu_probe.sh"


def to_wsl(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    suffix = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


class EphemeralDpuProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle"
        self.output = self.root / "output"
        self.audio = self.root / "audio"
        self.fake_bin = self.root / "fake-bin"
        self.proc_modules = self.root / "proc-modules"
        self.fdt = self.root / "fdt.txt"
        self.dev_root = self.root / "dev"
        self.dmesg_log = self.root / "dmesg.log"
        self.smoke_count = self.root / "smoke-count"
        self.mount_log = self.root / "mount-log"
        self.audio_loaded = self.root / "audio-loaded"
        self.dexplorer_lib_target = self.root / "dexplorer-lib-target"

        for path in (
            self.bundle / "runtime",
            self.bundle / "dnndk" / "pkgs" / "bin",
            self.bundle / "dnndk" / "pkgs" / "lib",
            self.audio / "board-tools",
            self.fake_bin,
            self.dev_root,
            self.dexplorer_lib_target,
        ):
            path.mkdir(parents=True, exist_ok=True)

        (self.bundle / "runtime" / "dpu-4.14.0-xilinx.ko").write_bytes(b"module")
        (self.bundle / "dnndk" / "pkgs" / "lib" / "libn2cube.so").write_bytes(b"n2")
        (self.bundle / "dnndk" / "pkgs" / "lib" / "libhineon.so").write_bytes(b"hi")
        self.proc_modules.write_text("", encoding="utf-8")
        self.fdt.write_text("compatible=deephi,dpu\n", encoding="utf-8")
        (self.dev_root / "dpu").write_text("device", encoding="utf-8")
        self.dmesg_log.write_text("boot complete\n", encoding="utf-8")
        self._write_executable(
            self.fake_bin / "uname",
            """#!/bin/sh
            case "$1" in
              -m) printf '%s\n' "${FAKE_UNAME_M:-armv7l}" ;;
              -r) printf '%s\n' "${FAKE_UNAME_R:-4.14.0-xilinx}" ;;
              *) exit 2 ;;
            esac
            """,
        )
        self._write_executable(
            self.fake_bin / "modinfo",
            """#!/bin/sh
            [ "$1" = "-F" ] && [ "$2" = "vermagic" ] || exit 2
            printf '%s\n' "${FAKE_VERMAGIC:-4.14.0-xilinx SMP mod_unload ARMv7}"
            """,
        )
        self._write_executable(
            self.fake_bin / "insmod",
            """#!/bin/sh
            printf 'dpu 1 0 - Live 0x00000000\n' > "$FAKE_PROC_MODULES"
            """,
        )
        self._write_executable(
            self.fake_bin / "dmesg",
            """#!/bin/sh
            cat "$FAKE_DMESG_LOG"
            """,
        )
        self._write_executable(
            self.bundle / "dnndk" / "pkgs" / "bin" / "dexplorer",
            """#!/bin/sh
            [ "$1" = "-w" ] || exit 7
            echo 'DPU Core 0: B1152'
            """,
        )
        self._write_executable(
            self.fake_bin / "mount",
            """#!/bin/sh
            printf 'mount:%s:%s:%s\n' "$1" "$2" "$3" >> "$FAKE_MOUNT_LOG"
            [ "$1" = "--bind" ]
            """,
        )
        self._write_executable(
            self.fake_bin / "umount",
            """#!/bin/sh
            printf 'umount:%s\n' "$1" >> "$FAKE_MOUNT_LOG"
            """,
        )
        self._write_executable(
            self.bundle / "runtime" / "dpu_smoke",
            """#!/bin/sh
            count=0
            [ ! -f "$FAKE_SMOKE_COUNT" ] || count=$(cat "$FAKE_SMOKE_COUNT")
            count=$((count + 1))
            printf '%s\n' "$count" > "$FAKE_SMOKE_COUNT"
            if [ -n "${FAKE_INJECT_DMESG:-}" ] && [ "$count" -eq 1 ]; then
              printf '%s\n' "$FAKE_INJECT_DMESG" >> "$FAKE_DMESG_LOG"
            fi
            [ "$count" -ne "${FAKE_SMOKE_FAIL_AT:-0}" ]
            """,
        )
        (self.audio / "env.sh").write_text(
            "PATH=$PATH\nexport PATH\n", encoding="utf-8", newline="\n"
        )
        self._write_executable(
            self.audio / "load-usb-audio.sh",
            """#!/bin/sh
            : > "$FAKE_AUDIO_LOADED"
            """,
        )
        self._write_executable(
            self.audio / "board-tools" / "xvf3800_probe.sh",
            """#!/bin/sh
            [ -f "$FAKE_AUDIO_LOADED" ] || exit 3
            report=
            while [ "$#" -gt 0 ]; do
              case "$1" in --report) report=$2; shift 2;; *) shift;; esac
            done
            cat > "$report" <<'EOF'
DEVICE_FOUND=yes
USB_ID=2886:001a
PCM_DEVICE=hw:0,0
SELECTED_FORMAT=S16_LE
SELECTED_CHANNELS=2
SELECTED_RATE=16000
SUPPORTED_CAPTURE=yes
EOF
            """,
        )
        self._write_executable(
            self.audio / "board-tools" / "xvf3800_record.sh",
            """#!/bin/sh
            output=
            while [ "$#" -gt 0 ]; do
              case "$1" in --output) output=$2; shift 2;; *) shift;; esac
            done
            printf 'fake-wave' > "$output"
            echo CAPTURE_STATUS=complete
            """,
        )
        self._write_executable(
            self.audio / "wav_probe",
            """#!/bin/sh
            echo CHANNELS=2
            echo SAMPLE_RATE=16000
            echo BITS_PER_SAMPLE=16
            echo "SIGNAL=${FAKE_SIGNAL:-present}"
            """,
        )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def _write_executable(path: Path, content: str):
        path.write_text(textwrap.dedent(content), encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def run_probe(self, **overrides):
        env = {
            "PATH": f"{to_wsl(self.fake_bin)}:/usr/bin:/bin",
            "DPU_PROBE_FDT": to_wsl(self.fdt),
            "DPU_PROBE_PROC_MODULES": to_wsl(self.proc_modules),
            "DPU_PROBE_DEV_ROOT": to_wsl(self.dev_root),
            "FAKE_PROC_MODULES": to_wsl(self.proc_modules),
            "FAKE_DMESG_LOG": to_wsl(self.dmesg_log),
            "FAKE_SMOKE_COUNT": to_wsl(self.smoke_count),
            "FAKE_MOUNT_LOG": to_wsl(self.mount_log),
            "FAKE_AUDIO_LOADED": to_wsl(self.audio_loaded),
            "DPU_PROBE_DEXPLORER_LIB_TARGET": to_wsl(self.dexplorer_lib_target),
            "DPU_PROBE_ITERATIONS": "3",
        }
        env.update({key: str(value) for key, value in overrides.items()})
        assignments = " ".join(f"{key}='{value}'" for key, value in env.items())
        command = (
            f"env {assignments} bash {to_wsl(PROBE)} "
            f"--bundle-dir {to_wsl(self.bundle)} "
            f"--audio-root {to_wsl(self.audio)} "
            f"--output-dir {to_wsl(self.output)}"
        )
        return subprocess.run(
            ["wsl.exe", "bash", "-lc", command],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

    def assert_rejected(self, result, reason):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"REASON={reason}", result.stdout + result.stderr)

    def test_rejects_wrong_architecture(self):
        self.assert_rejected(self.run_probe(FAKE_UNAME_M="aarch64"), "wrong_architecture")

    def test_rejects_wrong_kernel_release(self):
        self.assert_rejected(self.run_probe(FAKE_UNAME_R="5.10.0"), "wrong_kernel_release")

    def test_rejects_missing_dpu_device_tree_node(self):
        self.fdt.write_text("compatible=xlnx,zynq-7000\n", encoding="utf-8")
        self.assert_rejected(self.run_probe(), "missing_dpu_device_tree_node")

    def test_rejects_module_vermagic_mismatch(self):
        self.assert_rejected(
            self.run_probe(FAKE_VERMAGIC="4.19.0 SMP mod_unload ARMv7"),
            "module_vermagic_mismatch",
        )

    def test_rejects_absent_dpu_character_device(self):
        (self.dev_root / "dpu").unlink()
        self.assert_rejected(self.run_probe(), "missing_dpu_character_device")

    def test_rejects_nonzero_smoke_iteration(self):
        self.assert_rejected(self.run_probe(FAKE_SMOKE_FAIL_AT="2"), "dpu_smoke_failed")

    def test_rejects_new_dpu_timeout_in_kernel_log(self):
        self.assert_rejected(
            self.run_probe(FAKE_INJECT_DMESG="dpu timeout waiting for interrupt"),
            "new_kernel_fault",
        )

    def test_rejects_new_usb_disconnect_in_kernel_log(self):
        self.assert_rejected(
            self.run_probe(FAKE_INJECT_DMESG="usb 1-1: USB disconnect, device number 3"),
            "new_usb_fault",
        )

    def test_rejects_silent_audio(self):
        self.assert_rejected(self.run_probe(FAKE_SIGNAL="absent"), "audio_signal_absent")

    def test_success_keeps_reports_and_wav_under_output_directory(self):
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("DPU_PROBE_STATUS=pass", result.stdout)
        self.assertEqual(self.smoke_count.read_text(encoding="utf-8").strip(), "4")
        self.assertTrue(self.audio_loaded.exists())
        mount_log = self.mount_log.read_text(encoding="utf-8")
        self.assertIn("mount:--bind:", mount_log)
        self.assertIn(f"umount:{to_wsl(self.dexplorer_lib_target)}", mount_log)
        created = [path for path in self.output.rglob("*") if path.is_file()]
        self.assertTrue(created)
        self.assertTrue(all(self.output in path.parents for path in created))
        script = PROBE.read_text(encoding="utf-8")
        self.assertNotIn("/lib/modules", script)
        self.assertNotIn("/etc/", script)
        self.assertNotIn("cp ", script)
        self.assertNotIn("rm ", script)
        self.assertNotIn("rmmod", script)


if __name__ == "__main__":
    unittest.main()
