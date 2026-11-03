import hashlib
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "guest" / "build_soundjie_live_tools.sh"
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")


def bash_path(path: Path) -> str:
    path = Path(path).resolve()
    return f"/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SoundjieLiveArmBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.toolchain = self.root / "toolchain"
        self.toolchain.mkdir()
        self.compiler_log = self.root / "compiler.log"
        self.logmel = self.root / "logmel_cli"
        self.logmel.write_bytes(b"preverified-arm-logmel\n")
        self.logmel.chmod(0o755)
        self._write_toolchain()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _exe(path, content):
        path.write_text(textwrap.dedent(content), encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def _write_toolchain(self):
        self._exe(self.toolchain / "arm-linux-gnueabihf-g++", """#!/bin/sh
            echo "$*" >> "$LIVE_BUILD_COMPILER_LOG"
            output=
            while [ $# -gt 0 ]; do
              case "$1" in -o) output=$2; shift 2;; *) shift;; esac
            done
            [ -n "$output" ] || exit 64
            case "$output" in *selector*) printf 'fake-arm-selector\n' > "$output";; *) printf 'fake-arm-fuser\n' > "$output";; esac
            chmod 755 "$output"
        """)
        self._exe(self.toolchain / "arm-linux-gnueabihf-readelf", """#!/bin/sh
            mode=${LIVE_BUILD_AUDIT_MODE:-ok}
            case "$1" in
              -h)
                [ "$mode" != wrong-machine ] || { echo 'Machine: Advanced Micro Devices X86-64'; exit 0; }
                echo 'Machine: ARM'; echo 'Flags: 0x5000400, Version5 EABI, hard-float ABI' ;;
              -A) echo 'Tag_CPU_arch: v7'; echo 'Tag_ABI_VFP_args: VFP registers' ;;
              --version-info)
                [ "$mode" != new-glibc ] || { echo 'Name: GLIBC_2.28'; exit 0; }
                echo 'Name: GLIBC_2.4' ;;
              -d) echo 'Shared library: [libm.so.6]'; echo 'Shared library: [libc.so.6]' ;;
              -l)
                if [ "$mode" = wrong-loader ]; then
                  echo '[Requesting program interpreter: /lib/ld-linux.so.3]'
                else
                  echo '[Requesting program interpreter: /lib/ld-linux-armhf.so.3]'
                fi ;;
              *) exit 64 ;;
            esac
        """)

    def run_build(self, output, **overrides):
        env = os.environ.copy()
        env.update({
            "LIVE_BUILD_COMPILER_LOG": bash_path(self.compiler_log),
            "PATH": f"{bash_path(self.toolchain)}:/usr/bin:/bin",
        })
        env.update({key: str(value) for key, value in overrides.items()})
        return subprocess.run(
            [
                str(GIT_BASH), "-c", 'PATH="$1:$PATH"; export PATH; shift; exec sh "$@"',
                "arm-build-test", bash_path(self.toolchain), bash_path(BUILD),
                "--repo-root", bash_path(ROOT), "--output-dir", bash_path(output),
                "--logmel-cli", bash_path(self.logmel),
            ],
            env=env, text=True, encoding="utf-8", errors="replace", capture_output=True,
        )

    def test_build_script_pins_cpp11_and_arm_hard_float(self):
        output = self.root / "build"
        result = self.run_build(output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self.compiler_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertIn("-std=c++11", command)
            self.assertIn("-march=armv7-a", command)
            self.assertIn("-mfloat-abi=hard", command)
            self.assertIn("-mfpu=vfpv3-d16", command)
            self.assertIn("-static-libgcc", command)
            self.assertIn("-static-libstdc++", command)
        self.assertIn("live_window_selector.cpp", commands[0])
        self.assertIn("wav_pcm.cpp", commands[0])
        self.assertIn("live_result_fuser.cpp", commands[1])

    def test_rejects_wrong_machine_glibc_or_dynamic_loader(self):
        for mode in ("wrong-machine", "new-glibc", "wrong-loader"):
            with self.subTest(mode=mode):
                output = self.root / f"build-{mode}"
                result = self.run_build(output, LIVE_BUILD_AUDIT_MODE=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(output.exists())

    def test_manifest_binds_source_and_binary_hashes(self):
        output = self.root / "manifest-build"
        result = self.run_build(output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [line.split("\t") for line in (output / "build-manifest.tsv").read_text(
            encoding="utf-8").splitlines()]
        by_path = {row[1]: (row[0], row[2]) for row in rows}
        expected_sources = (
            "src/wav_pcm.cpp", "src/wav_pcm.h", "src/live_window_selector.cpp",
            "src/live_window_selector.h", "src/live_window_selector_cli.cpp",
            "src/live_result_fuser.cpp", "src/live_result_fuser.h",
            "src/live_result_fuser_cli.cpp",
        )
        for relative in expected_sources:
            self.assertEqual(by_path[relative], ("source", digest(ROOT / relative)))
        for name in ("live_window_selector", "live_result_fuser", "logmel_cli"):
            self.assertEqual(by_path[name], ("binary", digest(output / name)))
        report = (output / "build-report.txt").read_text(encoding="utf-8")
        self.assertEqual(report.count("MACHINE=ARM"), 3)
        self.assertEqual(report.count("HARD_FLOAT=yes"), 3)
        self.assertEqual(report.count("GLIBC_MAX_OK=yes"), 3)
        self.assertEqual(report.count("ELF_INTERPRETER=/lib/ld-linux-armhf.so.3"), 3)
        self.assertIn(f"LOGMEL_INPUT_SHA256={digest(self.logmel)}", report)

    def test_build_is_create_only(self):
        output = self.root / "owned"
        output.mkdir()
        marker = output / "user-file.txt"
        marker.write_text("keep\n", encoding="utf-8")
        result = self.run_build(output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("output directory already exists", result.stderr)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")
        self.assertFalse(self.compiler_log.exists())


if __name__ == "__main__":
    unittest.main()
