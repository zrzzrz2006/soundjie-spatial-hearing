import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "host" / "build_logmel_parity_bundle.py"
BUILD_ARM = ROOT / "scripts" / "guest" / "build_logmel_cli.sh"
BOARD = ROOT / "scripts" / "board" / "run_logmel_parity.sh"
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
CASES = (
    "silence", "impulse", "sine_440", "dual_tone", "noise_seeded",
    "clipping", "stereo_antiphase", "xvf3800-start-0000000000",
    "xvf3800-start-0000072000", "xvf3800-start-0000144000",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bash_path(path):
    path = Path(path).resolve()
    return f"/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


def load_builder():
    spec = importlib.util.spec_from_file_location("logmel_parity_bundle", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogMelParityBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        records = []
        names = [("runtime/logmel_cli", "cli", "0755"),
                 ("config/logmel_frontend_v1.json", "config", "0644")]
        for case in CASES:
            names.extend(((f"inputs/{case}.wav", f"{case}.wav", "0644"),
                          (f"golden/{case}.f32le", f"{case}.f32", "0644"),
                          (f"golden/{case}.i8", f"{case}.i8", "0644")))
        names.append(("probe/run_logmel_parity.sh", "probe", "0755"))
        for index, (target, source, mode) in enumerate(names):
            path = self.sources / source
            if target == "config/logmel_frontend_v1.json":
                path.write_bytes((ROOT / "config/logmel_frontend_v1.json").read_bytes())
            elif target.endswith(".i8") and "xvf3800" in target:
                case_index = int(target.split("start-")[1].split(".")[0]) // 72000
                values = ((np.arange(6144) * (case_index * 12 + 1) + case_index)
                          % 121 - 60).astype(np.int8)
                path.write_bytes(values.tobytes())
            else:
                path.write_bytes(f"payload-{index}".encode())
            records.append({"name": target, "source": str(path), "size": path.stat().st_size,
                            "sha256": sha256(path), "mode": mode})
        self.config = self.root / "bundle.json"
        self.config.write_text(json.dumps({
            "schema": 2, "input_scale": 128.0, "cases": list(CASES),
            "normalization_profile": "bounded-log-power-64-v1",
            "config_sha256": next(item["sha256"] for item in records
                                  if item["name"] == "config/logmel_frontend_v1.json"),
            "archive_members": records,
        }), encoding="utf-8")

        self.toolchain = self.root / "fake-arm-toolchain"
        self.toolchain.mkdir()
        compiler = self.toolchain / "arm-linux-gnueabihf-g++"
        compiler.write_text(
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = --version ]; then\n"
            "  echo 'arm-linux-gnueabihf-g++ (test toolchain) 6.3.0'\n"
            "  exit 0\n"
            "fi\n"
            "out=\n"
            "while [ $# -gt 0 ]; do\n"
            "  if [ \"$1\" = -o ]; then out=$2; shift 2; else shift; fi\n"
            "done\n"
            "printf 'test-arm-elf' > \"$out\"\n",
            encoding="ascii", newline="\n")
        readelf = self.toolchain / "arm-linux-gnueabihf-readelf"
        readelf.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  -h) printf 'Machine: ARM\\nFlags: Version5 EABI\\n' ;;\n"
            "  -A) printf 'Tag_CPU_arch: %s\\nTag_ABI_VFP_args: %s\\n' "
            "\"${TEST_CPU_ARCH:-v7}\" \"${TEST_VFP_ARGS:-VFP registers}\" ;;\n"
            "  --version-info) printf 'Name: GLIBC_%s\\n' \"${TEST_GLIBC:-2.17}\" ;;\n"
            "  -d) printf 'NEEDED Shared library: [%s]\\n' \"${TEST_NEEDED:-libm.so.6}\" ;;\n"
            "  -l) printf 'Requesting program interpreter: /lib/ld-linux-armhf.so.3\\n' ;;\n"
            "  *) exit 64 ;;\n"
            "esac\n",
            encoding="ascii", newline="\n")
        os.chmod(compiler, 0o755)
        os.chmod(readelf, 0o755)

    def tearDown(self):
        self.temporary.cleanup()

    def test_bundle_is_deterministic_and_exactly_allowlisted(self):
        builder = load_builder()
        a = builder.build_bundle(self.config, self.root / "a")
        b = builder.build_bundle(self.config, self.root / "b")
        self.assertEqual(a["SJD2P.TGZ"].sha256, b["SJD2P.TGZ"].sha256)
        import tarfile
        with tarfile.open(self.root / "a" / "SJD2P.TGZ", "r:gz") as archive:
            members = archive.getmembers()
            self.assertTrue(all(item.mtime == 0 and item.uid == 0 and item.gid == 0
                                for item in members))
            names = [item.name for item in members]
            self.assertIn("manifest/cases.tsv", names)
            self.assertIn("manifest/contract.tsv", names)
            self.assertIn("manifest/SHA256SUMS", names)
            self.assertEqual(len(names), 36)
            cases = archive.extractfile("manifest/cases.tsv").read().decode("ascii")
            self.assertTrue(all(line.split("\t")[4] == "0"
                                for line in cases.splitlines()))
            contract = archive.extractfile("manifest/contract.tsv").read().decode("ascii")
            self.assertIn("NORMALIZATION_PROFILE\tbounded-log-power-64-v1\n", contract)
            self.assertIn("INPUT_SCALE\t128\n", contract)

    def test_bundle_rejects_bad_hash_path_traversal_and_overwrite(self):
        builder = load_builder()
        data = json.loads(self.config.read_text(encoding="utf-8"))
        data["archive_members"][0]["sha256"] = "0" * 64
        bad_hash = self.root / "bad-hash.json"
        bad_hash.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            builder.build_bundle(bad_hash, self.root / "bad-a")
        data = json.loads(self.config.read_text(encoding="utf-8"))
        data["archive_members"][2]["name"] = "../escape"
        bad_path = self.root / "bad-path.json"
        bad_path.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            builder.build_bundle(bad_path, self.root / "bad-b")
        builder.build_bundle(self.config, self.root / "built")
        with self.assertRaises(FileExistsError):
            builder.build_bundle(self.config, self.root / "built")

    def test_bundle_rejects_nonfinite_scale_and_packages_validated_snapshot(self):
        builder = load_builder()
        data = json.loads(self.config.read_text(encoding="utf-8"))
        data["input_scale"] = float("inf")
        nonfinite = self.root / "nonfinite.json"
        nonfinite.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input scale"):
            builder.build_bundle(nonfinite, self.root / "nonfinite-out")

        target = Path(data["archive_members"][0]["source"])
        expected = json.loads(self.config.read_text(encoding="utf-8"))["archive_members"][0]
        original_temporary_directory = builder.tempfile.TemporaryDirectory

        class MutatingTemporaryDirectory:
            def __init__(self, *args, **kwargs):
                self.inner = original_temporary_directory(*args, **kwargs)

            def __enter__(self):
                directory = self.inner.__enter__()
                target.write_bytes(b"changed-after-validation")
                return directory

            def __exit__(self, *args):
                return self.inner.__exit__(*args)

        builder.tempfile.TemporaryDirectory = MutatingTemporaryDirectory
        try:
            result = builder.build_bundle(self.config, self.root / "snapshot")
        finally:
            builder.tempfile.TemporaryDirectory = original_temporary_directory
        import tarfile
        with tarfile.open(result["SJD2P.TGZ"].path, "r:gz") as archive:
            payload = archive.extractfile("runtime/logmel_cli").read()
        self.assertEqual(len(payload), expected["size"])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), expected["sha256"])

    def test_bundle_rejects_old_profile_and_collapsed_real_tensors(self):
        builder = load_builder()
        for name, mutate in (
                ("schema", lambda data: data.update(schema=1)),
                ("profile", lambda data: data.update(normalization_profile="identity-64-v1"))):
            data = json.loads(self.config.read_text(encoding="utf-8"))
            mutate(data)
            path = self.root / f"bad-{name}.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.subTest(name=name), self.assertRaises(ValueError):
                builder.build_bundle(path, self.root / f"bad-{name}")

        data = json.loads(self.config.read_text(encoding="utf-8"))
        for item in data["archive_members"]:
            if item["name"].endswith(".i8") and "xvf3800" in item["name"]:
                source = Path(item["source"])
                source.write_bytes(bytes([128]) * 6144)
                item["sha256"] = sha256(source)
                item["size"] = 6144
        collapsed = self.root / "collapsed.json"
        collapsed.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            builder.build_bundle(collapsed, self.root / "collapsed")

    def test_arm_build_audits_abi_and_glibc(self):
        output, report = self.root / "logmel_cli", self.root / "arm-report.txt"
        env = os.environ.copy()
        env["PATH"] = str(self.toolchain) + os.pathsep + env["PATH"]
        result = subprocess.run([
            str(GIT_BASH), bash_path(BUILD_ARM), "--mode", "arm",
            "--repo-root", bash_path(ROOT), "--output", bash_path(output),
            "--report", bash_path(report),
        ], env=env, text=True, encoding="utf-8", errors="replace", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = report.read_text(encoding="ascii")
        self.assertIn("MACHINE=ARM", text)
        self.assertIn("EABI=Version5", text)
        self.assertIn("HARD_FLOAT=yes", text)
        self.assertIn("GLIBC_MAX_OK=yes", text)
        self.assertIn("CPU_ARCH=v7", text)
        self.assertIn("MAX_GLIBC=2.17", text)
        self.assertIn("DT_NEEDED=libm.so.6", text)
        self.assertTrue(output.is_file())

    def test_arm_build_rejects_wrong_cpu_float_new_glibc_and_n2cube(self):
        variants = (
            ("armv6", {"TEST_CPU_ARCH": "v6"}, "ARMv7"),
            ("soft", {"TEST_VFP_ARGS": "Base AAPCS"}, "hard-float"),
            ("glibc", {"TEST_GLIBC": "2.18"}, "glibc"),
            ("n2cube", {"TEST_NEEDED": "libn2cube.so"}, "N2Cube"),
        )
        for name, overrides, reason in variants:
            with self.subTest(name=name):
                output, report = self.root / f"{name}-cli", self.root / f"{name}-report.txt"
                env = os.environ.copy()
                env["PATH"] = str(self.toolchain) + os.pathsep + env["PATH"]
                env.update(overrides)
                result = subprocess.run([
                    str(GIT_BASH), bash_path(BUILD_ARM), "--mode", "arm",
                    "--repo-root", bash_path(ROOT), "--output", bash_path(output),
                    "--report", bash_path(report),
                ], env=env, text=True, encoding="utf-8", errors="replace", capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(reason, result.stderr)
                self.assertFalse(output.exists())
                self.assertFalse(report.exists())


class LogMelParityBoardScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        for folder in ("runtime", "config", "inputs", "golden", "probe", "manifest"):
            (self.bundle / folder).mkdir(parents=True, exist_ok=True)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        uname = self.tools / "uname"
        uname.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  -m) printf '%s\\n' \"${TEST_UNAME_M:-armv7l}\" ;;\n"
            "  -r) printf '%s\\n' \"${TEST_UNAME_R:-4.14.0-xilinx}\" ;;\n"
            "  *) exit 64 ;;\n"
            "esac\n",
            encoding="ascii", newline="\n")
        os.chmod(uname, 0o755)
        runner = self.bundle / "runtime" / "logmel_cli"
        script = "#!/bin/sh\nwhile [ \"$#\" -gt 0 ]; do\n  case \"$1\" in\n    --float-output) f=$2; shift 2;;\n    --int8-output) i=$2; shift 2;;\n    --report) r=$2; shift 2;;\n    --golden-float) gf=$2; shift 2;;\n    --golden-int8) gi=$2; shift 2;;\n    *) shift 2;;\n  esac\ndone\ncp \"$gf\" \"$f\"\ncp \"$gi\" \"$i\"\ncase \"${TEST_RUNNER_MODE:-pass}\" in\n  float-fail) printf 'LOGMEL_STATUS=fail\\nMAX_ABS=0.1\\nRMSE=0.01\\nINT8_EXACT=yes\\n' > \"$r\" ;;\n  int8-fail) printf 'LOGMEL_STATUS=pass\\nMAX_ABS=0\\nRMSE=0\\nINT8_EXACT=no\\n' > \"$r\" ;;\n  *) printf 'LOGMEL_STATUS=pass\\nMAX_ABS=0\\nRMSE=0\\nINT8_EXACT=yes\\n' > \"$r\" ;;\nesac\n"
        runner.write_text(script, encoding="ascii", newline="\n")
        os.chmod(runner, 0o755)
        (self.bundle / "config" / "logmel_frontend_v1.json").write_bytes(
            (ROOT / "config/logmel_frontend_v1.json").read_bytes())
        case_lines = []
        for case in CASES:
            (self.bundle / "inputs" / f"{case}.wav").write_bytes(b"wav")
            (self.bundle / "golden" / f"{case}.f32le").write_bytes(b"f" * 24576)
            index = CASES.index(case)
            values = ((np.arange(6144) * (index * 12 + 1) + index)
                      % 121 - 60).astype(np.int8)
            (self.bundle / "golden" / f"{case}.i8").write_bytes(values.tobytes())
            case_lines.append(
                f"{case}\tinputs/{case}.wav\tgolden/{case}.f32le\t"
                f"golden/{case}.i8\t0\t128\n")
        (self.bundle / "manifest" / "cases.tsv").write_text(
            "".join(case_lines), encoding="ascii", newline="\n")
        (self.bundle / "probe" / "run_logmel_parity.sh").write_text(
            "#!/bin/sh\n", encoding="ascii", newline="\n")
        (self.bundle / "manifest" / "contract.tsv").write_text(
            "SCHEMA\t2\nNORMALIZATION_PROFILE\tbounded-log-power-64-v1\n"
            "INPUT_SCALE\t128\n"
            f"CONFIG_SHA256\t{sha256(self.bundle / 'config/logmel_frontend_v1.json')}\n",
            encoding="ascii", newline="\n")
        self.write_sums()

    def write_sums(self):
        names = ["runtime/logmel_cli", "config/logmel_frontend_v1.json"]
        for case in CASES:
            names.extend((f"inputs/{case}.wav", f"golden/{case}.f32le",
                          f"golden/{case}.i8"))
        names.extend(("probe/run_logmel_parity.sh", "manifest/cases.tsv",
                      "manifest/contract.tsv"))
        sums = "".join(f"{sha256(self.bundle / name)}  {name}\n" for name in names)
        (self.bundle / "manifest" / "SHA256SUMS").write_text(sums, encoding="ascii", newline="\n")

    def tearDown(self):
        self.temporary.cleanup()

    def run_script(self, output, **overrides):
        env = os.environ.copy()
        env.update(overrides)
        return subprocess.run([
            str(GIT_BASH), "-c", 'PATH="$1:$PATH"; export PATH; shift; exec "$@"',
            "board-test", bash_path(self.tools), bash_path(BOARD),
            "--bundle-dir", bash_path(self.bundle),
            "--output-dir", bash_path(output),
        ], env=env, text=True, encoding="utf-8", errors="replace", capture_output=True)

    def test_board_script_passes_valid_case_and_rejects_existing_output(self):
        output = self.root / "out"
        passed = self.run_script(output)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self.assertIn("LOGMEL_PARITY_STATUS=pass", passed.stdout)
        report = (output / "board-report.txt").read_text(encoding="ascii")
        self.assertIn("PARITY_MAX_ABS=0", report)
        self.assertIn("PARITY_MAX_RMSE=0", report)
        self.assertIn("INT8_EXACT_CASES=10", report)
        summary = (output / "case-summary.tsv").read_text(encoding="ascii")
        lines = summary.splitlines()
        self.assertEqual(lines[0], "CASE\tMAX_ABS\tRMSE\tINT8_EXACT\tFIRST_MAX_ERROR_INDEX\tFIRST_INT8_MISMATCH_INDEX")
        self.assertEqual(len(lines), 11)
        self.assertTrue(lines[1].startswith("silence\t0\t0\tyes\t"))
        repeated = self.run_script(output)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("output_already_exists", repeated.stdout)

    def test_board_script_rejects_wrong_arch_and_hash_mismatch(self):
        wrong_arch = self.run_script(self.root / "wrong", TEST_UNAME_M="x86_64")
        self.assertNotEqual(wrong_arch.returncode, 0)
        self.assertIn("wrong_architecture", wrong_arch.stdout)
        with (self.bundle / "inputs" / "silence.wav").open("ab") as stream:
            stream.write(b"corrupt")
        bad_hash = self.run_script(self.root / "bad-hash")
        self.assertNotEqual(bad_hash.returncode, 0)
        self.assertIn("bundle_hash_mismatch", bad_hash.stdout)

    def test_board_script_rejects_wrong_kernel_unsafe_sums_and_incomplete_cases(self):
        wrong_kernel = self.run_script(self.root / "wrong-kernel", TEST_UNAME_R="6.1.0")
        self.assertNotEqual(wrong_kernel.returncode, 0)
        self.assertIn("wrong_kernel_release", wrong_kernel.stdout)

        sums = self.bundle / "manifest" / "SHA256SUMS"
        sums.write_text(sums.read_text(encoding="ascii") + "0" * 64 + "  ../outside\n",
                        encoding="ascii", newline="\n")
        unsafe = self.run_script(self.root / "unsafe")
        self.assertNotEqual(unsafe.returncode, 0)
        self.assertIn("unsafe_checksum_manifest", unsafe.stdout)

        self.write_sums()
        cases = self.bundle / "manifest" / "cases.tsv"
        lines = cases.read_text(encoding="ascii").splitlines(keepends=True)
        cases.write_text("".join(lines[:-1]), encoding="ascii", newline="\n")
        self.write_sums()
        incomplete = self.run_script(self.root / "incomplete")
        self.assertNotEqual(incomplete.returncode, 0)
        self.assertIn("case_manifest_mismatch", incomplete.stdout)

    def test_board_script_rejects_source_offset_for_pre_extracted_wav(self):
        cases = self.bundle / "manifest" / "cases.tsv"
        lines = cases.read_text(encoding="ascii").splitlines(keepends=True)
        fields = lines[8].rstrip("\n").split("\t")
        fields[4] = "72000"
        lines[8] = "\t".join(fields) + "\n"
        cases.write_text("".join(lines), encoding="ascii", newline="\n")
        self.write_sums()
        result = self.run_script(self.root / "source-offset")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("case_manifest_mismatch", result.stdout)

    def test_board_script_rejects_float_threshold_and_int8_mismatch_reports(self):
        float_fail = self.run_script(self.root / "float-fail", TEST_RUNNER_MODE="float-fail")
        self.assertNotEqual(float_fail.returncode, 0)
        self.assertIn("silence_report_failed", float_fail.stdout)
        int8_fail = self.run_script(self.root / "int8-fail", TEST_RUNNER_MODE="int8-fail")
        self.assertNotEqual(int8_fail.returncode, 0)
        self.assertIn("silence_int8_mismatch", int8_fail.stdout)

    def test_board_script_rejects_real_tensor_saturation_before_frontend(self):
        tensor = self.bundle / "golden" / f"{CASES[-3]}.i8"
        values = bytearray(tensor.read_bytes())
        values[:62] = bytes([128]) * 62
        tensor.write_bytes(values)
        self.write_sums()
        result = self.run_script(self.root / "saturated")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("tensor_quality_failed", result.stdout)


if __name__ == "__main__":
    unittest.main()
