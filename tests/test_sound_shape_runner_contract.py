import hashlib
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "sound_shape_runner.cpp"


def wsl_path(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


class SoundShapeRunnerContractTests(unittest.TestCase):
    def test_runner_pins_single_input_and_logits_output_cli(self):
        text = RUNNER.read_text(encoding="utf-8")
        for token in (
            "--kernel", "--input-node", "--output-node", "--input",
            "--iterations", "--dump", "--report", "zero", "pattern",
        ):
            self.assertIn(token, text)

    def test_runner_repopulates_input_before_every_dpu_run(self):
        text = RUNNER.read_text(encoding="utf-8")
        loop = text.index("for (int iteration")
        refill = text.index("copy_input", loop)
        run = text.index("dpuRunTask", loop)
        self.assertLess(refill, run)

    def test_runner_rejects_empty_or_all_zero_logits(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("invalid output tensor", text)
        self.assertIn("all-zero logits", text)
        self.assertIn("MISMATCH_ITERATIONS", text)

    def test_runner_adds_exact_file_input_and_provenance_contract(self):
        text = RUNNER.read_text(encoding="utf-8")
        for token in (
            "--input-file", "--model-file", "INPUT_SOURCE=file",
            "INPUT_BYTES=", "INPUT_SCALE=", "MODEL_SHA256=",
            "INPUT_SHA256=", "OUTPUT_SHA256=", "OUTPUT_VALUES=",
            "dpuGetInputTensorScale", "std::isfinite", "sha256_file",
        ):
            self.assertIn(token, text)
        self.assertIn("input file must be exactly 6144 bytes", text)
        self.assertIn("input file and --input are mutually exclusive", text)
        self.assertIn("--model-file is required for file input", text)

    def test_file_buffer_is_copied_before_every_run_without_requantization(self):
        text = RUNNER.read_text(encoding="utf-8")
        loop = text.index("for (int iteration")
        copy = text.index("copy_input", loop)
        run = text.index("dpuRunTask", loop)
        self.assertLess(copy, run)

    def test_describe_tensor_is_mutually_exclusive_and_never_runs_dpu(self):
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("--describe-tensor", text)
        self.assertIn("describe mode is mutually exclusive", text)
        self.assertIn("dpuGetOutputTensorScale", text)
        self.assertIn("OUTPUT_SCALE=", text)
        self.assertIn("OUTPUT_SIZE=", text)
        describe = text.index("if (opt.describe_tensor)")
        loop = text.index("for (int iteration")
        self.assertLess(describe, loop)
        self.assertNotIn("dpuRunTask", text[describe:loop])


class SoundShapeRunnerCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        (cls.root / "n2cube.h").write_text(textwrap.dedent(r"""
            #ifndef TEST_N2CUBE_H_
            #define TEST_N2CUBE_H_
            #include <cstdint>
            struct DPUKernel {};
            struct DPUTask {};
            const int N2CUBE_EXCEPTION_MODE_RET_ERR_CODE = 1;
            const int T_MODE_NORMAL = 0;
            void dpuSetExceptionMode(int);
            int dpuOpen();
            int dpuClose();
            DPUKernel* dpuLoadKernel(const char*);
            int dpuDestroyKernel(DPUKernel*);
            DPUTask* dpuCreateTask(DPUKernel*, int);
            int dpuDestroyTask(DPUTask*);
            int dpuGetInputTensorSize(DPUTask*, const char*, int = 0);
            int8_t* dpuGetInputTensorAddress(DPUTask*, const char*, int = 0);
            float dpuGetInputTensorScale(DPUTask*, const char*, int = 0);
            float dpuGetOutputTensorScale(DPUTask*, const char*, int = 0);
            int dpuRunTask(DPUTask*);
            int dpuGetOutputTensorSize(DPUTask*, const char*, int = 0);
            int8_t* dpuGetOutputTensorAddress(DPUTask*, const char*, int = 0);
            #endif
        """), encoding="utf-8", newline="\n")
        stub = cls.root / "n2cube_stub.cpp"
        stub.write_text(textwrap.dedent(r"""
            #include "n2cube.h"
            #include <cstdlib>
            static DPUKernel kernel;
            static DPUTask task;
            static int8_t input[6144];
            static int8_t output[8];
            void dpuSetExceptionMode(int) {}
            int dpuOpen() { return 0; }
            int dpuClose() { return 0; }
            DPUKernel* dpuLoadKernel(const char*) { return &kernel; }
            int dpuDestroyKernel(DPUKernel*) { return 0; }
            DPUTask* dpuCreateTask(DPUKernel*, int) { return &task; }
            int dpuDestroyTask(DPUTask*) { return 0; }
            int dpuGetInputTensorSize(DPUTask*, const char*, int) { return 6144; }
            int8_t* dpuGetInputTensorAddress(DPUTask*, const char*, int) { return input; }
            float dpuGetInputTensorScale(DPUTask*, const char*, int) {
              const char* value = std::getenv("SJD_TEST_INPUT_SCALE");
              return value ? std::strtof(value, nullptr) : 8.0f;
            }
            float dpuGetOutputTensorScale(DPUTask*, const char*, int) {
              const char* value = std::getenv("SJD_TEST_OUTPUT_SCALE");
              return value ? std::strtof(value, nullptr) : 0.0625f;
            }
            int dpuRunTask(DPUTask*) {
              if (std::getenv("SJD_TEST_FORBID_DPU_RUN")) std::abort();
              for (int i = 0; i < 8; ++i) output[i] = static_cast<int8_t>(input[i] + i + 1);
              return 0;
            }
            int dpuGetOutputTensorSize(DPUTask*, const char*, int) { return 8; }
            int8_t* dpuGetOutputTensorAddress(DPUTask*, const char*, int) { return output; }
        """), encoding="utf-8", newline="\n")
        cls.binary = cls.root / "sound_shape_runner"
        linked_model_sha = hashlib.sha256(b"verified-model").hexdigest()
        command = [
            "wsl.exe", "g++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
            f'-DSOUNDJIE_LINKED_MODEL_SHA256="{linked_model_sha}"',
            f"-I{wsl_path(cls.root)}", f"-I{wsl_path(ROOT / 'src')}",
            wsl_path(RUNNER), wsl_path(ROOT / "src" / "tensor_stats.cpp"),
            wsl_path(ROOT / "src" / "sha256.cpp"), wsl_path(stub),
            "-o", wsl_path(cls.binary),
        ]
        result = subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                                capture_output=True)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_cli(self, *args, env=None):
        environment = os.environ.copy()
        if env:
            environment.update(env)
            forwarded = ":".join(env)
            environment["WSLENV"] = ":".join(
                item for item in (environment.get("WSLENV", ""), forwarded) if item)
        return subprocess.run(["wsl.exe", wsl_path(self.binary), *args], text=True,
                              encoding="utf-8", errors="replace", capture_output=True,
                              env=environment)

    def test_cli_rejects_conflicting_inputs_missing_model_and_describe_outputs(self):
        base = ["--kernel", "sound_shape_probe", "--input-node", "conv1"]
        conflict = self.run_cli(*base, "--output-node", "logits", "--input", "zero",
                                "--input-file", "/tmp/x", "--iterations", "1",
                                "--dump", "/tmp/d", "--report", "/tmp/r")
        self.assertEqual(conflict.returncode, 64)
        self.assertIn("mutually exclusive", conflict.stderr)
        no_model = self.run_cli(*base, "--output-node", "logits", "--input-file", "/tmp/x",
                                "--iterations", "1", "--dump", "/tmp/d", "--report", "/tmp/r")
        self.assertEqual(no_model.returncode, 64)
        self.assertIn("--model-file is required", no_model.stderr)
        bad_describe = self.run_cli(*base, "--describe-tensor", "--output-node", "logits",
                                    "--iterations", "1")
        self.assertEqual(bad_describe.returncode, 64)
        self.assertIn("describe mode is mutually exclusive", bad_describe.stderr)

    def test_describe_reports_contract_without_running_task(self):
        result = self.run_cli("--kernel", "sound_shape_probe", "--input-node", "conv1",
                              "--output-node", "logits",
                              "--describe-tensor", env={"SJD_TEST_FORBID_DPU_RUN": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("INPUT_SIZE=6144", result.stdout)
        self.assertIn("INPUT_SCALE=8", result.stdout)
        self.assertIn("OUTPUT_SIZE=8", result.stdout)
        self.assertIn("OUTPUT_SCALE=", result.stdout)
        self.assertIn("DPU_RUN_CALLS=0", result.stdout)

    def test_describe_rejects_nonpositive_and_nonfinite_scale(self):
        for scale in ("0", "-1", "nan"):
            with self.subTest(scale=scale):
                result = self.run_cli(
                    "--kernel", "sound_shape_probe", "--input-node", "conv1",
                    "--output-node", "logits",
                    "--describe-tensor", env={"SJD_TEST_INPUT_SCALE": scale})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid input tensor scale", result.stderr)
        for scale in ("0", "-1", "nan"):
            with self.subTest(output_scale=scale):
                result = self.run_cli(
                    "--kernel", "sound_shape_probe", "--input-node", "conv1",
                    "--output-node", "logits", "--describe-tensor",
                    env={"SJD_TEST_OUTPUT_SCALE": scale})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid output tensor scale", result.stderr)

    def test_file_mode_hashes_exact_tensor_model_and_output(self):
        tensor = self.root / "tensor.i8"
        model = self.root / "model.elf"
        dump = self.root / "logits.bin"
        report = self.root / "report.txt"
        tensor.write_bytes(bytes(index % 127 for index in range(6144)))
        model.write_bytes(b"verified-model")
        result = self.run_cli(
            "--kernel", "sound_shape_probe", "--input-node", "conv1",
            "--output-node", "logits", "--input-file", wsl_path(tensor),
            "--model-file", wsl_path(model), "--iterations", "2",
            "--dump", wsl_path(dump), "--report", wsl_path(report))
        self.assertEqual(result.returncode, 0, result.stderr)
        fields = dict(line.split("=", 1) for line in report.read_text(encoding="ascii").splitlines())
        self.assertEqual(fields["INPUT_SOURCE"], "file")
        self.assertEqual(fields["INPUT_BYTES"], "6144")
        self.assertEqual(fields["INPUT_SHA256"], hashlib.sha256(tensor.read_bytes()).hexdigest())
        self.assertEqual(fields["MODEL_SHA256"], hashlib.sha256(model.read_bytes()).hexdigest())
        self.assertEqual(fields["OUTPUT_SHA256"], hashlib.sha256(dump.read_bytes()).hexdigest())
        self.assertEqual(len(fields["OUTPUT_VALUES"].split(",")), 8)
        self.assertEqual(fields["OUTPUT_SCALE"], "0.0625")
        self.assertEqual(fields["DPU_RUN_CALLS"], "2")

        wrong_model = self.root / "wrong-model.elf"
        wrong_model.write_bytes(b"not-the-linked-model")
        rejected = self.run_cli(
            "--kernel", "sound_shape_probe", "--input-node", "conv1",
            "--output-node", "logits", "--input-file", wsl_path(tensor),
            "--model-file", wsl_path(wrong_model), "--iterations", "1",
            "--dump", "/tmp/wrong-model.bin", "--report", "/tmp/wrong-model.txt")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("model file does not match linked model", rejected.stderr)

    def test_legacy_modes_preserve_report_and_reject_same_output_path(self):
        for mode in ("zero", "pattern"):
            dump = self.root / f"{mode}.bin"
            report = self.root / f"{mode}.txt"
            result = self.run_cli(
                "--kernel", "sound_shape_probe", "--input-node", "conv1",
                "--output-node", "logits", "--input", mode, "--iterations", "2",
                "--dump", wsl_path(dump), "--report", wsl_path(report))
            self.assertEqual(result.returncode, 0, result.stderr)
            text = report.read_text(encoding="ascii")
            self.assertIn(f"INPUT_MODE={mode}", text)
            self.assertIn("DPU_RUN_CALLS=2", text)
            self.assertNotIn("INPUT_SCALE=", text)
        same = self.root / "same-output"
        rejected = self.run_cli(
            "--kernel", "sound_shape_probe", "--input-node", "conv1",
            "--output-node", "logits", "--input", "zero", "--iterations", "1",
            "--dump", wsl_path(same), "--report", wsl_path(same))
        self.assertEqual(rejected.returncode, 64)
        self.assertIn("dump and report paths must differ", rejected.stderr)

        symlink_target = f"/tmp/sjd-output-{os.getpid()}"
        symlink_alias = symlink_target + "-link"
        subprocess.run(["wsl.exe", "rm", "-f", symlink_target, symlink_alias], check=True)
        subprocess.run(["wsl.exe", "ln", "-s", symlink_target, symlink_alias], check=True)
        try:
            rejected = self.run_cli(
                "--kernel", "sound_shape_probe", "--input-node", "conv1",
                "--output-node", "logits", "--input", "zero", "--iterations", "1",
                "--dump", symlink_target, "--report", symlink_alias)
            self.assertEqual(rejected.returncode, 64)
            self.assertIn("dump and report paths must differ", rejected.stderr)
        finally:
            subprocess.run(["wsl.exe", "rm", "-f", symlink_target, symlink_alias], check=True)

        alias = self.root / "alias-output"
        lexical_alias = wsl_path(alias.parent) + "/./alias-output"
        rejected = self.run_cli(
            "--kernel", "sound_shape_probe", "--input-node", "conv1",
            "--output-node", "logits", "--input", "zero", "--iterations", "1",
            "--dump", wsl_path(alias), "--report", lexical_alias)
        self.assertEqual(rejected.returncode, 64)
        self.assertIn("dump and report paths must differ", rejected.stderr)

        hardlink_dump = self.root / "hardlink-dump"
        hardlink_report = self.root / "hardlink-report"
        hardlink_dump.write_bytes(b"existing")
        os.link(hardlink_dump, hardlink_report)
        rejected = self.run_cli(
            "--kernel", "sound_shape_probe", "--input-node", "conv1",
            "--output-node", "logits", "--input", "zero", "--iterations", "1",
            "--dump", wsl_path(hardlink_dump), "--report", wsl_path(hardlink_report))
        self.assertEqual(rejected.returncode, 64)
        self.assertIn("dump and report paths must differ", rejected.stderr)

    def test_file_mode_rejects_short_and_long_tensor(self):
        model = self.root / "size-model.elf"
        model.write_bytes(b"model")
        for name, size in (("short", 6143), ("long", 6145)):
            tensor = self.root / f"{name}.i8"
            tensor.write_bytes(b"x" * size)
            result = self.run_cli(
                "--kernel", "sound_shape_probe", "--input-node", "conv1",
                "--output-node", "logits", "--input-file", wsl_path(tensor),
                "--model-file", wsl_path(model), "--iterations", "1",
                "--dump", f"/tmp/{name}.bin", "--report", f"/tmp/{name}.txt")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly 6144 bytes", result.stderr)


if __name__ == "__main__":
    unittest.main()
