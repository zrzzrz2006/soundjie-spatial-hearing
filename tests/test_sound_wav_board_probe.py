import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "scripts" / "board" / "run_sound_wav_dpu.sh"
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
CASES = (
    "xvf3800-start-0000000000",
    "xvf3800-start-0000072000",
    "xvf3800-start-0000144000",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bash_path(path):
    path = Path(path).resolve()
    return f"/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


class SoundWavBoardProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        for folder in ("runtime", "model", "dnndk/pkgs/bin", "dnndk/pkgs/lib",
                       "frontend", "config", "inputs", "golden", "probe", "manifest"):
            (self.bundle / folder).mkdir(parents=True, exist_ok=True)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.counter = self.root / "counter"
        self.counter.write_text("0", encoding="ascii")
        self.dmesg = self.root / "dmesg"
        self.dmesg.write_text("baseline\n", encoding="ascii")
        self.proc_modules = self.root / "modules"
        self.proc_modules.write_text("dpu 40960 0 - Live 0x0\n", encoding="ascii")
        self.dev = self.root / "dev"
        self.dev.mkdir()
        (self.dev / "dpu").write_bytes(b"")
        self.fdt = self.root / "fdt"
        self.fdt.write_bytes(b"deephi,dpu\0")
        self.lib_target = self.root / "lib-target"
        self.lib_target.mkdir()

        self._tool("uname", """#!/bin/sh
case "$1" in
  -m) echo "${TEST_UNAME_M:-armv7l}" ;;
  -r) echo "${TEST_UNAME_R:-4.14.0-xilinx}" ;;
  *) exit 64 ;;
esac
""")
        self._tool("modinfo", "#!/bin/sh\necho '4.14.0-xilinx SMP preempt mod_unload modversions ARMv7 p2v8'\n")
        self._tool("mount", "#!/bin/sh\nexit 0\n")
        self._tool("umount", "#!/bin/sh\nexit 0\n")
        self._tool("dmesg", "#!/bin/sh\ncat \"$DPU_TEST_DMESG\"\n")

        (self.bundle / "runtime/dpu-4.14.0-xilinx.ko").write_bytes(b"module")
        (self.bundle / "model/dpu_sound_shape_probe.elf").write_bytes(b"verified-model")
        (self.bundle / "dnndk/pkgs/lib/libn2cube.so").write_bytes(b"n2cube")
        (self.bundle / "dnndk/pkgs/lib/libhineon.so").write_bytes(b"hineon")
        (self.bundle / "frontend/logmel_cli").write_bytes(b"frontend")
        (self.bundle / "config/logmel_frontend_v1.json").write_bytes(
            (ROOT / "config/logmel_frontend_v1.json").read_bytes())
        (self.bundle / "manifest/dpu-input-contract.txt").write_text(
            "INPUT_SIZE=6144\nINPUT_SCALE=128\n", encoding="ascii")
        (self.bundle / "probe/run_sound_wav_dpu.sh").write_text("#!/bin/sh\n", encoding="ascii")
        self._write_runner()
        self._write_dexplorer()
        case_lines = []
        for index, case in enumerate(CASES):
            (self.bundle / f"inputs/{case}.wav").write_bytes(b"wav" + bytes([index]))
            (self.bundle / f"golden/{case}.f32le").write_bytes(b"f" * 24576)
            values = ((np.arange(6144) * (index * 12 + 1) + index)
                      % 121 - 60).astype(np.int8)
            (self.bundle / f"golden/{case}.i8").write_bytes(values.tobytes())
            start = int(case.rsplit("-", 1)[1])
            case_lines.append(f"{case}\tinputs/{case}.wav\tgolden/{case}.f32le\t"
                              f"golden/{case}.i8\t{start}\t128\n")
        (self.bundle / "manifest/cases.tsv").write_text(
            "".join(case_lines), encoding="ascii", newline="\n")
        model_sha = sha256(self.bundle / "model/dpu_sound_shape_probe.elf")
        runner_sha = sha256(self.bundle / "runtime/sound_shape_runner")
        (self.bundle / "manifest/contract.tsv").write_text(
            "SCHEMA\t2\nNORMALIZATION_PROFILE\tbounded-log-power-64-v1\n"
            f"CONFIG_SHA256\t{sha256(self.bundle / 'config/logmel_frontend_v1.json')}\n"
            "INPUT_SCALE\t128\nINPUT_SIZE\t6144\nMODEL_KERNEL\tsound_shape_probe\n"
            "INPUT_NODE\tconv1\nOUTPUT_NODE\tlogits\n"
            f"MODEL_SHA256\t{model_sha}\nRUNNER_SHA256\t{runner_sha}\n",
            encoding="ascii", newline="\n")
        self.write_sums()

    def tearDown(self):
        self.temporary.cleanup()

    def _tool(self, name, text):
        path = self.tools / name
        path.write_text(text, encoding="ascii", newline="\n")
        os.chmod(path, 0o755)

    def _write_dexplorer(self):
        path = self.bundle / "dnndk/pkgs/bin/dexplorer"
        path.write_text("""#!/bin/sh
case "$1" in
  -w) echo 'DPU Arch        : B1152F' ;;
  -s) value=$(cat "$DPU_TEST_COUNTER"); [ "${TEST_STATIC_COUNTERS:-0}" = 1 ] && value=0; echo "Core 0 schedule : $value"; echo "Core 0 interrupt: $value" ;;
  *) exit 64 ;;
esac
""", encoding="ascii", newline="\n")
        os.chmod(path, 0o755)

    def _write_runner(self):
        path = self.bundle / "runtime/sound_shape_runner"
        path.write_text("""#!/bin/sh
for arg in "$@"; do
  [ "$arg" = --describe-tensor ] && { echo DPU_TENSOR_STATUS=pass; echo KERNEL=sound_shape_probe; echo INPUT_NODE=conv1; echo INPUT_SIZE=6144; echo INPUT_SCALE=${TEST_RUNTIME_SCALE:-128}; echo DPU_RUN_CALLS=0; exit 0; }
done
[ "${TEST_RUNNER_FAIL:-0}" != 1 ] || exit 9
while [ $# -gt 0 ]; do
  case "$1" in
    --input-file) input=$2; shift 2 ;;
    --model-file) model=$2; shift 2 ;;
    --iterations) calls=$2; shift 2 ;;
    --dump) dump=$2; shift 2 ;;
    --report) report=$2; shift 2 ;;
    *) shift 2 ;;
  esac
done
if [ "${TEST_ZERO_OUTPUT:-0}" = 1 ]; then
  dd if=/dev/zero of="$dump" bs=1 count=8 2>/dev/null
else
  dd if="$input" of="$dump" bs=1 count=8 2>/dev/null
fi
old=$(cat "$DPU_TEST_COUNTER")
echo $((old + calls)) > "$DPU_TEST_COUNTER"
[ "${TEST_NEW_FAULT:-0}" != 1 ] || echo 'dpu timeout injected' >> "$DPU_TEST_DMESG"
input_sha=$(sha256sum "$input" | awk '{print $1}')
model_sha=$(sha256sum "$model" | awk '{print $1}')
output_sha=$(sha256sum "$dump" | awk '{print $1}')
mismatch=${TEST_MISMATCH_ITERATIONS:-0}
cat > "$report" <<EOF
DPU_SOUND_STATUS=pass
INPUT_MODE=file
INPUT_SCALE=${TEST_RUNTIME_SCALE:-128}
INPUT_SOURCE=file
INPUT_BYTES=6144
MODEL_SHA256=$model_sha
INPUT_SHA256=$input_sha
OUTPUT_SHA256=$output_sha
OUTPUT_VALUES=1,2,3,4,5,6,7,8
DPU_RUN_CALLS=$calls
MISMATCH_ITERATIONS=$mismatch
RUN_TOTAL_US=20
EOF
""", encoding="ascii", newline="\n")
        os.chmod(path, 0o755)

    def write_sums(self):
        names = []
        for path in sorted(self.bundle.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                names.append(path.relative_to(self.bundle).as_posix())
        text = "".join(f"{sha256(self.bundle / name)}  {name}\n" for name in names)
        (self.bundle / "manifest/SHA256SUMS").write_text(text, encoding="ascii", newline="\n")

    def run_probe(self, output_name, **overrides):
        env = os.environ.copy()
        env.update({
            "DPU_WAV_FDT": bash_path(self.fdt),
            "DPU_WAV_PROC_MODULES": bash_path(self.proc_modules),
            "DPU_WAV_DEV_ROOT": bash_path(self.dev),
            "DPU_WAV_LIB_TARGET": bash_path(self.lib_target),
            "DPU_TEST_COUNTER": bash_path(self.counter),
            "DPU_TEST_DMESG": bash_path(self.dmesg),
        })
        env.update(overrides)
        return subprocess.run([
            str(GIT_BASH), "-c", 'PATH="$1:$PATH"; export PATH; shift; exec "$@"',
            "board-test", bash_path(self.tools), bash_path(BOARD),
            "--bundle-dir", bash_path(self.bundle),
            "--output-dir", bash_path(self.root / output_name),
        ], env=env, text=True, encoding="utf-8", errors="replace", capture_output=True)

    def test_valid_bundle_runs_two_inferences_per_real_window(self):
        result = self.run_probe("pass")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DPU_WAV_STATUS=pass", result.stdout)
        self.assertIn("TOTAL_DPU_RUN_CALLS=6", result.stdout)
        self.assertIn("SCHEDULE_DELTA=6", result.stdout)
        self.assertIn("INTERRUPT_DELTA=6", result.stdout)

    def test_rejects_hash_and_tensor_size(self):
        with (self.bundle / f"inputs/{CASES[0]}.wav").open("ab") as stream:
            stream.write(b"corrupt")
        self.assertIn("bundle_hash_mismatch", self.run_probe("hash").stdout)

        self.tearDown()
        self.setUp()
        (self.bundle / f"golden/{CASES[0]}.i8").write_bytes(b"x" * 6143)
        self.write_sums()
        self.assertIn("tensor_size_mismatch", self.run_probe("size").stdout)

    def test_rejects_file_not_listed_in_checksum_manifest(self):
        (self.bundle / "runtime" / "unlisted-tool").write_bytes(b"unexpected")
        result = self.run_probe("extra")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unlisted_bundle_file", result.stdout)

    def test_rejects_scale_runner_zero_and_repeat_drift(self):
        self.assertIn("runtime_scale_mismatch", self.run_probe("scale", TEST_RUNTIME_SCALE="8").stdout)
        self.assertIn("runner_failed", self.run_probe("runner", TEST_RUNNER_FAIL="1").stdout)
        self.assertIn("all_zero_output", self.run_probe("zero", TEST_ZERO_OUTPUT="1").stdout)
        self.assertIn("repeat_mismatch", self.run_probe("drift", TEST_MISMATCH_ITERATIONS="1").stdout)

    def test_rejects_static_counters_new_fault_and_existing_output(self):
        self.assertIn("counter_delta_mismatch", self.run_probe("static", TEST_STATIC_COUNTERS="1").stdout)
        self.counter.write_text("0", encoding="ascii")
        self.assertIn("new_kernel_fault", self.run_probe("fault", TEST_NEW_FAULT="1").stdout)
        existing = self.root / "existing"
        existing.mkdir()
        self.assertIn("output_already_exists", self.run_probe("existing").stdout)

    def test_rejects_profile_before_any_dpu_call(self):
        contract = self.bundle / "manifest/contract.tsv"
        contract.write_text(contract.read_text(encoding="ascii").replace(
            "bounded-log-power-64-v1", "identity-64-v1"), encoding="ascii", newline="\n")
        self.write_sums()
        result = self.run_probe("profile")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("contract_invalid", result.stdout)
        self.assertEqual(self.counter.read_text(encoding="ascii"), "0")


if __name__ == "__main__":
    unittest.main()
