import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "scripts" / "board" / "run_soundjie_d3.sh"
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
CLASS_IDS = (
    "fire_smoke_alarm", "emergency_siren", "car_horn", "glass_breaking",
    "door_knock_bell", "distress_voice", "normal_speech", "background_unknown",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bash_path(path):
    path = Path(path).resolve()
    return f"/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


class SoundjieD3BoardProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        for folder in ("runtime", "model", "dnndk/pkgs/bin", "dnndk/pkgs/lib",
                       "config", "inputs", "demo", "probe", "manifest"):
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
case "$1" in -m) echo armv7l ;; -r) echo 4.14.0-xilinx ;; *) exit 64 ;; esac
""")
        self._tool("modinfo", "#!/bin/sh\necho '4.14.0-xilinx SMP ARMv7'\n")
        self._tool("mount", "#!/bin/sh\nexit 0\n")
        self._tool("umount", "#!/bin/sh\nexit 0\n")
        self._tool("dmesg", "#!/bin/sh\ncat \"$DPU_TEST_DMESG\"\n")

        (self.bundle / "runtime/dpu-4.14.0-xilinx.ko").write_bytes(b"module")
        (self.bundle / "model/dpu_soundjie_d3.elf").write_bytes(b"verified-model")
        (self.bundle / "dnndk/pkgs/lib/libn2cube.so").write_bytes(b"n2cube")
        (self.bundle / "dnndk/pkgs/lib/libhineon.so").write_bytes(b"hineon")
        (self.bundle / "config/logmel_frontend_v1.json").write_text("{}", encoding="ascii")
        (self.bundle / "config/soundjie_d3_model.json").write_text("{}", encoding="ascii")
        (self.bundle / "model/labels.txt").write_text("\n".join(CLASS_IDS) + "\n", encoding="ascii")
        (self.bundle / "manifest/soundjie-d3-model-manifest.json").write_text("{}", encoding="ascii")
        (self.bundle / "manifest/prepared-manifest.jsonl").write_text("", encoding="ascii")
        (self.bundle / "manifest/demo.jsonl").write_text("", encoding="ascii")
        (self.bundle / "manifest/calibration-manifest.json").write_text("{}", encoding="ascii")
        (self.bundle / "manifest/ATTRIBUTION.md").write_text("fixture attribution\n", encoding="ascii")
        (self.bundle / "probe/run_soundjie_d3.sh").write_text("#!/bin/sh\n", encoding="ascii")
        self._write_runner()
        self._write_dexplorer()

        case_lines = []
        for class_index, class_id in enumerate(CLASS_IDS):
            for split, ordinal in (("test", 0), ("demo", 0), ("demo", 1)):
                case_id = f"{split}-{class_index}-{ordinal}"
                tensor = self.bundle / f"inputs/{case_id}.i8"
                values = ((np.arange(6144) + class_index * 9 + ordinal) % 241 - 120).astype(np.int8)
                tensor.write_bytes(values.tobytes())
                wav = "-"
                if split == "demo":
                    wav = f"demo/{case_id}.wav"
                    (self.bundle / wav).write_bytes(b"wav" + bytes([class_index, ordinal]))
                case_lines.append("\t".join((
                    case_id, split, f"inputs/{case_id}.i8", wav, str(class_index), class_id,
                    f"recording:{case_id}", f"group:{case_id}", sha256(tensor),
                )) + "\n")
        (self.bundle / "manifest/cases.tsv").write_text("".join(case_lines), encoding="ascii", newline="\n")
        model_sha = sha256(self.bundle / "model/dpu_soundjie_d3.elf")
        runner_sha = sha256(self.bundle / "runtime/sound_shape_runner")
        (self.bundle / "manifest/contract.tsv").write_text(
            "SCHEMA\t1\nCLASS_COUNT\t8\nCASE_COUNT\t24\n"
            "MODEL_KERNEL\tsoundjie_d3\nINPUT_NODE\tconv1\nOUTPUT_NODE\tfc8\n"
            "INPUT_SHAPE\t1,1,64,96\nINPUT_SIZE\t6144\nINPUT_SCALE\t128\n"
            "OUTPUT_SIZE\t8\nOUTPUT_SCALE\t0.125\n"
            f"MODEL_SHA256\t{model_sha}\nRUNNER_SHA256\t{runner_sha}\n"
            f"LABELS_SHA256\t{sha256(self.bundle / 'model/labels.txt')}\n",
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
  -w) echo 'DPU Arch : B1152F' ;;
  -s) value=$(cat "$DPU_TEST_COUNTER"); [ "${TEST_STATIC_COUNTERS:-0}" = 1 ] && value=0; echo "Core 0 schedule : $value"; echo "Core 0 interrupt: $value" ;;
  *) exit 64 ;;
esac
""", encoding="ascii", newline="\n")
        os.chmod(path, 0o755)

    def _write_runner(self):
        path = self.bundle / "runtime/sound_shape_runner"
        path.write_text("""#!/bin/sh
describe=0
for arg in "$@"; do [ "$arg" = --describe-tensor ] && describe=1; done
if [ "$describe" = 1 ]; then
  cat <<EOF
DPU_TENSOR_STATUS=pass
KERNEL=soundjie_d3
INPUT_NODE=${TEST_INPUT_NODE:-conv1}
OUTPUT_NODE=${TEST_OUTPUT_NODE:-fc8}
INPUT_SIZE=${TEST_INPUT_SIZE:-6144}
INPUT_SCALE=${TEST_INPUT_SCALE:-128}
OUTPUT_SIZE=${TEST_OUTPUT_SIZE:-8}
OUTPUT_SCALE=${TEST_OUTPUT_SCALE:-0.125}
DPU_RUN_CALLS=0
EOF
  exit 0
fi
while [ $# -gt 0 ]; do
  case "$1" in
    --input-file) input=$2; shift 2 ;; --model-file) model=$2; shift 2 ;;
    --iterations) calls=$2; shift 2 ;; --dump) dump=$2; shift 2 ;;
    --report) report=$2; shift 2 ;; *) shift 2 ;;
  esac
done
count=${TEST_OUTPUT_COUNT:-8}
if [ "${TEST_ZERO_OUTPUT:-0}" = 1 ]; then
  dd if=/dev/zero of="$dump" bs=1 count="$count" 2>/dev/null
else
  dd if="$input" of="$dump" bs=1 count="$count" 2>/dev/null
fi
old=$(cat "$DPU_TEST_COUNTER"); echo $((old + calls)) > "$DPU_TEST_COUNTER"
values=$(od -An -td1 -v "$dump" | awk '{for(i=1;i<=NF;i++){if(n++)printf ",";printf "%d",$i}}')
cat > "$report" <<EOF
DPU_SOUND_STATUS=pass
INPUT_SCALE=${TEST_INPUT_SCALE:-128}
OUTPUT_SCALE=${TEST_OUTPUT_SCALE:-0.125}
INPUT_SOURCE=file
INPUT_BYTES=6144
MODEL_SHA256=$(sha256sum "$model" | awk '{print $1}')
INPUT_SHA256=$(sha256sum "$input" | awk '{print $1}')
OUTPUT_SHA256=$(sha256sum "$dump" | awk '{print $1}')
OUTPUT_VALUES=$values
OUTPUT_BYTES=$count
DPU_RUN_CALLS=$calls
MISMATCH_ITERATIONS=0
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

    def run_probe(self, name, **overrides):
        env = os.environ.copy()
        env.update({
            "SJD3_FDT": bash_path(self.fdt),
            "SJD3_PROC_MODULES": bash_path(self.proc_modules),
            "SJD3_DEV_ROOT": bash_path(self.dev),
            "SJD3_LIB_TARGET": bash_path(self.lib_target),
            "DPU_TEST_COUNTER": bash_path(self.counter),
            "DPU_TEST_DMESG": bash_path(self.dmesg),
        })
        env.update(overrides)
        return subprocess.run([
            str(GIT_BASH), "-c", 'PATH="$1:$PATH"; export PATH; shift; exec "$@"',
            "board-test", bash_path(self.tools), bash_path(BOARD),
            "--bundle-dir", bash_path(self.bundle),
            "--output-dir", bash_path(self.root / name),
        ], env=env, text=True, encoding="utf-8", errors="replace", capture_output=True)

    def assert_pre_dpu_rejected(self, result, reason):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(reason, result.stdout)
        self.assertEqual(self.counter.read_text(encoding="ascii"), "0")
        self.assertIn("DPU_RUN_CALLS=0", result.stdout)

    def test_fails_before_dpu_on_shape_hash_fixpoint_or_node_drift(self):
        first = self.bundle / "inputs/test-0-0.i8"
        first.write_bytes(first.read_bytes() + b"x")
        self.assert_pre_dpu_rejected(self.run_probe("hash"), "bundle_hash_mismatch")

        self.tearDown(); self.setUp()
        self.assert_pre_dpu_rejected(self.run_probe("shape", TEST_INPUT_SIZE="6000"),
                                     "runtime_shape_mismatch")
        self.assert_pre_dpu_rejected(self.run_probe("fix", TEST_OUTPUT_SCALE="0.25"),
                                     "runtime_fixpoint_mismatch")
        self.assert_pre_dpu_rejected(self.run_probe("node", TEST_OUTPUT_NODE="wrong"),
                                     "runtime_node_mismatch")

    def test_fails_on_zero_or_non_eight_output(self):
        zero = self.run_probe("zero", TEST_ZERO_OUTPUT="1")
        self.assertNotEqual(zero.returncode, 0)
        self.assertIn("all_zero_output", zero.stdout)
        self.counter.write_text("0", encoding="ascii")
        bad_count = self.run_probe("count", TEST_OUTPUT_COUNT="7")
        self.assertNotEqual(bad_count.returncode, 0)
        self.assertIn("output_value_count_wrong", bad_count.stdout)

    def test_success_requires_schedule_and_interrupt_delta(self):
        result = self.run_probe("pass")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SJD3_STATUS=pass", result.stdout)
        self.assertIn("TOTAL_DPU_RUN_CALLS=24", result.stdout)
        predictions = self.root / "pass/predictions.jsonl"
        rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(len(row["raw_logits"]) == 8 for row in rows))
        self.assertTrue(all(len(row["dequantized_logits"]) == 8 for row in rows))
        self.assertTrue(all(0.0 <= row["confidence"] <= 1.0 for row in rows))

        self.tearDown(); self.setUp()
        static = self.run_probe("static", TEST_STATIC_COUNTERS="1")
        self.assertNotEqual(static.returncode, 0)
        self.assertIn("counter_delta_mismatch", static.stdout)


if __name__ == "__main__":
    unittest.main()
