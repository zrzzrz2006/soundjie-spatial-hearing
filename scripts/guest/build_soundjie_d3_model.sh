#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --input-root DIR --platform-dcf FILE --dnnc-bin FILE --dnnc-lib-dir DIR --output-dir DIR --report FILE" >&2
  exit 64
}

INPUT_ROOT=""; PLATFORM_DCF=""; DNNC_BIN=""; DNNC_LIBRARY_PATH=""; OUTPUT_DIR=""; REPORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root) INPUT_ROOT=$2; shift 2 ;;
    --platform-dcf) PLATFORM_DCF=$2; shift 2 ;;
    --dnnc-bin) DNNC_BIN=$2; shift 2 ;;
    --dnnc-lib-dir) DNNC_LIBRARY_PATH=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --report) REPORT=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$INPUT_ROOT" && -n "$PLATFORM_DCF" && -n "$DNNC_BIN" && -n "$DNNC_LIBRARY_PATH" && -n "$OUTPUT_DIR" && -n "$REPORT" ]] || usage
[[ ! -e "$OUTPUT_DIR" ]] || { echo "output directory already exists: $OUTPUT_DIR" >&2; exit 3; }
[[ ! -e "$REPORT" ]] || { echo "report already exists: $REPORT" >&2; exit 3; }
for file in deploy.prototxt soundjie_d3_float32.caffemodel calibration.npy calibration-targets.npy calibration-manifest.json demo-manifest.json parity-inputs.npy pytorch-logits.npy fused-manifest.json caffe-export-manifest.json; do
  [[ -s "$INPUT_ROOT/$file" ]] || { echo "missing input: $file" >&2; exit 4; }
done
[[ -s "$PLATFORM_DCF" ]] || { echo "missing platform DCF" >&2; exit 4; }
[[ -x "$DNNC_BIN" ]] || { echo "missing executable legacy dnnc: $DNNC_BIN" >&2; exit 4; }
[[ -d "$DNNC_LIBRARY_PATH" ]] || { echo "missing legacy dnnc library directory: $DNNC_LIBRARY_PATH" >&2; exit 4; }
: "${PS1:=}"
source /etc/profile.d/conda.sh
conda activate vitis-ai-caffe
for tool in python3 vai_q_caffe sha256sum; do
  command -v "$tool" >/dev/null || { echo "missing tool: $tool" >&2; exit 5; }
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AUDITOR="$REPO_ROOT/scripts/host/audit_soundjie_d3_model.py"
TMP_DIR="$(mktemp -d "$(dirname "$OUTPUT_DIR")/.soundjie_d3.XXXXXX")"
FAILED_DIR="${OUTPUT_DIR}.failed"
cleanup() {
  status=$?
  if [[ -n "${TMP_DIR:-}" && -d "$TMP_DIR" ]]; then
    if [[ $status -ne 0 && ! -e "$FAILED_DIR" ]]; then
      mv "$TMP_DIR" "$FAILED_DIR"
      echo "preserved failed build diagnostics: $FAILED_DIR" >&2
    else
      rm -rf -- "$TMP_DIR"
    fi
  fi
  return "$status"
}
trap cleanup EXIT

cat > "$TMP_DIR/architecture.json" <<EOF
{"target":"dpuv2","dcf":"$PLATFORM_DCF","cpu_arch":"arm32"}
EOF
printf '%s\n' 'TARGET=1152FA' 'CPU=arm32' 'ABI=1' 'KERNELS=soundjie_d3' > "$TMP_DIR/compile-contract.txt"
LD_LIBRARY_PATH="$DNNC_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$DNNC_BIN" --version 2>&1 | tee "$TMP_DIR/dnnc-version.txt"
grep -Eq '^dnnc version v2\.(04|05)$' "$TMP_DIR/dnnc-version.txt" || {
  echo "legacy dnnc version is not v2.04/v2.05" >&2; exit 6;
}
grep -Eq '^DPU Target : v1\.4\.0$' "$TMP_DIR/dnnc-version.txt" || {
  echo "legacy dnnc target is not v1.4.0" >&2; exit 6;
}
python3 "$AUDITOR" verify-calibration \
  --calibration-manifest "$INPUT_ROOT/calibration-manifest.json" \
  --demo-manifest "$INPUT_ROOT/demo-manifest.json"
python3 "$AUDITOR" make-assets --input-root "$INPUT_ROOT" --work-dir "$TMP_DIR"

# The calibration model and HDF5 tensor are prepared by the host auditor from
# calibration-manifest.json after proving no recording/source-group overlaps
# demo-manifest.json.  Quantization therefore never sees demo material.
printf '%q ' vai_q_caffe quantize \
  -model "$TMP_DIR/quantize.prototxt" \
  -weights "$INPUT_ROOT/soundjie_d3_float32.caffemodel" \
  -output_dir "$TMP_DIR/quantized" -calib_iter 64 \
  > "$TMP_DIR/quantize-command.txt"
printf '\n' >> "$TMP_DIR/quantize-command.txt"
vai_q_caffe quantize \
  -model "$TMP_DIR/quantize.prototxt" \
  -weights "$INPUT_ROOT/soundjie_d3_float32.caffemodel" \
  -output_dir "$TMP_DIR/quantized" -calib_iter 64 2>&1 | tee "$TMP_DIR/vai_q_caffe.log"

python3 "$AUDITOR" make-runtime \
  --input "$TMP_DIR/quantized/deploy.prototxt" \
  --output "$TMP_DIR/quantized/runtime-deploy.prototxt"

python3 "$AUDITOR" evaluate \
  --prototxt "$TMP_DIR/quantized/runtime-deploy.prototxt" \
  --caffemodel "$TMP_DIR/quantized/deploy.caffemodel" \
  --inputs "$INPUT_ROOT/parity-inputs.npy" \
  --reference-logits "$INPUT_ROOT/pytorch-logits.npy" \
  --targets-manifest "$INPUT_ROOT/fused-manifest.json" \
  --output "$TMP_DIR/quantization-metrics.json"

mkdir "$TMP_DIR/compiled"
mkdir "$TMP_DIR/dnnc-logs"
printf '%q ' "$DNNC_BIN" --parser caffe \
  --prototxt "$TMP_DIR/quantized/runtime-deploy.prototxt" \
  --caffemodel "$TMP_DIR/quantized/deploy.caffemodel" \
  --dpu 1152FA --cpu_arch arm32 --abi 1 --mode normal \
  --output_dir "$TMP_DIR/compiled" --net_name soundjie_d3 \
  > "$TMP_DIR/compile-command.txt"
printf '\n' >> "$TMP_DIR/compile-command.txt"
LD_LIBRARY_PATH="$DNNC_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
GLOG_logtostderr=1 GLOG_log_dir="$TMP_DIR/dnnc-logs" \
  "$DNNC_BIN" \
  --parser caffe \
  --prototxt "$TMP_DIR/quantized/runtime-deploy.prototxt" \
  --caffemodel "$TMP_DIR/quantized/deploy.caffemodel" \
  --dpu 1152FA --cpu_arch arm32 --abi 1 --mode normal \
  --output_dir "$TMP_DIR/compiled" --net_name soundjie_d3 \
  2>&1 | tee "$TMP_DIR/dnnc-v2.05.log"

if grep -Eiq 'CPUKernel|unsupported[[:space:]]+(layer|operator|op)' "$TMP_DIR/dnnc-v2.05.log"; then
  echo "CPUKernel or unsupported operation found" >&2; exit 9
fi
mapfile -t ELFS < <(find "$TMP_DIR/compiled" -maxdepth 1 -type f -name 'dpu_*.elf' -print)
[[ ${#ELFS[@]} -eq 1 ]] || { echo "expected exactly one DPU ELF" >&2; exit 10; }
python3 "$AUDITOR" audit \
  --contract "$REPO_ROOT/config/soundjie_d3_compile.json" \
  --compile-log "$TMP_DIR/dnnc-v2.05.log" \
  --elf "${ELFS[0]}" --kernel-info "$TMP_DIR/dnnc-v2.05.log" \
  --prototxt "$TMP_DIR/quantized/runtime-deploy.prototxt" \
  --fused-manifest "$INPUT_ROOT/fused-manifest.json" \
  --caffe-export-manifest "$INPUT_ROOT/caffe-export-manifest.json" \
  --float-deploy "$INPUT_ROOT/deploy.prototxt" \
  --float-caffemodel "$INPUT_ROOT/soundjie_d3_float32.caffemodel" \
  --quantized-caffemodel "$TMP_DIR/quantized/deploy.caffemodel" \
  --compiler-binary "$DNNC_BIN" \
  --compiler-version "$TMP_DIR/dnnc-version.txt" \
  --platform-dcf "$PLATFORM_DCF" \
  --quantize-command "$TMP_DIR/quantize-command.txt" \
  --compile-command "$TMP_DIR/compile-command.txt" \
  --metrics "$TMP_DIR/quantization-metrics.json" \
  --calibration-manifest "$INPUT_ROOT/calibration-manifest.json" \
  --output "$TMP_DIR/soundjie-d3-model-manifest.json"
echo "RESULT=pass" > "$TMP_DIR/soundjie-d3-model-report.txt"
echo "ABI=1" >> "$TMP_DIR/soundjie-d3-model-report.txt"
python3 - "$TMP_DIR/soundjie-d3-model-manifest.json" >> "$TMP_DIR/soundjie-d3-model-report.txt" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
e=value["compile_evidence"]; m=value["quantization_metrics"]
print("DNNC_VERSION=" + e["dnnc_version"])
print("DPU_TARGET=" + e["dpu_target"])
print("DPU=" + e["dpu"])
print("KERNEL=" + e["kernels"][0])
print("OUTPUT_COUNT=" + str(e["output_count"]))
print("REFERENCE_MACRO_F1=" + str(m["reference_macro_f1"]))
print("QUANTIZED_MACRO_F1=" + str(m["quantized_macro_f1"]))
print("MACRO_F1_DELTA=" + str(m["macro_f1_delta"]))
PY
mv "$TMP_DIR" "$OUTPUT_DIR"; TMP_DIR=""
cp "$OUTPUT_DIR/soundjie-d3-model-report.txt" "$REPORT"
echo "RESULT=pass"
