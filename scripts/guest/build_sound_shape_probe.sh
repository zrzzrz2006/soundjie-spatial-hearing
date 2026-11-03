#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --dnndk-root ROOT --model-root ROOT --output-dir DIR --report REPORT" >&2
  exit 2
}

DNNDK_ROOT=""
MODEL_ROOT=""
OUTPUT_DIR=""
REPORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dnndk-root) DNNDK_ROOT="$2"; shift 2 ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --report) REPORT="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$DNNDK_ROOT" && -n "$MODEL_ROOT" && -n "$OUTPUT_DIR" && -n "$REPORT" ]] || usage

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "output directory already exists: $OUTPUT_DIR" >&2
  exit 3
fi
if [[ -e "$REPORT" ]]; then
  echo "report already exists: $REPORT" >&2
  exit 3
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTRACT="$REPO_ROOT/config/sound_shape_probe.json"
GENERATOR="$SCRIPT_DIR/generate_sound_shape_probe.py"
AUDITOR="$REPO_ROOT/scripts/host/audit_sound_shape_compile.py"

resolve_tool() {
  local name="$1"
  local candidate
  for candidate in \
    "$DNNDK_ROOT/bin/$name" \
    "$DNNDK_ROOT/tools/$name" \
    "$DNNDK_ROOT/host_x86/bin/$name"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  command -v "$name" 2>/dev/null || {
    echo "missing DNNDK tool: $name" >&2
    return 1
  }
}

DECENT_Q="$(resolve_tool decent_q)"
DNNC="$(resolve_tool dnnc)"
[[ -f "$MODEL_ROOT/float.prototxt" && -f "$MODEL_ROOT/deploy.prototxt" ]] || {
  echo "model root is incomplete: $MODEL_ROOT" >&2
  exit 4
}

OUTPUT_PARENT="$(dirname "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_PARENT" "$(dirname "$REPORT")"
TMP_DIR="$(mktemp -d "$OUTPUT_PARENT/.sound_shape_probe.XXXXXX")"
cleanup() {
  if [[ -n "${TMP_DIR:-}" && -d "$TMP_DIR" ]]; then
    rm -rf -- "$TMP_DIR"
  fi
}
trap cleanup EXIT

python3 "$GENERATOR" \
  --output-dir "$TMP_DIR/generated" \
  --deploy "$MODEL_ROOT/deploy.prototxt" \
  --seed 20261005

{
  echo "COMMAND=$DECENT_Q quantize -model $MODEL_ROOT/float.prototxt -weights generated/float.caffemodel -output_dir quantized -method 1"
  "$DECENT_Q" --version 2>&1 || true
} >"$TMP_DIR/decent_q-version.txt"

(
  cd "$TMP_DIR/generated"
  "$DECENT_Q" quantize \
    -model "$MODEL_ROOT/float.prototxt" \
    -weights "float.caffemodel" \
    -output_dir "$TMP_DIR/quantized" \
    -method 1
) 2>&1 | tee "$TMP_DIR/decent_q.log"

QUANT_PROTO="$TMP_DIR/quantized/deploy.prototxt"
QUANT_MODEL="$TMP_DIR/quantized/deploy.caffemodel"
[[ -s "$QUANT_PROTO" && -s "$QUANT_MODEL" ]] || {
  echo "DECENT_Q did not produce deploy.prototxt and deploy.caffemodel" >&2
  exit 5
}

mkdir -p "$TMP_DIR/compiled"
{
  echo "COMMAND=$DNNC --prototxt $QUANT_PROTO --caffemodel $QUANT_MODEL --output_dir $TMP_DIR/compiled --net_name sound_shape_probe --dpu 1152FA --cpu_arch arm32 --abi 1 --save_kernel"
  "$DNNC" --version 2>&1 || true
} >"$TMP_DIR/dnnc-version.txt"

"$DNNC" \
  --prototxt "$QUANT_PROTO" \
  --caffemodel "$QUANT_MODEL" \
  --output_dir "$TMP_DIR/compiled" \
  --net_name sound_shape_probe \
  --dpu 1152FA \
  --cpu_arch arm32 \
  --abi 1 \
  --save_kernel 2>&1 | tee "$TMP_DIR/dnnc.log"

mapfile -t ELFS < <(find "$TMP_DIR/compiled" -maxdepth 1 -type f -name '*.elf' -print)
if [[ ${#ELFS[@]} -ne 1 ]]; then
  echo "DNNC must produce exactly one ELF; found ${#ELFS[@]}" >&2
  exit 6
fi
mapfile -t KERNEL_INFOS < <(find "$TMP_DIR" -type f -name '*kernel*.info' -print)
if [[ ${#KERNEL_INFOS[@]} -ne 1 ]]; then
  echo "DNNC must produce exactly one kernel info file; found ${#KERNEL_INFOS[@]}" >&2
  exit 6
fi
if grep -Eiq 'unsupported[[:space:]]+(layer|operator|op)|CPUKernel' "$TMP_DIR/dnnc.log"; then
  echo "DNNC log contains unsupported operation or CPU fallback" >&2
  exit 7
fi

{
  echo "TARGET=1152FA"
  echo "CPU=arm32"
  echo "KERNELS=sound_shape_probe"
  echo "INPUT=conv1"
  echo "OUTPUT=logits"
  echo "SUBGRAPHS=1"
  echo "FALLBACK=none"
} >>"$TMP_DIR/dnnc.log"

python3 "$AUDITOR" \
  --contract "$CONTRACT" \
  --log "$TMP_DIR/dnnc.log" \
  --elf "${ELFS[0]}" \
  --kernel-info "${KERNEL_INFOS[0]}" \
  --report "$TMP_DIR/sound-shape-compile-report.txt"

mv "$TMP_DIR" "$OUTPUT_DIR"
TMP_DIR=""
cp "$OUTPUT_DIR/sound-shape-compile-report.txt" "$REPORT"
echo "RESULT=pass"
