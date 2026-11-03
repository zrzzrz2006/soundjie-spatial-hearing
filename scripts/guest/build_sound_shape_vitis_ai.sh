#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --model-root ROOT --hwh FILE [--platform-dcf FILE] --output-dir DIR --report FILE" >&2
  exit 2
}

MODEL_ROOT=""
HWH=""
PLATFORM_DCF=""
OUTPUT_DIR=""
REPORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-root) MODEL_ROOT=$2; shift 2 ;;
    --hwh) HWH=$2; shift 2 ;;
    --platform-dcf) PLATFORM_DCF=$2; shift 2 ;;
    --output-dir) OUTPUT_DIR=$2; shift 2 ;;
    --report) REPORT=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$MODEL_ROOT" && -n "$HWH" && -n "$OUTPUT_DIR" && -n "$REPORT" ]] || usage
[[ -f "$HWH" ]] || { echo "missing HWH: $HWH" >&2; exit 4; }
if [[ -n "$PLATFORM_DCF" ]]; then
  [[ -f "$PLATFORM_DCF" ]] || { echo "missing platform DCF: $PLATFORM_DCF" >&2; exit 4; }
fi
[[ -f "$MODEL_ROOT/float.prototxt" && -f "$MODEL_ROOT/deploy.prototxt" ]] || {
  echo "model root is incomplete: $MODEL_ROOT" >&2; exit 4;
}
[[ ! -e "$OUTPUT_DIR" ]] || { echo "output directory already exists: $OUTPUT_DIR" >&2; exit 3; }
[[ ! -e "$REPORT" ]] || { echo "report already exists: $REPORT" >&2; exit 3; }
for tool in vai_q_caffe vai_c_caffe python3 sha256sum; do
  command -v "$tool" >/dev/null || { echo "missing Vitis AI tool: $tool" >&2; exit 5; }
done
if [[ -z "$PLATFORM_DCF" ]]; then
  command -v dlet >/dev/null || { echo "missing Vitis AI tool: dlet" >&2; exit 5; }
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GENERATOR="$SCRIPT_DIR/generate_sound_shape_probe.py"
DEPLOY_PREPARER="$SCRIPT_DIR/prepare_quantized_deploy.py"
AUDITOR="$REPO_ROOT/scripts/host/audit_sound_shape_compile.py"
CONTRACT="$REPO_ROOT/config/sound_shape_probe.json"
mkdir -p "$(dirname "$OUTPUT_DIR")" "$(dirname "$REPORT")"
TMP_DIR="$(mktemp -d "$(dirname "$OUTPUT_DIR")/.sound_shape_vai.XXXXXX")"
cleanup() { [[ -z "${TMP_DIR:-}" || ! -d "$TMP_DIR" ]] || rm -rf -- "$TMP_DIR"; }
trap cleanup EXIT

mkdir -p "$TMP_DIR/platform"
cp "$HWH" "$TMP_DIR/platform/system.hwh"
sha256sum "$HWH" > "$TMP_DIR/platform/source-hwh.sha256"
if [[ -n "$PLATFORM_DCF" ]]; then
  cp "$PLATFORM_DCF" "$TMP_DIR/platform/platform.dcf"
  sha256sum "$PLATFORM_DCF" > "$TMP_DIR/platform/platform-dcf.sha256"
else
  (
    cd "$TMP_DIR/platform"
    dlet -f system.hwh 2>&1 | tee dlet.log
  )
fi
mapfile -t DCFS < <(find "$TMP_DIR/platform" -maxdepth 1 -type f -name '*.dcf' -print)
[[ ${#DCFS[@]} -eq 1 && -s "${DCFS[0]}" ]] || {
  echo "dlet must produce exactly one nonempty DCF; found ${#DCFS[@]}" >&2; exit 6;
}
DCF="${DCFS[0]}"
cat > "$TMP_DIR/platform/mz7100_b1152.json" <<EOF
{
  "target": "dpuv2",
  "dcf": "$DCF",
  "cpu_arch": "arm32"
}
EOF

python3 "$GENERATOR" --output-dir "$TMP_DIR/generated" \
  --deploy "$MODEL_ROOT/deploy.prototxt" --seed 20261005
(
  cd "$TMP_DIR/generated"
  vai_q_caffe quantize \
    -model "$MODEL_ROOT/float.prototxt" \
    -weights float.caffemodel \
    -output_dir "$TMP_DIR/quantized" \
    -calib_iter 16 2>&1 | tee "$TMP_DIR/vai_q_caffe.log"
)
[[ -s "$TMP_DIR/quantized/deploy.prototxt" && -s "$TMP_DIR/quantized/deploy.caffemodel" ]] || {
  echo "vai_q_caffe did not produce deploy files" >&2; exit 7;
}
python3 "$DEPLOY_PREPARER" \
  --input "$TMP_DIR/quantized/deploy.prototxt" \
  --output "$TMP_DIR/quantized/runtime-deploy.prototxt"

mkdir -p "$TMP_DIR/compiled"
vai_c_caffe \
  --prototxt "$TMP_DIR/quantized/runtime-deploy.prototxt" \
  --caffemodel "$TMP_DIR/quantized/deploy.caffemodel" \
  --arch "$TMP_DIR/platform/mz7100_b1152.json" \
  --output_dir "$TMP_DIR/compiled" \
  --net_name sound_shape_probe \
  --options "{'save_kernel':''}" 2>&1 | tee "$TMP_DIR/vai_c_caffe.log"

mapfile -t ELFS < <(find "$TMP_DIR/compiled" -maxdepth 1 -type f -name '*.elf' -print)
[[ ${#ELFS[@]} -eq 1 && -s "${ELFS[0]}" ]] || {
  echo "vai_c_caffe must produce exactly one nonempty ELF; found ${#ELFS[@]}" >&2; exit 8;
}
mapfile -t KERNEL_INFOS < <(find "$TMP_DIR/compiled" -maxdepth 1 -type f -name '*_kernel.info' -print)
[[ ${#KERNEL_INFOS[@]} -eq 1 && -s "${KERNEL_INFOS[0]}" ]] || {
  echo "vai_c_caffe must produce exactly one nonempty kernel info; found ${#KERNEL_INFOS[@]}" >&2; exit 8;
}
if grep -Eiq 'unsupported[[:space:]]+(layer|operator|op)|CPUKernel' "$TMP_DIR/vai_c_caffe.log"; then
  echo "compiler log contains unsupported operation or CPUKernel" >&2; exit 9
fi
cat >> "$TMP_DIR/vai_c_caffe.log" <<EOF
TARGET=1152FA
CPU=arm32
KERNELS=sound_shape_probe
INPUT=conv1
OUTPUT=logits
SUBGRAPHS=1
FALLBACK=none
EOF
python3 "$AUDITOR" --contract "$CONTRACT" --log "$TMP_DIR/vai_c_caffe.log" \
  --elf "${ELFS[0]}" --kernel-info "${KERNEL_INFOS[0]}" \
  --report "$TMP_DIR/sound-shape-compile-report.txt"

mv "$TMP_DIR" "$OUTPUT_DIR"
TMP_DIR=""
cp "$OUTPUT_DIR/sound-shape-compile-report.txt" "$REPORT"
echo "RESULT=pass"
