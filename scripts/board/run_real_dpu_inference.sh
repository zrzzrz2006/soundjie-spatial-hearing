#!/bin/sh
set -eu

bundle_dir=
output_dir=
level=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bundle-dir) bundle_dir=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --level) level=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

fail() {
  echo "DPU_INFERENCE_STATUS=fail"
  echo "REASON=$1"
  exit "${2:-1}"
}

case "$bundle_dir:$output_dir" in
  /*:/*) ;;
  *) fail path_must_be_absolute 64 ;;
esac
[ "$level" = yolo ] || fail unsupported_level 64

module="$bundle_dir/runtime/dpu-4.14.0-xilinx.ko"
runner="$bundle_dir/runtime/dpu_yolo_probe"
dexplorer="$bundle_dir/dnndk/pkgs/bin/dexplorer"
dnndk_lib="$bundle_dir/dnndk/pkgs/lib"
for required in "$module" "$runner" "$dexplorer" "$dnndk_lib/libn2cube.so" "$dnndk_lib/libhineon.so"; do
  [ -f "$required" ] || fail missing_required_file
done

[ "$(uname -m)" = armv7l ] || fail wrong_architecture
[ "$(uname -r)" = 4.14.0-xilinx ] || fail wrong_kernel_release
fdt=${DPU_INFERENCE_FDT:-/sys/firmware/fdt}
[ -r "$fdt" ] && grep -a -q 'deephi,dpu' "$fdt" || fail missing_dpu_device_tree_node
vermagic=$(modinfo -F vermagic "$module" 2>/dev/null || true)
case "$vermagic" in 4.14.0-xilinx*) ;; *) fail module_vermagic_mismatch ;; esac

proc_modules=${DPU_INFERENCE_PROC_MODULES:-/proc/modules}
dev_root=${DPU_INFERENCE_DEV_ROOT:-/dev}
if ! grep -q '^dpu ' "$proc_modules" 2>/dev/null; then
  insmod "$module" || fail module_load_failed
fi
grep -q '^dpu ' "$proc_modules" 2>/dev/null || fail module_not_registered
[ -e "$dev_root/dpu" ] || fail missing_dpu_character_device

mkdir -p "$output_dir"
zero_dir="$output_dir/zero"
pattern_dir="$output_dir/pattern"
stress_dir="$output_dir/stress"
for path in "$zero_dir" "$pattern_dir" "$stress_dir"; do
  [ ! -e "$path" ] || fail output_already_exists
  mkdir -p "$path"
done

before="$output_dir/dexplorer-before.txt"
after="$output_dir/dexplorer-after.txt"
kernel_log="$output_dir/level-a-kernel-log.txt"
counter_report="$output_dir/level-a-counter-delta.txt"
summary="$output_dir/level-a-board-report.txt"
dmesg_before=$(dmesg | wc -l)

lib_target=${DPU_INFERENCE_LIB_TARGET:-/usr/local/lib}
[ -d "$lib_target" ] || fail missing_dexplorer_lib_target
LD_LIBRARY_PATH="$dnndk_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
mounted=no
cleanup_mount() {
  if [ "$mounted" = yes ]; then
    umount "$lib_target" >/dev/null 2>&1 || true
  fi
}
trap cleanup_mount 0 HUP INT TERM
mount --bind "$dnndk_lib" "$lib_target" || fail dexplorer_bind_mount_failed
mounted=yes
"$dexplorer" -w > "$output_dir/dexplorer-whoami.txt" 2>&1 || fail dexplorer_failed
grep -Eqi 'B1152|DPU' "$output_dir/dexplorer-whoami.txt" || fail dpu_signature_missing
"$dexplorer" -s > "$before" 2>&1 || fail dexplorer_status_failed

output_args='--output-node layer81_conv --output-node layer93_conv --output-node layer105_conv --output-node layer117_conv'

# shellcheck disable=SC2086
"$runner" --kernel yolo --input-node layer0_conv $output_args --input zero --iterations 1 \
  --dump-dir "$zero_dir" --report "$zero_dir/report.txt" || fail zero_inference_failed
# shellcheck disable=SC2086
"$runner" --kernel yolo --input-node layer0_conv $output_args --input pattern --iterations 1 \
  --dump-dir "$pattern_dir" --report "$pattern_dir/report.txt" || fail pattern_inference_failed
stress_iterations=${DPU_INFERENCE_STRESS_ITERATIONS:-100}
case "$stress_iterations" in ''|*[!0-9]*) fail invalid_stress_iterations 64 ;; esac
[ "$stress_iterations" -gt 0 ] || fail invalid_stress_iterations 64
# shellcheck disable=SC2086
"$runner" --kernel yolo --input-node layer0_conv $output_args --input pattern --iterations "$stress_iterations" \
  --dump-dir "$stress_dir" --report "$stress_dir/report.txt" || fail stress_inference_failed

for item in "$zero_dir:1" "$pattern_dir:1" "$stress_dir:$stress_iterations"; do
  dir=${item%:*}; calls=${item##*:}
  grep -q '^DPU_RUN_STATUS=pass$' "$dir/report.txt" || fail runner_report_failed
  grep -q "^DPU_RUN_CALLS=$calls$" "$dir/report.txt" || fail runner_call_count_wrong
done
grep -q '^MISMATCH_ITERATIONS=0$' "$stress_dir/report.txt" || fail stress_repeat_mismatch

hash_outputs() {
  dir=$1
  manifest=$2
  : > "$manifest"
  for node in layer81_conv layer93_conv layer105_conv layer117_conv; do
    file="$dir/$node.bin"
    [ -s "$file" ] || fail empty_output_dump
    if ! od -An -tu1 "$file" | awk '{ for (i=1; i<=NF; i++) if ($i != 0) found=1 } END { exit(found ? 0 : 1) }'; then
      fail all_zero_output_dump
    fi
    digest=$(sha256sum "$file" | awk '{print $1}')
    echo "$node $digest" >> "$manifest"
  done
}
hash_outputs "$zero_dir" "$zero_dir/SHA256SUMS"
hash_outputs "$pattern_dir" "$pattern_dir/SHA256SUMS"
hash_outputs "$stress_dir" "$stress_dir/SHA256SUMS"
cmp -s "$zero_dir/SHA256SUMS" "$pattern_dir/SHA256SUMS" && fail zero_pattern_outputs_identical
cmp -s "$pattern_dir/SHA256SUMS" "$stress_dir/SHA256SUMS" || fail pattern_stress_outputs_differ

"$dexplorer" -s > "$after" 2>&1 || fail dexplorer_status_failed
if ! umount "$lib_target"; then fail dexplorer_unmount_failed; fi
mounted=no

counter_total() {
  awk -v key="$2" '
    index(tolower($0), key) {
      for (i=NF; i>=1; i--) {
        value=$i; gsub(/[^0-9]/, "", value)
        if (value ~ /^[0-9]+$/) { total += value; found=1; break }
      }
    }
    END { if (!found) exit 1; print total }
  ' "$1"
}
schedule_before=$(counter_total "$before" schedule) || fail schedule_counter_missing
schedule_after=$(counter_total "$after" schedule) || fail schedule_counter_missing
interrupt_before=$(counter_total "$before" interrupt) || fail interrupt_counter_missing
interrupt_after=$(counter_total "$after" interrupt) || fail interrupt_counter_missing
schedule_delta=$((schedule_after - schedule_before))
interrupt_delta=$((interrupt_after - interrupt_before))
[ "$schedule_delta" -gt 0 ] || fail schedule_counter_not_advanced
[ "$interrupt_delta" -gt 0 ] || fail interrupt_counter_not_advanced

dmesg | tail -n "+$((dmesg_before + 1))" > "$kernel_log"
if grep -Eqi 'dpu.*(timeout|hang|fault|serror)|dma.*fault|irq.*(timeout|error)|usb.*(disconnect|reset)|mmc.*(error|timeout)|ext4.*(error|abort)' "$kernel_log"; then
  fail new_kernel_fault
fi

cat > "$counter_report" <<EOF
SCHEDULE_BEFORE=$schedule_before
SCHEDULE_AFTER=$schedule_after
SCHEDULE_DELTA=$schedule_delta
INTERRUPT_BEFORE=$interrupt_before
INTERRUPT_AFTER=$interrupt_after
INTERRUPT_DELTA=$interrupt_delta
EOF
cat > "$summary" <<EOF
DPU_INFERENCE_STATUS=pass
LEVEL=yolo
ARCH=armv7l
KERNEL=4.14.0-xilinx
MODEL_KERNEL=yolo
INPUT_NODE=layer0_conv
OUTPUT_NODE_COUNT=4
ZERO_ITERATIONS=1
PATTERN_ITERATIONS=1
STRESS_ITERATIONS=$stress_iterations
SCHEDULE_DELTA=$schedule_delta
INTERRUPT_DELTA=$interrupt_delta
ZERO_PATTERN_DIFFER=yes
PATTERN_STRESS_MATCH=yes
NEW_KERNEL_FAULTS=0
EOF
cat "$summary"
