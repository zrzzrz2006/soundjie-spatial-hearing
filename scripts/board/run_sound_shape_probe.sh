#!/bin/sh
set -eu

bundle_dir=
output_dir=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bundle-dir) bundle_dir=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

fail() { echo "DPU_SOUND_STATUS=fail"; echo "REASON=$1"; exit "${2:-1}"; }
case "$bundle_dir:$output_dir" in /*:/*) ;; *) fail path_must_be_absolute 64 ;; esac

module="$bundle_dir/runtime/dpu-4.14.0-xilinx.ko"
runner="$bundle_dir/runtime/sound_shape_probe"
model="$bundle_dir/model/dpu_sound_shape_probe.elf"
dexplorer="$bundle_dir/dnndk/pkgs/bin/dexplorer"
libs="$bundle_dir/dnndk/pkgs/lib"
for path in "$module" "$runner" "$model" "$dexplorer" "$libs/libn2cube.so" "$libs/libhineon.so"; do
  [ -f "$path" ] || fail missing_required_file
done

[ "$(uname -m)" = armv7l ] || fail wrong_architecture
[ "$(uname -r)" = 4.14.0-xilinx ] || fail wrong_kernel_release
fdt=${DPU_SOUND_FDT:-/sys/firmware/fdt}
[ -r "$fdt" ] && grep -a -q 'deephi,dpu' "$fdt" || fail missing_dpu_device_tree_node
case "$(modinfo -F vermagic "$module" 2>/dev/null || true)" in 4.14.0-xilinx*) ;; *) fail module_vermagic_mismatch ;; esac
proc_modules=${DPU_SOUND_PROC_MODULES:-/proc/modules}
dev_root=${DPU_SOUND_DEV_ROOT:-/dev}
grep -q '^dpu ' "$proc_modules" 2>/dev/null || insmod "$module" || fail module_load_failed
grep -q '^dpu ' "$proc_modules" 2>/dev/null || fail module_not_registered
[ -e "$dev_root/dpu" ] || fail missing_dpu_character_device

[ ! -e "$output_dir" ] || fail output_already_exists
mkdir -p "$output_dir/zero" "$output_dir/pattern" "$output_dir/stress"
before="$output_dir/dexplorer-before.txt"
after="$output_dir/dexplorer-after.txt"
dmesg_before=$(dmesg | wc -l)

lib_target=${DPU_SOUND_LIB_TARGET:-/usr/local/lib}
[ -d "$lib_target" ] || fail missing_dexplorer_lib_target
export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mounted=no
cleanup() { [ "$mounted" != yes ] || umount "$lib_target" >/dev/null 2>&1 || true; }
trap cleanup 0 HUP INT TERM
mount --bind "$libs" "$lib_target" || fail dexplorer_bind_mount_failed
mounted=yes
"$dexplorer" -w > "$output_dir/dexplorer-whoami.txt" 2>&1 || fail dexplorer_failed
grep -Eqi 'B1152|DPU' "$output_dir/dexplorer-whoami.txt" || fail dpu_signature_missing
"$dexplorer" -s > "$before" 2>&1 || fail dexplorer_status_failed

run_one() {
  mode=$1; iterations=$2; dir=$3
  "$runner" --kernel sound_shape_probe --input-node conv1 --output-node logits \
    --input "$mode" --iterations "$iterations" --dump "$dir/logits.bin" \
    --report "$dir/report.txt" || fail "${mode}_inference_failed"
  grep -q '^DPU_SOUND_STATUS=pass$' "$dir/report.txt" || fail runner_report_failed
  grep -q "^DPU_RUN_CALLS=$iterations$" "$dir/report.txt" || fail runner_call_count_wrong
  grep -q '^MISMATCH_ITERATIONS=0$' "$dir/report.txt" || fail stress_repeat_mismatch
  [ -s "$dir/logits.bin" ] || fail empty_output_dump
  od -An -tu1 "$dir/logits.bin" | awk '{for(i=1;i<=NF;i++) if($i!=0) found=1} END{exit(found?0:1)}' || fail all_zero_output_dump
  sha256sum "$dir/logits.bin" | awk '{print $1}' > "$dir/logits.sha256"
}
run_one zero 1 "$output_dir/zero"
run_one pattern 1 "$output_dir/pattern"
stress_iterations=${DPU_SOUND_STRESS_ITERATIONS:-100}
case "$stress_iterations" in ''|*[!0-9]*) fail invalid_stress_iterations 64 ;; esac
[ "$stress_iterations" -gt 0 ] || fail invalid_stress_iterations 64
run_one pattern "$stress_iterations" "$output_dir/stress"
cmp -s "$output_dir/zero/logits.sha256" "$output_dir/pattern/logits.sha256" && fail zero_pattern_outputs_identical
cmp -s "$output_dir/pattern/logits.sha256" "$output_dir/stress/logits.sha256" || fail pattern_stress_outputs_differ

"$dexplorer" -s > "$after" 2>&1 || fail dexplorer_status_failed
umount "$lib_target" || fail dexplorer_unmount_failed
mounted=no
counter() {
  awk -v key="$2" 'index(tolower($0),key){for(i=NF;i>=1;i--){v=$i;gsub(/[^0-9]/,"",v);if(v~/^[0-9]+$/){n+=v;f=1;break}}} END{if(!f)exit 1;print n}' "$1"
}
schedule_before=$(counter "$before" schedule) || fail schedule_counter_missing
schedule_after=$(counter "$after" schedule) || fail schedule_counter_missing
interrupt_before=$(counter "$before" interrupt) || fail interrupt_counter_missing
interrupt_after=$(counter "$after" interrupt) || fail interrupt_counter_missing
schedule_delta=$((schedule_after - schedule_before))
interrupt_delta=$((interrupt_after - interrupt_before))
[ "$schedule_delta" -gt 0 ] || fail schedule_counter_not_advanced
[ "$interrupt_delta" -gt 0 ] || fail interrupt_counter_not_advanced

kernel_log="$output_dir/kernel-new.txt"
dmesg | tail -n "+$((dmesg_before + 1))" > "$kernel_log"
if grep -Eqi 'dpu.*(timeout|hang|fault|serror)|dma.*fault|irq.*(timeout|error)|usb.*(disconnect|reset)|mmc.*(error|timeout)|ext4.*(error|abort)' "$kernel_log"; then
  fail new_kernel_fault
fi
output_sha=$(cat "$output_dir/pattern/logits.sha256")
cat > "$output_dir/board-report.txt" <<EOF
DPU_SOUND_STATUS=pass
MODEL_KERNEL=sound_shape_probe
INPUT_NODE=conv1
OUTPUT_NODE=logits
OUTPUT_BYTES=8
OUTPUT_SHA256=$output_sha
STRESS_ITERATIONS=$stress_iterations
SCHEDULE_DELTA=$schedule_delta
INTERRUPT_DELTA=$interrupt_delta
NEW_KERNEL_FAULTS=0
EOF
cat "$output_dir/board-report.txt"
