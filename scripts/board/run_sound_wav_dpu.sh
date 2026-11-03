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

fail() { echo "DPU_WAV_STATUS=fail"; echo "REASON=$1"; exit "${2:-1}"; }
case "$bundle_dir:$output_dir" in /*:/*) ;; *) fail path_must_be_absolute 64 ;; esac
[ ! -e "$output_dir" ] || fail output_already_exists

module="$bundle_dir/runtime/dpu-4.14.0-xilinx.ko"
runner="$bundle_dir/runtime/sound_shape_runner"
model="$bundle_dir/model/dpu_sound_shape_probe.elf"
dexplorer="$bundle_dir/dnndk/pkgs/bin/dexplorer"
libs="$bundle_dir/dnndk/pkgs/lib"
cases="$bundle_dir/manifest/cases.tsv"
contract="$bundle_dir/manifest/contract.tsv"
sums="$bundle_dir/manifest/SHA256SUMS"
for path in "$module" "$runner" "$model" "$dexplorer" \
  "$libs/libn2cube.so" "$libs/libhineon.so" "$cases" "$contract" "$sums"; do
  [ -f "$path" ] || fail missing_required_file
done

validate_sums() {
  awk '
    length($1) != 64 { exit 1 }
    { hash=$1; gsub(/[0-9a-f]/, "", hash); if (hash != "") exit 1 }
    substr($0,65,2) != "  " { exit 1 }
    {
      name=substr($0,67)
      if (name == "" || name ~ /^\// || name ~ /\\/ || name ~ /(^|\/)\.\.($|\/)/ || seen[name]++) exit 1
    }
  ' "$sums"
}
validate_sums || fail unsafe_checksum_manifest
actual_files=$(cd "$bundle_dir" && find . -type f | sed 's#^\./##' | sort)
expected_files=$({ awk '{print substr($0,67)}' "$sums"; printf '%s\n' manifest/SHA256SUMS; } | sort)
[ "$actual_files" = "$expected_files" ] || fail unlisted_bundle_file
(cd "$bundle_dir" && sha256sum -c manifest/SHA256SUMS >/dev/null 2>&1) || fail bundle_hash_mismatch

expected_cases=$(printf 'xvf3800-start-0000000000\tinputs/xvf3800-start-0000000000.wav\tgolden/xvf3800-start-0000000000.f32le\tgolden/xvf3800-start-0000000000.i8\t0\t128\nxvf3800-start-0000072000\tinputs/xvf3800-start-0000072000.wav\tgolden/xvf3800-start-0000072000.f32le\tgolden/xvf3800-start-0000072000.i8\t72000\t128\nxvf3800-start-0000144000\tinputs/xvf3800-start-0000144000.wav\tgolden/xvf3800-start-0000144000.f32le\tgolden/xvf3800-start-0000144000.i8\t144000\t128\n')
[ "$(cat "$cases")" = "$expected_cases" ] || fail case_manifest_mismatch

contract_value() {
  awk -F '\t' -v key="$1" '$1==key { if (++count > 1) exit 2; value=$2 }
    END { if (count != 1) exit 1; print value }' "$contract"
}
input_scale=$(contract_value INPUT_SCALE) || fail contract_invalid
input_size=$(contract_value INPUT_SIZE) || fail contract_invalid
schema=$(contract_value SCHEMA) || fail contract_invalid
profile=$(contract_value NORMALIZATION_PROFILE) || fail contract_invalid
config_sha=$(contract_value CONFIG_SHA256) || fail contract_invalid
kernel=$(contract_value MODEL_KERNEL) || fail contract_invalid
input_node=$(contract_value INPUT_NODE) || fail contract_invalid
output_node=$(contract_value OUTPUT_NODE) || fail contract_invalid
model_sha=$(contract_value MODEL_SHA256) || fail contract_invalid
runner_sha=$(contract_value RUNNER_SHA256) || fail contract_invalid
[ "$input_scale" = 128 ] && [ "$input_size" = 6144 ] || fail contract_invalid
[ "$schema" = 2 ] && [ "$profile" = bounded-log-power-64-v1 ] || fail contract_invalid
[ "$(sha256sum "$bundle_dir/config/logmel_frontend_v1.json" | awk '{print $1}')" = "$config_sha" ] || fail contract_invalid
[ "$kernel" = sound_shape_probe ] && [ "$input_node" = conv1 ] && [ "$output_node" = logits ] || fail contract_invalid
[ "$(sha256sum "$model" | awk '{print $1}')" = "$model_sha" ] || fail model_hash_mismatch
[ "$(sha256sum "$runner" | awk '{print $1}')" = "$runner_sha" ] || fail runner_hash_mismatch

tab=$(printf '\t')
tensor_paths=
tensor_hashes=
while IFS="$tab" read -r case_name wav_path float_path tensor_path start scale; do
  [ -f "$bundle_dir/$wav_path" ] && [ -f "$bundle_dir/$float_path" ] && [ -f "$bundle_dir/$tensor_path" ] || fail missing_case_file
  [ "$(wc -c < "$bundle_dir/$tensor_path" | tr -d ' ')" = 6144 ] || fail tensor_size_mismatch
  [ "$scale" = "$input_scale" ] || fail case_scale_mismatch
  tensor="$bundle_dir/$tensor_path"
  stats=$(od -An -td1 -v "$tensor" | awk '
    { for(i=1;i<=NF;i++){ seen[$i]=1; if($i==-128) low++; if($i==127) high++; count++ } }
    END { for(value in seen) unique++; print unique, low+0, high+0, count+0 }') || fail tensor_stats_failed
  set -- $stats
  [ "$4" = 6144 ] && [ "$1" -ge 32 ] || fail tensor_quality_failed
  [ $(( $2 * 100 )) -lt "$4" ] && [ $(( $3 * 100 )) -lt "$4" ] || fail tensor_quality_failed
  hash=$(sha256sum "$tensor" | awk '{print $1}')
  case "$tensor_hashes" in *" $hash "*) fail tensor_hash_not_distinct ;; esac
  tensor_hashes="$tensor_hashes $hash "
  tensor_paths="$tensor_paths $tensor"
done < "$cases"
set -- $tensor_paths
[ "$#" -eq 3 ] || fail case_manifest_mismatch
t1=$1; t2=$2; t3=$3
for pair in "$t1 $t2" "$t1 $t3" "$t2 $t3"; do
  set -- $pair
  differing=$(cmp -l "$1" "$2" 2>/dev/null | wc -l | tr -d ' ' || true)
  [ "$differing" -ge 3072 ] || fail tensor_pairwise_difference_low
done

[ "$(uname -m)" = armv7l ] || fail wrong_architecture
[ "$(uname -r)" = 4.14.0-xilinx ] || fail wrong_kernel_release
fdt=${DPU_WAV_FDT:-/sys/firmware/fdt}
[ -r "$fdt" ] && grep -a -q 'deephi,dpu' "$fdt" || fail missing_dpu_device_tree_node
case "$(modinfo -F vermagic "$module" 2>/dev/null || true)" in 4.14.0-xilinx*) ;; *) fail module_vermagic_mismatch ;; esac
proc_modules=${DPU_WAV_PROC_MODULES:-/proc/modules}
dev_root=${DPU_WAV_DEV_ROOT:-/dev}
grep -q '^dpu ' "$proc_modules" 2>/dev/null || insmod "$module" || fail module_load_failed
grep -q '^dpu ' "$proc_modules" 2>/dev/null || fail module_not_registered
[ -e "$dev_root/dpu" ] || fail missing_dpu_character_device

mkdir -p "$output_dir"
dmesg_before=$(dmesg | wc -l)
lib_target=${DPU_WAV_LIB_TARGET:-/usr/local/lib}
[ -d "$lib_target" ] || fail missing_dexplorer_lib_target
export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mounted=no
cleanup() { [ "$mounted" != yes ] || umount "$lib_target" >/dev/null 2>&1 || true; }
trap cleanup 0 HUP INT TERM
mount --bind "$libs" "$lib_target" || fail dexplorer_bind_mount_failed
mounted=yes
"$dexplorer" -w > "$output_dir/dexplorer-whoami.txt" 2>&1 || fail dexplorer_failed
grep -Eqi 'B1152|DPU' "$output_dir/dexplorer-whoami.txt" || fail dpu_signature_missing
"$dexplorer" -s > "$output_dir/dexplorer-before.txt" 2>&1 || fail dexplorer_status_failed

counter() {
  awk -v key="$2" 'index(tolower($0),key) { for(i=NF;i>=1;i--) { value=$i; gsub(/[^0-9]/,"",value); if(value~/^[0-9]+$/){total+=value;found=1;break} } }
    END { if(!found) exit 1; print total }' "$1"
}
schedule_before=$(counter "$output_dir/dexplorer-before.txt" schedule) || fail schedule_counter_missing
interrupt_before=$(counter "$output_dir/dexplorer-before.txt" interrupt) || fail interrupt_counter_missing

"$runner" --kernel "$kernel" --input-node "$input_node" --describe-tensor > "$output_dir/tensor-description.txt" 2>&1 || fail describe_failed
grep -q '^DPU_TENSOR_STATUS=pass$' "$output_dir/tensor-description.txt" || fail describe_failed
grep -q "^INPUT_SIZE=$input_size$" "$output_dir/tensor-description.txt" || fail runtime_size_mismatch
runtime_scale=$(awk -F= '$1=="INPUT_SCALE" {print $2}' "$output_dir/tensor-description.txt")
awk -v actual="$runtime_scale" -v expected="$input_scale" 'BEGIN { exit !(actual+0 == expected+0 && actual+0 > 0) }' || fail runtime_scale_mismatch
grep -q '^DPU_RUN_CALLS=0$' "$output_dir/tensor-description.txt" || fail describe_ran_dpu
"$dexplorer" -s > "$output_dir/dexplorer-after-describe.txt" 2>&1 || fail dexplorer_status_failed
[ "$(counter "$output_dir/dexplorer-after-describe.txt" schedule)" = "$schedule_before" ] || fail describe_advanced_counter
[ "$(counter "$output_dir/dexplorer-after-describe.txt" interrupt)" = "$interrupt_before" ] || fail describe_advanced_counter

field() { awk -F= -v key="$2" '$1==key { if(++count>1) exit 2; value=$2 } END{ if(count!=1) exit 1; print value }' "$1"; }
total_calls=0
while IFS="$tab" read -r case_name wav_path float_path tensor_path start scale; do
  case_dir="$output_dir/$case_name"
  mkdir -p "$case_dir"
  tensor="$bundle_dir/$tensor_path"
  dump="$case_dir/logits.bin"
  report="$case_dir/report.txt"
  "$runner" --kernel "$kernel" --input-node "$input_node" --output-node "$output_node" \
    --input-file "$tensor" --model-file "$model" --iterations 2 \
    --dump "$dump" --report "$report" || fail runner_failed
  [ "$(field "$report" DPU_SOUND_STATUS)" = pass ] || fail runner_report_failed
  [ "$(field "$report" INPUT_SOURCE)" = file ] || fail runner_report_failed
  [ "$(field "$report" INPUT_BYTES)" = 6144 ] || fail runner_report_failed
  awk -v actual="$(field "$report" INPUT_SCALE)" -v expected="$input_scale" 'BEGIN {exit !(actual+0 == expected+0)}' || fail runtime_scale_mismatch
  [ "$(field "$report" MODEL_SHA256)" = "$model_sha" ] || fail model_hash_mismatch
  [ "$(field "$report" INPUT_SHA256)" = "$(sha256sum "$tensor" | awk '{print $1}')" ] || fail input_hash_mismatch
  [ "$(field "$report" DPU_RUN_CALLS)" = 2 ] || fail runner_call_count_wrong
  [ "$(field "$report" MISMATCH_ITERATIONS)" = 0 ] || fail repeat_mismatch
  [ "$(field "$report" OUTPUT_VALUES | awk -F, '{print NF}')" = 8 ] || fail output_value_count_wrong
  [ -s "$dump" ] || fail empty_output
  od -An -tu1 "$dump" | awk '{for(i=1;i<=NF;i++) if($i!=0) found=1} END{exit(found?0:1)}' || fail all_zero_output
  [ "$(field "$report" OUTPUT_SHA256)" = "$(sha256sum "$dump" | awk '{print $1}')" ] || fail output_hash_mismatch
  total_calls=$((total_calls + 2))
done < "$cases"

"$dexplorer" -s > "$output_dir/dexplorer-after.txt" 2>&1 || fail dexplorer_status_failed
schedule_after=$(counter "$output_dir/dexplorer-after.txt" schedule) || fail schedule_counter_missing
interrupt_after=$(counter "$output_dir/dexplorer-after.txt" interrupt) || fail interrupt_counter_missing
schedule_delta=$((schedule_after - schedule_before))
interrupt_delta=$((interrupt_after - interrupt_before))
[ "$schedule_delta" -eq "$total_calls" ] && [ "$interrupt_delta" -eq "$total_calls" ] || fail counter_delta_mismatch

dmesg | tail -n "+$((dmesg_before + 1))" > "$output_dir/kernel-new.txt"
if grep -Eqi 'dpu.*(timeout|hang|fault|serror)|dma.*fault|irq.*(timeout|error)|mmc.*(error|timeout)|ext4.*(error|abort)' "$output_dir/kernel-new.txt"; then
  fail new_kernel_fault
fi
umount "$lib_target" || fail dexplorer_unmount_failed
mounted=no

cat > "$output_dir/board-report.txt" <<EOF
DPU_WAV_STATUS=pass
INPUT_SCALE=$input_scale
INPUT_SIZE=$input_size
REAL_WINDOW_COUNT=3
RUNS_PER_WINDOW=2
TOTAL_DPU_RUN_CALLS=$total_calls
SCHEDULE_DELTA=$schedule_delta
INTERRUPT_DELTA=$interrupt_delta
NEW_KERNEL_FAULTS=0
EOF
cat "$output_dir/board-report.txt"
