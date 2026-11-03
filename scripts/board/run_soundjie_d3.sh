#!/bin/sh
set -eu

bundle_dir=
output_dir=
dpu_calls=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bundle-dir) bundle_dir=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

fail() { echo "SJD3_STATUS=fail"; echo "REASON=$1"; echo "DPU_RUN_CALLS=$dpu_calls"; exit "${2:-1}"; }
case "$bundle_dir:$output_dir" in /*:/*) ;; *) fail path_must_be_absolute 64 ;; esac
[ ! -e "$output_dir" ] || fail output_already_exists

module="$bundle_dir/runtime/dpu-4.14.0-xilinx.ko"
runner="$bundle_dir/runtime/sound_shape_runner"
model="$bundle_dir/model/dpu_soundjie_d3.elf"
dexplorer="$bundle_dir/dnndk/pkgs/bin/dexplorer"
libs="$bundle_dir/dnndk/pkgs/lib"
labels="$bundle_dir/model/labels.txt"
cases="$bundle_dir/manifest/cases.tsv"
contract="$bundle_dir/manifest/contract.tsv"
sums="$bundle_dir/manifest/SHA256SUMS"
for path in "$module" "$runner" "$model" "$dexplorer" "$labels" "$cases" "$contract" "$sums" \
  "$libs/libn2cube.so" "$libs/libhineon.so"; do
  [ -f "$path" ] || fail missing_required_file
done

awk '
  length($1) != 64 { exit 1 }
  { hash=$1; gsub(/[0-9a-f]/, "", hash); if (hash != "") exit 1 }
  substr($0,65,2) != "  " { exit 1 }
  { name=substr($0,67); if(name=="" || name~/^\// || name~/\\/ || name~/(^|\/)\.\.($|\/)/ || seen[name]++) exit 1 }
' "$sums" || fail unsafe_checksum_manifest
actual_files=$(cd "$bundle_dir" && find . -type f | sed 's#^\./##' | sort)
expected_files=$({ awk '{print substr($0,67)}' "$sums"; printf '%s\n' manifest/SHA256SUMS; } | sort)
[ "$actual_files" = "$expected_files" ] || fail unlisted_bundle_file
(cd "$bundle_dir" && sha256sum -c manifest/SHA256SUMS >/dev/null 2>&1) || fail bundle_hash_mismatch

value() {
  awk -F '\t' -v key="$1" '$1==key {if(++n>1)exit 2;v=$2} END{if(n!=1)exit 1;print v}' "$contract"
}
schema=$(value SCHEMA) || fail contract_invalid
class_count=$(value CLASS_COUNT) || fail contract_invalid
case_count=$(value CASE_COUNT) || fail contract_invalid
kernel=$(value MODEL_KERNEL) || fail contract_invalid
input_node=$(value INPUT_NODE) || fail contract_invalid
output_node=$(value OUTPUT_NODE) || fail contract_invalid
input_shape=$(value INPUT_SHAPE) || fail contract_invalid
input_size=$(value INPUT_SIZE) || fail contract_invalid
input_scale=$(value INPUT_SCALE) || fail contract_invalid
output_size=$(value OUTPUT_SIZE) || fail contract_invalid
output_scale=$(value OUTPUT_SCALE) || fail contract_invalid
model_sha=$(value MODEL_SHA256) || fail contract_invalid
runner_sha=$(value RUNNER_SHA256) || fail contract_invalid
labels_sha=$(value LABELS_SHA256) || fail contract_invalid
[ "$schema:$class_count:$case_count" = 1:8:24 ] || fail contract_invalid
[ "$kernel:$input_node:$output_node" = soundjie_d3:conv1:fc8 ] || fail contract_invalid
[ "$input_shape:$input_size:$output_size" = 1,1,64,96:6144:8 ] || fail contract_invalid
awk -v a="$input_scale" -v b="$output_scale" 'BEGIN{exit !(a+0>0 && b+0>0)}' || fail contract_invalid
[ "$(sha256sum "$model" | awk '{print $1}')" = "$model_sha" ] || fail model_hash_mismatch
[ "$(sha256sum "$runner" | awk '{print $1}')" = "$runner_sha" ] || fail runner_hash_mismatch
[ "$(sha256sum "$labels" | awk '{print $1}')" = "$labels_sha" ] || fail labels_hash_mismatch
awk 'BEGIN{e[1]="fire_smoke_alarm";e[2]="emergency_siren";e[3]="car_horn";e[4]="glass_breaking";
           e[5]="door_knock_bell";e[6]="distress_voice";e[7]="normal_speech";e[8]="background_unknown"}
     {sub(/\r$/,"");if(NR>8 || $0!=e[NR])exit 1} END{if(NR!=8)exit 1}' "$labels" || fail labels_contract_mismatch

tab=$(printf '\t')
rows=0
while IFS="$tab" read -r case_id split tensor_path wav_path target_index target_id recording_id source_group tensor_sha; do
  rows=$((rows + 1))
  case "$case_id:$split:$tensor_path:$target_index:$target_id" in
    *[!a-zA-Z0-9_.:/-]*) fail case_manifest_invalid ;;
  esac
  [ "$tensor_path" = "inputs/$case_id.i8" ] || fail case_manifest_invalid
  tensor="$bundle_dir/$tensor_path"
  [ -f "$tensor" ] || fail missing_case_file
  [ "$(wc -c < "$tensor" | tr -d ' ')" = "$input_size" ] || fail tensor_size_mismatch
  [ "$(sha256sum "$tensor" | awk '{print $1}')" = "$tensor_sha" ] || fail input_hash_mismatch
  if [ "$split" = demo ]; then
    [ "$wav_path" = "demo/$case_id.wav" ] && [ -f "$bundle_dir/$wav_path" ] || fail missing_demo_wav
  else
    [ "$split" = test ] && [ "$wav_path" = - ] || fail case_manifest_invalid
  fi
done < "$cases"
[ "$rows" -eq "$case_count" ] || fail case_manifest_invalid

[ "$(uname -m)" = armv7l ] || fail wrong_architecture
[ "$(uname -r)" = 4.14.0-xilinx ] || fail wrong_kernel_release
fdt=${SJD3_FDT:-/sys/firmware/fdt}
[ -r "$fdt" ] && grep -a -q 'deephi,dpu' "$fdt" || fail missing_dpu_device_tree_node
case "$(modinfo -F vermagic "$module" 2>/dev/null || true)" in 4.14.0-xilinx*) ;; *) fail module_vermagic_mismatch ;; esac
proc_modules=${SJD3_PROC_MODULES:-/proc/modules}
dev_root=${SJD3_DEV_ROOT:-/dev}
grep -q '^dpu ' "$proc_modules" 2>/dev/null || insmod "$module" || fail module_load_failed
grep -q '^dpu ' "$proc_modules" 2>/dev/null || fail module_not_registered
[ -e "$dev_root/dpu" ] || fail missing_dpu_character_device

mkdir -p "$output_dir"
dmesg_before=$(dmesg | wc -l)
lib_target=${SJD3_LIB_TARGET:-/usr/local/lib}
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
  awk -v key="$2" 'index(tolower($0),key){for(i=NF;i>=1;i--){v=$i;gsub(/[^0-9]/,"",v);if(v~/^[0-9]+$/){t+=v;f=1;break}}}END{if(!f)exit 1;print t}' "$1"
}
schedule_before=$(counter "$output_dir/dexplorer-before.txt" schedule) || fail schedule_counter_missing
interrupt_before=$(counter "$output_dir/dexplorer-before.txt" interrupt) || fail interrupt_counter_missing

"$runner" --kernel "$kernel" --input-node "$input_node" --output-node "$output_node" \
  --describe-tensor > "$output_dir/tensor-description.txt" 2>&1 || fail describe_failed
field() { awk -F= -v key="$2" '$1==key{if(++n>1)exit 2;v=$2}END{if(n!=1)exit 1;print v}' "$1"; }
description="$output_dir/tensor-description.txt"
[ "$(field "$description" DPU_TENSOR_STATUS)" = pass ] || fail describe_failed
[ "$(field "$description" INPUT_NODE):$(field "$description" OUTPUT_NODE)" = "$input_node:$output_node" ] || fail runtime_node_mismatch
[ "$(field "$description" INPUT_SIZE):$(field "$description" OUTPUT_SIZE)" = "$input_size:$output_size" ] || fail runtime_shape_mismatch
awk -v a="$(field "$description" INPUT_SCALE)" -v b="$input_scale" \
       -v c="$(field "$description" OUTPUT_SCALE)" -v d="$output_scale" \
       'BEGIN{exit !(a+0==b+0 && c+0==d+0 && a+0>0 && c+0>0)}' || fail runtime_fixpoint_mismatch
[ "$(field "$description" DPU_RUN_CALLS)" = 0 ] || fail describe_ran_dpu

predictions="$output_dir/predictions.jsonl"
: > "$predictions"
while IFS="$tab" read -r case_id split tensor_path wav_path target_index target_id recording_id source_group tensor_sha; do
  case_dir="$output_dir/$case_id"
  mkdir "$case_dir"
  tensor="$bundle_dir/$tensor_path"
  dump="$case_dir/logits.i8"
  report="$case_dir/report.txt"
  "$runner" --kernel "$kernel" --input-node "$input_node" --output-node "$output_node" \
    --input-file "$tensor" --model-file "$model" --iterations 1 \
    --dump "$dump" --report "$report" || fail runner_failed
  dpu_calls=$((dpu_calls + 1))
  [ "$(field "$report" DPU_SOUND_STATUS)" = pass ] || fail runner_report_failed
  [ "$(field "$report" DPU_RUN_CALLS)" = 1 ] || fail runner_call_count_wrong
  [ "$(field "$report" INPUT_SHA256)" = "$tensor_sha" ] || fail input_hash_mismatch
  [ "$(field "$report" MODEL_SHA256)" = "$model_sha" ] || fail model_hash_mismatch
  awk -v a="$(field "$report" INPUT_SCALE)" -v b="$input_scale" \
       -v c="$(field "$report" OUTPUT_SCALE)" -v d="$output_scale" \
       'BEGIN{exit !(a+0==b+0 && c+0==d+0)}' || fail runtime_fixpoint_mismatch
  values=$(field "$report" OUTPUT_VALUES) || fail runner_report_failed
  [ "$(printf '%s\n' "$values" | awk -F, '{print NF}')" = "$output_size" ] || fail output_value_count_wrong
  [ "$(wc -c < "$dump" | tr -d ' ')" = "$output_size" ] || fail output_value_count_wrong
  printf '%s\n' "$values" | awk -F, '{for(i=1;i<=NF;i++)if($i!=0)f=1}END{exit(f?0:1)}' || fail all_zero_output
  [ "$(field "$report" OUTPUT_SHA256)" = "$(sha256sum "$dump" | awk '{print $1}')" ] || fail output_hash_mismatch
  result=$(printf '%s\n' "$values" | awk -F, -v scale="$output_scale" '
    {m=-1e300;best=0;for(i=1;i<=NF;i++){v[i]=$i*scale;if(v[i]>m){m=v[i];best=i-1}}
     sum=0;for(i=1;i<=NF;i++)sum+=exp(v[i]-m);conf=exp(v[best+1]-m)/sum;
     printf "%d\t%.9g\t[",best,conf;for(i=1;i<=NF;i++){if(i>1)printf ",";printf "%.9g",v[i]}printf "]"}')
  predicted_index=$(printf '%s' "$result" | cut -f1)
  confidence=$(printf '%s' "$result" | cut -f2)
  dequantized=$(printf '%s' "$result" | cut -f3)
  predicted_id=$(sed -n "$((predicted_index + 1))p" "$labels")
  runtime_us=$(field "$report" RUN_TOTAL_US)
  printf '{"case_id":"%s","split":"%s","expected_index":%s,"expected_id":"%s","predicted_index":%s,"predicted_id":"%s","confidence":%s,"raw_logits":[%s],"dequantized_logits":%s,"runtime_us":%s,"input_sha256":"%s","output_sha256":"%s"}\n' \
    "$case_id" "$split" "$target_index" "$target_id" "$predicted_index" "$predicted_id" \
    "$confidence" "$values" "$dequantized" "$runtime_us" "$tensor_sha" \
    "$(field "$report" OUTPUT_SHA256)" >> "$predictions"
done < "$cases"

"$dexplorer" -s > "$output_dir/dexplorer-after.txt" 2>&1 || fail dexplorer_status_failed
schedule_after=$(counter "$output_dir/dexplorer-after.txt" schedule) || fail schedule_counter_missing
interrupt_after=$(counter "$output_dir/dexplorer-after.txt" interrupt) || fail interrupt_counter_missing
schedule_delta=$((schedule_after - schedule_before))
interrupt_delta=$((interrupt_after - interrupt_before))
[ "$schedule_delta" -eq "$dpu_calls" ] && [ "$interrupt_delta" -eq "$dpu_calls" ] || fail counter_delta_mismatch
dmesg | tail -n "+$((dmesg_before + 1))" > "$output_dir/kernel-new.txt"
grep -Eqi 'dpu.*(timeout|hang|fault|serror)|dma.*fault|irq.*(timeout|error)|mmc.*(error|timeout)|ext4.*(error|abort)' "$output_dir/kernel-new.txt" && fail new_kernel_fault
umount "$lib_target" || fail dexplorer_unmount_failed
mounted=no
cat > "$output_dir/board-report.txt" <<EOF
SJD3_STATUS=pass
CASE_COUNT=$case_count
TOTAL_DPU_RUN_CALLS=$dpu_calls
SCHEDULE_DELTA=$schedule_delta
INTERRUPT_DELTA=$interrupt_delta
OUTPUT_SCALE=$output_scale
EOF
cat "$output_dir/board-report.txt"
