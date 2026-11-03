#!/bin/sh
set -eu

bundle_dir=
output_dir=
expected_class=
dpu_calls=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bundle-dir) bundle_dir=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --expected-class) expected_class=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

fail() {
  echo "LIVE_STATUS=fail"
  echo "REASON=$1"
  echo "DPU_RUN_CALLS=$dpu_calls"
  exit "${2:-1}"
}

case "$bundle_dir:$output_dir" in /*:/*) ;; *) fail path_must_be_absolute 64 ;; esac
[ ! -e "$output_dir" ] || fail output_already_exists

module="$bundle_dir/runtime/dpu-4.14.0-xilinx.ko"
runner="$bundle_dir/runtime/sound_shape_runner"
model="$bundle_dir/model/dpu_soundjie_d3.elf"
dexplorer="$bundle_dir/dnndk/pkgs/bin/dexplorer"
libs="$bundle_dir/dnndk/pkgs/lib"
labels_en="$bundle_dir/model/labels.txt"
labels_zh="$bundle_dir/model/display_labels_zh.txt"
frontend_config="$bundle_dir/config/logmel_frontend_v1.json"
live_config="$bundle_dir/config/soundjie_live_v1.tsv"
audio_loader="$bundle_dir/audio/load-usb-audio.sh"
audio_probe="$bundle_dir/audio/board-tools/xvf3800_probe.sh"
audio_record="$bundle_dir/audio/board-tools/xvf3800_record.sh"
arecord_bin="$bundle_dir/audio/bin/arecord"
alsa_lib="$bundle_dir/audio/lib"
alsa_config="$bundle_dir/audio/share/alsa/alsa.conf"
wav_probe="$bundle_dir/runtime/wav_probe"
selector="$bundle_dir/runtime/live_window_selector"
logmel="$bundle_dir/runtime/logmel_cli"
fuser="$bundle_dir/runtime/live_result_fuser"
contract="$bundle_dir/manifest/contract.tsv"
sums="$bundle_dir/manifest/SHA256SUMS"

for path in "$module" "$runner" "$model" "$dexplorer" "$labels_en" "$labels_zh" \
  "$frontend_config" "$live_config" "$audio_loader" "$audio_probe" "$audio_record" \
  "$arecord_bin" "$alsa_lib/libasound.so.2" "$alsa_config" "$wav_probe" "$selector" "$logmel" "$fuser" "$contract" "$sums" \
  "$libs/libn2cube.so" "$libs/libhineon.so"; do
  [ -f "$path" ] || fail missing_required_file
done

awk '
  length($1) != 64 { exit 1 }
  { hash=$1; gsub(/[0-9a-f]/, "", hash); if(hash!="")exit 1 }
  substr($0,65,2) != "  " { exit 1 }
  { name=substr($0,67); if(name=="" || name~/^\// || name~/\\/ || name~/(^|\/)\.\.($|\/)/ || seen[name]++)exit 1 }
' "$sums" || fail unsafe_checksum_manifest
actual_files=$(cd "$bundle_dir" && find . -type f | sed 's#^\./##' | sort)
expected_files=$({ awk '{print substr($0,67)}' "$sums"; printf '%s\n' manifest/SHA256SUMS; } | sort)
[ "$actual_files" = "$expected_files" ] || fail unlisted_bundle_file
(cd "$bundle_dir" && sha256sum -c manifest/SHA256SUMS >/dev/null 2>&1) || fail bundle_hash_mismatch

value() {
  awk -F '\t' -v key="$1" '$1==key{if(++n>1)exit 2;v=$2}END{if(n!=1)exit 1;print v}' "$contract"
}
schema=$(value SCHEMA) || fail contract_invalid
class_count=$(value CLASS_COUNT) || fail contract_invalid
kernel=$(value MODEL_KERNEL) || fail contract_invalid
input_node=$(value INPUT_NODE) || fail contract_invalid
output_node=$(value OUTPUT_NODE) || fail contract_invalid
input_shape=$(value INPUT_SHAPE) || fail contract_invalid
input_size=$(value INPUT_SIZE) || fail contract_invalid
input_scale=$(value INPUT_SCALE) || fail contract_invalid
output_size=$(value OUTPUT_SIZE) || fail contract_invalid
output_scale=$(value OUTPUT_SCALE) || fail contract_invalid
audio_usb_id=$(value AUDIO_USB_ID) || fail contract_invalid
audio_format=$(value AUDIO_FORMAT) || fail contract_invalid
audio_channels=$(value AUDIO_CHANNELS) || fail contract_invalid
audio_rate=$(value AUDIO_RATE) || fail contract_invalid
capture_seconds=$(value CAPTURE_SECONDS) || fail contract_invalid
model_sha=$(value MODEL_SHA256) || fail contract_invalid
runner_sha=$(value RUNNER_SHA256) || fail contract_invalid
[ "$schema:$class_count:$kernel:$input_node:$output_node" = "1:8:soundjie_d3:conv1:fc8" ] || fail contract_invalid
[ "$input_shape:$input_size:$input_scale:$output_size:$output_scale" = "1,1,64,96:6144:128:8:0.0625" ] || fail contract_invalid
[ "$audio_usb_id:$audio_format:$audio_channels:$audio_rate:$capture_seconds" = "2886:001a:S16_LE:2:16000:6" ] || fail contract_invalid
[ "$(sha256sum "$model" | awk '{print $1}')" = "$model_sha" ] || fail model_hash_mismatch
[ "$(sha256sum "$runner" | awk '{print $1}')" = "$runner_sha" ] || fail runner_hash_mismatch
awk 'BEGIN{e[1]="fire_smoke_alarm";e[2]="emergency_siren";e[3]="car_horn";e[4]="glass_breaking";
           e[5]="door_knock_bell";e[6]="distress_voice";e[7]="normal_speech";e[8]="background_unknown"}
     {sub(/\r$/,"");if(NR>8 || $0!=e[NR])exit 1}END{if(NR!=8)exit 1}' "$labels_en" || fail labels_contract_mismatch
[ "$(awk 'END{print NR}' "$labels_zh")" = 8 ] || fail display_labels_contract_mismatch
if [ -n "$expected_class" ]; then
  grep -F -x -q "$expected_class" "$labels_en" || fail expected_class_invalid 64
fi

[ "$(uname -m)" = armv7l ] || fail wrong_architecture
[ "$(uname -r)" = 4.14.0-xilinx ] || fail wrong_kernel_release
fdt=${SJD3_LIVE_FDT:-/sys/firmware/fdt}
[ -r "$fdt" ] && grep -a -q 'deephi,dpu' "$fdt" || fail missing_dpu_device_tree_node
case "$(modinfo -F vermagic "$module" 2>/dev/null || true)" in 4.14.0-xilinx*) ;; *) fail module_vermagic_mismatch ;; esac
proc_modules=${SJD3_LIVE_PROC_MODULES:-/proc/modules}
proc_root=${SJD3_LIVE_PROC_ROOT:-/proc}
dev_root=${SJD3_LIVE_DEV_ROOT:-/dev}
grep -q '^dpu ' "$proc_modules" 2>/dev/null || insmod "$module" || fail module_load_failed
grep -q '^dpu ' "$proc_modules" 2>/dev/null || fail module_not_registered
[ -e "$dev_root/dpu" ] || fail missing_dpu_character_device

mkdir -p "$output_dir"
dmesg_before=$(dmesg | wc -l)
lib_target=${SJD3_LIVE_LIB_TARGET:-/usr/local/lib}
[ -d "$lib_target" ] || fail missing_dexplorer_lib_target
export LD_LIBRARY_PATH="$libs:$alsa_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ALSA_CONFIG_PATH="$alsa_config"
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
    'BEGIN{exit !(a+0==b+0 && c+0==d+0)}' || fail runtime_fixpoint_mismatch
[ "$(field "$description" DPU_RUN_CALLS)" = 0 ] || fail describe_ran_dpu

"$audio_loader" || fail audio_module_load_failed
"$audio_probe" --proc-root "$proc_root" --arecord-bin "$arecord_bin" \
  --report "$output_dir/capture-report.txt" >/dev/null || fail audio_probe_failed
grep -q "^USB_ID=$audio_usb_id$" "$output_dir/capture-report.txt" || fail wrong_audio_usb_id
grep -q "^SELECTED_FORMAT=$audio_format$" "$output_dir/capture-report.txt" || fail wrong_audio_format
grep -q "^SELECTED_CHANNELS=$audio_channels$" "$output_dir/capture-report.txt" || fail wrong_audio_channels
grep -q "^SELECTED_RATE=$audio_rate$" "$output_dir/capture-report.txt" || fail wrong_audio_rate
grep -q '^SUPPORTED_CAPTURE=yes$' "$output_dir/capture-report.txt" || fail audio_capture_unsupported
"$audio_record" --seconds "$capture_seconds" --output "$output_dir/capture.wav" \
  --proc-root "$proc_root" --arecord-bin "$arecord_bin" --report "$output_dir/capture-report.txt" \
  --probe-script "$audio_probe" > "$output_dir/capture-command.txt" 2>&1 || fail audio_record_failed
"$wav_probe" "$output_dir/capture.wav" > "$output_dir/capture-wav-report.txt" 2>&1 || fail wav_probe_failed
grep -q "^CHANNELS=$audio_channels$" "$output_dir/capture-wav-report.txt" || fail recorded_audio_channels_wrong
grep -q "^SAMPLE_RATE=$audio_rate$" "$output_dir/capture-wav-report.txt" || fail recorded_audio_rate_wrong
grep -q '^BITS_PER_SAMPLE=16$' "$output_dir/capture-wav-report.txt" || fail recorded_audio_format_wrong
grep -q '^SIGNAL=present$' "$output_dir/capture-wav-report.txt" || fail audio_signal_absent

selector_status=0
"$selector" --wav "$output_dir/capture.wav" --config "$live_config" \
  --metrics "$output_dir/window-metrics.tsv" --selected "$output_dir/selected-windows.tsv" || selector_status=$?
[ "$selector_status" -eq 0 ] || { [ "$selector_status" -eq 2 ] && fail audio_signal_absent; fail window_selection_failed; }
selected_body="$output_dir/selected-body.tsv"
sed '1d' "$output_dir/selected-windows.tsv" > "$selected_body"
selected_count=$(awk 'NF{n++}END{print n+0}' "$selected_body")
[ "$selected_count" -ge 1 ] && [ "$selected_count" -le 3 ] || fail selected_window_count_wrong

predictions="$output_dir/window-predictions.tsv"
printf 'rank\tstart_sample\truntime_us\traw_logits_csv\tinput_sha256\toutput_sha256\n' > "$predictions"
tab=$(printf '\t')
while IFS="$tab" read -r rank start_sample rms peak clipping_rate; do
  case "$rank:$start_sample" in *[!0-9:]*) fail selected_window_invalid ;; esac
  case_dir="$output_dir/window-$rank"
  mkdir "$case_dir"
  float_output="$case_dir/logmel.f32le"
  tensor="$case_dir/tensor.i8"
  logmel_report="$case_dir/logmel-report.txt"
  "$logmel" --config "$frontend_config" --wav "$output_dir/capture.wav" \
    --start-sample "$start_sample" --input-scale "$input_scale" \
    --float-output "$float_output" --int8-output "$tensor" --report "$logmel_report" || fail logmel_failed
  [ "$(field "$logmel_report" LOGMEL_STATUS)" = pass ] || fail logmel_report_failed
  [ "$(field "$logmel_report" INT8_BYTES):$(field "$logmel_report" INPUT_SCALE)" = "$input_size:$input_scale" ] || fail logmel_contract_mismatch
  [ "$(wc -c < "$tensor" | tr -d ' ')" = "$input_size" ] || fail tensor_size_mismatch
  dump="$case_dir/logits.i8"
  runner_report="$case_dir/runner-report.txt"
  "$runner" --kernel "$kernel" --input-node "$input_node" --output-node "$output_node" \
    --input-file "$tensor" --model-file "$model" --iterations 1 \
    --dump "$dump" --report "$runner_report" || fail runner_failed
  dpu_calls=$((dpu_calls + 1))
  [ "$(field "$runner_report" DPU_SOUND_STATUS)" = pass ] || fail runner_report_failed
  [ "$(field "$runner_report" DPU_RUN_CALLS)" = 1 ] || fail runner_call_count_wrong
  [ "$(field "$runner_report" MODEL_SHA256)" = "$model_sha" ] || fail model_hash_mismatch
  [ "$(field "$runner_report" INPUT_SHA256)" = "$(sha256sum "$tensor" | awk '{print $1}')" ] || fail input_hash_mismatch
  values=$(field "$runner_report" OUTPUT_VALUES) || fail runner_report_failed
  [ "$(printf '%s\n' "$values" | awk -F, '{print NF}')" = "$output_size" ] || fail output_value_count_wrong
  [ "$(wc -c < "$dump" | tr -d ' ')" = "$output_size" ] || fail output_value_count_wrong
  output_sha=$(field "$runner_report" OUTPUT_SHA256) || fail runner_report_failed
  [ "$output_sha" = "$(sha256sum "$dump" | awk '{print $1}')" ] || fail output_hash_mismatch
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$rank" "$start_sample" \
    "$(field "$runner_report" RUN_TOTAL_US)" "$values" \
    "$(field "$runner_report" INPUT_SHA256)" "$output_sha" >> "$predictions"
done < "$selected_body"
[ "$dpu_calls" -eq "$selected_count" ] || fail runner_call_count_wrong

"$dexplorer" -s > "$output_dir/dexplorer-after.txt" 2>&1 || fail dexplorer_status_failed
schedule_after=$(counter "$output_dir/dexplorer-after.txt" schedule) || fail schedule_counter_missing
interrupt_after=$(counter "$output_dir/dexplorer-after.txt" interrupt) || fail interrupt_counter_missing
schedule_delta=$((schedule_after - schedule_before))
interrupt_delta=$((interrupt_after - interrupt_before))
[ "$schedule_delta" -eq "$dpu_calls" ] && [ "$interrupt_delta" -eq "$dpu_calls" ] || fail counter_delta_mismatch
dmesg | tail -n "+$((dmesg_before + 1))" > "$output_dir/kernel-new.txt"
grep -Eqi 'dpu.*(timeout|hang|fault|serror)|dma.*fault|irq.*(timeout|error)|usb.*(disconnect|reset|error)|reset.*usb|mmc.*(error|timeout)|ext4.*(error|abort)' "$output_dir/kernel-new.txt" && fail new_kernel_fault
umount "$lib_target" || fail dexplorer_unmount_failed
mounted=no

set -- "$fuser" --predictions "$predictions" --labels-en "$labels_en" --labels-zh "$labels_zh" \
  --output-scale "$output_scale" --session-result "$output_dir/session-result.json" \
  --display-status "$output_dir/display-status.txt"
[ -z "$expected_class" ] || set -- "$@" --expected-class "$expected_class"
"$@" > "$output_dir/fuser-stdout.txt" 2>&1 || fail result_fusion_failed
[ -f "$output_dir/session-result.json" ] && [ -f "$output_dir/display-status.txt" ] || fail result_publish_failed
cat "$output_dir/fuser-stdout.txt"
