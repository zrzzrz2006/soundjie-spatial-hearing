#!/bin/sh
set -eu

bundle_dir=
audio_root=
output_dir=

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bundle-dir) bundle_dir=$2; shift 2 ;;
    --audio-root) audio_root=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

fail() {
  echo "DPU_PROBE_STATUS=fail"
  echo "REASON=$1"
  exit "${2:-1}"
}

for required in "$bundle_dir" "$audio_root" "$output_dir"; do
  case "$required" in
    /*) ;;
    *) fail path_must_be_absolute 64 ;;
  esac
done

module="$bundle_dir/runtime/dpu-4.14.0-xilinx.ko"
smoke="$bundle_dir/runtime/dpu_smoke"
dexplorer="$bundle_dir/dnndk/pkgs/bin/dexplorer"
dnndk_lib="$bundle_dir/dnndk/pkgs/lib"
audio_env="$audio_root/env.sh"
audio_loader="$audio_root/load-usb-audio.sh"
audio_probe="$audio_root/board-tools/xvf3800_probe.sh"
audio_record="$audio_root/board-tools/xvf3800_record.sh"
wav_probe="$audio_root/wav_probe"

for required_file in "$module" "$smoke" "$dexplorer" "$audio_env" "$audio_loader" "$audio_probe" "$audio_record" "$wav_probe"; do
  [ -f "$required_file" ] || fail missing_required_file
done

mkdir -p "$output_dir"
summary="$output_dir/dpu-probe-report.txt"
dexplorer_log="$output_dir/dexplorer.txt"
smoke_log="$output_dir/dpu-smoke.txt"
kernel_log="$output_dir/dpu-kernel-interval.txt"
audio_report="$output_dir/xvf3800-probe.txt"
audio_record_log="$output_dir/xvf3800-record.txt"
wav_report="$output_dir/xvf3800-wav.txt"
wav_file="$output_dir/xvf3800-regression.wav"

arch=$(uname -m)
[ "$arch" = armv7l ] || fail wrong_architecture
kernel=$(uname -r)
[ "$kernel" = 4.14.0-xilinx ] || fail wrong_kernel_release

fdt_path=${DPU_PROBE_FDT:-/sys/firmware/fdt}
[ -r "$fdt_path" ] || fail missing_dpu_device_tree_node
grep -a -q 'deephi,dpu' "$fdt_path" || fail missing_dpu_device_tree_node

vermagic=$(modinfo -F vermagic "$module" 2>/dev/null || true)
case "$vermagic" in
  4.14.0-xilinx*) ;;
  *) fail module_vermagic_mismatch ;;
esac

proc_modules=${DPU_PROBE_PROC_MODULES:-/proc/modules}
dev_root=${DPU_PROBE_DEV_ROOT:-/dev}
dmesg_before=$(dmesg | wc -l)

if ! grep -q '^dpu ' "$proc_modules" 2>/dev/null; then
  insmod "$module" || fail module_load_failed
fi
grep -q '^dpu ' "$proc_modules" 2>/dev/null || fail module_not_registered
[ -e "$dev_root/dpu" ] || fail missing_dpu_character_device

LD_LIBRARY_PATH="$dnndk_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH

dexplorer_lib_target=${DPU_PROBE_DEXPLORER_LIB_TARGET:-/usr/local/lib}
[ -d "$dexplorer_lib_target" ] || fail missing_dexplorer_lib_target
dexplorer_lib_mounted=no
cleanup_dexplorer_mount() {
  if [ "$dexplorer_lib_mounted" = yes ]; then
    umount "$dexplorer_lib_target" >/dev/null 2>&1 || true
  fi
}
trap cleanup_dexplorer_mount 0 HUP INT TERM
mount --bind "$dnndk_lib" "$dexplorer_lib_target" || fail dexplorer_bind_mount_failed
dexplorer_lib_mounted=yes
if ! "$dexplorer" -w > "$dexplorer_log" 2>&1; then
  fail dexplorer_failed
fi
umount "$dexplorer_lib_target" || fail dexplorer_unmount_failed
dexplorer_lib_mounted=no
grep -Eqi 'B1152|DPU' "$dexplorer_log" || fail dpu_signature_missing

: > "$smoke_log"
if "$smoke" >> "$smoke_log" 2>&1; then
  echo 'SMOKE_SINGLE=pass' >> "$smoke_log"
else
  fail dpu_smoke_failed
fi

iterations=${DPU_PROBE_ITERATIONS:-100}
case "$iterations" in
  ''|*[!0-9]*) fail invalid_iteration_count 64 ;;
esac
[ "$iterations" -gt 0 ] || fail invalid_iteration_count 64

i=1
while [ "$i" -le "$iterations" ]; do
  if "$smoke" >> "$smoke_log" 2>&1; then
    echo "SMOKE_ITERATION_$i=pass" >> "$smoke_log"
  else
    echo "SMOKE_ITERATION_$i=fail" >> "$smoke_log"
    fail dpu_smoke_failed
  fi
  i=$((i + 1))
done

dmesg | tail -n "+$((dmesg_before + 1))" > "$kernel_log"
if grep -Eqi 'dpu.*(timeout|hang|fault|serror)|dma.*fault|irq.*(timeout|error)' "$kernel_log"; then
  fail new_kernel_fault
fi
if grep -Eqi 'usb.*(disconnect|reset)|reset.*usb' "$kernel_log"; then
  fail new_usb_fault
fi

. "$audio_env"
"$audio_loader" || fail audio_module_load_failed
"$audio_probe" --report "$audio_report" >/dev/null || fail audio_probe_failed
grep -q '^USB_ID=2886:001a$' "$audio_report" || fail wrong_audio_usb_id
grep -q '^SELECTED_FORMAT=S16_LE$' "$audio_report" || fail wrong_audio_format
grep -q '^SELECTED_CHANNELS=2$' "$audio_report" || fail wrong_audio_channels
grep -q '^SELECTED_RATE=16000$' "$audio_report" || fail wrong_audio_rate
grep -q '^SUPPORTED_CAPTURE=yes$' "$audio_report" || fail audio_capture_unsupported

"$audio_record" --seconds 10 --output "$wav_file" --report "$audio_report" > "$audio_record_log" 2>&1 || fail audio_record_failed
"$wav_probe" "$wav_file" > "$wav_report" 2>&1 || fail wav_probe_failed
grep -q '^CHANNELS=2$' "$wav_report" || fail recorded_audio_channels_wrong
grep -q '^SAMPLE_RATE=16000$' "$wav_report" || fail recorded_audio_rate_wrong
grep -q '^BITS_PER_SAMPLE=16$' "$wav_report" || fail recorded_audio_format_wrong
grep -q '^SIGNAL=present$' "$wav_report" || fail audio_signal_absent

cat > "$summary" <<EOF
DPU_PROBE_STATUS=pass
ARCH=$arch
KERNEL=$kernel
MODULE_VERMAGIC=$vermagic
SMOKE_ITERATIONS=$iterations
USB_ID=2886:001a
AUDIO_FORMAT=S16_LE
AUDIO_CHANNELS=2
AUDIO_RATE=16000
AUDIO_SIGNAL=present
EOF

cat "$summary"
