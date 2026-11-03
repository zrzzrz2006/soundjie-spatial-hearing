#!/bin/bash
set -eu

audit=""
kernel_source=""
toolchain_prefix=""
artifact_root=""
work_root=""
expected_kernel="4.14.0-xilinx"
dry_run=0
jobs=4

while [ "$#" -gt 0 ]; do
  case "$1" in
    --audit) audit=$2; shift 2 ;;
    --kernel-source) kernel_source=$2; shift 2 ;;
    --toolchain-prefix) toolchain_prefix=$2; shift 2 ;;
    --artifact-root) artifact_root=$2; shift 2 ;;
    --work-root) work_root=$2; shift 2 ;;
    --expected-kernel) expected_kernel=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

[ -f "$audit" ] || { echo "audit report missing: $audit" >&2; exit 66; }

value() {
  sed -n "s/^$1=//p" "$audit" | tail -n 1 | tr -d '\r'
}

kernel_release=$(value KERNEL_RELEASE)
action=$(value ACTION)

if [ "$action" = "BLOCK_KERNEL_MISMATCH" ] || [ "$kernel_release" != "$expected_kernel" ]; then
  echo "BLOCK_KERNEL_MISMATCH: expected $expected_kernel, got ${kernel_release:-unknown}" >&2
  exit 65
fi

case "$action" in
  USE_BUILTIN)
    echo "MODULE_BUILD=skip"
    echo "MODULE_REASON=built-in-support"
    exit 0
    ;;
  BUILD_MODULES|BUILD_MODULES_AND_ALSA)
    echo "MODULE_BUILD=required"
    echo "MODULE_TARGETS=snd-hwdep.ko,snd-rawmidi.ko,snd-usbmidi-lib.ko,snd-usb-audio.ko"
    ;;
  *)
    echo "unsupported audit action for module build: ${action:-missing}" >&2
    exit 65
    ;;
esac

[ "$dry_run" -eq 0 ] || exit 0
[ -d "$kernel_source" ] || { echo "kernel source missing: $kernel_source" >&2; exit 66; }
[ -x "${toolchain_prefix}gcc" ] || { echo "cross compiler missing: ${toolchain_prefix}gcc" >&2; exit 66; }
[ -n "$artifact_root" ] || { echo "artifact root is required" >&2; exit 64; }

work_root=${work_root:-$artifact_root/work}
build_dir="$work_root/kernel-src"
module_dir="$artifact_root/modules/$expected_kernel"
mkdir -p "$build_dir" "$module_dir"

if [ ! -f "$build_dir/.codex-writable-copy" ]; then
  rmdir "$build_dir" 2>/dev/null || true
  cp -a "$kernel_source" "$build_dir"
  : > "$build_dir/.codex-writable-copy"
fi

cp "$kernel_source/.config" "$work_root/target.config"
cp "$kernel_source/Module.symvers" "$work_root/target.Module.symvers"
make -C "$build_dir" ARCH=arm CROSS_COMPILE="$toolchain_prefix" \
  HOSTCFLAGS=-fcommon mrproper
cp "$work_root/target.config" "$build_dir/.config"

if grep -q '^# CONFIG_SND_USB_AUDIO is not set$' "$build_dir/.config"; then
  sed -i 's/^# CONFIG_SND_USB_AUDIO is not set$/CONFIG_SND_USB_AUDIO=m/' "$build_dir/.config"
elif grep -q '^CONFIG_SND_USB_AUDIO=' "$build_dir/.config"; then
  sed -i 's/^CONFIG_SND_USB_AUDIO=.*/CONFIG_SND_USB_AUDIO=m/' "$build_dir/.config"
else
  printf '%s\n' 'CONFIG_SND_USB_AUDIO=m' >> "$build_dir/.config"
fi

make -C "$build_dir" ARCH=arm CROSS_COMPILE="$toolchain_prefix" \
  HOSTCFLAGS=-fcommon \
  olddefconfig prepare modules_prepare
cp "$work_root/target.Module.symvers" "$build_dir/Module.symvers"
make -C "$build_dir" ARCH=arm CROSS_COMPILE="$toolchain_prefix" \
  HOSTCFLAGS=-fcommon \
  -j"$jobs" M=sound modules

for module in snd-hwdep snd-rawmidi snd-usbmidi-lib snd-usb-audio; do
  source_module=$(find "$build_dir/sound" -type f -name "$module.ko" | head -n 1)
  [ -n "$source_module" ] || { echo "$module.ko was not produced" >&2; exit 70; }
  cp -f "$source_module" "$module_dir/"
done

cat > "$module_dir/module-load-order.txt" <<'EOF'
snd-hwdep.ko
snd-rawmidi.ko
snd-usbmidi-lib.ko
snd-usb-audio.ko
EOF

for module in "$module_dir"/*.ko; do
  "${toolchain_prefix}readelf" -h "$module" | grep -q 'Machine:.*ARM'
done
(cd "$module_dir" && sha256sum ./*.ko module-load-order.txt > SHA256SUMS)
(cd "$module_dir" && sha256sum -c SHA256SUMS >/dev/null)
echo "MODULE_OUTPUT=$module_dir"
