#!/bin/bash
set -eu

fixture=""
expected_kernel="4.14.0-xilinx"
kernel_source=""
rootfs=""
dtb=""
output=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --fixture) fixture=$2; shift 2 ;;
    --expected-kernel) expected_kernel=$2; shift 2 ;;
    --kernel-source) kernel_source=$2; shift 2 ;;
    --rootfs) rootfs=$2; shift 2 ;;
    --dtb) dtb=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

config_value() {
  key=$1
  config=$2
  if grep -q "^${key}=y$" "$config"; then echo y
  elif grep -q "^${key}=m$" "$config"; then echo m
  elif grep -q "^# ${key} is not set$" "$config"; then echo n
  else echo unknown
  fi
}

if [ -n "$fixture" ]; then
  # Fixtures contain only constant KEY=VALUE lines controlled by this repository.
  . "$fixture"
else
  [ -d "$kernel_source" ] || { echo "kernel source missing: $kernel_source" >&2; exit 66; }
  [ -d "$rootfs" ] || { echo "rootfs missing: $rootfs" >&2; exit 66; }
  [ -f "$dtb" ] || { echo "device tree blob missing: $dtb" >&2; exit 66; }

  config="$kernel_source/.config"
  [ -f "$config" ] || { echo "kernel config missing: $config" >&2; exit 66; }

  if [ -f "$kernel_source/include/config/kernel.release" ]; then
    KERNEL_RELEASE=$(cat "$kernel_source/include/config/kernel.release")
  elif [ -d "$rootfs/lib/modules/$expected_kernel" ]; then
    KERNEL_RELEASE=$expected_kernel
  else
    KERNEL_RELEASE=$(find "$rootfs/lib/modules" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | head -n 1)
  fi

  CONFIG_USB=$(config_value CONFIG_USB "$config")
  CONFIG_USB_EHCI_HCD=$(config_value CONFIG_USB_EHCI_HCD "$config")
  CONFIG_SND=$(config_value CONFIG_SND "$config")
  CONFIG_SND_USB_AUDIO=$(config_value CONFIG_SND_USB_AUDIO "$config")

  if [ -x "$rootfs/usr/bin/arecord" ] || [ -x "$rootfs/bin/arecord" ]; then
    ARECORD_PRESENT=yes
  else
    ARECORD_PRESENT=no
  fi

  tmp_dts=$(mktemp)
  trap 'rm -f "$tmp_dts"' EXIT
  dtc -I dtb -O dts -o "$tmp_dts" "$dtb" 2>/dev/null
  usb_block=$(awk '
    /usb@e0002000[[:space:]]*\{/ {inside=1}
    inside {
      print
      opens=gsub(/\{/, "{")
      closes=gsub(/\}/, "}")
      if (opens > 0) seen_open=1
      depth += opens - closes
      if (seen_open && depth <= 0) exit
    }
  ' "$tmp_dts")
  USB0_DT_STATUS=$(printf '%s\n' "$usb_block" | sed -n 's/.*status[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)
  USB0_DR_MODE=$(printf '%s\n' "$usb_block" | sed -n 's/.*dr_mode[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)
  USB0_DT_STATUS=${USB0_DT_STATUS:-unknown}
  USB0_DR_MODE=${USB0_DR_MODE:-unknown}
fi

case "$KERNEL_RELEASE" in
  "$expected_kernel")
    if [ "$USB0_DT_STATUS" != "okay" ] || [ "$USB0_DR_MODE" != "host" ]; then
      ACTION=FIX_DEVICE_TREE
    elif [ "$CONFIG_USB" = n ] || [ "$CONFIG_USB" = unknown ] || \
         [ "$CONFIG_USB_EHCI_HCD" = n ] || [ "$CONFIG_USB_EHCI_HCD" = unknown ] || \
         [ "$CONFIG_SND" = n ] || [ "$CONFIG_SND" = unknown ]; then
      ACTION=BUILD_USB_STACK
    elif [ "$CONFIG_SND_USB_AUDIO" = n ] || [ "$CONFIG_SND_USB_AUDIO" = unknown ] || \
         [ "$ARECORD_PRESENT" != yes ]; then
      ACTION=BUILD_MODULES_AND_ALSA
    else
      ACTION=USE_BUILTIN
    fi
    ;;
  *) ACTION=BLOCK_KERNEL_MISMATCH ;;
esac

report=$(cat <<EOF
USB0_DT_STATUS=$USB0_DT_STATUS
USB0_DR_MODE=$USB0_DR_MODE
KERNEL_RELEASE=$KERNEL_RELEASE
CONFIG_USB=$CONFIG_USB
CONFIG_USB_EHCI_HCD=$CONFIG_USB_EHCI_HCD
CONFIG_SND=$CONFIG_SND
CONFIG_SND_USB_AUDIO=$CONFIG_SND_USB_AUDIO
ARECORD_PRESENT=$ARECORD_PRESENT
ACTION=$ACTION
EOF
)

printf '%s\n' "$report"
if [ -n "$output" ]; then
  mkdir -p "$(dirname "$output")"
  printf '%s\n' "$report" > "$output"
fi
