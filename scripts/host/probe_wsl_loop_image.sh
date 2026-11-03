#!/usr/bin/env bash
set -euo pipefail
image=/mnt/f/MLK/DPU_ZYNQ7000/tools/loop_probe_16m.img
truncate -s 16M "$image"
device=$(losetup --find --show "$image")
trap 'losetup -d "$device" 2>/dev/null || true' EXIT
printf 'LOOP=%s\n' "$device"
mkfs.ext4 -F "$device" >/dev/null
losetup -d "$device"
trap - EXIT
stat -c 'SIZE=%s' "$image"
