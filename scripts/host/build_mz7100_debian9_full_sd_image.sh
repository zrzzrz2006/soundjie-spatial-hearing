#!/usr/bin/env bash
set -euo pipefail

IMAGE_SIZE=15728640000
OUTPUT_ROOT=/mnt/f/MLK/DPU_ZYNQ7000/artifacts/mz7100_debian9_restored_sd
IMAGE="$OUTPUT_ROOT/mz7100_debian9_dpuv1_dpuy1_15gb.img"
MANIFEST="$IMAGE.manifest.txt"
TOOLS_ROOT=/mnt/f/MLK/DPU_ZYNQ7000/tools/debs/dosfstools/usr/sbin
MKFS_FAT="$TOOLS_ROOT/mkfs.fat"
FSCK_FAT="$TOOLS_ROOT/fsck.fat"
BOOT_SOURCE=/mnt/f/MLK/DPU_ZYNQ7000/backups/sd_recovery_2026-10-03/restored_original_boot
DPU_SOURCE=/mnt/f/MLK/DPU_ZYNQ7000/artifacts/mz7100_dpu_2017_probe_v1/prepared-v2
MODEL_SOURCE=/mnt/f/MLK/DPU_ZYNQ7000/artifacts/dpu_real_inference_v1/prepared-level-a
VENDOR_ARCHIVE=/mnt/f/MLK/MZ7100FA/08_linux/03_demo/MZ7100_PKG/part_1/deploy_images_7100.tar.gz
MOUNT_ROOT=/mnt/mz7100_image_build
BOOT_MOUNT="$MOUNT_ROOT/boot"
ROOT_MOUNT="$MOUNT_ROOT/rootfs"

log() { printf '[MZ7100-IMAGE] %s\n' "$*"; }
fail() { printf '[MZ7100-IMAGE] ERROR: %s\n' "$*" >&2; exit 1; }

[[ $(id -u) -eq 0 ]] || fail 'This script must run as WSL root.'
[[ -x "$MKFS_FAT" && -x "$FSCK_FAT" ]] || fail 'Portable dosfstools are missing.'
[[ -f "$VENDOR_ARCHIVE" ]] || fail "Missing vendor archive: $VENDOR_ARCHIVE"
mkdir -p "$OUTPUT_ROOT" "$BOOT_MOUNT" "$ROOT_MOUNT"
[[ ! -e "$IMAGE" ]] || fail "Refusing to overwrite existing image: $IMAGE"

declare -A expected_hashes=(
  [BOOT.bin]=992f696208b3cefdecb091dd0f1fb1053e26505199c38e841685d61d8cd53983
  [uImage]=ed95081d1cbe16d9b35dcdd5b0e1b0db4063b384c889f138dc5a037193677398
  [devicetree.dtb]=84a613762b49e32d07223e425f1c2526c872ae832a98986bd814a082286bc701
  [uEnv.txt]=900ee46eb0f86c3c90f4f40b0d5c2d05e46df1e06be1f483192084cfcde6bbb3
  [DPUV1.BIT]=6377393ebe3538022a5e92023c9f60a428bfffae3c31780c3ff98ba9ad5f9ff3
  [DPUV1.DTB]=aa6cf706c49070d3595f31d01d78b93616458ae02967cd58ee2383e3a7c5a453
  [DPUV1.TGZ]=8a7ea530d5001fe23aea95019d754b489b89bb222b25efb4074ba1ace2c43e04
  [DPUV1.SHA]=d02a94632c3970fbb47eaa508751de83e82ff0be8a032db570a49690fade408c
  [DPUY1.TGZ]=da6cbe7e28823255bca0dd34cce8a56a3a469ae9c8d861ea5d88fbcb1a40dade
  [DPUY1.SHA]=9ff089acce41974c8fb9e2c690de2cf77aa27e1a053ba3bbdb1a828b31355f53
)

source_for() {
    case "$1" in
        BOOT.bin|uImage|devicetree.dtb|uEnv.txt) printf '%s/%s' "$BOOT_SOURCE" "$1" ;;
        DPUV1.*) printf '%s/%s' "$DPU_SOURCE" "$1" ;;
        DPUY1.*) printf '%s/%s' "$MODEL_SOURCE" "$1" ;;
        *) return 1 ;;
    esac
}

for name in "${!expected_hashes[@]}"; do
    src=$(source_for "$name")
    [[ -f "$src" ]] || fail "Missing source file: $src"
    hash=$(sha256sum "$src" | awk '{print $1}')
    [[ "$hash" == "${expected_hashes[$name]}" ]] || fail "Source hash mismatch: $src"
done

log "Creating $IMAGE ($IMAGE_SIZE bytes)."
truncate -s "$IMAGE_SIZE" "$IMAGE"
LOOP=$(losetup --find --show --partscan "$IMAGE")

cleanup() {
    mountpoint -q "$BOOT_MOUNT" && umount "$BOOT_MOUNT" || true
    mountpoint -q "$ROOT_MOUNT" && umount "$ROOT_MOUNT" || true
    [[ -n "${LOOP:-}" ]] && losetup -d "$LOOP" 2>/dev/null || true
}
trap cleanup EXIT

printf 'label: dos\nunit: sectors\n\nstart=2048,size=202752,type=c,bootable\nstart=204800,type=83\n' | sfdisk --wipe always "$LOOP"
partx -u "$LOOP"
udevadm settle
sleep 2
P1="${LOOP}p1"
P2="${LOOP}p2"
[[ -b "$P1" && -b "$P2" ]] || fail "Partition nodes missing: $P1 $P2"

"$MKFS_FAT" -F 32 -n boot "$P1"
mkfs.ext4 -F -L rootfs "$P2"
mount -t vfat "$P1" "$BOOT_MOUNT"
mount -t ext4 "$P2" "$ROOT_MOUNT"

log 'Extracting the vendor Debian 9 root filesystem.'
tar -xOf "$VENDOR_ARCHIVE" deploy_images_7100/sdcard_images/rootfs/rootfs.tar.gz \
    | tar -xzpf - --numeric-owner -C "$ROOT_MOUNT"

log 'Adding restored boot files and both DPU bundles.'
for name in BOOT.bin uImage devicetree.dtb uEnv.txt DPUV1.BIT DPUV1.DTB DPUV1.TGZ DPUV1.SHA DPUY1.TGZ DPUY1.SHA; do
    cp "$(source_for "$name")" "$BOOT_MOUNT/$name"
done
sync

for name in "${!expected_hashes[@]}"; do
    hash=$(sha256sum "$BOOT_MOUNT/$name" | awk '{print $1}')
    [[ "$hash" == "${expected_hashes[$name]}" ]] || fail "Image readback hash mismatch: $name"
done

umount "$BOOT_MOUNT"
umount "$ROOT_MOUNT"
"$FSCK_FAT" -n "$P1"
e2fsck -fn "$P2"
losetup -d "$LOOP"
LOOP=
trap - EXIT

log 'Calculating full-image SHA-256.'
image_hash=$(sha256sum "$IMAGE" | awk '{print $1}')
cat > "$MANIFEST" <<EOF
image=$IMAGE
image_size=$IMAGE_SIZE
sha256=$image_hash
partition1_start=1048576
partition1_size=103809024
partition2_start=104857600
boot_files=BOOT.bin,uImage,devicetree.dtb,uEnv.txt,DPUV1.BIT,DPUV1.DTB,DPUV1.TGZ,DPUV1.SHA,DPUY1.TGZ,DPUY1.SHA
built=$(date --iso-8601=seconds)
EOF
log "IMAGE_COMPLETE sha256=$image_hash"

