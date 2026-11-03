#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --repo-root DIR --dnndk-root DIR --model-elf FILE --output FILE" >&2
  exit 64
}

repo_root=
dnndk_root=
model_elf=
output=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) repo_root=$2; shift 2 ;;
    --dnndk-root) dnndk_root=$2; shift 2 ;;
    --model-elf) model_elf=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    *) usage ;;
  esac
done
[[ -n "$repo_root" && -n "$dnndk_root" && -n "$model_elf" && -n "$output" ]] || usage
[[ ! -e "$output" ]] || { echo "output already exists: $output" >&2; exit 3; }
for path in \
  "$repo_root/src/sound_shape_runner.cpp" "$repo_root/src/tensor_stats.cpp" \
  "$repo_root/src/sha256.cpp" "$repo_root/src/sha256.h" \
  "$dnndk_root/include/n2cube.h" "$dnndk_root/lib/libn2cube.so" \
  "$dnndk_root/lib/libhineon.so" "$model_elf"; do
  [[ -f "$path" ]] || { echo "missing build input: $path" >&2; exit 4; }
done
for tool in arm-linux-gnueabihf-g++ arm-linux-gnueabihf-readelf sha256sum mktemp cp chmod; do
  command -v "$tool" >/dev/null || { echo "missing build tool: $tool" >&2; exit 5; }
done

mkdir -p "$(dirname "$output")"
temporary="${output}.new.$$"
model_snapshot_dir=$(mktemp -d "${output}.model.XXXXXX")
model_snapshot="$model_snapshot_dir/model.elf"
cleanup() {
  [[ ! -e "$temporary" ]] || rm -f -- "$temporary"
  [[ ! -e "$model_snapshot" ]] || rm -f -- "$model_snapshot"
  [[ ! -d "$model_snapshot_dir" ]] || rmdir -- "$model_snapshot_dir"
}
trap cleanup EXIT

cp -- "$model_elf" "$model_snapshot"
chmod 400 "$model_snapshot"
model_sha=$(sha256sum "$model_snapshot" | awk '{print $1}')
model_define="-DSOUNDJIE_LINKED_MODEL_SHA256=\"$model_sha\""
arm-linux-gnueabihf-g++ -std=c++11 -O2 -Wall -Wextra -Werror \
  -march=armv7-a -mfloat-abi=hard -mfpu=vfpv3-d16 \
  -static-libgcc -static-libstdc++ \
  "$model_define" \
  -I"$dnndk_root/include" -I"$repo_root/src" \
  "$repo_root/src/sound_shape_runner.cpp" "$repo_root/src/tensor_stats.cpp" \
  "$repo_root/src/sha256.cpp" \
  "$model_snapshot" -L"$dnndk_root/lib" -Wl,-rpath,/usr/lib \
  -ln2cube -lhineon -lpthread -ldl -o "$temporary"

header=$(arm-linux-gnueabihf-readelf -h "$temporary")
attributes=$(arm-linux-gnueabihf-readelf -A "$temporary")
versions=$(arm-linux-gnueabihf-readelf --version-info "$temporary")
grep -q 'Machine:.*ARM' <<<"$header"
grep -q 'Version5 EABI' <<<"$header"
grep -q 'Tag_CPU_arch: v7' <<<"$attributes"
grep -q 'Tag_ABI_VFP_args: VFP registers' <<<"$attributes"
grep -q 'GLIBC_2.17' <<<"$versions"
max_glibc=$(grep -oE 'GLIBC_[0-9]+\.[0-9]+' <<<"$versions" |
  sed 's/^GLIBC_//' | sort -t. -k1,1n -k2,2n | tail -1)
glibc_major=${max_glibc%%.*}
glibc_minor=${max_glibc#*.}
if (( glibc_major > 2 || (glibc_major == 2 && glibc_minor > 17) )); then
  echo "runner requires a newer glibc than 2.17" >&2
  exit 6
fi
[[ "$(sha256sum "$model_snapshot" | awk '{print $1}')" == "$model_sha" ]] || {
  echo "model snapshot changed during build" >&2
  exit 6
}

ln "$temporary" "$output" || { echo "refusing to replace output: $output" >&2; exit 3; }
rm -f -- "$temporary"
rm -f -- "$model_snapshot"
rmdir -- "$model_snapshot_dir"
trap - EXIT
echo "OUTPUT=$output"
echo "MAX_GLIBC=$max_glibc"
echo "SHA256=$(sha256sum "$output" | awk '{print $1}')"
