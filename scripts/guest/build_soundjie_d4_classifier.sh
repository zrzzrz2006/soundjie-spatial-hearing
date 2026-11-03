#!/usr/bin/env bash
set -euo pipefail
repo_root= dnndk_root= model_elf= output_dir=
physical_outputs=8
storage_outputs=
kernel_name=soundjie_d3
while [[ $# -gt 0 ]]; do
  [[ $# -ge 2 ]] || exit 64
  case "$1" in
    --repo-root) repo_root=$2 ;;
    --dnndk-root) dnndk_root=$2 ;;
    --model-elf) model_elf=$2 ;;
    --output-dir) output_dir=$2 ;;
    --physical-outputs) physical_outputs=$2 ;;
    --storage-outputs) storage_outputs=$2 ;;
    --kernel-name) kernel_name=$2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
  shift 2
done
[[ "$kernel_name" =~ ^[a-zA-Z_][a-zA-Z0-9_]{0,63}$ ]] || { echo 'invalid kernel name' >&2; exit 64; }
model_basename="dpu_${kernel_name}.elf"
[[ "$physical_outputs" == 8 || "$physical_outputs" == 6 || "$physical_outputs" == 5 ]] || exit 64
storage_outputs=${storage_outputs:-$physical_outputs}
[[ "$storage_outputs" == "$physical_outputs" || ( "$physical_outputs" == 5 && "$storage_outputs" == 6 ) ]] || exit 64
[[ -n "$repo_root" && -n "$dnndk_root" && -n "$model_elf" && -n "$output_dir" ]] || exit 64
[[ ! -e "$output_dir" && ! -L "$output_dir" ]] || { echo 'output already exists' >&2; exit 3; }
for input in "$repo_root/src/soundjie_d4_classifier.cpp" "$model_elf" \
  "$repo_root/config/logmel_frontend_v1.json" "$dnndk_root/include/n2cube.h" \
  "$dnndk_root/lib/libn2cube.so" "$dnndk_root/lib/libhineon.so"; do
  [[ -f "$input" ]] || { echo "missing input: $input" >&2; exit 4; }
done
compiler=arm-linux-gnueabihf-g++
readelf=arm-linux-gnueabihf-readelf
for tool in "$compiler" "$readelf" sha256sum mktemp; do command -v "$tool" >/dev/null; done
mkdir -p "$(dirname "$output_dir")"
stage=$(mktemp -d "$(dirname "$output_dir")/.d4-classifier.XXXXXX")
# Keep a failed staging directory for diagnosis; never remove caller-owned files.
cp "$model_elf" "$stage/model.elf"
cp "$repo_root/config/logmel_frontend_v1.json" "$stage/frontend.json"
model_sha=$(sha256sum "$stage/model.elf" | awk '{print $1}')
frontend_sha=$(sha256sum "$stage/frontend.json" | awk '{print $1}')
"$compiler" -std=c++11 -O2 -Wall -Wextra -Werror \
  -march=armv7-a -mfloat-abi=hard -mfpu=vfpv3-d16 -static-libgcc -static-libstdc++ \
  "-DSOUNDJIE_LINKED_MODEL_SHA256=\"$model_sha\"" \
  "-DSOUNDJIE_FRONTEND_SHA256=\"$frontend_sha\"" \
  "-DSOUNDJIE_PHYSICAL_OUTPUTS=$physical_outputs" \
  "-DSOUNDJIE_STORAGE_OUTPUTS=$storage_outputs" \
  "-DSOUNDJIE_KERNEL_NAME=\"$kernel_name\"" \
  "-DSOUNDJIE_MODEL_BASENAME=\"$model_basename\"" \
  -I"$repo_root/src" -I"$dnndk_root/include" \
  "$repo_root/src/soundjie_d4_classifier.cpp" "$repo_root/src/logmel_frontend.cpp" \
  "$repo_root/src/fft_radix2.cpp" "$repo_root/src/sha256.cpp" "$stage/model.elf" \
  -L"$dnndk_root/lib" -Wl,-rpath,/usr/lib -ln2cube -lhineon -lpthread -ldl \
  -o "$stage/soundjie_d4_classifier"
header=$("$readelf" -h "$stage/soundjie_d4_classifier")
attributes=$("$readelf" -A "$stage/soundjie_d4_classifier")
versions=$("$readelf" --version-info "$stage/soundjie_d4_classifier")
program_headers=$("$readelf" -l "$stage/soundjie_d4_classifier")
dynamic=$("$readelf" -d "$stage/soundjie_d4_classifier")
grep -q 'Machine:.*ARM' <<<"$header"
grep -q 'Version5 EABI' <<<"$header"
grep -q 'Tag_CPU_arch: v7' <<<"$attributes"
grep -q 'Tag_ABI_VFP_args: VFP registers' <<<"$attributes"
grep -q 'Requesting program interpreter: /lib/ld-linux-armhf.so.3' <<<"$program_headers"
max_glibc=$(grep -oE 'GLIBC_[0-9]+\.[0-9]+' <<<"$versions" | sed 's/^GLIBC_//' | sort -t. -k1,1n -k2,2n | tail -1)
[[ -n "$max_glibc" ]] || exit 6
major=${max_glibc%%.*}; minor=${max_glibc#*.}
(( major < 2 || (major == 2 && minor <= 17) )) || { echo 'GLIBC too new' >&2; exit 6; }
[[ "$(sha256sum "$stage/model.elf" | awk '{print $1}')" == "$model_sha" ]]
[[ "$(sha256sum "$stage/frontend.json" | awk '{print $1}')" == "$frontend_sha" ]]
{
  echo 'MACHINE=ARM'; echo 'CPU_ARCH=v7'; echo 'HARD_FLOAT=yes'
  echo 'ELF_INTERPRETER=/lib/ld-linux-armhf.so.3'
  echo "MAX_GLIBC=$max_glibc"
  echo "MODEL_SHA256=$model_sha"
  echo "FRONTEND_SHA256=$frontend_sha"
  echo "PHYSICAL_OUTPUTS=$physical_outputs"
  echo "STORAGE_OUTPUTS=$storage_outputs"
  echo "KERNEL_NAME=$kernel_name"
  echo "MODEL_BASENAME=$model_basename"
  echo "SHA256=$(sha256sum "$stage/soundjie_d4_classifier" | awk '{print $1}')"
  echo "DT_NEEDED=$(sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p' <<<"$dynamic" | paste -sd, -)"
} > "$stage/build-report.txt"
mv -T -n "$stage" "$output_dir"
[[ ! -d "$stage" ]] || { echo 'refusing concurrent output replacement' >&2; exit 3; }
cat "$output_dir/build-report.txt"
