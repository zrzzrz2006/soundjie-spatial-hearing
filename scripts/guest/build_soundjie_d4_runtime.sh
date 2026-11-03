#!/bin/sh
set -eu

repo_root=
output_dir=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo-root) repo_root=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
[ -n "$repo_root" ] && [ -d "$repo_root" ] || { echo 'missing repository root' >&2; exit 64; }
[ -n "$output_dir" ] || { echo 'missing output directory' >&2; exit 64; }
[ ! -e "$output_dir" ] || { echo "output directory already exists: $output_dir" >&2; exit 3; }

compiler=arm-linux-gnueabihf-g++
readelf=arm-linux-gnueabihf-readelf
for tool in "$compiler" "$readelf" sha256sum mktemp mv awk; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing build tool: $tool" >&2; exit 4; }
done
sources='src/pcm_ring_buffer.cpp src/soundjie_d4_state.cpp src/soundjie_d4_runtime.cpp src/fft_radix2.cpp src/known_clip_matcher.cpp src/traffic_fine_model.cpp'
for source in $sources; do
  [ -f "$repo_root/$source" ] || { echo "missing source: $source" >&2; exit 5; }
done

parent=${output_dir%/*}
[ "$parent" != "$output_dir" ] || parent=.
[ -d "$parent" ] || { echo "output parent does not exist: $parent" >&2; exit 2; }
stage=$(mktemp -d "$parent/.soundjie-d4-runtime.XXXXXX")
cleanup() {
  [ ! -d "$stage" ] || find "$stage" -depth -mindepth 1 -delete >/dev/null 2>&1 || true
  [ ! -d "$stage" ] || rmdir "$stage" >/dev/null 2>&1 || true
}
trap cleanup 0 HUP INT TERM

flags='-std=c++11 -O2 -Wall -Wextra -Werror -march=armv7-a -mfloat-abi=hard -mfpu=neon -static-libgcc -static-libstdc++ -pthread'
# shellcheck disable=SC2086
"$compiler" $flags -I"$repo_root/src" \
  "$repo_root/src/pcm_ring_buffer.cpp" \
  "$repo_root/src/soundjie_d4_state.cpp" \
  "$repo_root/src/soundjie_d4_runtime.cpp" \
  "$repo_root/src/fft_radix2.cpp" \
  "$repo_root/src/known_clip_matcher.cpp" \
  "$repo_root/src/traffic_fine_model.cpp" \
  -o "$stage/soundjie_d4_runtime"

header=$("$readelf" -h "$stage/soundjie_d4_runtime")
attributes=$("$readelf" -A "$stage/soundjie_d4_runtime")
program_headers=$("$readelf" -l "$stage/soundjie_d4_runtime")
versions=$("$readelf" --version-info "$stage/soundjie_d4_runtime")
dynamic=$("$readelf" -d "$stage/soundjie_d4_runtime")
printf '%s\n' "$header" | grep -q 'Machine:.*ARM' || { echo 'not ARM' >&2; exit 6; }
printf '%s\n' "$attributes" | grep -q 'Tag_CPU_arch: v7' || { echo 'not ARMv7' >&2; exit 6; }
printf '%s\n' "$attributes" | grep -q 'Tag_ABI_VFP_args: VFP registers' || { echo 'not hard-float' >&2; exit 6; }
interpreter=$(printf '%s\n' "$program_headers" | sed -n 's/.*Requesting program interpreter: \([^]]*\).*/\1/p' | head -1)
[ "$interpreter" = /lib/ld-linux-armhf.so.3 ] || { echo 'wrong loader' >&2; exit 6; }
max_glibc=$(printf '%s\n' "$versions" | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sed 's/^GLIBC_//' | sort -t. -k1,1n -k2,2n | tail -1)
major=${max_glibc%%.*}; minor=${max_glibc#*.}
[ "$major" -lt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -le 17 ]; } || { echo 'GLIBC newer than 2.17' >&2; exit 6; }
dt_needed=$(printf '%s\n' "$dynamic" | awk '/Shared library:/ {gsub(/.*\[/,""); gsub(/\].*/,""); printf "%s%s",s,$0; s=","} END {if(s=="") printf "none"}')

{
  echo 'BUILD_SCHEMA=1'
  echo 'MACHINE=ARM'
  echo 'CPU_ARCH=v7'
  echo 'HARD_FLOAT=yes'
  echo "ELF_INTERPRETER=$interpreter"
  echo "MAX_GLIBC=$max_glibc"
  echo "DT_NEEDED=$dt_needed"
  echo "SHA256=$(sha256sum "$stage/soundjie_d4_runtime" | awk '{print $1}')"
} > "$stage/build-report.txt"
mv -T "$stage" "$output_dir"
trap - 0 HUP INT TERM
echo "OUTPUT_DIR=$output_dir"
