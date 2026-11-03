#!/bin/sh
set -eu

repo_root=
output_dir=
logmel_cli=

usage() {
  echo "usage: $0 --repo-root DIR --output-dir NEW_DIR --logmel-cli ARM_FILE"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo-root) repo_root=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --logmel-cli) logmel_cli=$2; shift 2 ;;
    --help) usage; exit 0 ;;
    *) usage >&2; exit 64 ;;
  esac
done

[ -n "$repo_root" ] && [ -n "$output_dir" ] && [ -n "$logmel_cli" ] || { usage >&2; exit 64; }
[ -d "$repo_root" ] || { echo "repository root does not exist" >&2; exit 2; }
[ -f "$logmel_cli" ] || { echo "verified logmel_cli does not exist" >&2; exit 2; }
[ ! -e "$output_dir" ] || { echo "output directory already exists: $output_dir" >&2; exit 3; }

compiler=arm-linux-gnueabihf-g++
readelf=arm-linux-gnueabihf-readelf
for tool in "$compiler" "$readelf" sha256sum mktemp cp chmod mv; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing build tool: $tool" >&2; exit 4; }
done

sources='src/wav_pcm.cpp
src/wav_pcm.h
src/live_window_selector.cpp
src/live_window_selector.h
src/live_window_selector_cli.cpp
src/live_result_fuser.cpp
src/live_result_fuser.h
src/live_result_fuser_cli.cpp'
printf '%s\n' "$sources" | while IFS= read -r relative; do
  [ -f "$repo_root/$relative" ] || { echo "missing source: $relative" >&2; exit 5; }
done

parent=${output_dir%/*}
[ "$parent" != "$output_dir" ] || parent=.
[ -d "$parent" ] || { echo "output parent does not exist: $parent" >&2; exit 2; }
stage=$(mktemp -d "$parent/.soundjie-live-build.XXXXXX")
cleanup() {
  if [ -d "$stage" ]; then
    find "$stage" -depth -mindepth 1 -delete >/dev/null 2>&1 || true
    rmdir "$stage" >/dev/null 2>&1 || true
  fi
}
trap cleanup 0 HUP INT TERM

flags='-std=c++11 -O2 -Wall -Wextra -Werror -march=armv7-a -mfloat-abi=hard -mfpu=vfpv3-d16 -static-libgcc -static-libstdc++'
# shellcheck disable=SC2086
"$compiler" $flags -I"$repo_root/src" \
  "$repo_root/src/wav_pcm.cpp" "$repo_root/src/live_window_selector.cpp" \
  "$repo_root/src/live_window_selector_cli.cpp" -o "$stage/live_window_selector"
# shellcheck disable=SC2086
"$compiler" $flags -I"$repo_root/src" \
  "$repo_root/src/live_result_fuser.cpp" "$repo_root/src/live_result_fuser_cli.cpp" \
  -o "$stage/live_result_fuser"
cp "$logmel_cli" "$stage/logmel_cli"
chmod 755 "$stage/live_window_selector" "$stage/live_result_fuser" "$stage/logmel_cli"

report="$stage/build-report.txt"
printf 'BUILD_SCHEMA=1\nTOOLCHAIN_PREFIX=arm-linux-gnueabihf-\nVMDK_MODE=not_used\nVMDK_UNMOUNTED=not_applicable\nNBD_DISCONNECTED=not_applicable\nLOGMEL_INPUT_SHA256=%s\n' \
  "$(sha256sum "$logmel_cli" | awk '{print $1}')" > "$report"

audit_binary() {
  name=$1
  binary=$2
  header=$("$readelf" -h "$binary")
  attributes=$("$readelf" -A "$binary")
  versions=$("$readelf" --version-info "$binary")
  dynamic=$("$readelf" -d "$binary")
  program_headers=$("$readelf" -l "$binary")
  printf '%s\n' "$header" | grep -q 'Machine:.*ARM' || { echo "$name is not ARM" >&2; return 1; }
  printf '%s\n' "$header" | grep -q 'Version5 EABI' || { echo "$name is not EABI5" >&2; return 1; }
  printf '%s\n' "$attributes" | grep -q 'Tag_CPU_arch: v7' || { echo "$name is not ARMv7" >&2; return 1; }
  printf '%s\n' "$attributes" | grep -q 'Tag_ABI_VFP_args: VFP registers' || { echo "$name is not hard-float" >&2; return 1; }
  max_glibc=$(printf '%s\n' "$versions" | grep -oE 'GLIBC_[0-9]+\.[0-9]+' | sed 's/^GLIBC_//' | sort -t. -k1,1n -k2,2n | tail -1)
  [ -n "$max_glibc" ] || { echo "$name has no auditable glibc version" >&2; return 1; }
  major=${max_glibc%%.*}
  minor=${max_glibc#*.}
  if [ "$major" -gt 2 ] || { [ "$major" -eq 2 ] && [ "$minor" -gt 17 ]; }; then
    echo "$name requires GLIBC newer than 2.17" >&2
    return 1
  fi
  interpreter=$(printf '%s\n' "$program_headers" | sed -n 's/.*Requesting program interpreter: \([^]]*\).*/\1/p' | head -1)
  [ "$interpreter" = /lib/ld-linux-armhf.so.3 ] || { echo "$name has wrong dynamic loader" >&2; return 1; }
  needed=$(printf '%s\n' "$dynamic" | sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p' | paste -sd, -)
  printf '%s\n' "$needed" | grep -Eqi 'n2cube|hineon' && { echo "$name has forbidden DPU dependency" >&2; return 1; }
  {
    echo "BEGIN_TOOL=$name"
    echo 'MACHINE=ARM'
    echo 'EABI=Version5'
    echo 'CPU_ARCH=v7'
    echo 'HARD_FLOAT=yes'
    echo 'GLIBC_MAX_OK=yes'
    echo "MAX_GLIBC=$max_glibc"
    echo "ELF_INTERPRETER=$interpreter"
    echo "DT_NEEDED=$needed"
    echo "SHA256=$(sha256sum "$binary" | awk '{print $1}')"
    echo "END_TOOL=$name"
  } >> "$report"
}

audit_binary live_window_selector "$stage/live_window_selector"
audit_binary live_result_fuser "$stage/live_result_fuser"
audit_binary logmel_cli "$stage/logmel_cli"

manifest="$stage/build-manifest.tsv"
printf '%s\n' "$sources" | while IFS= read -r relative; do
  printf 'source\t%s\t%s\n' "$relative" "$(sha256sum "$repo_root/$relative" | awk '{print $1}')"
done > "$manifest"
for name in live_window_selector live_result_fuser logmel_cli; do
  printf 'binary\t%s\t%s\n' "$name" "$(sha256sum "$stage/$name" | awk '{print $1}')" >> "$manifest"
done

[ ! -e "$output_dir" ] || { echo "output directory appeared during build: $output_dir" >&2; exit 3; }
mv -T "$stage" "$output_dir"
trap - 0 HUP INT TERM
echo "OUTPUT_DIR=$output_dir"
for name in live_window_selector live_result_fuser logmel_cli; do
  sha256sum "$output_dir/$name"
done
