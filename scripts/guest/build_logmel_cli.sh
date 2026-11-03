#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 --mode native|arm --repo-root DIR --output FILE [--report FILE]"
}

mode=
repo_root=
output=
report=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help) usage; exit 0 ;;
    --mode) mode=$2; shift 2 ;;
    --repo-root) repo_root=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --report) report=$2; shift 2 ;;
    *) usage >&2; exit 64 ;;
  esac
done
[[ "$mode" == native || "$mode" == arm ]] || { usage >&2; exit 64; }
[[ -n "$repo_root" && -n "$output" ]] || { usage >&2; exit 64; }
[[ ! -e "$output" ]] || { echo "output already exists: $output" >&2; exit 3; }
[[ -z "$report" || ! -e "$report" ]] || { echo "report already exists: $report" >&2; exit 3; }

sources=(
  "$repo_root/src/wav_pcm.cpp"
  "$repo_root/src/fft_radix2.cpp"
  "$repo_root/src/logmel_frontend.cpp"
  "$repo_root/src/logmel_cli.cpp"
)
for source in "${sources[@]}"; do
  [[ -f "$source" ]] || { echo "missing source: $source" >&2; exit 4; }
done

compiler=g++
[[ "$mode" == native ]] || compiler=arm-linux-gnueabihf-g++
command -v "$compiler" >/dev/null || { echo "missing compiler: $compiler" >&2; exit 5; }
mkdir -p "$(dirname "$output")"
temporary="${output}.new.$$"
cleanup() { [[ ! -e "$temporary" ]] || rm -f -- "$temporary"; }
trap cleanup EXIT

flags=(-std=c++11 -O2 -Wall -Wextra -Werror -I"$repo_root/src")
[[ "$mode" == native ]] || flags+=(
  -march=armv7-a -mfloat-abi=hard -mfpu=vfpv3-d16
  -static-libgcc -static-libstdc++
)
"$compiler" "${flags[@]}" "${sources[@]}" -o "$temporary"

if [[ "$mode" == arm ]]; then
  readelf=arm-linux-gnueabihf-readelf
  command -v "$readelf" >/dev/null || { echo "missing build tool: $readelf" >&2; exit 5; }
  [[ -n "$report" ]] || { echo "--report is required for arm mode" >&2; exit 64; }
  header=$($readelf -h "$temporary")
  attributes=$($readelf -A "$temporary")
  versions=$($readelf --version-info "$temporary")
  dynamic=$($readelf -d "$temporary")
  program_headers=$($readelf -l "$temporary")
  grep -q 'Machine:.*ARM' <<<"$header" || { echo "binary is not ARM" >&2; exit 6; }
  grep -q 'Version5 EABI' <<<"$header" || { echo "binary is not EABI5" >&2; exit 6; }
  grep -q 'Tag_CPU_arch: v7' <<<"$attributes" || {
    echo "binary is not ARMv7" >&2
    exit 6
  }
  grep -q 'Tag_ABI_VFP_args: VFP registers' <<<"$attributes" || {
    echo "binary is not hard-float ABI" >&2
    exit 6
  }
  max_glibc=$(grep -oE 'GLIBC_[0-9]+\.[0-9]+' <<<"$versions" |
    sed 's/^GLIBC_//' | sort -t. -k1,1n -k2,2n | tail -1)
  [[ -n "$max_glibc" ]] || { echo "binary has no auditable glibc version" >&2; exit 6; }
  glibc_major=${max_glibc%%.*}
  glibc_minor=${max_glibc#*.}
  if (( glibc_major > 2 || (glibc_major == 2 && glibc_minor > 17) )); then
    echo "binary requires a newer glibc than 2.17" >&2
    exit 6
  fi
  if grep -Eqi 'n2cube' <<<"$dynamic"; then
    echo "binary has a forbidden N2Cube dependency" >&2
    exit 6
  fi
  needed=$(sed -n 's/.*Shared library: \[\([^]]*\)\].*/\1/p' <<<"$dynamic" | paste -sd, -)
  interpreter=$(sed -n 's/.*Requesting program interpreter: \([^]]*\).*/\1/p' \
    <<<"$program_headers" | head -1)
  report_tmp="${report}.new.$$"
  cleanup_report() { [[ ! -e "$report_tmp" ]] || rm -f -- "$report_tmp"; }
  trap 'cleanup; cleanup_report' EXIT
  {
    echo 'MACHINE=ARM'
    echo 'EABI=Version5'
    echo 'HARD_FLOAT=yes'
    echo 'GLIBC_MAX_OK=yes'
    echo 'CPU_ARCH=v7'
    echo "MAX_GLIBC=$max_glibc"
    echo "ELF_INTERPRETER=$interpreter"
    echo "DT_NEEDED=$needed"
    echo "COMPILER=$($compiler --version | head -1)"
  } > "$report_tmp"
fi
mv "$temporary" "$output"
if [[ "$mode" == arm ]]; then mv "$report_tmp" "$report"; fi
trap - EXIT
echo "OUTPUT=$output"
sha256sum "$output"
