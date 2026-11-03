#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 REPO_ROOT DNNDK_PKG_ROOT MODEL_ELF OUTPUT" >&2
  exit 64
fi

repo_root="$(realpath "$1")"
dnndk_root="$(realpath "$2")"
model_elf="$(realpath "$3")"
output="$4"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cxx="$($script_dir/discover_arm_toolchain.sh)"

mkdir -p "$(dirname "$output")"
"$cxx" \
  -std=c++11 -O2 -Wall -Wextra \
  -static-libgcc -static-libstdc++ \
  -I"$dnndk_root/include" -I"$repo_root/src" \
  "$repo_root/src/dpu_tensor_runner.cpp" \
  "$repo_root/src/tensor_stats.cpp" \
  "$model_elf" \
  -L"$dnndk_root/lib" -Wl,-rpath,/usr/lib \
  -ln2cube -lhineon -lpthread -ldl \
  -o "$output"

readelf -h "$output" | grep -E 'Class:|Machine:|Flags:'
readelf -d "$output" | grep -E 'NEEDED|RPATH|RUNPATH'
readelf --version-info "$output" | grep 'Name: GLIBC_' | sort -u
sha256sum "$output"
