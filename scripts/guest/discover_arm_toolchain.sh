#!/usr/bin/env bash
set -euo pipefail

# Print a verified ARMv7 hard-float C++ compiler path.  Set CXX to override
# discovery when running inside a prepared VM or chroot.
candidates=()
if [[ -n "${CXX:-}" ]]; then
  candidates+=("$CXX")
fi
if command -v arm-linux-gnueabihf-g++ >/dev/null 2>&1; then
  candidates+=("$(command -v arm-linux-gnueabihf-g++)")
fi
while IFS= read -r candidate; do
  candidates+=("$candidate")
done < <(find /opt /tools /mnt/f/MLK/DPU_ZYNQ7000/tools -type f \
  -name arm-linux-gnueabihf-g++ 2>/dev/null | sort)

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
printf 'int main() { return 0; }\n' >"$work_dir/probe.cpp"

for compiler in "${candidates[@]}"; do
  [[ -x "$compiler" ]] || continue
  if "$compiler" -std=c++11 "$work_dir/probe.cpp" -o "$work_dir/probe" \
      >/dev/null 2>&1 &&
     readelf -h "$work_dir/probe" 2>/dev/null | grep -q 'Machine:.*ARM' &&
     readelf -h "$work_dir/probe" 2>/dev/null | grep -q 'Version5 EABI' &&
     readelf -A "$work_dir/probe" 2>/dev/null | grep -q 'Tag_ABI_VFP_args: VFP registers'; then
    printf '%s\n' "$compiler"
    exit 0
  fi
done

echo 'No verified ARMv7 EABI5 hard-float C++ toolchain found.' >&2
exit 1
