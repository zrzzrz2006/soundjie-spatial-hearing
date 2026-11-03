#!/bin/sh
set -eu

output=xvf3800-diagnostics.tar.gz
proc_root=/proc
arecord_bin=arecord
probe_script="$(dirname "$0")/xvf3800_probe.sh"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) output=$2; shift 2 ;;
    --proc-root) proc_root=$2; shift 2 ;;
    --arecord-bin) arecord_bin=$2; shift 2 ;;
    --probe-script) probe_script=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
run_capture() {
  name=$1
  shift
  { "$@"; } > "$work/$name.txt" 2>&1 || true
}

run_capture uname uname -a
if command -v lsusb >/dev/null 2>&1; then run_capture lsusb lsusb; fi
if command -v dmesg >/dev/null 2>&1; then dmesg 2>&1 | tail -n 300 > "$work/dmesg-tail.txt" || true; fi
if [ -r "$proc_root/asound/cards" ]; then cp "$proc_root/asound/cards" "$work/asound-cards.txt"; fi
run_capture arecord-list "$arecord_bin" -l
for source in "$proc_root"/asound/card[0-9]*/id "$proc_root"/asound/card[0-9]*/usbid "$proc_root"/asound/card[0-9]*/stream0; do
  [ -r "$source" ] || continue
  safe=$(printf '%s' "$source" | tr '/:' '__')
  cp "$source" "$work/$safe.txt"
done
"$probe_script" --proc-root "$proc_root" --arecord-bin "$arecord_bin" --report "$work/probe-report.env" > "$work/probe-output.txt" 2>&1 || true
run_capture loaded-modules cat "$proc_root/modules"

mkdir -p "$(dirname "$output")"
tar -czf "$output" -C "$work" .
printf 'DIAGNOSTICS_ARCHIVE=%s\n' "$output"
