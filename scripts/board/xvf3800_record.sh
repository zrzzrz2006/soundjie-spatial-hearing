#!/bin/sh
set -eu

seconds=10
output=
proc_root=/proc
arecord_bin=arecord
report=
probe_script="$(dirname "$0")/xvf3800_probe.sh"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --seconds) seconds=$2; shift 2 ;;
    --output) output=$2; shift 2 ;;
    --proc-root) proc_root=$2; shift 2 ;;
    --arecord-bin) arecord_bin=$2; shift 2 ;;
    --report) report=$2; shift 2 ;;
    --probe-script) probe_script=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

case "$seconds" in *[!0-9]*|'') echo "seconds must be a positive integer" >&2; exit 64 ;; esac
[ "$seconds" -gt 0 ] || { echo "seconds must be greater than zero" >&2; exit 64; }
[ -n "$output" ] || { echo "--output is required" >&2; exit 64; }

cleanup_report=no
if [ -z "$report" ]; then
  report=$(mktemp)
  cleanup_report=yes
fi
trap '[ "$cleanup_report" = yes ] && rm -f "$report"' EXIT

"$probe_script" --proc-root "$proc_root" --arecord-bin "$arecord_bin" --report "$report" >/dev/null
value() { sed -n "s/^$1=//p" "$report" | head -n 1; }

device=$(value PCM_DEVICE)
format=$(value SELECTED_FORMAT)
channels=$(value SELECTED_CHANNELS)
rate=$(value SELECTED_RATE)
supported=$(value SUPPORTED_CAPTURE)
[ "$supported" = yes ] || { echo "capture is not supported by discovered hardware parameters" >&2; exit 5; }

mkdir -p "$(dirname "$output")"
"$arecord_bin" -D "$device" -t wav -f "$format" -c "$channels" -r "$rate" -d "$seconds" "$output"
[ -s "$output" ] || { echo "capture output is empty: $output" >&2; exit 6; }

cat <<EOF
CAPTURE_STATUS=complete
CAPTURE_DEVICE=$device
CAPTURE_FORMAT=$format
CAPTURE_CHANNELS=$channels
CAPTURE_RATE=$rate
CAPTURE_SECONDS=$seconds
CAPTURE_OUTPUT=$output
EOF
