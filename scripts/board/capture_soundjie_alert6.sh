#!/bin/sh
# Bounded native stereo capture only; owned caller must stop the demo first.
set -eu
bundle=${1:?bundle}
output=${2:?new /run directory}
seconds=${3:?duration seconds}
case "$output" in /run/sj-alert6-replay-*) :;; *) exit 64;; esac
[ "$(dirname "$output")" = /run ] || exit 64
case "$seconds" in ''|*[!0-9]*) exit 64;; esac
[ "$seconds" -ge 30 ] && [ "$seconds" -le 180 ] || exit 64
[ ! -e "$output" ] || exit 3
[ "$(stat -f -c %T /run)" = tmpfs ] || exit 64
export LD_LIBRARY_PATH="$bundle/audio/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ALSA_CONFIG_PATH="$bundle/audio/share/alsa/alsa.conf"
mkdir "$output"
printf '%s\n' "$$" > "$output/owner.pid"
printf 'SJ_ALERT6_CAPTURE_START seconds=%s\n' "$seconds"
set +e
"$bundle/audio/bin/arecord" -D hw:0,0 -t wav -f S16_LE -c 2 -r 16000 -d "$seconds" \
  "$output/capture.wav" > "$output/arecord.stdout" 2> "$output/arecord.stderr"
code=$?
set -e
printf '%s\n' "$code" > "$output/arecord.exit"
[ "$code" -eq 0 ] || exit "$code"
if grep -i -E 'overrun|underrun|error|failed' "$output/arecord.stderr"; then
  echo 'SJ_ALERT6_CAPTURE_INVALID'; exit 2
fi
sha256sum "$output/capture.wav" > "$output/capture.sha256"
gzip -1 -c "$output/capture.wav" > "$output/capture.wav.gz"
sha256sum "$output/capture.wav.gz" > "$output/archive.sha256"
wc -c "$output/capture.wav" "$output/capture.wav.gz"
cat "$output/capture.sha256"
echo SJ_ALERT6_CAPTURE_DONE
