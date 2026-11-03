#!/bin/sh
set -eu

bundle_dir=${1:?bundle directory required}
bundle_dir=$(CDPATH= cd -- "$bundle_dir" && pwd)
export SOUNDJIE_BUNDLE_DIR="$bundle_dir"
export LD_LIBRARY_PATH="$bundle_dir/dnndk/pkgs/lib:$bundle_dir/audio/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ALSA_CONFIG_PATH="$bundle_dir/audio/share/alsa/alsa.conf"
output_dir=${2:?new session output directory required}
current_json=${3:-/run/soundjie-d4/current.json}
device=${SOUNDJIE_PCM_DEVICE:-hw:0,0}
arecord_bin="$bundle_dir/audio/bin/arecord"
runtime="$bundle_dir/runtime/soundjie_d4_runtime"
classifier="$bundle_dir/runtime/soundjie_d4_classifier"

for executable in "$arecord_bin" "$runtime" "$classifier"; do
  [ -x "$executable" ] || { echo "missing executable: $executable" >&2; exit 66; }
done
mkdir -p "$(dirname "$current_json")"

exec "$arecord_bin" -D "$device" -t raw -f S16_LE -c 2 -r 16000 | \
  "$runtime" --input - --output-dir "$output_dir" --current "$current_json" \
    --window-samples 16000 --hop-samples 12000 --cadence-ms 750 \
    --classifier-command "$classifier" --retain-windows "${SOUNDJIE_RETAIN_WINDOWS:-0}"
