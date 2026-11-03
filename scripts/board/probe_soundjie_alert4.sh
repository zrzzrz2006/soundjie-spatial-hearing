#!/bin/sh
# Manual only: verify new linked DPU on stored validation PCM before live capture.
set -eu
overlay=$(CDPATH= cd -- "${1:?prepared alert4bg overlay}" && pwd -P)
patch=$(CDPATH= cd -- "${2:?verified alert4bg package}" && pwd -P)
output=${3:?new direct /run output}
case "$overlay:$patch:$output" in /run/*:/run/*:/run/*) :;; *) exit 64;; esac
case "$output" in *'/../'*|*'/..'|*'/./'*|*'/.'|*'//'*) exit 64;; esac
[ "$(dirname -- "$output")" = /run ] && [ "$(stat -f -c %T /run)" = tmpfs ] || exit 64
[ ! -e "$output" ] && [ ! -L "$output" ] || exit 3
(cd "$patch" && sha256sum -c SHA256SUMS)
[ "$(cat "$overlay/config/class-contract.txt")" = soundjie-alert4bg-v2 ] || exit 65
for name in model/dpu_soundjie_d3.elf runtime/soundjie_d4_classifier; do
  [ "$(sha256sum "$overlay/$name" | awk '{print $1}')" = "$(sha256sum "$patch/$name" | awk '{print $1}')" ] || exit 65
done
mkdir "$output"
export SOUNDJIE_BUNDLE_DIR="$overlay"
export LD_LIBRARY_PATH="$overlay/dnndk/pkgs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
count=0
for pcm in "$patch"/probe/*.s16; do
  name=${pcm##*/}
  timeout -k 1 10 "$overlay/runtime/soundjie_d4_classifier" "$pcm" >"$output/$name.txt" 2>"$output/$name.stderr"
  count=$((count+1))
done
[ "$count" = 10 ] || exit 65
printf 'DPU_PCM_PROBE_COMMANDS_COMPLETE=%s\n' "$count"
# Host must still compare the numeric results; this line is not an accuracy pass.
