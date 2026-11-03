#!/bin/sh
# Only prepare a RAM overlay; never mount/write SD, program PL or start capture.
set -eu
base=${1:?existing verified D4 bundle required}
patch=${2:?TV patch directory required}
overlay=${3:?new direct /run directory required}
base=$(CDPATH= cd -- "$base" && pwd -P)
patch=$(CDPATH= cd -- "$patch" && pwd -P)
case "$overlay" in /run/*) :;; *) echo 'overlay must be below /run' >&2; exit 64;; esac
case "$overlay" in *'/../'*|*'/..'|*'/./'*|*'/.'|*'//'*) echo 'noncanonical overlay' >&2; exit 64;; esac
parent=$(CDPATH= cd -- "$(dirname -- "$overlay")" && pwd -P)
[ "$parent" = /run ] && [ "$(stat -f -c %T "$parent")" = tmpfs ] || exit 64
[ ! -e "$overlay" ] && [ ! -L "$overlay" ] || { echo 'overlay already exists' >&2; exit 3; }
(cd "$patch" && sha256sum -c SHA256SUMS)
expected=66a317bc2c21e9d155d38e195466219bba0a7e722a01d480dbab95c0d930e7cd
[ "$(sha256sum "$base/model/dpu_soundjie_d3.elf" | awk '{print $1}')" = "$expected" ] || { echo 'DPU baseline mismatch' >&2; exit 65; }
for name in soundjie_d4_runtime soundjie_d4_classifier soundjie_hdmi_ui; do
  [ -x "$patch/runtime/$name" ] || exit 66
done
[ -d "$base/runtime" ] && [ -d "$base/assets" ] || exit 66
mkdir "$overlay" "$overlay/runtime" "$overlay/assets"
for item in "$base"/*; do
  name=${item##*/};case "$name" in runtime|assets|tv) :;; *) ln -s "$item" "$overlay/$name";; esac
done
for item in "$base/runtime"/*; do
  name=${item##*/};case "$name" in soundjie_d4_runtime|soundjie_d4_classifier|soundjie_hdmi_ui) :;; *) ln -s "$item" "$overlay/runtime/$name";; esac
done
for item in "$base/assets"/*; do
  name=${item##*/};[ "$name" = hdmi ] || ln -s "$item" "$overlay/assets/$name"
done
for name in soundjie_d4_runtime soundjie_d4_classifier soundjie_hdmi_ui; do ln -s "$patch/runtime/$name" "$overlay/runtime/$name"; done
ln -s "$patch/assets/hdmi" "$overlay/assets/hdmi"
ln -s "$patch/tv" "$overlay/tv"
printf 'TV_OVERLAY=%s\n' "$overlay"
