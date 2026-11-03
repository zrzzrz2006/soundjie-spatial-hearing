#!/bin/sh
# Prepare a RAM-only overlay. Does not start capture, program PL, or write SD.
set -eu
base=${1:?existing verified bundle required}
patch=${2:?verified diagnostic patch directory required}
overlay=${3:?new /run overlay directory required}
base=$(CDPATH= cd -- "$base" && pwd)
patch=$(CDPATH= cd -- "$patch" && pwd)
case "$overlay" in /run/*) :;; *) echo 'overlay must be below /run' >&2; exit 64;; esac
case "$overlay" in *'/../'*|*'/..'|*'/./'*|*'/.'|*'//'*) echo 'noncanonical overlay path' >&2; exit 64;; esac
parent=$(CDPATH= cd -- "$(dirname -- "$overlay")" && pwd -P)
[ "$parent" = /run ] && [ "$(stat -f -c %T "$parent")" = tmpfs ] || {
  echo 'overlay parent must be real /run on tmpfs' >&2; exit 64;
}
[ ! -e "$overlay" ] && [ ! -L "$overlay" ] || { echo 'overlay already exists' >&2; exit 3; }
(cd "$patch" && sha256sum -c SHA256SUMS)
expected=66a317bc2c21e9d155d38e195466219bba0a7e722a01d480dbab95c0d930e7cd
actual=$(sha256sum "$base/model/dpu_soundjie_d3.elf" | awk '{print $1}')
[ "$actual" = "$expected" ] || { echo 'diagnostic classifier requires frozen baseline model' >&2; exit 65; }
for item in soundjie_d4_runtime soundjie_d4_classifier soundjie_hdmi_ui; do
  [ -x "$patch/runtime/$item" ] || { echo 'missing diagnostic executable' >&2; exit 66; }
done
[ -d "$base/runtime" ] || exit 66
mkdir "$overlay"
for item in "$base"/*; do
  name=${item##*/}
  [ "$name" = runtime ] || ln -s "$item" "$overlay/$name"
done
mkdir "$overlay/runtime"
for item in "$base/runtime"/*; do
  name=${item##*/}
  case "$name" in soundjie_d4_runtime|soundjie_d4_classifier|soundjie_hdmi_ui) :;;
    *) ln -s "$item" "$overlay/runtime/$name";;
  esac
done
for item in soundjie_d4_runtime soundjie_d4_classifier soundjie_hdmi_ui; do
  ln -s "$patch/runtime/$item" "$overlay/runtime/$item"
done
printf 'DIAGNOSTIC_OVERLAY=%s\n' "$overlay"
printf '%s\n' 'Baseline model; raw uncalibrated scores; no accuracy improvement claimed.'
