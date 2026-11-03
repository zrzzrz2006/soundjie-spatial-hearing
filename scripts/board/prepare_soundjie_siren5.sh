#!/bin/sh
# Create-only tmpfs overlay. No programming, SD writes, capture, or power changes.
set -eu
base=$(CDPATH= cd -- "${1:?verified dual TV overlay required}" && pwd -P)
patch=$(CDPATH= cd -- "${2:?verified siren5 patch required}" && pwd -P)
overlay=${3:?new direct /run directory required}
case "$base:$patch" in /run/*:/run/*) :;; *) exit 64;; esac
case "$overlay" in /run/*) :;; *) exit 64;; esac
case "$overlay" in *'/../'*|*'/..'|*'/./'*|*'/.'|*'//'*) exit 64;; esac
parent=$(CDPATH= cd -- "$(dirname -- "$overlay")" && pwd -P)
[ "$parent" = /run ] && [ "$(stat -f -c %T "$parent")" = tmpfs ] || exit 64
[ "$(stat -f -c %T "$base")" = tmpfs ] && [ "$(stat -f -c %T "$patch")" = tmpfs ] || exit 64
[ ! -e "$overlay" ] && [ ! -L "$overlay" ] || exit 3
(cd "$patch" && sha256sum -c SHA256SUMS)
[ "$(sha256sum "$base/model/dpu_soundjie_d3.elf" | awk '{print $1}')" = 66a317bc2c21e9d155d38e195466219bba0a7e722a01d480dbab95c0d930e7cd ] || exit 65
[ "$(sha256sum "$base/config/logmel_frontend_v1.json" | awk '{print $1}')" = bd11b2f97b4b82e34d39a78121df92b9d1c3620de1ab296934b04c1ff995a0b1 ] || exit 65
[ "$(sha256sum "$base/tv/reference.features" | awk '{print $1}')" = 839537e996650ba41dce80f5229774e8978f45dac2aab6922acfbf383ad324b7 ] || exit 65
[ "$(sha256sum "$base/tv/traffic.bin" | awk '{print $1}')" = d153c13085569058067ccc33d45b6093daaaf4abf887fbd69691b6377ee5bff5 ] || exit 65
for file in "$base/audio/bin/arecord" "$base/runtime/xvf3800_doa" "$base/tv/cues.tsv" "$base/config/soundjie_d4_runtime.tsv"; do [ -f "$file" ] || exit 66; done
mkdir "$overlay" "$overlay/runtime" "$overlay/config" "$overlay/assets"
for item in "$base"/*; do
  name=${item##*/};case "$name" in runtime|config|assets|model|board|manifest) :;; *) ln -s "$item" "$overlay/$name";; esac
done
for item in "$base/runtime"/*; do
  name=${item##*/};case "$name" in soundjie_d4_runtime|soundjie_d4_classifier|soundjie_hdmi_ui) :;; *) ln -s "$item" "$overlay/runtime/$name";; esac
done
for item in "$base/config"/*; do
  name=${item##*/};case "$name" in class-contract.txt|classes.json) :;; *) ln -s "$item" "$overlay/config/$name";; esac
done
for item in "$base/assets"/*; do
  name=${item##*/};[ "$name" = hdmi ] || ln -s "$item" "$overlay/assets/$name"
done
for name in soundjie_d4_runtime soundjie_d4_classifier soundjie_hdmi_ui; do ln -s "$patch/runtime/$name" "$overlay/runtime/$name"; done
ln -s "$patch/model" "$overlay/model"
ln -s "$patch/board" "$overlay/board"
ln -s "$patch/assets/hdmi" "$overlay/assets/hdmi"
ln -s "$patch/config/class-contract.txt" "$overlay/config/class-contract.txt"
ln -s "$patch/config/classes.json" "$overlay/config/classes.json"
printf 'SIREN5_OVERLAY=%s\n' "$overlay"
