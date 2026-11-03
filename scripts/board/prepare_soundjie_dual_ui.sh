#!/bin/sh
# Add UI/atlas only. All earlier DPU, PCM, matcher and model files remain intact.
set -eu
base=$(CDPATH= cd -- "${1:?verified continuity overlay required}" && pwd -P)
patch=$(CDPATH= cd -- "${2:?verified dual UI patch required}" && pwd -P)
overlay=${3:?new direct /run directory required}
case "$base:$patch" in /run/*:/run/*) :;; *) exit 64;; esac
case "$overlay" in /run/*) :;; *) exit 64;; esac
case "$overlay" in *'/../'*|*'/..'|*'/./'*|*'/.'|*'//'*) exit 64;; esac
parent=$(CDPATH= cd -- "$(dirname -- "$overlay")" && pwd -P)
[ "$parent" = /run ] && [ "$(stat -f -c %T "$parent")" = tmpfs ] || exit 64
[ "$(stat -f -c %T "$base")" = tmpfs ] && [ "$(stat -f -c %T "$patch")" = tmpfs ] || exit 64
[ ! -e "$overlay" ] && [ ! -L "$overlay" ] || exit 3
(cd "$patch" && sha256sum -c SHA256SUMS)
[ -x "$patch/soundjie_hdmi_ui" ] && [ -f "$patch/soundjie_glyph_atlas.bin" ] || exit 66
[ "$(sha256sum "$base/model/dpu_soundjie_d3.elf" | awk '{print $1}')" = 66a317bc2c21e9d155d38e195466219bba0a7e722a01d480dbab95c0d930e7cd ] || exit 65
[ "$(sha256sum "$base/runtime/soundjie_d4_runtime" | awk '{print $1}')" = a8321f50c44ef89808456820afe8c14e128ac4aefd15d4afb98001ffd8a0d00d ] || exit 65
[ "$(sha256sum "$base/runtime/soundjie_d4_classifier" | awk '{print $1}')" = 2e869dd02e22e71251a76de5936c474d89bdc3990f63d559937831e3003a96dd ] || exit 65
[ "$(sha256sum "$base/runtime/soundjie_hdmi_ui" | awk '{print $1}')" = 2999e53ed3dff98888e2eb0c68bdba0b5fbf54e0a0d408f5125aef4988722c98 ] || exit 65
[ "$(sha256sum "$base/tv/reference.features" | awk '{print $1}')" = 839537e996650ba41dce80f5229774e8978f45dac2aab6922acfbf383ad324b7 ] || exit 65
[ "$(sha256sum "$base/tv/traffic.bin" | awk '{print $1}')" = d153c13085569058067ccc33d45b6093daaaf4abf887fbd69691b6377ee5bff5 ] || exit 65
mkdir "$overlay" "$overlay/runtime" "$overlay/assets" "$overlay/assets/hdmi"
for item in "$base"/*; do
    name=${item##*/};case "$name" in runtime|assets) :;; *) ln -s "$item" "$overlay/$name";; esac
done
for item in "$base/runtime"/*; do
    name=${item##*/};[ "$name" = soundjie_hdmi_ui ] || ln -s "$item" "$overlay/runtime/$name"
done
for item in "$base/assets"/*; do
    name=${item##*/};[ "$name" = hdmi ] || ln -s "$item" "$overlay/assets/$name"
done
for item in "$base/assets/hdmi"/*; do
    name=${item##*/};[ "$name" = soundjie_glyph_atlas.bin ] || ln -s "$item" "$overlay/assets/hdmi/$name"
done
ln -s "$patch/soundjie_hdmi_ui" "$overlay/runtime/soundjie_hdmi_ui"
ln -s "$patch/soundjie_glyph_atlas.bin" "$overlay/assets/hdmi/soundjie_glyph_atlas.bin"
printf 'DUAL_UI_OVERLAY=%s\n' "$overlay"
