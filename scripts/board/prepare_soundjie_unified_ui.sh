#!/bin/sh
# Create a new RAM overlay for the reviewed UI. Never overwrite the old bundle.
set -eu
base=$(CDPATH= cd -- "${1:?verified original application required}" && pwd -P)
patch=$(CDPATH= cd -- "${2:?verified UI patch required}" && pwd -P)
overlay=${3:?new direct /run directory required}
case "$base:$patch:$overlay" in /run/*:/run/*:/run/*) :;; *) exit 64;; esac
case "$overlay" in *'/../'*|*'/..'|*'/./'*|*'/.'|*'//'*) exit 64;; esac
[ "$(dirname -- "$overlay")" = /run ] || exit 64
[ "$(stat -f -c %T /run)" = tmpfs ] || exit 64
[ "$(stat -f -c %T "$base")" = tmpfs ] || exit 64
[ "$(stat -f -c %T "$patch")" = tmpfs ] || exit 64
[ ! -e "$overlay" ] && [ ! -L "$overlay" ] || exit 3
(cd "$base" && sha256sum -c SHA256SUMS)
(cd "$patch" && sha256sum -c SHA256SUMS)
check() { [ "$(sha256sum "$1" | awk '{print $1}')" = "$2" ] || exit 65; }
check "$base/runtime/soundjie_hdmi_ui" e99425dd280734702f309d5bd2f513456d18798878ab2a4269922bb1ca57d7f8
check "$base/assets/hdmi/soundjie_glyph_atlas.bin" acbb38d5d5e6121f26fcfd1b5935af19c6a2903b7cdf02afd826faebfed1f2b0
check "$base/model/dpu_soundjie_d3.elf" bfaf1f963a6cc9b6a00a4023de36385a858c38efc1541a5dafb286205927e8e6
check "$base/runtime/soundjie_d4_runtime" 06c6546b0e58f0f31fa89b7aa3ef7b7516e67e0e8714c96198fe7e13bbcf8adc
case "$(sha256sum "$patch/runtime/soundjie_hdmi_ui" | awk '{print $1}')" in
  5d1ebbb1e51cab7f3466305bc15c65ddc2638e54e9f489b638b946c75cafc9fa|\
  1efeea939c3a8aba2f403d4cd5a4cf0d446c27f5fb2d41160a9eae91f8f25d0a|\
  0159b622f72d6cf2cb9fdd27d1f1d1d50e2cd2d6a8bd026cd09e49e30ba8f1ca) :;;
  *) exit 65;;
esac
check "$patch/assets/hdmi/soundjie_glyph_atlas.bin" c62a459e5b1838c55b036e3cbd522097f1650e68ad3ce5c885320f9874c52833
[ -x "$patch/runtime/soundjie_hdmi_ui" ] || exit 66
mkdir "$overlay" "$overlay/runtime" "$overlay/assets" "$overlay/assets/hdmi" "$overlay/manifest"
for item in "$base"/*; do
    name=${item##*/};case "$name" in runtime|assets|manifest|SHA256SUMS) :;; *) ln -s "$item" "$overlay/$name";; esac
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
ln -s "$patch/runtime/soundjie_hdmi_ui" "$overlay/runtime/soundjie_hdmi_ui"
ln -s "$patch/assets/hdmi/soundjie_glyph_atlas.bin" "$overlay/assets/hdmi/soundjie_glyph_atlas.bin"
ln -s "$patch/manifest/ui-update.json" "$overlay/manifest/ui-update.json"
ln -s "$base/manifest" "$overlay/manifest/original-release"
printf 'RAM_UI_OVERLAY=%s\n' "$overlay"
