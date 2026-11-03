#!/bin/sh
# Add only the revised matcher/runtime in RAM; preserve earlier binaries/data.
set -eu
base=$(CDPATH= cd -- "${1:?verified calibrated TV overlay required}" && pwd -P)
patch=$(CDPATH= cd -- "${2:?verified runtime patch required}" && pwd -P)
overlay=${3:?new direct /run directory required}
case "$base:$patch" in /run/*:/run/*) :;; *) exit 64;; esac
case "$overlay" in /run/*) :;; *) exit 64;; esac
case "$overlay" in *'/../'*|*'/..'|*'/./'*|*'/.'|*'//'*) exit 64;; esac
parent=$(CDPATH= cd -- "$(dirname -- "$overlay")" && pwd -P)
[ "$parent" = /run ] && [ "$(stat -f -c %T "$parent")" = tmpfs ] || exit 64
[ "$(stat -f -c %T "$base")" = tmpfs ] && [ "$(stat -f -c %T "$patch")" = tmpfs ] || exit 64
[ ! -e "$overlay" ] && [ ! -L "$overlay" ] || exit 3
(cd "$patch" && sha256sum -c SHA256SUMS)
[ -x "$patch/soundjie_d4_runtime" ] || exit 66
[ "$(sha256sum "$base/runtime/soundjie_d4_runtime" | awk '{print $1}')" = 8c308e3505f91c30869441fd3a4bf5570079bb1193361d9637fb7630ddfa9b85 ] || exit 65
[ "$(sha256sum "$base/tv/reference.features" | awk '{print $1}')" = 839537e996650ba41dce80f5229774e8978f45dac2aab6922acfbf383ad324b7 ] || exit 65
[ "$(sha256sum "$base/tv/traffic.bin" | awk '{print $1}')" = d153c13085569058067ccc33d45b6093daaaf4abf887fbd69691b6377ee5bff5 ] || exit 65
mkdir "$overlay" "$overlay/runtime"
for item in "$base"/*; do
    name=${item##*/};[ "$name" = runtime ] || ln -s "$item" "$overlay/$name"
done
for item in "$base/runtime"/*; do
    name=${item##*/};[ "$name" = soundjie_d4_runtime ] || ln -s "$item" "$overlay/runtime/$name"
done
ln -s "$patch/soundjie_d4_runtime" "$overlay/runtime/soundjie_d4_runtime"
printf 'TV_CONTINUITY_OVERLAY=%s\n' "$overlay"
