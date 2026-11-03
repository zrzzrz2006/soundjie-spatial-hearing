#!/bin/sh
# Data-only enrollment overlay: new /run directories, existing programs unchanged.
set -eu
base=$(CDPATH= cd -- "${1:?verified TV overlay required}" && pwd -P)
data=$(CDPATH= cd -- "${2:?calibration data directory required}" && pwd -P)
overlay=${3:?new direct /run directory required}
case "$base:$data" in /run/*:/run/*) :;; *) exit 64;; esac
case "$overlay" in /run/*) :;; *) exit 64;; esac
case "$overlay" in *'/../'*|*'/..'|*'/./'*|*'/.'|*'//'*) exit 64;; esac
parent=$(CDPATH= cd -- "$(dirname -- "$overlay")" && pwd -P)
[ "$parent" = /run ] && [ "$(stat -f -c %T "$parent")" = tmpfs ] || exit 64
[ "$(stat -f -c %T "$base")" = tmpfs ] && [ "$(stat -f -c %T "$data")" = tmpfs ] || exit 64
[ ! -e "$overlay" ] && [ ! -L "$overlay" ] || exit 3
(cd "$data" && sha256sum -c SHA256SUMS)
for item in reference.features traffic.bin report.json; do [ -f "$data/$item" ] || exit 66; done
# Enroll only the audited TV1 program and baseline, not an arbitrary D4 build.
[ "$(sha256sum "$base/runtime/soundjie_d4_runtime" | awk '{print $1}')" = 8c308e3505f91c30869441fd3a4bf5570079bb1193361d9637fb7630ddfa9b85 ] || exit 65
[ "$(sha256sum "$base/tv/reference.features" | awk '{print $1}')" = 5ecf799805da26b59b82b128c3801c519ec467f8c4487f28a7a2efbc5110ea73 ] || exit 65
[ "$(sha256sum "$base/tv/traffic.bin" | awk '{print $1}')" = 2c5a08559bb64b2fa6010792c051952c97d460abb83bd99c4d182b6d1bfb0114 ] || exit 65
mkdir "$overlay" "$overlay/tv"
for item in "$base"/*; do
    name=${item##*/}
    [ "$name" = tv ] || ln -s "$item" "$overlay/$name"
done
for item in "$base/tv"/*; do
    name=${item##*/}
    case "$name" in reference.features|traffic.bin) :;; *) ln -s "$item" "$overlay/tv/$name";; esac
done
ln -s "$data/reference.features" "$overlay/tv/reference.features"
ln -s "$data/traffic.bin" "$overlay/tv/traffic.bin"
printf 'TV_CALIBRATED_OVERLAY=%s\n' "$overlay"
