#!/bin/bash
set -eu

audit=""
artifact_root=""
work_root=""
toolchain_prefix=""
alsa_lib_tarball=""
alsa_utils_tarball=""
dry_run=0
jobs=4

while [ "$#" -gt 0 ]; do
  case "$1" in
    --audit) audit=$2; shift 2 ;;
    --artifact-root) artifact_root=$2; shift 2 ;;
    --work-root) work_root=$2; shift 2 ;;
    --toolchain-prefix) toolchain_prefix=$2; shift 2 ;;
    --alsa-lib-tarball) alsa_lib_tarball=$2; shift 2 ;;
    --alsa-utils-tarball) alsa_utils_tarball=$2; shift 2 ;;
    --jobs) jobs=$2; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

[ -f "$audit" ] || { echo "audit report missing: $audit" >&2; exit 66; }
value() { sed -n "s/^$1=//p" "$audit" | tail -n 1 | tr -d '\r'; }
action=$(value ACTION)
arecord_present=$(value ARECORD_PRESENT)

if [ "$action" = "BLOCK_KERNEL_MISMATCH" ]; then
  echo "BLOCK_KERNEL_MISMATCH" >&2
  exit 65
fi
if [ "$arecord_present" = "yes" ]; then
  echo "ALSA_PACKAGE=skip"
  exit 0
fi

echo "ALSA_PACKAGE=required"
[ "$dry_run" -eq 0 ] || exit 0
[ -n "$artifact_root" ] || { echo "artifact root is required" >&2; exit 64; }
[ -x "${toolchain_prefix}gcc" ] || { echo "cross compiler missing: ${toolchain_prefix}gcc" >&2; exit 66; }
[ -f "$alsa_lib_tarball" ] || { echo "alsa-lib tarball missing: $alsa_lib_tarball" >&2; exit 66; }
[ -f "$alsa_utils_tarball" ] || { echo "alsa-utils tarball missing: $alsa_utils_tarball" >&2; exit 66; }

work_root=${work_root:-$artifact_root/work}
work="$work_root/alsa-cross"
stage="$work_root/alsa-stage-arm"
bundle="$artifact_root/alsa-runtime"
mkdir -p "$work" "$stage" "$bundle/usr/bin" "$bundle/usr/lib" "$bundle/usr/share"

tar -xf "$alsa_lib_tarball" -C "$work"
tar -xf "$alsa_utils_tarball" -C "$work"
lib_src=$(find "$work" -mindepth 1 -maxdepth 1 -type d -name 'alsa-lib-*' | head -n 1)
utils_src=$(find "$work" -mindepth 1 -maxdepth 1 -type d -name 'alsa-utils-*' | head -n 1)
[ -n "$lib_src" ] && [ -n "$utils_src" ] || { echo "ALSA source extraction failed" >&2; exit 70; }

export CC="${toolchain_prefix}gcc"
export AR="${toolchain_prefix}ar"
export RANLIB="${toolchain_prefix}ranlib"
export STRIP="${toolchain_prefix}strip"
export LD="${toolchain_prefix}ld"
export NM="${toolchain_prefix}nm"
export OBJDUMP="${toolchain_prefix}objdump"

mkdir -p "$work/lib-build-arm" "$work/utils-build-arm"
(cd "$work/lib-build-arm" && "$lib_src/configure" \
  --host=arm-linux-gnueabihf --prefix=/usr --disable-python)
make -C "$work/lib-build-arm" -j"$jobs"
make -C "$work/lib-build-arm" DESTDIR="$stage" install

export PKG_CONFIG_PATH=
export PKG_CONFIG_LIBDIR="$stage/usr/lib/pkgconfig"
export CPPFLAGS="-I$stage/usr/include"
export LDFLAGS="-L$stage/usr/lib -Wl,-rpath,/usr/lib"
(cd "$work/utils-build-arm" && "$utils_src/configure" \
  --host=arm-linux-gnueabihf --prefix=/usr \
  --disable-alsamixer --disable-xmlto --disable-nls)
make -C "$work/utils-build-arm/aplay" -j"$jobs"

cp "$work/utils-build-arm/aplay/aplay" "$bundle/usr/bin/aplay"
cp "$work/utils-build-arm/aplay/aplay" "$bundle/usr/bin/arecord"
cp "$stage/usr/lib/libasound.so.2.0.0" "$bundle/usr/lib/libasound.so.2.0.0"
cp "$stage/usr/lib/libasound.so.2.0.0" "$bundle/usr/lib/libasound.so.2"
cp -a "$stage/usr/share/alsa" "$bundle/usr/share/"
"${toolchain_prefix}strip" --strip-unneeded "$bundle/usr/bin/aplay" "$bundle/usr/bin/arecord"
"${toolchain_prefix}strip" --strip-unneeded "$bundle/usr/lib/libasound.so.2.0.0" "$bundle/usr/lib/libasound.so.2"
"${toolchain_prefix}readelf" -h "$bundle/usr/bin/aplay" | grep -q 'Machine:.*ARM'
(cd "$bundle" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
(cd "$bundle" && sha256sum -c SHA256SUMS >/dev/null)
echo "ALSA_OUTPUT=$bundle"
