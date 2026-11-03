#!/bin/sh
set -eu
repo_root= output_dir=
while [ "$#" -gt 0 ]; do case "$1" in --repo-root) repo_root=$2;shift 2;; --output-dir) output_dir=$2;shift 2;; *) exit 64;; esac; done
[ -d "$repo_root" ] && [ -n "$output_dir" ] || exit 64
[ ! -e "$output_dir" ] || { echo 'output directory already exists' >&2; exit 3; }
parent=${output_dir%/*}; [ "$parent" != "$output_dir" ] || parent=.; [ -d "$parent" ] || exit 2
stage=$(mktemp -d "$parent/.soundjie-hdmi.XXXXXX")
cleanup(){ [ ! -d "$stage" ] || find "$stage" -depth -mindepth 1 -delete >/dev/null 2>&1 || true; [ ! -d "$stage" ] || rmdir "$stage" 2>/dev/null || true; }
trap cleanup 0 HUP INT TERM
compiler=arm-linux-gnueabihf-g++; readelf=arm-linux-gnueabihf-readelf
flags='-std=c++11 -O2 -Wall -Wextra -Werror -march=armv7-a -mfloat-abi=hard -mfpu=vfpv3-d16 -static-libgcc -static-libstdc++'
# shellcheck disable=SC2086
"$compiler" $flags -I"$repo_root/src" "$repo_root/src/framebuffer_surface.cpp" "$repo_root/src/soundjie_hdmi_ui.cpp" "$repo_root/src/soundjie_d4_doa_stream.cpp" "$repo_root/src/soundjie_d4_state.cpp" "$repo_root/src/xvf3800_control_protocol.cpp" "$repo_root/src/sha256.cpp" -o "$stage/soundjie_hdmi_ui"
header=$("$readelf" -h "$stage/soundjie_hdmi_ui"); attrs=$("$readelf" -A "$stage/soundjie_hdmi_ui"); ph=$("$readelf" -l "$stage/soundjie_hdmi_ui"); versions=$("$readelf" --version-info "$stage/soundjie_hdmi_ui"); dynamic=$("$readelf" -d "$stage/soundjie_hdmi_ui")
printf '%s\n' "$header"|grep -q 'Machine:.*ARM'; printf '%s\n' "$attrs"|grep -q 'Tag_CPU_arch: v7'; printf '%s\n' "$attrs"|grep -q 'Tag_ABI_VFP_args: VFP registers'
interpreter=$(printf '%s\n' "$ph"|sed -n 's/.*Requesting program interpreter: \([^]]*\).*/\1/p'|head -1); [ "$interpreter" = /lib/ld-linux-armhf.so.3 ]
max_glibc=$(printf '%s\n' "$versions"|grep -oE 'GLIBC_[0-9]+\.[0-9]+'|sed 's/GLIBC_//'|sort -t. -k1,1n -k2,2n|tail -1)
dt_needed=$(printf '%s\n' "$dynamic"|awk '/Shared library:/{gsub(/.*\[/,"");gsub(/\].*/,"");printf "%s%s",s,$0;s=","}END{if(s=="")printf "none"}')
{ echo BUILD_SCHEMA=1; echo MACHINE=ARM; echo CPU_ARCH=v7; echo HARD_FLOAT=yes; echo "ELF_INTERPRETER=$interpreter"; echo "MAX_GLIBC=$max_glibc"; echo "DT_NEEDED=$dt_needed"; echo "SHA256=$(sha256sum "$stage/soundjie_hdmi_ui"|awk '{print $1}')"; } > "$stage/build-report.txt"
mv -T "$stage" "$output_dir"; trap - 0 HUP INT TERM; echo "OUTPUT_DIR=$output_dir"
