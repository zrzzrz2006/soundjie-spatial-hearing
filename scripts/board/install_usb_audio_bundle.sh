#!/bin/sh
set -eu

bundle=
install_root=/opt/xvf3800
expected_kernel=4.14.0-xilinx
dry_run=no

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bundle) bundle=$2; shift 2 ;;
    --install-root) install_root=$2; shift 2 ;;
    --expected-kernel) expected_kernel=$2; shift 2 ;;
    --dry-run) dry_run=yes; shift ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

[ -n "$bundle" ] || { echo "--bundle is required" >&2; exit 64; }
[ -d "$bundle/modules/$expected_kernel" ] || { echo "module bundle missing" >&2; exit 66; }
[ -x "$bundle/alsa-runtime/usr/bin/arecord" ] || { echo "ALSA runtime missing" >&2; exit 66; }
[ -d "$bundle/board-tools" ] || { echo "board tools missing" >&2; exit 66; }

actual_kernel=$(uname -r)
actual_machine=$(uname -m)
[ "$actual_kernel" = "$expected_kernel" ] || { echo "kernel mismatch: $actual_kernel" >&2; exit 65; }
case "$actual_machine" in armv7l|armv7*) ;; *) echo "machine mismatch: $actual_machine" >&2; exit 65 ;; esac
[ -e /lib/ld-linux-armhf.so.3 ] || { echo "ARM hard-float loader missing: /lib/ld-linux-armhf.so.3" >&2; exit 65; }

if [ -e "$install_root" ] && [ "$(find "$install_root" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
  echo "refusing to overwrite non-empty install root: $install_root" >&2
  exit 73
fi

if [ "$dry_run" = yes ]; then
  cat <<EOF
INSTALL_STATUS=dry-run
KERNEL_RELEASE=$actual_kernel
MACHINE=$actual_machine
INSTALL_ROOT=$install_root
EOF
  exit 0
fi

mkdir -p "$install_root"
cp -R "$bundle/modules" "$install_root/"
cp -R "$bundle/alsa-runtime" "$install_root/"
cp -R "$bundle/board-tools" "$install_root/"
if [ -f "$bundle/wav_probe" ]; then cp "$bundle/wav_probe" "$install_root/"; fi

cat > "$install_root/env.sh" <<EOF
XVF3800_ROOT=$install_root
export XVF3800_ROOT
PATH=\$XVF3800_ROOT/alsa-runtime/usr/bin:\$XVF3800_ROOT/board-tools:\$PATH
LD_LIBRARY_PATH=\$XVF3800_ROOT/alsa-runtime/usr/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
ALSA_CONFIG_PATH=\$XVF3800_ROOT/alsa-runtime/usr/share/alsa/alsa.conf
export PATH LD_LIBRARY_PATH ALSA_CONFIG_PATH
EOF

cat > "$install_root/load-usb-audio.sh" <<EOF
#!/bin/sh
set -eu
module_dir=$install_root/modules/$expected_kernel
for module in snd-hwdep.ko snd-rawmidi.ko snd-usbmidi-lib.ko snd-usb-audio.ko; do
  module_name=\$(printf '%s' "\${module%.ko}" | tr '-' '_')
  if ! grep -q "^\$module_name " /proc/modules 2>/dev/null; then
    insmod "\$module_dir/\$module"
  fi
done
EOF
chmod +x "$install_root/load-usb-audio.sh" "$install_root/board-tools"/*.sh
printf 'INSTALL_STATUS=complete\nINSTALL_ROOT=%s\n' "$install_root"
