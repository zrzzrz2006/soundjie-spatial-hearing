#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
module_dir="$root/modules"
for module in snd-hwdep.ko snd-rawmidi.ko snd-usbmidi-lib.ko snd-usb-audio.ko; do
  module_name=$(printf '%s' "${module%.ko}" | tr '-' '_')
  if ! grep -q "^$module_name " /proc/modules 2>/dev/null; then
    insmod "$module_dir/$module"
  fi
done
