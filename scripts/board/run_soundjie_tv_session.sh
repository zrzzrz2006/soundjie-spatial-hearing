#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
export SOUNDJIE_TV_ENABLE=1
export SOUNDJIE_RETAIN_WINDOWS=${SOUNDJIE_RETAIN_WINDOWS:-120}
# This is deliberately manual. No autostart/boot files are altered.
exec sh "$here/run_soundjie_d4_session.sh" "$@"
