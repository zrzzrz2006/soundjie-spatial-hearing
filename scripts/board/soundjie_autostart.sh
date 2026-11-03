#!/bin/sh
# Read-only SD demo init. Original Linux root and app archive remain unchanged.
# --check-bundle is read-only and can run in an existing session.
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
APP_SHA=0e89305ba1822fe5b8f7a41fb0b6919b09b1212eca0e458b9e51046bb762d7af
bundle=/run/sj-auto-release/sj-two-clip-release
session=/run/sj-auto-session

check_bundle() {
    checked=$1
    (cd "$checked" && sha256sum -c SHA256SUMS) || return
    for bin in soundjie_d4_runtime soundjie_d4_classifier xvf3800_doa soundjie_hdmi_ui; do
        [ -x "$checked/runtime/$bin" ] || return 66
    done
    [ "$(cat "$checked/config/class-contract.txt")" = soundjie-alert5bg-v3 ] || return 65
    [ "$(cat "$checked/config/hop-samples.txt")" = 8000 ] || return 65
    [ -f "$checked/tv/bank.tsv" ] || return 66
    printf 'SJ_AUTOSTART_BUNDLE_CHECK_OK\n'
}

case "${1:-}" in
    --check-bundle)
        [ "$#" -eq 2 ] || exit 64
        check_bundle "$2"
        exit $?;;
    --prepare)
        # A separate shell makes errexit effective even when PID1 inspects
        # this command's exit code. Never run this inside the existing demo.
        [ "$(uname -s)" = Linux ] && [ "$PPID" -eq 1 ] &&
            [ -f /run/sj-init ] && [ -f /run/sj-boot/SJAUTO1.SH ] || {
                echo 'prepare requires the cold-boot bootstrap as PID1' >&2; exit 64;
            }
        set -eu
        launched=
        trap 'rc=$?; if [ "$rc" -ne 0 ] && [ -n "$launched" ]; then kill -TERM "$launched" 2>/dev/null || :; wait "$launched" 2>/dev/null || :; fi' EXIT
        mount -t proc proc /proc
        mount -t sysfs sysfs /sys
        awk '$2=="/" && $3=="ext4" && $4 ~ /(^|,)ro(,|$)/ {found=1} END {exit !found}' /proc/mounts
        [ ! -e "$session" ] && [ ! -e /run/sj-auto-release ]
        printf '%s  %s\n' "$APP_SHA" /run/sj-boot/SJ6V1.TGZ | sha256sum -c -
        mkdir /run/sj-auto-release
        tar -xzf /run/sj-boot/SJ6V1.TGZ -C /run/sj-auto-release
        check_bundle "$bundle"
        umount /run/sj-boot
        sh "$bundle/audio/load-usb-audio.sh"
        # Wait at most ten seconds for this exact card, not an arbitrary USB
        # device that happens to use card zero.
        count=0
        until [ -c /dev/snd/pcmC0D0c ] && grep -q 'reSpeaker XVF3800' /proc/asound/cards; do
            count=$((count + 1))
            [ "$count" -le 40 ] || { echo 'XVF3800 capture card timeout' >&2; exit 66; }
            sleep .25
        done
        grep -q 'reSpeaker XVF3800' /proc/asound/card0/stream0
        insmod "$bundle/runtime/dpu-4.14.0-xilinx.ko"
        [ -c /dev/dpu ] && [ -c /dev/fb0 ]
        mount --bind "$bundle/dnndk/pkgs/lib" /usr/local/lib
        export LD_LIBRARY_PATH="$bundle/dnndk/pkgs/lib:$bundle/audio/lib"
        export ALSA_CONFIG_PATH="$bundle/audio/share/alsa/alsa.conf"
        timeout -k 1 8 "$bundle/dnndk/pkgs/bin/dexplorer" -w > /run/sj-auto-dpu.txt
        grep -q 'B1152F' /run/sj-auto-dpu.txt
        SOUNDJIE_RETAIN_WINDOWS=0 SOUNDJIE_LOG_MAX_BYTES=1048576 setsid sh \
            "$bundle/board/run_soundjie_tv_session.sh" "$bundle" "$session" /dev/fb0 \
            < /dev/null > /run/sj-auto-supervisor.log 2>&1 &
        launched=$!
        printf '%s\n' "$launched" > /run/sj-auto-owner.pid
        # A created UI log is not readiness: its initial frames can be stale
        # while the first audio window is still being collected.
        timeout -k 1 15 python3 - "$session" <<'PY'
import json, os, sys, time
from pathlib import Path
s = Path(sys.argv[1])
deadline = time.monotonic() + 12
while time.monotonic() < deadline:
    if (s/'status.json').exists():
        raise SystemExit('application exited during startup')
    try:
        raw = (s/'logs/ui.jsonl').read_text(encoding='utf-8')
        lines = raw.splitlines()
        if not raw.endswith('\n'):
            lines = lines[:-1]
        u = json.loads(lines[-1])
        if all(u[k] for k in ('usb_ok','dpu_ok','framebuffer_ok','classification_fresh')) and u['display_class_count'] == 6:
            print('SJ_AUTO_HEALTH_OK')
            break
    except (OSError, ValueError, KeyError, IndexError):
        pass
    time.sleep(.25)
else:
    raise SystemExit('HDMI/DPU/USB startup readiness timeout')
PY
        printf 'SJ_AUTO_APP_STARTED=%s\n' "$session"
        exit 0;;
    '') :;;
    *) echo 'unknown startup mode' >&2; exit 64;;
esac

[ "$(uname -s)" = Linux ] && [ "$$" -eq 1 ] || { echo 'automatic startup requires PID1; use --check-bundle for inspection' >&2; exit 64; }
# No automatic reboot/retry on hardware faults. The established supervisor
# clears stale HDMI state and stops its children if a runtime process fails.
if /bin/sh "$0" --prepare > /run/sj-startup.log 2>&1; then
    printf '\nSJ_AUTO_READY: application running; serial is maintenance only.\n'
else
    printf '\nSJ_AUTO_FAILED: inspect /run/sj-startup.log; do not remove SD.\n'
    cat /run/sj-startup.log
fi
printf 'Use normal application stop and halt before power removal.\n'
# If a maintenance shell exits or the serial peer disconnects, PID1 must live.
# The shell reaps its known children; all app workers are owned by the existing
# session supervisor, which also records a terminal failure state.
while :; do
    /bin/sh -i < /dev/console > /dev/console 2>&1
    sleep 1
done
