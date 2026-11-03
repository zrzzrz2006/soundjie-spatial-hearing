#!/bin/sh
# Foreground, fail-stop session owner. Does not boot/program/shut down hardware.
set -eu
bundle=${1:?bundle directory required}
session=${2:?new absolute session directory required}
framebuffer=${3:-/dev/fb0}
bundle=$(CDPATH= cd -- "$bundle" && pwd)
case "$session" in /*) :;; *) echo 'session must be absolute' >&2; exit 64;; esac
[ ! -e "$session" ] || { echo 'session already exists' >&2; exit 3; }
for tool in setsid timeout mkfifo sleep df awk dirname tr; do
  command -v "$tool" >/dev/null 2>&1 || { echo "missing host tool: $tool" >&2; exit 66; }
done
# Sessions store bounded routine history. Keep room for terminal publication
# instead of filling tmpfs and leaving a stale alert on HDMI. Never delete logs
# or touch user files automatically. Producers rotate their own bounded files;
# explicit SOUNDJIE_LOG_MAX_BYTES=0 retains full diagnostics until this guard.
available_kb() {
  space_report=$(LC_ALL=C df -Pk "$1") || return 1
  printf '%s\n' "$space_report" | awk '
    NR == 2 { if (NF < 4 || $4 !~ /^[0-9]+$/) exit 1; print $4; found=1 }
    END { if (!found) exit 1 }'
}
space_kb=$(available_kb "$(dirname -- "$session")") || {
  echo 'cannot verify session filesystem space' >&2; exit 66;
}
[ "$space_kb" -ge 16384 ] || { echo 'session filesystem has less than 16 MiB free' >&2; exit 66; }
capture="$bundle/audio/bin/arecord"
runtime="$bundle/runtime/soundjie_d4_runtime"
classifier="$bundle/runtime/soundjie_d4_classifier"
doa="$bundle/runtime/xvf3800_doa"
ui="$bundle/runtime/soundjie_hdmi_ui"
atlas="$bundle/assets/hdmi/soundjie_glyph_atlas.bin"
config="$bundle/config/soundjie_d4_runtime.tsv"
set --
case "${SOUNDJIE_TV_ENABLE:-0}" in
  0) :;;
  1)
    for required in "$bundle/tv/traffic.bin"; do
      [ -f "$required" ] || { echo "missing TV file: $required" >&2; exit 66; }
    done
    if [ -f "$bundle/tv/bank.tsv" ]; then
      set -- --tv-bank "$bundle/tv/bank.tsv" --traffic-model "$bundle/tv/traffic.bin"
      [ -f "$bundle/tv/dialogue-horn-start.txt" ] && [ -f "$bundle/tv/dialogue-horn-end.txt" ] || { echo 'missing dialogue interval' >&2; exit 66; }
      set -- "$@" --tv-dialogue-horn-start "$(tr -d '\r' < "$bundle/tv/dialogue-horn-start.txt")" --tv-dialogue-horn-end "$(tr -d '\r' < "$bundle/tv/dialogue-horn-end.txt")"
    else
      for required in "$bundle/tv/reference.features" "$bundle/tv/cues.tsv"; do
        [ -f "$required" ] || { echo "missing TV file: $required" >&2; exit 66; }
      done
      set -- --tv-reference "$bundle/tv/reference.features" --tv-cues "$bundle/tv/cues.tsv" --traffic-model "$bundle/tv/traffic.bin"
    fi
    ;;
  *) echo 'SOUNDJIE_TV_ENABLE must be 0 or 1' >&2; exit 64;;
esac
class_contract=soundjie-legacy8-v1
if [ -f "$bundle/config/class-contract.txt" ]; then
  class_contract=$(cat "$bundle/config/class-contract.txt")
  case "$class_contract" in
    soundjie-siren5-v1) :;;
    soundjie-alert4bg-v2|soundjie-alert5bg-v3)
      [ -f "$bundle/config/alert-thresholds.txt" ] || { echo 'missing alert thresholds' >&2; exit 65; }
      set -- "$@" --alert-thresholds "$(cat "$bundle/config/alert-thresholds.txt")";;
    *) echo 'unknown bundle class contract' >&2; exit 65;;
  esac
  set -- "$@" --class-contract "$class_contract"
fi
hop_samples=12000
if [ -f "$bundle/config/hop-samples.txt" ]; then hop_samples=$(cat "$bundle/config/hop-samples.txt"); fi
hop_samples=${SOUNDJIE_HOP_SAMPLES:-$hop_samples}
case "$hop_samples" in 4000|6400|8000|12000) :;; *) echo 'unsupported hop samples' >&2; exit 65;; esac
cadence_ms=$((hop_samples / 16))
log_max_bytes=${SOUNDJIE_LOG_MAX_BYTES:-1048576}
case "$log_max_bytes" in ''|*[!0-9]*) echo 'invalid log limit' >&2; exit 65;; esac
if [ "$log_max_bytes" != 0 ]; then
  [ "$log_max_bytes" -ge 4096 ] && [ "$log_max_bytes" -le 16777216 ] || { echo 'invalid log limit' >&2; exit 65; }
fi
bounded_logs=0
if [ -f "$bundle/config/log-policy.txt" ]; then
  [ "$(cat "$bundle/config/log-policy.txt")" = bounded-jsonl-v1 ] || { echo 'unknown log capability' >&2; exit 65; }
  bounded_logs=1
  set -- "$@" --log-max-bytes "$log_max_bytes"
fi
for executable in "$capture" "$runtime" "$classifier" "$doa" "$ui"; do
  [ -x "$executable" ] || { echo "missing executable: $executable" >&2; exit 66; }
done
for required in "$atlas" "$config"; do
  [ -f "$required" ] || { echo "missing file: $required" >&2; exit 66; }
done
export SOUNDJIE_BUNDLE_DIR="$bundle"
export LD_LIBRARY_PATH="$bundle/dnndk/pkgs/lib:$bundle/audio/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ALSA_CONFIG_PATH="$bundle/audio/share/alsa/alsa.conf"
mkdir "$session"
mkdir "$session/state" "$session/logs"
printf '%s\n' "$$" > "$session/supervisor.pid"
mkfifo "$session/state/pcm.fifo"
current="$session/state/current.json"
capture_pid= runtime_pid= doa_pid= ui_pid=
reason=supervisor_error
child_exit=-1
result=2

cleanup() {
  trap - EXIT HUP INT TERM
  set +e
  # Only the four groups started by this invocation. Never killall/pkill.
  for pid in $capture_pid $runtime_pid $doa_pid $ui_pid; do kill -TERM "-$pid" 2>/dev/null; done
  sleep .3
  for pid in $capture_pid $runtime_pid $doa_pid $ui_pid; do kill -KILL "-$pid" 2>/dev/null; done
  for pid in $capture_pid $runtime_pid $doa_pid $ui_pid; do wait "$pid" 2>/dev/null; done
  # Producers are stopped before publishing the terminal error/unavailable state.
  session_state=fault
  [ "$reason" != requested_stop ] || session_state=stopped
  printf '{"session_state":"%s","class_contract":"%s","classification_rejected":false,"class_scores":[],"sequence":0,"timestamp_ms":0,"class_id":-1,"label_zh":"不可用","confidence":0,"top3":[],"doa_raw_deg":null,"doa_deg":null,"doa_sector":null,"doa_valid":false,"doa_reason":"session_stopped","rms":0,"dbfs":-96,"peak":0,"clipping_rate":0,"audio_state":"unavailable","usb_ok":false,"dpu_ok":false,"framebuffer_ok":false,"inference_ms":0,"end_to_end_ms":0}\n' "$session_state" "$class_contract" > "$current.stop"
  mv "$current.stop" "$current"
  timeout -k 1 3 "$ui" --class-contract "$class_contract" --snapshot "$current" --atlas "$atlas" --framebuffer "$framebuffer" > "$session/logs/final-display.json" 2> "$session/logs/final-display.stderr"
  clear_exit=$?
  printf '{"reason":"%s","child_exit":%s,"clear_exit":%s,"exit_code":%s}\n' "$reason" "$child_exit" "$clear_exit" "$result" > "$session/status.json"
  if [ "$clear_exit" -ne 0 ]; then echo 'terminal display clear failed; inspect final-display.stderr' >&2; fi
  exit "$result"
}
trap cleanup EXIT
trap 'reason=requested_stop; result=0; exit 0' HUP INT TERM

# Noninteractive shell has job control disabled; setsid does not need to fork.
# Each wrapper records the command's exit separately (including a clean EOF).
start() {
  role=$1; source=$2; destination=$3; shift 3
  setsid sh -c '
    marker=$1; source=$2; destination=$3; shift 3
    "$@" < "$source" > "$destination" 2> "$marker.stderr"
    code=$?
    printf "%s\n" "$code" > "$marker.exit"
    exit "$code"
  ' d4-child "$session/logs/$role" "$source" "$destination" "$@" &
  started_pid=$!
}
start_capture() {
  # XVF3800 verified period: 800 frames = 50 ms at 16 kHz. The default 2000
  # frames rounded a 400 ms inference hop to 500 ms. Keep 500 ms overrun margin.
  case "$class_contract" in
    soundjie-alert4bg-v2|soundjie-alert5bg-v3) set -- --period-size=800 --buffer-size=8000;;
    *) set --;;
  esac
  start capture /dev/null "$session/state/pcm.fifo" "$capture" \
    -D "${SOUNDJIE_PCM_DEVICE:-hw:0,0}" -t raw -f S16_LE -c 2 -r 16000 "$@"
}
start_capture
capture_pid=$started_pid
start runtime "$session/state/pcm.fifo" "$session/logs/runtime.stdout" "$runtime" \
  --input - --output-dir "$session/inference" --current "$current" \
  --window-samples 16000 --hop-samples "$hop_samples" --cadence-ms "$cadence_ms" \
  --classifier-command "$classifier" --classifier-timeout-ms "${SOUNDJIE_CLASSIFIER_TIMEOUT_MS:-5000}" \
  --retain-windows "${SOUNDJIE_RETAIN_WINDOWS:-0}" "$@"
runtime_pid=$started_pid
set --
[ "$bounded_logs" -eq 0 ] || set -- --log-max-bytes "$log_max_bytes"
start doa /dev/null "$session/logs/doa.stdout" "$doa" \
  --jsonl "$session/state/doa.jsonl" --interval-ms 100 "$@"
doa_pid=$started_pid
ui_stdout="$session/logs/ui.jsonl"
if [ "$bounded_logs" -eq 1 ]; then
  ui_stdout="$session/logs/ui.stdout"
  set -- "$@" --jsonl "$session/logs/ui.jsonl"
fi
start ui /dev/null "$ui_stdout" "$ui" "$@" \
  --class-contract "$class_contract" --snapshot "$current" --doa-jsonl "$session/state/doa.jsonl" \
  --runtime-config "$config" --atlas "$atlas" --framebuffer "$framebuffer" \
  --frames 0 --interval-ms 100 --snapshot-stale-ms 2000
ui_pid=$started_pid
printf 'D4_SESSION_STARTED=%s\n' "$session"
space_polls=0
while :; do
  for role in capture runtime doa ui; do
    case "$role" in capture) pid=$capture_pid;; runtime) pid=$runtime_pid;; doa) pid=$doa_pid;; ui) pid=$ui_pid;; esac
    if [ -f "$session/logs/$role.exit" ]; then
      reason=${role}_exited
      child_exit=$(head -n 1 "$session/logs/$role.exit")
      case "$child_exit" in ''|*[!0-9]*) child_exit=-1;; esac
      exit 2
    fi
    if ! kill -0 "$pid" 2>/dev/null; then reason=${role}_disappeared; exit 2; fi
  done
  space_polls=$((space_polls + 1))
  if [ "$space_polls" -ge 50 ]; then
    space_polls=0
    space_kb=$(available_kb "$session") || { reason=storage_check_failed; exit 2; }
    if [ "$space_kb" -lt 16384 ]; then reason=storage_low; exit 2; fi
  fi
  # A group SIGTERM also interrupts sleep. Do not let errexit bypass the
  # pending signal trap on older dash versions used by the board.
  sleep .1 || :
done
