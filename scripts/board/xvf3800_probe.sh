#!/bin/sh
set -eu

proc_root=/proc
arecord_bin=arecord
report=
identity_regex='ReSpeaker|XVF3800|XMOS|USB Audio'

while [ "$#" -gt 0 ]; do
  case "$1" in
    --proc-root) proc_root=$2; shift 2 ;;
    --arecord-bin) arecord_bin=$2; shift 2 ;;
    --report) report=$2; shift 2 ;;
    --identity-regex) identity_regex=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

emit_report() {
  content=$1
  printf '%s\n' "$content"
  if [ -n "$report" ]; then
    mkdir -p "$(dirname "$report")"
    printf '%s\n' "$content" > "$report"
  fi
}

asound_root="$proc_root/asound"
cards_file="$asound_root/cards"
if [ ! -r "$cards_file" ]; then
  emit_report 'DEVICE_FOUND=no
REASON=alsa_cards_unavailable'
  exit 2
fi

card_index=
card_id=
usb_id=
for card_dir in "$asound_root"/card[0-9]*; do
  [ -d "$card_dir" ] || continue
  index=${card_dir##*/card}
  identity=
  for source in "$card_dir/id" "$card_dir/usbid" "$card_dir/stream0"; do
    if [ -r "$source" ]; then
      identity="$identity $(cat "$source")"
    fi
  done
  identity="$identity $(grep -A1 "^[[:space:]]*$index[[:space:]]" "$cards_file" 2>/dev/null || true)"
  if printf '%s\n' "$identity" | grep -Eqi "$identity_regex"; then
    card_index=$index
    card_id=$(cat "$card_dir/id" 2>/dev/null || printf 'unknown')
    usb_id=$(cat "$card_dir/usbid" 2>/dev/null || printf 'unknown')
    break
  fi
done

if [ -z "$card_index" ]; then
  emit_report 'DEVICE_FOUND=no
REASON=matching_usb_audio_identity_not_found'
  exit 3
fi

listing=$($arecord_bin -l 2>&1 || true)
device_index=$(printf '%s\n' "$listing" | sed -n "s/^card[[:space:]]*$card_index:.*device[[:space:]]*\([0-9][0-9]*\):.*/\1/p" | head -n 1)
device_index=${device_index:-0}
pcm_device="hw:$card_index,$device_index"

hw_params=$($arecord_bin -D "$pcm_device" --dump-hw-params -t raw -d 1 /dev/null 2>&1 || true)
formats=$(printf '%s\n' "$hw_params" | sed -n 's/^[[:space:]]*FORMAT:[[:space:]]*//p' | head -n 1)
channels=$(printf '%s\n' "$hw_params" | sed -n 's/^[[:space:]]*CHANNELS:[[:space:]]*//p' | head -n 1)
rates=$(printf '%s\n' "$hw_params" | sed -n 's/^[[:space:]]*RATE:[[:space:]]*//p' | head -n 1)

if printf '%s\n' "$formats" | grep -Eq '(^|[[:space:]])S16_LE($|[[:space:]])'; then
  selected_format=S16_LE
elif printf '%s\n' "$formats" | grep -Eq '(^|[[:space:]])S32_LE($|[[:space:]])'; then
  selected_format=S32_LE
else
  emit_report "DEVICE_FOUND=yes
CARD_INDEX=$card_index
CARD_ID=$card_id
USB_ID=$usb_id
PCM_DEVICE=$pcm_device
FORMATS=$formats
CHANNELS=$channels
RATES=$rates
SUPPORTED_CAPTURE=no
REASON=no_supported_pcm_format"
  exit 4
fi

channel_numbers=$(printf '%s\n' "$channels" | grep -oE '[0-9]+' || true)
channel_min=$(printf '%s\n' "$channel_numbers" | head -n 1)
channel_max=$(printf '%s\n' "$channel_numbers" | tail -n 1)
if [ -z "$channel_min" ] || [ -z "$channel_max" ]; then
  selected_channels=
elif [ "$channel_min" -le 6 ] && [ "$channel_max" -ge 6 ]; then
  selected_channels=6
elif [ "$channel_min" -le 2 ] && [ "$channel_max" -ge 2 ]; then
  selected_channels=2
else
  selected_channels=
fi

rate_numbers=$(printf '%s\n' "$rates" | grep -oE '[0-9]+' || true)
rate_min=$(printf '%s\n' "$rate_numbers" | head -n 1)
rate_max=$(printf '%s\n' "$rate_numbers" | tail -n 1)
selected_rate=
if [ -n "$rate_min" ] && [ -n "$rate_max" ]; then
  if [ "$rate_min" -le 48000 ] && [ "$rate_max" -ge 48000 ]; then
    selected_rate=48000
  elif [ "$rate_min" -le 16000 ] && [ "$rate_max" -ge 16000 ]; then
    selected_rate=16000
  elif [ "$rate_min" -eq "$rate_max" ]; then
    selected_rate=$rate_min
  fi
fi

if [ -z "$selected_channels" ] || [ -z "$selected_rate" ]; then
  supported=no
  reason=unsupported_channel_or_rate_range
  status=5
else
  supported=yes
  reason=ready
  status=0
fi

emit_report "DEVICE_FOUND=yes
CARD_INDEX=$card_index
CARD_ID=$card_id
USB_ID=$usb_id
PCM_DEVICE=$pcm_device
FORMATS=$formats
CHANNELS=$channels
RATES=$rates
SELECTED_FORMAT=$selected_format
SELECTED_CHANNELS=$selected_channels
SELECTED_RATE=$selected_rate
SUPPORTED_CAPTURE=$supported
REASON=$reason"
exit "$status"
