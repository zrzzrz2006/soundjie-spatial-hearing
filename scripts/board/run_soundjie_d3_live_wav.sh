#!/bin/sh
set -eu

bundle_dir=
wav_member=
output_dir=
expected_class=
dpu_calls=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --bundle-dir) bundle_dir=$2; shift 2 ;;
    --wav) wav_member=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    --expected-class) expected_class=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

fail() {
  echo "LIVE_WAV_STATUS=fail"
  echo "REASON=$1"
  echo "DPU_RUN_CALLS=$dpu_calls"
  exit "${2:-1}"
}

case "$wav_member" in demo/*) ;; *) fail unsafe_wav_member 64 ;; esac
case "$wav_member" in *\\*|*..*|/*) fail unsafe_wav_member 64 ;; esac
case "$bundle_dir:$output_dir" in /*:/*) ;; *) fail path_must_be_absolute 64 ;; esac
[ ! -e "$output_dir" ] || fail output_already_exists

sums="$bundle_dir/manifest/SHA256SUMS"
playback="$bundle_dir/manifest/demo-playback.tsv"
contract="$bundle_dir/manifest/contract.tsv"
runner="$bundle_dir/runtime/sound_shape_runner"
model="$bundle_dir/model/dpu_soundjie_d3.elf"
logmel="$bundle_dir/runtime/logmel_cli"
frontend="$bundle_dir/config/logmel_frontend_v1.json"
labels="$bundle_dir/model/labels.txt"
for path in "$sums" "$playback" "$contract" "$runner" "$model" "$logmel" "$frontend" "$labels" "$bundle_dir/$wav_member"; do
  [ -f "$path" ] || fail missing_required_file
done

awk 'length($1)!=64{exit 1}{h=$1;gsub(/[0-9a-f]/,"",h);if(h!="")exit 1}
     substr($0,65,2)!="  "{exit 1}{n=substr($0,67);if(n==""||n~/^\//||n~/\\/||n~/(^|\/)\.\.($|\/)/||seen[n]++)exit 1}' "$sums" || fail unsafe_checksum_manifest
actual=$(cd "$bundle_dir" && find . -type f | sed 's#^\./##' | sort)
listed=$({ awk '{print substr($0,67)}' "$sums"; printf '%s\n' manifest/SHA256SUMS; } | sort)
[ "$actual" = "$listed" ] || fail unlisted_bundle_file
(cd "$bundle_dir" && sha256sum -c manifest/SHA256SUMS >/dev/null 2>&1) || fail bundle_hash_mismatch

tab=$(printf '\t')
row=$(awk -F '\t' -v wav="$wav_member" 'NR>1&&$2==wav{if(++n>1)exit 2;print}END{if(n!=1)exit 1}' "$playback") || fail wav_not_allowlisted
IFS="$tab" read -r case_id wav_path tensor_path direct_path EXPECTED_TENSOR_SHA256 DIRECT_OUTPUT_SHA256 target_index target_id source_id license creator source_url license_url modification_notice <<EOF
$row
EOF
[ "$wav_path" = "$wav_member" ] || fail wav_not_allowlisted
case "$tensor_path:$direct_path" in inputs/*:direct/*) ;; *) fail demo_contract_invalid ;; esac
for path in "$bundle_dir/$tensor_path" "$bundle_dir/$direct_path"; do [ -f "$path" ] || fail missing_required_file; done
[ "$(sha256sum "$bundle_dir/$tensor_path" | awk '{print $1}')" = "$EXPECTED_TENSOR_SHA256" ] || fail demo_tensor_hash_mismatch
[ "$(sha256sum "$bundle_dir/$direct_path" | awk '{print $1}')" = "$DIRECT_OUTPUT_SHA256" ] || fail direct_output_hash_mismatch
[ -z "$expected_class" ] || [ "$expected_class" = "$target_id" ] || fail expected_class_invalid 64

value() { awk -F '\t' -v key="$1" '$1==key{if(++n>1)exit 2;v=$2}END{if(n!=1)exit 1;print v}' "$contract"; }
[ "$(value SCHEMA):$(value CLASS_COUNT):$(value MODEL_KERNEL):$(value INPUT_NODE):$(value OUTPUT_NODE)" = "1:8:soundjie_d3:conv1:fc8" ] || fail contract_invalid
[ "$(value INPUT_SIZE):$(value INPUT_SCALE):$(value OUTPUT_SIZE):$(value OUTPUT_SCALE)" = "6144:128:8:0.0625" ] || fail contract_invalid
[ "$(sha256sum "$model" | awk '{print $1}')" = "$(value MODEL_SHA256)" ] || fail model_hash_mismatch
[ "$(sha256sum "$runner" | awk '{print $1}')" = "$(value RUNNER_SHA256)" ] || fail runner_hash_mismatch

mkdir "$output_dir"
field() { awk -F= -v key="$2" '$1==key{if(++n>1)exit 2;v=$2}END{if(n!=1)exit 1;print v}' "$1"; }
"$runner" --kernel soundjie_d3 --input-node conv1 --output-node fc8 --describe-tensor > "$output_dir/tensor-description.txt" 2>&1 || fail describe_failed
[ "$(field "$output_dir/tensor-description.txt" DPU_RUN_CALLS)" = 0 ] || fail describe_ran_dpu
[ "$(field "$output_dir/tensor-description.txt" INPUT_SIZE):$(field "$output_dir/tensor-description.txt" OUTPUT_SIZE)" = "6144:8" ] || fail runtime_shape_mismatch

"$logmel" --config "$frontend" --wav "$bundle_dir/$wav_member" --start-sample 0 --input-scale 128 \
  --float-output "$output_dir/logmel.f32le" --int8-output "$output_dir/tensor.i8" --report "$output_dir/logmel-report.txt" || fail logmel_failed
[ "$(sha256sum "$output_dir/tensor.i8" | awk '{print $1}')" = "$EXPECTED_TENSOR_SHA256" ] || fail frontend_tensor_mismatch

"$runner" --kernel soundjie_d3 --input-node conv1 --output-node fc8 --input-file "$output_dir/tensor.i8" \
  --model-file "$model" --iterations 1 --dump "$output_dir/logits.i8" --report "$output_dir/runner-report.txt" || fail runner_failed
dpu_calls=$((dpu_calls + 1))
[ "$(field "$output_dir/runner-report.txt" DPU_SOUND_STATUS)" = pass ] || fail runner_report_failed
[ "$(field "$output_dir/runner-report.txt" DPU_RUN_CALLS)" = 1 ] || fail runner_call_count_wrong
[ "$(field "$output_dir/runner-report.txt" MODEL_SHA256)" = "$(value MODEL_SHA256)" ] || fail model_hash_mismatch
[ "$(field "$output_dir/runner-report.txt" INPUT_SHA256)" = "$EXPECTED_TENSOR_SHA256" ] || fail input_hash_mismatch
[ "$(sha256sum "$output_dir/logits.i8" | awk '{print $1}')" = "$DIRECT_OUTPUT_SHA256" ] || fail direct_output_mismatch

values=$(field "$output_dir/runner-report.txt" OUTPUT_VALUES) || fail runner_report_failed
[ "$(printf '%s\n' "$values" | awk -F, '{print NF}')" = 8 ] || fail output_value_count_wrong
predicted_index=$(printf '%s\n' "$values" | awk -F, '{best=1;bestv=$1;for(i=2;i<=NF;i++)if($i>bestv){best=i;bestv=$i}print best-1}')
predicted_id=$(sed -n "$((predicted_index + 1))p" "$labels")
match=no; [ "$predicted_id" != "$target_id" ] || match=yes
cat > "$output_dir/session-result.txt" <<EOF
LIVE_WAV_STATUS=pass
CASE_ID=$case_id
EXPECTED_CLASS=$target_id
PREDICTED_CLASS=$predicted_id
PREDICTION_MATCH=$match
DPU_RUN_CALLS=$dpu_calls
DIRECT_OUTPUT_SHA256=$DIRECT_OUTPUT_SHA256
EOF
cat "$output_dir/session-result.txt"
