#!/bin/sh
set -eu

bundle_dir=
output_dir=
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bundle-dir) bundle_dir=$2; shift 2 ;;
    --output-dir) output_dir=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

fail() { echo "LOGMEL_PARITY_STATUS=fail"; echo "REASON=$1"; exit "${2:-1}"; }
case "$bundle_dir:$output_dir" in /*:/*) ;; *) fail path_must_be_absolute 64 ;; esac
[ "$(uname -m)" = armv7l ] || fail wrong_architecture
[ "$(uname -r)" = 4.14.0-xilinx ] || fail wrong_kernel_release
[ ! -e "$output_dir" ] || fail output_already_exists
for path in runtime/logmel_cli config/logmel_frontend_v1.json manifest/cases.tsv manifest/contract.tsv manifest/SHA256SUMS; do
  [ -f "$bundle_dir/$path" ] || fail missing_required_file
done
(awk '
  BEGIN {
    count = split("silence impulse sine_440 dual_tone noise_seeded clipping stereo_antiphase xvf3800-start-0000000000 xvf3800-start-0000072000 xvf3800-start-0000144000", cases, " ")
    expected[++n] = "runtime/logmel_cli"
    expected[++n] = "config/logmel_frontend_v1.json"
    for (i = 1; i <= count; i++) {
      expected[++n] = "inputs/" cases[i] ".wav"
      expected[++n] = "golden/" cases[i] ".f32le"
      expected[++n] = "golden/" cases[i] ".i8"
    }
    expected[++n] = "probe/run_logmel_parity.sh"
    expected[++n] = "manifest/cases.tsv"
    expected[++n] = "manifest/contract.tsv"
  }
  NF != 2 || length($1) != 64 || $1 ~ /[^0-9a-f]/ || $2 != expected[NR] { exit 1 }
  END { if (NR != n) exit 1 }
' "$bundle_dir/manifest/SHA256SUMS") || fail unsafe_checksum_manifest
(cd "$bundle_dir" && sha256sum -c manifest/SHA256SUMS >/dev/null) || fail bundle_hash_mismatch
actual_files=$(cd "$bundle_dir" && find . -type f | sed 's#^\./##' | sort)
expected_files=$({ awk '{print $2}' "$bundle_dir/manifest/SHA256SUMS"; printf '%s\n' manifest/SHA256SUMS; } | sort)
[ "$actual_files" = "$expected_files" ] || fail unlisted_bundle_file
contract_value() {
  awk -F '\t' -v key="$1" '$1==key { if (++count > 1) exit 2; value=$2 }
    END { if (count != 1) exit 1; print value }' "$bundle_dir/manifest/contract.tsv"
}
[ "$(contract_value SCHEMA)" = 2 ] || fail contract_invalid
[ "$(contract_value NORMALIZATION_PROFILE)" = bounded-log-power-64-v1 ] || fail contract_invalid
[ "$(contract_value INPUT_SCALE)" = 128 ] || fail contract_invalid
[ "$(contract_value CONFIG_SHA256)" = "$(sha256sum "$bundle_dir/config/logmel_frontend_v1.json" | awk '{print $1}')" ] || fail contract_invalid
(awk -F '\t' '
  BEGIN {
    count = split("silence impulse sine_440 dual_tone noise_seeded clipping stereo_antiphase xvf3800-start-0000000000 xvf3800-start-0000072000 xvf3800-start-0000144000", cases, " ")
  }
  {
    i = NR
    if (i > count || NF != 6 || $1 != cases[i] ||
        $2 != "inputs/" cases[i] ".wav" ||
        $3 != "golden/" cases[i] ".f32le" ||
        $4 != "golden/" cases[i] ".i8" ||
        $5 != 0 || $6 != "128") exit 1
  }
  END { if (NR != count) exit 1 }
' "$bundle_dir/manifest/cases.tsv") || fail case_manifest_mismatch
t1="$bundle_dir/golden/xvf3800-start-0000000000.i8"
t2="$bundle_dir/golden/xvf3800-start-0000072000.i8"
t3="$bundle_dir/golden/xvf3800-start-0000144000.i8"
tensor_hashes=
for tensor in "$t1" "$t2" "$t3"; do
  [ "$(wc -c < "$tensor" | tr -d ' ')" = 6144 ] || fail tensor_size_mismatch
  stats=$(od -An -td1 -v "$tensor" | awk '
    { for(i=1;i<=NF;i++){ seen[$i]=1; if($i==-128) low++; if($i==127) high++; count++ } }
    END { for(value in seen) unique++; print unique, low+0, high+0, count+0 }') || fail tensor_stats_failed
  set -- $stats
  [ "$4" = 6144 ] && [ "$1" -ge 32 ] || fail tensor_quality_failed
  [ $(( $2 * 100 )) -lt "$4" ] && [ $(( $3 * 100 )) -lt "$4" ] || fail tensor_quality_failed
  hash=$(sha256sum "$tensor" | awk '{print $1}')
  case "$tensor_hashes" in *" $hash "*) fail tensor_hash_not_distinct ;; esac
  tensor_hashes="$tensor_hashes $hash "
done
pair_quality() {
  differing=$(cmp -l "$1" "$2" 2>/dev/null | wc -l | tr -d ' ' || true)
  [ "$differing" -ge 3072 ] || fail tensor_pairwise_difference_low
}
pair_quality "$t1" "$t2"
pair_quality "$t1" "$t3"
pair_quality "$t2" "$t3"
mkdir -p "$output_dir"
printf 'CASE\tMAX_ABS\tRMSE\tINT8_EXACT\tFIRST_MAX_ERROR_INDEX\tFIRST_INT8_MISMATCH_INDEX\n' > "$output_dir/case-summary.tsv"
report_value() {
  awk -F= -v key="$2" '$1==key { if (++count > 1) exit 2; value=$2 }
    END { if (count == 0) print "NA"; else print value }' "$1"
}
count=0
max_abs=0
max_rmse=0
exact_cases=0
while IFS='	' read -r name wav golden_float golden_int8 start scale; do
  [ -n "$name" ] || continue
  case "$name:$wav:$golden_float:$golden_int8" in *..*|*\\*) fail unsafe_manifest_path ;; esac
  case_dir="$output_dir/$name"
  mkdir "$case_dir"
  "$bundle_dir/runtime/logmel_cli" \
    --config "$bundle_dir/config/logmel_frontend_v1.json" \
    --wav "$bundle_dir/$wav" --start-sample "$start" --input-scale "$scale" \
    --float-output "$case_dir/features.f32le" --int8-output "$case_dir/tensor.i8" \
    --report "$case_dir/report.txt" \
    --golden-float "$bundle_dir/$golden_float" --golden-int8 "$bundle_dir/$golden_int8" \
    || fail "${name}_frontend_failed"
  grep -q '^LOGMEL_STATUS=pass$' "$case_dir/report.txt" || fail "${name}_report_failed"
  grep -q '^INT8_EXACT=yes$' "$case_dir/report.txt" || fail "${name}_int8_mismatch"
  [ "$(wc -c < "$case_dir/features.f32le")" -eq 24576 ] || fail "${name}_float_size"
  [ "$(wc -c < "$case_dir/tensor.i8")" -eq 6144 ] || fail "${name}_int8_size"
  sha256sum "$case_dir/features.f32le" "$case_dir/tensor.i8" > "$case_dir/SHA256SUMS"
  case_max_abs=$(report_value "$case_dir/report.txt" MAX_ABS) || fail "${name}_report_invalid"
  case_rmse=$(report_value "$case_dir/report.txt" RMSE) || fail "${name}_report_invalid"
  case_exact=$(report_value "$case_dir/report.txt" INT8_EXACT) || fail "${name}_report_invalid"
  first_max=$(report_value "$case_dir/report.txt" FIRST_MAX_ERROR_INDEX) || fail "${name}_report_invalid"
  first_int8=$(report_value "$case_dir/report.txt" FIRST_INT8_MISMATCH_INDEX) || fail "${name}_report_invalid"
  [ "$case_max_abs" != NA ] && [ "$case_rmse" != NA ] && [ "$case_exact" = yes ] || fail "${name}_report_invalid"
  max_abs=$(awk -v a="$max_abs" -v b="$case_max_abs" 'BEGIN { print (a+0 >= b+0) ? a : b }')
  max_rmse=$(awk -v a="$max_rmse" -v b="$case_rmse" 'BEGIN { print (a+0 >= b+0) ? a : b }')
  exact_cases=$((exact_cases + 1))
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$case_max_abs" "$case_rmse" \
    "$case_exact" "$first_max" "$first_int8" >> "$output_dir/case-summary.tsv"
  count=$((count + 1))
done < "$bundle_dir/manifest/cases.tsv"
[ "$count" -eq 10 ] || fail case_manifest_mismatch
cat > "$output_dir/board-report.txt" <<EOF
LOGMEL_PARITY_STATUS=pass
CASE_COUNT=$count
PARITY_MAX_ABS=$max_abs
PARITY_MAX_RMSE=$max_rmse
INT8_EXACT_CASES=$exact_cases
ARCH=armv7l
KERNEL=4.14.0-xilinx
EOF
cat "$output_dir/board-report.txt"
