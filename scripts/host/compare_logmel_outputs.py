#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np


FLOAT_COUNT = 64 * 96
FLOAT_BYTES = FLOAT_COUNT * 4
INT8_BYTES = FLOAT_COUNT
MAX_ABS_LIMIT = 1e-4
RMSE_LIMIT = 1e-5


def read_exact(path, size, label):
    data = Path(path).read_bytes()
    if len(data) != size:
        raise ValueError(f"{label} must contain exactly {size} bytes")
    return data


def compare(golden_f32, candidate_f32, golden_i8, candidate_i8):
    golden_float = np.frombuffer(read_exact(golden_f32, FLOAT_BYTES, "golden f32"), dtype="<f4")
    candidate_float = np.frombuffer(
        read_exact(candidate_f32, FLOAT_BYTES, "candidate f32"), dtype="<f4")
    if not np.isfinite(golden_float).all() or not np.isfinite(candidate_float).all():
        raise ValueError("float tensors must be finite")
    differences = np.abs(golden_float.astype(np.float64) - candidate_float.astype(np.float64))
    max_index = int(np.argmax(differences))
    max_abs = float(differences[max_index])
    rmse = float(math.sqrt(float(np.mean(differences * differences))))
    golden_bytes = read_exact(golden_i8, INT8_BYTES, "golden int8")
    candidate_bytes = read_exact(candidate_i8, INT8_BYTES, "candidate int8")
    mismatch = next((index for index, pair in enumerate(zip(golden_bytes, candidate_bytes))
                     if pair[0] != pair[1]), -1)
    passed = max_abs <= MAX_ABS_LIMIT and rmse <= RMSE_LIMIT and mismatch == -1
    return {
        "PARITY_STATUS": "pass" if passed else "fail",
        "FLOAT_COUNT": FLOAT_COUNT,
        "MAX_ABS": format(max_abs, ".17g"),
        "RMSE": format(rmse, ".17g"),
        "FIRST_MAX_ERROR_INDEX": max_index,
        "GOLDEN_AT_MAX_ERROR": format(float(golden_float[max_index]), ".9g"),
        "CANDIDATE_AT_MAX_ERROR": format(float(candidate_float[max_index]), ".9g"),
        "INT8_EXACT": "yes" if mismatch == -1 else "no",
        "FIRST_INT8_MISMATCH_INDEX": mismatch,
    }


def write_report(path, result):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to replace report: {path}")
    text = "".join(f"{key}={value}\n" for key, value in result.items())
    path.write_text(text, encoding="ascii", newline="\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-f32", required=True, type=Path)
    parser.add_argument("--candidate-f32", required=True, type=Path)
    parser.add_argument("--golden-i8", required=True, type=Path)
    parser.add_argument("--candidate-i8", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = compare(
            args.golden_f32, args.candidate_f32, args.golden_i8, args.candidate_i8)
        write_report(args.report, result)
        print(f"PARITY_STATUS={result['PARITY_STATUS']}")
        return 0 if result["PARITY_STATUS"] == "pass" else 1
    except Exception as error:
        print(f"PARITY_STATUS=fail\nREASON={error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
