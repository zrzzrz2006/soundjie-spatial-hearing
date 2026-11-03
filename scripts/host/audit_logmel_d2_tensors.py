#!/usr/bin/env python3
import hashlib
import itertools

import numpy as np


EXPECTED_PROFILE_ID = "bounded-log-power-64-v1"
EXPECTED_PROFILE_DOMAIN = "natural_log_mel_power"
EXPECTED_LOWER_BOUND = -23.025850929940457
EXPECTED_UPPER_BOUND = 5.9914645471079817
EXPECTED_TENSOR_BYTES = 64 * 96


def audit_manifest_contract(manifest):
    expected = {
        "schema": 2,
        "profile_id": EXPECTED_PROFILE_ID,
        "profile_domain": EXPECTED_PROFILE_DOMAIN,
        "profile_lower_bound": EXPECTED_LOWER_BOUND,
        "profile_upper_bound": EXPECTED_UPPER_BOUND,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"manifest {key} does not match D2 contract")
    return True


def _case_statistics(name, tensor):
    values = np.asarray(tensor)
    if values.dtype != np.dtype("int8") or values.shape not in (
            (EXPECTED_TENSOR_BYTES,), (64, 96)):
        raise ValueError(f"{name}: tensor must be [6144] or [64,96] int8 values")
    values = values.reshape(EXPECTED_TENSOR_BYTES)
    raw = values.tobytes(order="C")
    size = values.size
    low = int(np.count_nonzero(values == -128))
    high = int(np.count_nonzero(values == 127))
    return {
        "name": name,
        "size": int(size),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "unique_values": int(np.unique(values).size),
        "zero_count": int(np.count_nonzero(values == 0)),
        "negative_count": int(np.count_nonzero(values < 0)),
        "positive_count": int(np.count_nonzero(values > 0)),
        "saturated_low": low,
        "saturated_high": high,
        "saturated_low_rate": low / size,
        "saturated_high_rate": high / size,
    }


def audit_real_windows(cases, minimum_unique=32, maximum_saturation=0.01,
                       minimum_pairwise_difference=0.50):
    cases = list(cases)
    if len(cases) != 3:
        raise ValueError("D2 audit requires exactly three real windows")
    if not (1 <= minimum_unique <= 256):
        raise ValueError("invalid minimum_unique")
    if not (0.0 < maximum_saturation <= 1.0):
        raise ValueError("invalid maximum_saturation")
    if not (0.0 <= minimum_pairwise_difference <= 1.0):
        raise ValueError("invalid minimum_pairwise_difference")

    names = [item["name"] for item in cases]
    if len(set(names)) != 3 or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("real-window names must be unique non-empty strings")
    tensors = [np.asarray(item["tensor"]).reshape(EXPECTED_TENSOR_BYTES)
               for item in cases]
    statistics = [_case_statistics(name, tensor)
                  for name, tensor in zip(names, tensors)]
    for item in statistics:
        if item["unique_values"] < minimum_unique:
            raise ValueError(f"{item['name']}: too few unique INT8 values")
        if item["saturated_low_rate"] >= maximum_saturation:
            raise ValueError(f"{item['name']}: low saturation is not below limit")
        if item["saturated_high_rate"] >= maximum_saturation:
            raise ValueError(f"{item['name']}: high saturation is not below limit")
    if len({item["sha256"] for item in statistics}) != 3:
        raise ValueError("real-window INT8 hashes must be pairwise distinct")

    pairs = []
    for left, right in itertools.combinations(range(3), 2):
        differing = int(np.count_nonzero(tensors[left] != tensors[right]))
        ratio = differing / EXPECTED_TENSOR_BYTES
        pair = {
            "left": names[left], "right": names[right],
            "differing_bytes": differing, "difference_ratio": ratio,
        }
        pairs.append(pair)
        if ratio < minimum_pairwise_difference:
            raise ValueError(
                f"{names[left]} vs {names[right]}: pairwise difference below limit")
    return {"cases": statistics, "pairs": pairs}
