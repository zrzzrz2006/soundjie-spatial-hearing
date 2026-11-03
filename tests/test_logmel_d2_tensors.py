import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "host" / "audit_logmel_d2_tensors.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("audit_logmel_d2_tensors", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def qualified_tensors():
    index = np.arange(6144, dtype=np.int32)
    return [
        ((index % 121) - 60).astype(np.int8),
        (((index + 1) % 121) - 60).astype(np.int8),
        (((index * 37 + 11) % 121) - 60).astype(np.int8),
    ]


class LogMelD2TensorAuditTests(unittest.TestCase):
    def setUp(self):
        self.audit = load_auditor()

    def cases(self, tensors):
        return [{"name": f"real-{index}", "tensor": tensor}
                for index, tensor in enumerate(tensors)]

    def test_rejects_collapsed_and_31_unique_tensors(self):
        for tensor in (
                np.full(6144, -128, dtype=np.int8),
                np.zeros(6144, dtype=np.int8),
                (np.arange(6144) % 31).astype(np.int8)):
            tensors = qualified_tensors()
            tensors[0] = tensor
            with self.subTest(unique=len(np.unique(tensor))), self.assertRaises(ValueError):
                self.audit.audit_real_windows(self.cases(tensors))

    def test_rejects_saturation_equal_to_or_above_limit(self):
        for saturated, limit in ((61, 61 / 6144), (62, 0.01)):
            tensors = qualified_tensors()
            tensors[0][:saturated] = -128
            with self.subTest(saturated=saturated), self.assertRaises(ValueError):
                self.audit.audit_real_windows(
                    self.cases(tensors), maximum_saturation=limit)

    def test_pairwise_threshold_rejects_49_9_percent_and_accepts_50_percent(self):
        tensors = qualified_tensors()
        tensors[1] = tensors[0].copy()
        changed = int(6144 * 0.499)
        tensors[1][:changed] += 1
        with self.assertRaises(ValueError):
            self.audit.audit_real_windows(self.cases(tensors))

        tensors[1] = tensors[0].copy()
        tensors[1][:3072] += 1
        result = self.audit.audit_real_windows(self.cases(tensors))
        pair = next(item for item in result["pairs"]
                    if item["left"] == "real-0" and item["right"] == "real-1")
        self.assertEqual(pair["differing_bytes"], 3072)
        self.assertEqual(pair["difference_ratio"], 0.5)

    def test_rejects_three_identical_hashes(self):
        tensor = qualified_tensors()[0]
        with self.assertRaises(ValueError):
            self.audit.audit_real_windows(self.cases([tensor.copy()] * 3))

    def test_accepts_three_qualified_tensors_and_reports_full_statistics(self):
        tensors = qualified_tensors()
        tensors[0] = tensors[0].reshape(64, 96)
        result = self.audit.audit_real_windows(self.cases(tensors))
        self.assertEqual(len(result["cases"]), 3)
        self.assertEqual(len(result["pairs"]), 3)
        self.assertEqual(len({item["sha256"] for item in result["cases"]}), 3)
        for item in result["cases"]:
            self.assertGreaterEqual(item["unique_values"], 32)
            self.assertLess(item["saturated_low_rate"], 0.01)
            self.assertLess(item["saturated_high_rate"], 0.01)
            self.assertEqual(item["negative_count"] + item["zero_count"] +
                             item["positive_count"], 6144)

    def test_rejects_old_identity_manifest(self):
        old = {"schema": 1, "profile_id": "identity-64-v1", "cases": []}
        with self.assertRaises(ValueError):
            self.audit.audit_manifest_contract(old)


if __name__ == "__main__":
    unittest.main()
