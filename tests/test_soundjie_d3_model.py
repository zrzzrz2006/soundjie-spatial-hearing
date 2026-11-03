import importlib
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def load_training():
    sys.path.insert(0, str(REPO))
    try:
        return importlib.import_module("scripts.host.train_soundjie_d3")
    finally:
        sys.path.pop(0)


class SoundjieD3ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = load_training()
        cls.config = cls.training.load_model_config(REPO / "config" / "soundjie_d3_model.json")

    def test_exact_layer_names_shapes_and_eight_logits(self):
        torch = self.training.torch
        model = self.training.SoundjieCNN(self.config)
        names = [name for name, _ in model.named_modules() if name]
        for index in range(1, 5):
            self.assertIn(f"conv{index}", names)
            self.assertIn(f"bn{index}", names)
            self.assertIn(f"relu{index}", names)
            self.assertIn(f"pool{index}", names)
        self.assertIn("flatten", names)
        self.assertIn("fc8", names)
        output = model(torch.zeros(2, 1, 64, 96))
        self.assertEqual(tuple(output.shape), (2, 8))

    def test_rejects_class_or_frontend_hash_drift(self):
        good = {"class_map_sha256": "a" * 64, "frontend_sha256": "b" * 64}
        self.training.validate_training_contract(good, "a" * 64, "b" * 64)
        for class_hash, frontend_hash in (("c" * 64, "b" * 64), ("a" * 64, "d" * 64)):
            with self.assertRaisesRegex(self.training.TrainingError, "drift"):
                self.training.validate_training_contract(good, class_hash, frontend_hash)

    def test_model_config_pins_current_class_and_frontend_hashes(self):
        class_path = REPO / "config" / "soundjie_d3_classes.json"
        frontend_path = REPO / "config" / "logmel_frontend_v1.json"
        self.assertEqual(self.config.class_map_sha256, hashlib.sha256(class_path.read_bytes()).hexdigest())
        self.assertEqual(self.config.frontend_sha256, hashlib.sha256(frontend_path.read_bytes()).hexdigest())

    def test_checkpoint_contains_no_unapproved_operator(self):
        model = self.training.SoundjieCNN(self.config)
        self.training.audit_model_operators(model)
        allowed = {type(module).__name__ for _, module in model.named_modules()}
        self.assertTrue(allowed.issubset({"SoundjieCNN", "Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", "Flatten", "Linear"}))

    def test_eval_order_matches_labels_file(self):
        labels = (REPO / "models" / "soundjie_d3" / "labels.txt").read_text(encoding="utf-8").splitlines()
        report = self.training.evaluate_logits(
            np.eye(8, dtype=np.float32), np.arange(8, dtype=np.int64), labels
        )
        self.assertEqual(list(report.per_class), labels)
        self.assertEqual(report.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
