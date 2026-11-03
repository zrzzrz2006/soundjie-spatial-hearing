import importlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def load_training():
    sys.path.insert(0, str(REPO))
    try:
        return importlib.import_module("scripts.host.train_soundjie_d3")
    finally:
        sys.path.pop(0)


class SoundjieD3TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training = load_training()

    def test_32_example_overfit_reaches_95_percent(self):
        features = np.zeros((32, 1, 64, 96), dtype=np.float32)
        labels = np.repeat(np.arange(8, dtype=np.int64), 4)
        for index, label in enumerate(labels):
            features[index, 0, label * 8 : label * 8 + 6, 8:88] = 0.8
        result = self.training.overfit_arrays(features, labels, seed=20261013, epochs=80)
        self.assertGreaterEqual(result.accuracy, 0.95)

    def test_fixed_seed_repeats_logits(self):
        features = np.zeros((16, 1, 64, 96), dtype=np.float32)
        labels = np.repeat(np.arange(8, dtype=np.int64), 2)
        for index, label in enumerate(labels):
            features[index, 0, label * 8 : label * 8 + 4, :] = 0.6
        first = self.training.overfit_arrays(features, labels, seed=19, epochs=4)
        second = self.training.overfit_arrays(features, labels, seed=19, epochs=4)
        self.assertTrue(np.array_equal(first.logits, second.logits))

    def test_validation_never_uses_random_augmentation(self):
        feature = np.linspace(-1, 1, 64 * 96, dtype=np.float32).reshape(1, 64, 96)
        first = self.training.apply_augmentation(feature, "validation", 1)
        second = self.training.apply_augmentation(feature, "validation", 999)
        self.assertTrue(np.array_equal(first, feature))
        self.assertTrue(np.array_equal(first, second))

    def test_reports_all_eight_class_metrics_even_when_zero(self):
        labels = [f"class-{index}" for index in range(8)]
        logits = np.zeros((4, 8), dtype=np.float32)
        report = self.training.evaluate_logits(logits, np.array([0, 1, 2, 3]), labels)
        self.assertEqual(len(report.per_class), 8)
        self.assertTrue(all(set(values) == {"precision", "recall", "f1", "support"} for values in report.per_class.values()))

    def test_checkpoint_is_create_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            path.write_bytes(b"user-data")
            with self.assertRaisesRegex(self.training.TrainingError, "exists"):
                self.training.save_checkpoint_create_only(path, {"state": "fixture"})
            self.assertEqual(path.read_bytes(), b"user-data")

    def test_checkpoint_can_be_created_and_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            self.training.save_checkpoint_create_only(path, {"state": "fixture"})
            loaded = self.training.torch.load(path, weights_only=False)
            self.assertEqual(loaded, {"state": "fixture"})

    def test_prepared_loader_excludes_demo_and_rejects_hash_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            feature = np.zeros((1, 64, 96), dtype=np.float32)
            (root / "features" / "train").mkdir(parents=True)
            (root / "features" / "demo").mkdir(parents=True)
            np.save(root / "features" / "train" / "a.npy", feature)
            np.save(root / "features" / "demo" / "b.npy", feature)
            import hashlib
            digest = hashlib.sha256((root / "features" / "train" / "a.npy").read_bytes()).hexdigest()
            records = [
                {"split": "train", "target_index": 2, "feature_path": "features/train/a.npy", "feature_sha256": digest},
                {"split": "demo", "target_index": 7, "feature_path": "features/demo/b.npy", "feature_sha256": "0" * 64},
            ]
            manifest = root / "prepared-manifest.jsonl"
            manifest.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            features, labels, selected = self.training.load_prepared_arrays(manifest, {"train"})
            self.assertEqual(features.shape, (1, 1, 64, 96))
            self.assertEqual(labels.tolist(), [2])
            self.assertEqual([item["split"] for item in selected], ["train"])
            records[0]["feature_sha256"] = "f" * 64
            manifest.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            with self.assertRaisesRegex(self.training.TrainingError, "hash"):
                self.training.load_prepared_arrays(manifest, {"train"})

    def test_prepared_loader_rejects_target_id_class_map_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "feature.npy"
            np.save(path, np.zeros((1, 64, 96), dtype=np.float32))
            import hashlib
            record = {
                "split": "train", "target_index": 0, "target_id": "wrong-id",
                "feature_path": "feature.npy",
                "feature_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            manifest = root / "prepared-manifest.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(self.training.TrainingError, "class map"):
                self.training.load_prepared_arrays(manifest, {"train"}, [f"class-{index}" for index in range(8)])

    def test_class_weights_use_train_counts_and_cover_all_classes(self):
        targets = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64)
        weights = self.training.compute_class_weights(targets, 8)
        self.assertEqual(weights.shape, (8,))
        self.assertLess(weights[0], weights[1])
        with self.assertRaisesRegex(self.training.TrainingError, "missing"):
            self.training.compute_class_weights(targets[:-1], 8)

    def test_confusion_png_is_real_png_and_create_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "confusion.png"
            self.training.write_confusion_png(path, np.eye(8, dtype=np.int64))
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with self.assertRaisesRegex(self.training.TrainingError, "exists"):
                self.training.write_confusion_png(path, np.eye(8, dtype=np.int64))

    def test_online_augmentation_is_epoch_varying_and_repeatable(self):
        features = np.linspace(-1, 1, 4 * 64 * 96, dtype=np.float32).reshape(4, 1, 64, 96)
        backgrounds = features[2:]
        recipe = {
            "gain_db": [-3.0, 3.0], "time_shift_frames": 8,
            "reverb_decay": [0.05, 0.20], "reverb_taps": 3,
            "background_mix": [0.03, 0.15],
        }
        first = self.training.augment_training_epoch(features, backgrounds, recipe, 17, 2)
        repeated = self.training.augment_training_epoch(features, backgrounds, recipe, 17, 2)
        next_epoch = self.training.augment_training_epoch(features, backgrounds, recipe, 17, 3)
        self.assertTrue(np.array_equal(first, repeated))
        self.assertFalse(np.array_equal(first, next_epoch))

    def test_create_only_publish_preserves_racing_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"

            def race_publish(source, destination):
                Path(destination).write_bytes(b"racing-user-data")
                raise FileExistsError(destination)

            primitive = "rename" if self.training.os.name == "nt" else "link"
            with mock.patch.object(self.training.os, primitive, side_effect=race_publish):
                with self.assertRaisesRegex(self.training.TrainingError, "exists"):
                    self.training.save_checkpoint_create_only(path, {"state": "fixture"})
            self.assertEqual(path.read_bytes(), b"racing-user-data")

    def test_failed_overfit_gate_publishes_no_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            output = root / "output"
            args = Namespace(
                config=REPO / "config" / "soundjie_d3_training.json",
                prepared_root=prepared, samples=32, output_dir=output,
            )
            fake_features = np.zeros((32, 1, 64, 96), dtype=np.float32)
            fake_targets = np.repeat(np.arange(8, dtype=np.int64), 4)
            failed = self.training.OverfitResult(0.5, np.zeros((32, 8), dtype=np.float32))
            with mock.patch.object(self.training, "verify_prepared_contract", return_value={}) as verifier, \
                    mock.patch.object(self.training, "load_prepared_arrays", return_value=(fake_features, fake_targets, [])), \
                    mock.patch.object(self.training, "overfit_arrays", return_value=failed):
                with self.assertRaisesRegex(self.training.TrainingError, "gate"):
                    self.training._run_overfit(args)
            verifier.assert_called_once()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
