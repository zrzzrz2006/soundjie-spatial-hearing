import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
FRONTEND_PATH = REPO / "config" / "logmel_frontend_v1.json"


def load_modules():
    sys.path.insert(0, str(REPO))
    try:
        prepare = importlib.import_module("scripts.host.prepare_soundjie_d3")
        frontend = importlib.import_module("scripts.host.logmel_reference")
        return prepare, frontend
    finally:
        sys.path.pop(0)


class SoundjieD3FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepare, cls.frontend_module = load_modules()
        cls.frontend = cls.frontend_module.load_frontend_config(FRONTEND_PATH)

    def test_prepared_feature_matches_d2_reference_byte_for_byte(self):
        time = np.arange(16000, dtype=np.float64) / 16000.0
        samples = 0.3 * np.sin(2 * np.pi * 731.0 * time)
        expected = self.frontend_module.compute_logmel(samples, self.frontend).astype("<f4")
        actual = self.prepare.compute_feature(samples, self.frontend)
        self.assertEqual(actual.shape, (1, 64, 96))
        self.assertEqual(actual.dtype, np.dtype("<f4"))
        self.assertEqual(actual[0].tobytes(), expected.tobytes())

    def test_no_demo_path_or_source_group_in_training_features(self):
        split = [
            {"recording_id": "train-a", "path": "F:/data/train.wav", "source_group": "group-train", "speaker_or_uploader": "speaker-train", "sha256": "a" * 64}
        ]
        demo = [
            {"recording_id": "demo-a", "path": "F:/data/demo.wav", "source_group": "group-demo", "speaker_or_uploader": "speaker-demo", "sha256": "b" * 64}
        ]
        self.prepare.validate_demo_isolation(split, demo)
        for field in ("path", "source_group", "speaker_or_uploader", "sha256"):
            leaked = [dict(split[0])]
            leaked[0][field] = demo[0][field]
            with self.assertRaisesRegex(self.prepare.AudioPreparationError, "demo leakage"):
                self.prepare.validate_demo_isolation(leaked, demo)

    def test_no_silent_or_constant_feature_is_accepted(self):
        for samples in (np.zeros(16000), np.ones(16000) * 0.1):
            with self.assertRaisesRegex(self.prepare.AudioPreparationError, "silent|constant"):
                self.prepare.compute_feature(samples, self.frontend)

    def test_manifest_hash_changes_on_frontend_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            changed_path = Path(temporary) / "frontend.json"
            raw = FRONTEND_PATH.read_text(encoding="utf-8")
            changed_path.write_text(raw + "\n", encoding="utf-8")
            changed = self.frontend_module.load_frontend_config(changed_path)
            first = self.prepare.preparation_contract_hash(self.frontend, 20261013, "split-hash")
            second = self.prepare.preparation_contract_hash(changed, 20261013, "split-hash")
            self.assertNotEqual(first, second)

    def test_prepare_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "existing-output"
            existing.mkdir()
            marker = existing / "user-marker.txt"
            marker.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(self.prepare.AudioPreparationError, "already exists"):
                self.prepare.prepare_dataset(
                    Path(temporary) / "missing-splits.jsonl",
                    Path(temporary) / "missing-demo.jsonl",
                    FRONTEND_PATH,
                    REPO / "config" / "soundjie_d3_training.json",
                    existing,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
