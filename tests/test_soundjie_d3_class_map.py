import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXPECTED_IDS = [
    "fire_smoke_alarm",
    "emergency_siren",
    "car_horn",
    "glass_breaking",
    "door_knock_bell",
    "distress_voice",
    "normal_speech",
    "background_unknown",
]


def load_curator():
    sys.path.insert(0, str(REPO))
    try:
        return importlib.import_module("scripts.host.curate_soundjie_d3")
    finally:
        sys.path.pop(0)


class SoundjieD3ClassMapTests(unittest.TestCase):
    def setUp(self):
        self.curator = load_curator()
        self.class_path = REPO / "config" / "soundjie_d3_classes.json"
        self.labels_path = REPO / "models" / "soundjie_d3" / "labels.txt"

    def test_pins_exact_eight_class_order(self):
        class_map = self.curator.load_class_map(self.class_path)
        self.assertEqual([item.index for item in class_map.classes], list(range(8)))
        self.assertEqual([item.id for item in class_map.classes], EXPECTED_IDS)
        labels = self.labels_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(labels, EXPECTED_IDS)

    def test_rejects_alias_collision_and_unknown_target(self):
        payload = json.loads(self.class_path.read_text(encoding="utf-8"))
        payload["classes"][1]["source_labels"]["esc50"].append("clock_alarm")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "collision.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(self.curator.ClassMapError, "collision"):
                self.curator.load_class_map(path)

        payload = json.loads(self.class_path.read_text(encoding="utf-8"))
        payload["background_exclusions"].append(
            {"source_id": "esc50", "label": "dog", "target_id": "not_a_class"}
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unknown.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(self.curator.ClassMapError, "unknown target"):
                self.curator.load_class_map(path)

    def test_rejects_multitarget_recording(self):
        source = {
            "source_id": "esc50",
            "relative_path": "audio/example.wav",
            "path": str(Path(r"F:\MLK\DPU_ZYNQ7000\datasets\soundjie_v1\raw\example.wav")),
            "license": "CC-BY-NC-3.0",
            "license_decision": "approved",
            "source_group": "esc50:123",
            "speaker_or_uploader": "123",
            "original_labels": ["siren", "car_horn"],
            "sha256": "a" * 64,
        }
        result = self.curator.catalog_recording(source)
        self.assertEqual(result.reason, "multiple_targets")
        self.assertEqual(result.target_ids, ("emergency_siren", "car_horn"))

    def test_background_excludes_target_synonyms_and_ancestors(self):
        class_map = self.curator.load_class_map(self.class_path)
        forbidden = {
            "alarm",
            "siren",
            "horn",
            "glass",
            "door",
            "knock",
            "bell",
            "crying",
            "scream",
            "speech",
            "voice",
            "human_vocalization",
            "emergency",
        }
        self.assertTrue(forbidden.issubset(set(class_map.background_forbidden_terms)))
        background = next(item for item in class_map.classes if item.id == "background_unknown")
        background_labels = {
            label for labels in background.source_labels.values() for label in labels
        }
        self.assertTrue(background_labels)
        self.assertTrue(background_labels.isdisjoint(forbidden))

    def test_records_alarm_and_distress_proxy_semantics(self):
        class_map = self.curator.load_class_map(self.class_path)
        alarm = next(item for item in class_map.classes if item.id == "fire_smoke_alarm")
        distress = next(item for item in class_map.classes if item.id == "distress_voice")
        self.assertEqual(alarm.semantic_status, "proxy")
        self.assertIn("clock_alarm", alarm.source_labels["esc50"])
        self.assertIn("smoke detector", alarm.limitations.lower())
        self.assertEqual(distress.semantic_status, "proxy")
        self.assertIn("crying_baby", distress.source_labels["esc50"])
        self.assertIn("not scream", distress.limitations.lower())
        self.assertIn("not chinese distress speech", distress.limitations.lower())


if __name__ == "__main__":
    unittest.main()
