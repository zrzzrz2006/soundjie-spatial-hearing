import importlib
import sys
import unittest
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CLASS_IDS = (
    "fire_smoke_alarm",
    "emergency_siren",
    "car_horn",
    "glass_breaking",
    "door_knock_bell",
    "distress_voice",
    "normal_speech",
    "background_unknown",
)


def load_curator():
    sys.path.insert(0, str(REPO))
    try:
        return importlib.import_module("scripts.host.curate_soundjie_d3")
    finally:
        sys.path.pop(0)


def record(curator, number, class_index=0, group=None, speaker=None, digest=None, license_name="CC-BY-4.0"):
    class_id = CLASS_IDS[class_index]
    return curator.CatalogRecord(
        recording_id=f"fixture:{number}",
        source_id="fixture",
        source_group=group or f"group:{number}",
        speaker_or_uploader=speaker or f"speaker:{number}",
        license=license_name,
        original_labels=(class_id,),
        target_id=class_id,
        target_index=class_index,
        path=str(Path(r"F:\MLK\DPU_ZYNQ7000\datasets\soundjie_v1\fixture") / f"renamed-{number}.wav"),
        sha256=digest or f"{number:064x}"[-64:],
    )


class SoundjieD3SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.curator = load_curator()
        cls.config = cls.curator.load_split_config(REPO / "config" / "soundjie_d3_training.json")

    def test_group_never_crosses_splits_even_with_renamed_files(self):
        records = [record(self.curator, i) for i in range(12)]
        records.extend(
            [
                record(self.curator, 100, group="shared-source", speaker="uploader-a"),
                record(self.curator, 101, group="shared-source", speaker="uploader-a"),
            ]
        )
        manifest = self.curator.assign_splits(records, self.config)
        by_id = {item.recording_id: item.split for item in manifest.records}
        self.assertEqual(by_id["fixture:100"], by_id["fixture:101"])
        self.assertEqual(manifest.cross_split_group_count, 0)

    def test_speaker_never_crosses_speech_splits(self):
        records = [record(self.curator, i, class_index=6) for i in range(12)]
        records.extend(
            [
                record(self.curator, 200, class_index=6, group="chapter-a", speaker="speaker-z"),
                record(self.curator, 201, class_index=6, group="chapter-b", speaker="speaker-z"),
            ]
        )
        manifest = self.curator.assign_splits(records, self.config)
        by_id = {item.recording_id: item.split for item in manifest.records}
        self.assertEqual(by_id["fixture:200"], by_id["fixture:201"])
        self.assertEqual(manifest.cross_split_speaker_count, 0)

    def test_demo_has_two_or_three_per_class_and_no_overlap(self):
        records = []
        number = 1
        for class_index in range(8):
            for _ in range(8):
                records.append(record(self.curator, number, class_index=class_index))
                number += 1
        demo = self.curator.select_demo(records, 2, 3)
        counts = defaultdict(int)
        for item in demo.records:
            counts[item.target_id] += 1
            self.assertIn(item.selection_reason.split(":", 1)[0], {"CC0", "CC-BY", "other_approved"})
        self.assertEqual(set(counts), set(CLASS_IDS))
        self.assertTrue(all(2 <= count <= 3 for count in counts.values()))
        remaining = [item for item in records if item.recording_id not in demo.reserved_recording_ids]
        splits = self.curator.assign_splits(remaining, self.config)
        split_groups = {item.source_group for item in splits.records}
        split_hashes = {item.sha256 for item in splits.records}
        split_speakers = {item.speaker_or_uploader for item in splits.records}
        self.assertTrue(set(demo.reserved_source_groups).isdisjoint(split_groups))
        self.assertTrue(set(demo.reserved_sha256).isdisjoint(split_hashes))
        self.assertTrue(set(demo.reserved_speakers).isdisjoint(split_speakers))

    def test_content_hash_duplicate_is_single_group(self):
        duplicate = "d" * 64
        records = [record(self.curator, i) for i in range(12)]
        records.extend(
            [
                record(self.curator, 300, group="source-a", speaker="user-a", digest=duplicate),
                record(self.curator, 301, group="source-b", speaker="user-b", digest=duplicate),
            ]
        )
        manifest = self.curator.assign_splits(records, self.config)
        by_id = {item.recording_id: item.split for item in manifest.records}
        self.assertEqual(by_id["fixture:300"], by_id["fixture:301"])
        self.assertEqual(manifest.cross_split_hash_count, 0)

    def test_split_is_deterministic_and_preserves_all_classes(self):
        records = []
        number = 1
        for class_index in range(8):
            for _ in range(12):
                records.append(record(self.curator, number, class_index=class_index))
                number += 1
        first = self.curator.assign_splits(records, self.config)
        second = self.curator.assign_splits(list(reversed(records)), self.config)
        first_pairs = [(item.recording_id, item.split) for item in first.records]
        second_pairs = [(item.recording_id, item.split) for item in second.records]
        self.assertEqual(first_pairs, second_pairs)
        by_split = defaultdict(set)
        for item in first.records:
            by_split[item.split].add(item.target_id)
        self.assertEqual(set(by_split), {"train", "validation", "test"})
        self.assertTrue(all(ids == set(CLASS_IDS) for ids in by_split.values()))


if __name__ == "__main__":
    unittest.main()
