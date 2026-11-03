import copy
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class D4DataTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/host/build_soundjie_d4_manifest.py'
        self.assertTrue(path.exists(), 'D4 manifest validator is missing')
        spec = importlib.util.spec_from_file_location('d4_manifest', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def record(self, rid='r1', group='g1', purpose='train', **changes):
        row = dict(recording_id=rid, source_group=group, target_id='car_horn',
                   target_index=2, domain='direct', purpose=purpose,
                   sha256='a' * 64, parent_recording_id=None,
                   parent_sha256=None, license='CC-BY', original_url='https://example.org/audio',
                   path='F:/audio.wav', derivation=None, feature_sha256=None, conditions={})
        row.update(changes)
        return row

    def test_holdout_and_demo_shoot_are_excluded_from_training(self):
        rows = [self.record(), self.record('r2', 'g2', 'd5_holdout', sha256='b'*64),
                self.record('r3', 'g3', 'demo_shoot', sha256='c'*64),
                self.record('r4', 'g4', 'demo_calibration', sha256='d'*64)]
        parts = self.module.partition_records(rows)
        self.assertEqual([r['recording_id'] for r in parts['train']], ['r1'])
        self.assertEqual([r['recording_id'] for r in parts['demo_calibration']], ['r4'])

    def test_source_group_cannot_cross_purpose(self):
        with self.assertRaisesRegex(ValueError, 'source_group'):
            self.module.partition_records([self.record(), self.record('r2', purpose='d5_holdout', sha256='b'*64)])

    def test_augmentation_cannot_relabel_holdout_as_training(self):
        parent = self.record(purpose='d5_holdout')
        child = self.record('r2', purpose='train', domain='simulated_replay',
                            sha256='b'*64, parent_recording_id='r1', parent_sha256='a'*64,
                            derivation={'seed': 1})
        with self.assertRaises(ValueError):
            self.module.partition_records([parent, child])

    def test_parent_hash_and_group_must_match(self):
        child = self.record('r2', 'g2', domain='simulated_replay', sha256='b'*64,
                            parent_recording_id='r1', parent_sha256='f'*64, derivation={'seed': 1})
        with self.assertRaises(ValueError):
            self.module.partition_records([self.record(), child])

    def test_identical_audio_cannot_cross_purpose_under_another_group(self):
        with self.assertRaisesRegex(ValueError, 'audio hash'):
            self.module.partition_records([self.record(), self.record('r2', 'g2', 'validation')])

    def test_old_test_split_is_frozen_as_holdout_without_mutating_input(self):
        row = {'recording_id': 'r1', 'source_group': 'g1', 'target_id': 'car_horn',
               'target_index': 2, 'split': 'test', 'path': 'F:/audio.wav',
               'sha256': 'a'*64, 'license': 'CC-BY', 'source_id': 'esc50'}
        before = copy.deepcopy(row)
        converted = self.module.convert_d3([row], {'esc50': 'https://example.org/esc50'})
        self.assertEqual(converted[0]['purpose'], 'd5_holdout')
        self.assertEqual(row, before)

    def test_unknown_split_and_duplicate_record_fail_closed(self):
        with self.assertRaises(ValueError):
            self.module.convert_d3([{'split': 'other'}], {})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.module.partition_records([self.record(), self.record()])

    def test_separate_demo_manifest_without_split_is_preserved_as_demo_only(self):
        demo = self.record()
        demo['source_id'] = 'esc50'
        converted = self.module.convert_d3([], {'esc50': 'https://example.org/data'}, [demo])
        self.assertEqual(converted[0]['purpose'], 'demo_shoot')
        self.assertEqual(self.module.partition_records(converted)['train'], [])


if __name__ == '__main__':
    unittest.main()
