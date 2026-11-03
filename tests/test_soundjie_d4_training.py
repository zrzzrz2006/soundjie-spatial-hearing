import importlib.util
from pathlib import Path
import unittest
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class D4TrainingTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'scripts/host/augment_soundjie_d4.py'
        self.assertTrue(path.exists(), 'D4 waveform augmentation is missing')
        spec = importlib.util.spec_from_file_location('d4_aug', path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_seeded_augmentation_preserves_input_and_records_recipe(self):
        x = np.sin(np.arange(16000)*0.1)*0.3
        before = x.copy()
        a, ra = self.mod.augment_waveform(x, 42, 'train')
        b, rb = self.mod.augment_waveform(x, 42, 'train')
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(x, before)
        self.assertEqual(ra, rb)
        self.assertEqual(len(ra['recipe_sha256']), 64)
        self.assertEqual(a.shape, (16000,))
        self.assertTrue(np.isfinite(a).all())
        self.assertLessEqual(np.abs(a).max(), 1)
        self.assertFalse(np.array_equal(a, x))

    def test_holdout_demo_validation_cannot_be_augmented_for_training(self):
        for purpose in ('validation', 'demo_shoot', 'd5_holdout', 'demo_calibration'):
            with self.assertRaises(ValueError):
                self.mod.augment_waveform(np.ones(16000)*0.1, 1, purpose)

    def test_shift_does_not_wrap_tail_to_front(self):
        x = np.array([1., 2., 3., 4.])
        np.testing.assert_array_equal(self.mod.zero_shift(x, 2), [0, 0, 1, 2])
        np.testing.assert_array_equal(self.mod.zero_shift(x, -2), [3, 4, 0, 0])

    def test_nonfinite_and_wrong_shape_are_rejected(self):
        for x in (np.zeros(2), np.full(16000, np.nan)):
            with self.assertRaises(ValueError):
                self.mod.augment_waveform(x, 1, 'train')

    def test_balanced_epoch_bounds_background_despite_large_speech_pool(self):
        y = np.concatenate([np.full(2 if i != 6 else 200, i) for i in range(8)])
        indices = self.mod.balanced_indices(y, per_class=16, seed=3)
        np.testing.assert_array_equal(np.bincount(y[indices], minlength=8), np.full(8, 16))
        np.testing.assert_array_equal(indices, self.mod.balanced_indices(y, 16, 3))
        with self.assertRaises(ValueError):
            self.mod.balanced_indices(np.array([0, 1]), 16, 3)

    def test_gate_rejects_collapse_and_missing_class_support(self):
        y = np.arange(8)
        self.assertFalse(self.mod.candidate_gate(y, np.full(8, 7))['passed'])
        self.assertFalse(self.mod.candidate_gate(y[:4], y[:4])['passed'])
        report = self.mod.candidate_gate(y, y)
        self.assertTrue(report['passed'])
        self.assertEqual(report['predicted_class_count'], 8)
        self.assertEqual(report['focus_min_recall'], 1)


class D4PipelineTests(unittest.TestCase):
    def module(self):
        path = ROOT / 'scripts/host/train_soundjie_d4.py'
        self.assertTrue(path.exists(), 'D4 training pipeline is missing')
        spec = importlib.util.spec_from_file_location('d4_pipeline', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_temporal_windows_include_real_tail_and_deduplicate_boundaries(self):
        module = self.module()
        self.assertTrue(hasattr(module, 'temporal_windows'))
        x = np.cos(np.arange(48000)*.13)*.1
        x[:16000] *= 3
        row = dict(purpose='train', target_index=3)
        windows = module.temporal_windows(x, row, 7, [-8000, 8000, 16000], .15)
        self.assertEqual([w.start_sample for w in windows], [0, 8000, 16000])
        np.testing.assert_array_equal(windows[-1].samples, x[16000:32000])

    def test_temporal_windows_preserve_validation_and_filter_silent_tail(self):
        module = self.module()
        self.assertTrue(hasattr(module, 'temporal_windows'))
        x = np.r_[np.sin(np.arange(16000)*.13)*.3, np.zeros(32000)]
        for purpose in ('train', 'validation'):
            windows = module.temporal_windows(x, dict(purpose=purpose, target_index=3), 7, [16000], .15)
            self.assertEqual(len(windows), 1)
        with self.assertRaises(ValueError):
            module.temporal_windows(x, dict(purpose='demo_shoot', target_index=3), 7, [8000], .15)

    def test_selection_excludes_holdout_and_demo_and_caps_per_class(self):
        module = self.module()
        rows = [dict(recording_id=f'{purpose}-{c}-{i}', purpose=purpose, target_index=c)
                for purpose in ('train', 'validation', 'demo_shoot', 'd5_holdout')
                for c in range(8) for i in range(5)]
        selected = module.select_records(rows, 3, 2)
        self.assertEqual(len(selected), 48)
        self.assertEqual({r['purpose'] for r in selected}, {'train', 'validation'})
        self.assertEqual(selected, module.select_records(rows, 3, 2))

    def test_tiny_training_produces_compatible_checkpoint_and_refuses_overwrite(self):
        module = self.module()
        import torch
        rng = np.random.default_rng(6)
        x = rng.uniform(-1, 1, (8, 1, 64, 96)).astype(np.float32)
        y = np.arange(8, dtype=np.int64)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'run'
            cfg = dict(epochs=1, batch_size=8, per_class_epoch=1, learning_rate=.001,
                       weight_decay=.0001, threads=1)
            module.train_candidate(x, y, x, y, cfg, 11, output, {'fixture': True})
            checkpoint = torch.load(output / 'soundjie_d4_float32.pt', weights_only=False)
            self.assertEqual(len(checkpoint['labels']), 8)
            self.assertIn('fc8.weight', checkpoint['state_dict'])
            self.assertTrue((output / 'metrics.json').is_file())
            with self.assertRaises(FileExistsError):
                module.train_candidate(x, y, x, y, cfg, 11, output, {'fixture': True})


if __name__ == '__main__':
    unittest.main()
