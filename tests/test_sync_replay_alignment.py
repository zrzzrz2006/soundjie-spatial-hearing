import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('sync_experiment', Path(__file__).resolve().parents[1]/'evidence/soundjie-d4/analyze_sync_four_20261021.py')
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class AlignmentTests(unittest.TestCase):
    def test_three_repetitions_with_playback_gaps_and_inverted_gain(self):
        # A contiguous-three-window assumption must not include leading silence.
        fn = getattr(experiment, 'align_repetitions', None)
        self.assertTrue(callable(fn), 'gap-aware alignment missing')
        source = np.random.default_rng(12).normal(size=1600)
        recorded = np.random.default_rng(13).normal(0, .001, 12800)
        for start in [2300, 4700, 7100]:
            recorded[start:start+1600] -= .2*source
        starts, scores = fn(recorded, source)
        self.assertEqual(starts, [2300, 4700, 7100])
        self.assertTrue(all(abs(s)>.99 for s in scores))

    def test_no_signal_rejected(self):
        fn = getattr(experiment, 'align_repetitions', None)
        self.assertTrue(callable(fn), 'gap-aware alignment missing')
        with self.assertRaises(ValueError):
            fn(np.zeros(12800), np.ones(1600))


if __name__ == '__main__':
    unittest.main()
