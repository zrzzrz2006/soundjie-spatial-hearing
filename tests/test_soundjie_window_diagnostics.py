import unittest
import tempfile
from pathlib import Path
import numpy as np
from scripts.host import diagnose_soundjie_windows as diagnostic


class WindowDiagnosticsTests(unittest.TestCase):
    def test_accepts_real_supervisor_session_root_and_inference_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'inference').mkdir()
            (root/'inference/timeline.jsonl').write_text('',encoding='utf-8')
            self.assertEqual(diagnostic.resolve_board_session(root), (root/'inference').resolve())
            self.assertEqual(diagnostic.resolve_board_session(root/'inference'), (root/'inference').resolve())
            with self.assertRaises(ValueError):diagnostic.resolve_board_session(root/'missing')

    def test_sequential_windows_preserve_transient_tail_and_offsets(self):
        audio = np.arange(20001, dtype=np.float64) / 32768
        windows = diagnostic.sequential_windows(audio, hop=4000)
        self.assertEqual([x[0] for x in windows], [0, 4000, 4001])
        np.testing.assert_array_equal(windows[-1][1], audio[-16000:])
        self.assertEqual(diagnostic.sequential_windows(np.ones(16000))[0][0], 0)

    def test_softmax_is_stable_and_retains_all_classes(self):
        result = diagnostic.score_row(np.array([1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]))
        self.assertEqual(result['predicted_index'], 7)
        self.assertEqual(result['top2_indices'], [7, 6])
        self.assertAlmostEqual(sum(result['scores']), 1.)
        self.assertGreater(result['top2_margin'], 0)
        with self.assertRaises(ValueError):
            diagnostic.score_row(np.zeros(7))

    def test_rejection_summary_does_not_count_abstention_as_correct(self):
        scores = np.array([[.9, .1], [.51, .49], [.1, .9]])
        report = diagnostic.reject_summary(scores, np.array([0, 1, 0]), .7, .2)
        self.assertEqual(report['accepted'], 2)
        self.assertEqual(report['correct_accepted'], 1)
        self.assertAlmostEqual(report['accuracy_among_accepted'], .5)
        self.assertAlmostEqual(report['correct_over_all'], 1/3)


if __name__ == '__main__':
    unittest.main()
