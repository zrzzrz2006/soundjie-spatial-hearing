"""Replay safety, provenance, sample alignment and file round-trip tests."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import wave

import numpy as np


def fixtures():
    rows, sources, probs = [], [], []
    for target, kind in [(0, 'siren'), (1, 'horn'), (2, 'doorbell'),
                         (3, 'bark'), (4, 'ordinary_speech'), (4, 'glass')]:
        for n in range(2 if target < 4 else 1):
            name = '%s-%s' % (kind, n)
            row = dict(recording_id=name, purpose='validation', target_index=target,
                       parent_sha256=name, pcm_sha256_16k='pcm-' + name,
                       sampling_identity=name, source_group=name, uploader_group=name,
                       start_sample_16k=0, derivation=None)
            src = dict(row, sha256=name, negative_subtype=kind,
                       class_contract='soundjie-alert4bg-v2', license='CC0-1.0',
                       original_url='https://example.org/' + name)
            rows.append(row); sources.append(src)
            p = np.full(5, .0025); p[target] = .99; probs.append(p)
    return rows, sources, np.asarray(probs), [.95, .725, .675, .75]


class Alert4ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.replay = importlib.import_module('scripts.host.soundjie_alert4_replay')
        except ModuleNotFoundError as exc:
            raise AssertionError('The specified replay implementation is missing') from exc

    def workdir(self):
        root = os.environ.get('SOUNDJIE_REPLAY_TEST_ROOT')
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
        # Retain only this test's newly created files for diagnostics; touch no inputs.
        return Path(tempfile.mkdtemp(prefix='alert4-', dir=root))

    def test_negative_validation_start_keeps_window_center_and_zero_padding(self):
        clip, info = self.replay.centered_excerpt(np.arange(1., 11.), -2, 12, 8)
        np.testing.assert_array_equal(clip, [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8])
        np.testing.assert_array_equal(clip[2:10], [0, 0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(info['excerpt_start_sample_16k'], -4)
        self.assertEqual(info['left_padding_samples'], 4)
        self.assertEqual(info['validation_offset_samples'], 2)

    def test_right_boundary_is_padded_without_wrapping_to_source_start(self):
        clip, info = self.replay.centered_excerpt(np.arange(1., 11.), 6, 8, 4)
        np.testing.assert_array_equal(clip, [5, 6, 7, 8, 9, 10, 0, 0])
        self.assertEqual(info['right_padding_samples'], 2)
        self.assertEqual(info['excerpt_start_sample_16k'], 4)

    def test_empty_nonfinite_or_nonoverlapping_validation_is_rejected(self):
        for samples, start in [([], 0), ([np.nan], 0), ([np.inf], 0), ([1.], 100)]:
            with self.subTest(samples=samples, start=start), self.assertRaises(ValueError):
                self.replay.centered_excerpt(samples, start)

    def test_uniform_gain_preserves_relative_level_and_avoids_clipping(self):
        pcm, info = self.replay.uniform_gain_pcm(
            [np.array([-2., -1., 0., 1., 2.]), np.array([-.5, .5])])
        self.assertEqual(info['digital_gain'], .25)
        np.testing.assert_array_equal(pcm[0], [-16384, -8192, 0, 8192, 16384])
        np.testing.assert_array_equal(pcm[1], [-4096, 4096])
        self.assertEqual(info['clipped_samples'], 0)

    def test_silence_and_nonfinite_gain_inputs_do_not_make_a_demo(self):
        for clips in [[], [np.zeros(10)], [np.array([np.inf])]]:
            with self.subTest(clips=clips), self.assertRaises(ValueError):
                self.replay.uniform_gain_pcm(clips)

    def test_selection_is_correct_high_confidence_and_source_disjoint(self):
        rows, sources, p, thresholds = fixtures()
        # A higher-scoring alias of the first recording must not replace a
        # genuinely different source. The identity hash itself also catches it.
        rows.append(dict(rows[0], recording_id='siren-alias'))
        sources.append(dict(sources[0], recording_id='siren-alias'))
        p = np.vstack([p, [.999, .00025, .00025, .00025, .00025]])
        selected = self.replay.select_cases(rows, sources, p, thresholds)
        self.assertEqual([r['target_index'] for r in selected], [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        self.assertEqual(len({r['source']['source_group'] for r in selected}), 10)
        self.assertEqual(len({r['source']['sha256'] for r in selected}), 10)
        self.assertIn('siren-1', [r['recording_id'] for r in selected])
        self.assertEqual([r['source']['negative_subtype'] for r in selected if r['target_index'] == 4],
                         ['ordinary_speech', 'glass'])

    def test_selection_refuses_train_overlap_instead_of_claiming_independence(self):
        rows, sources, p, thresholds = fixtures()
        sources.append(dict(sources[0], purpose='train', recording_id='train-alias'))
        with self.assertRaises(ValueError):
            self.replay.select_cases(rows, sources, p, thresholds)

    def test_threshold_rejection_is_not_promoted_to_runner_up(self):
        rows, sources, p, thresholds = fixtures()
        p[0] = [.94, .015, .015, .015, .015]
        with self.assertRaises(ValueError):
            self.replay.select_cases(rows, sources, p, thresholds)

    def test_holdout_and_metadata_class_drift_are_rejected(self):
        for key, value in [('purpose', 'holdout'), ('target_index', 4)]:
            rows, sources, p, thresholds = fixtures()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.replay.select_cases(rows, sources, p, thresholds)

    def make_bundle(self):
        rows, sources, p, thresholds = fixtures()
        selected = self.replay.select_cases(rows, sources, p, thresholds)
        clips = [np.full(48000, i + 1, dtype='<i2') for i in range(10)]
        output = self.workdir() / 'playlist'
        plan = self.replay.write_bundle(output, clips, selected,
                                        {'contract': 'soundjie-alert4bg-v2'})
        return output, clips, plan

    def test_build_has_exact_silence_gaps_references_and_no_playback(self):
        fake = mock.Mock()
        with mock.patch.dict('sys.modules', {'winsound': fake}):
            output, clips, plan = self.make_bundle()
            fake.PlaySound.assert_not_called()
        checked = self.replay.verify_bundle(output)
        self.assertFalse(checked['independent_accuracy'])
        self.assertFalse(checked['board_tested'])
        self.assertEqual(plan['duration_s'], 54.)
        with wave.open(str(output / 'playback.wav'), 'rb') as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 16000))
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')
        self.assertEqual(pcm.size, 864000)
        self.assertFalse(np.any(pcm[:64000]))
        self.assertFalse(np.any(pcm[112000:144000]))
        self.assertFalse(np.any(pcm[832000:]))
        np.testing.assert_array_equal(pcm[64000:112000], clips[0])
        self.assertEqual(plan['events'][4]['expected_display'], '背景')
        self.assertFalse(plan['events'][4]['expected_main_score_visible'])
        self.assertFalse(plan['events'][4]['expected_alert'])

    def test_existing_output_and_existing_file_are_never_overwritten(self):
        output, clips, plan = self.make_bundle()
        before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.replay.write_bundle(output, clips, plan['events'], {})
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()})
        path = self.workdir() / 'already.wav'; path.write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            self.replay.write_wav(path, np.zeros(16000, dtype='<i2'))
        self.assertEqual(path.read_bytes(), b'original')

    def test_reference_or_schedule_tampering_prevents_playback(self):
        for target in ['schedule.json', 'references/01_siren.wav', 'playback.wav']:
            output, _, _ = self.make_bundle()
            path = output / target
            with path.open('ab') as f:
                f.write(b'changed')
            fake = mock.Mock()
            with self.subTest(target=target), mock.patch.dict('sys.modules', {'winsound': fake}):
                with self.assertRaises(ValueError):
                    self.replay.play(output)
                fake.PlaySound.assert_not_called()

    def test_explicit_play_records_unique_runs_without_overwriting_the_bundle(self):
        output, _, plan = self.make_bundle()
        before = (output / 'SHA256SUMS').read_bytes()
        # The Windows audio API is the only mocked side effect. Exercise the
        # real integrity verification and real create-only run-log writes.
        fake = mock.Mock(SND_FILENAME=0x20000, SND_NODEFAULT=2)
        with mock.patch.dict('sys.modules', {'winsound': fake}), \
             mock.patch.object(self.replay.time, 'perf_counter', side_effect=[0, 54, 100, 154]):
            self.replay.play(output)
            self.replay.play(output)
        logs = list(output.glob('playback-run-*.json'))
        self.assertEqual(len(logs), 2)
        for path in logs:
            result = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(result['elapsed_seconds'], 54)
            self.assertFalse(result['physical_reception_confirmed'])
            self.assertEqual(result['playback_sha256'], plan['playback_sha256'])
        self.assertEqual((output / 'SHA256SUMS').read_bytes(), before)
        self.replay.verify_bundle(output)


if __name__ == '__main__':
    unittest.main()
