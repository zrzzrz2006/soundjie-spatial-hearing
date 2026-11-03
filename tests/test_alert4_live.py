"""Live assessment must prove reception before interpreting model decisions."""
import importlib.util
import contextlib
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/host/assess_alert4_live.py'
RATE = 16000


def row(sequence, end, class_id=0, **changes):
    result = dict(sequence=sequence, window_end_sample=end,
                  capture_file='capture-%d.s16' % sequence if sequence <= 240 else None,
                  timestamp_ms=end / 16 + 10000, class_id=class_id,
                  class_contract='soundjie-alert4bg-v2', classification_rejected=False,
                  usb_ok=True, dpu_ok=True, end_to_end_ms=125,
                  inference_ms=.6, clip_match_ms=3, fine_model_ms=2, classifier_ms=120)
    result.update(changes)
    return result


def archive(members):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode='w:gz') as stream:
        for name, value in members:
            member = tarfile.TarInfo(name)
            if isinstance(value, bytes):
                member.size = len(value)
                stream.addfile(member, io.BytesIO(value))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = value
                stream.addfile(member)
    result.seek(0)
    return result


def payloads(rows, samples):
    return {r['capture_file']: samples[r['window_end_sample'] - RATE:r['window_end_sample']].tobytes()
            for r in rows if r['capture_file']}


class Alert4LiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = None
        if SOURCE.is_file():
            spec = importlib.util.spec_from_file_location('alert4_live_test_module', SOURCE)
            cls.module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.assertIsNotNone(self.module, 'live assessment helper is missing')
        self.samples = np.random.default_rng(42).integers(-5000, 5001, RATE * 4, dtype=np.int16)

    def test_reconstruction_preserves_exact_overlaps_and_sample_coordinates(self):
        rows = [row(1, RATE), row(2, RATE + 4000), row(3, RATE + 8000)]
        result = self.module.reconstruct(rows, payloads(rows, self.samples))
        np.testing.assert_array_equal(result['samples'], self.samples[:24000])
        self.assertTrue(result['observed'].all())
        self.assertEqual(result['report']['gaps'], [])

    def test_conflicting_overlap_or_non_32000_payload_is_rejected(self):
        rows = [row(1, RATE), row(2, RATE + 4000)]
        original = payloads(rows, self.samples)
        for changed in [b'x' + original['capture-2.s16'][1:],
                        original['capture-2.s16'][:-2], original['capture-2.s16'] + b'xx']:
            with self.subTest(length=len(changed)), self.assertRaises(ValueError):
                self.module.reconstruct(rows, dict(original, **{'capture-2.s16': changed}))

    def test_gap_is_missing_audio_not_inserted_silence_evidence(self):
        rows = [row(1, RATE), row(2, 40000)]
        result = self.module.reconstruct(rows, payloads(rows, self.samples))
        self.assertFalse(result['observed'][16000:24000].any())
        self.assertEqual(result['report']['gaps'], [{'start_sample': 16000, 'end_sample': 24000}])
        self.assertEqual(result['report']['missing_samples'], 8000)

    def test_missing_claimed_payload_and_missing_retained_payload_are_rejected(self):
        for capture in ['capture-1.s16', None]:
            with self.subTest(capture=capture), self.assertRaises(ValueError):
                self.module.reconstruct([row(1, RATE, capture_file=capture)], {})

    def test_retention_is_sequence_limit_not_first_240_inference_rows(self):
        rows = [row(240, RATE), row(241, RATE + 4000)]
        result = self.module.reconstruct(rows, {'capture-240.s16': self.samples[:RATE].tobytes()})
        self.assertEqual([r['sequence'] for r in result['windows']], [240])
        self.assertEqual(result['report']['unretained_windows'], 1)

    def test_archive_is_read_without_extracting_and_rejects_ambiguous_or_unsafe_payloads(self):
        timeline = (json.dumps(row(1, RATE)) + '\n').encode()
        good = [('session/inference/timeline.jsonl', timeline),
                ('session/inference/capture-1.s16', self.samples[:RATE].tobytes())]
        loaded = self.module.load_session(archive(good))
        self.assertEqual(len(loaded['timeline']), 1)
        self.assertEqual(len(loaded['payloads']['capture-1.s16']), 32000)
        for extra in [[('../escape', b'bad')], [('/absolute', b'bad')],
                      [('session/inference/capture-1.s16', b'bad')],
                      [('second/inference/timeline.jsonl', timeline)]]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.module.load_session(archive(good + extra))
        with self.assertRaises(ValueError):
            self.module.load_session(archive([good[0], (good[1][0], '/outside')]))

    def test_archive_missing_capture_or_malformed_json_is_rejected(self):
        for data in [(json.dumps(row(1, RATE)) + '\n').encode(), b'{"sequence":1,\n']:
            with self.subTest(data=data[:20]), self.assertRaises(ValueError):
                self.module.assess_loaded(self.module.load_session(archive([
                    ('inference/timeline.jsonl', data)])), {}, np.zeros(54 * RATE, dtype='<i2'))

    def test_normalized_alignment_finds_delay_and_inverted_attenuated_reception(self):
        reference = self.samples[:RATE * 3].astype(np.float64)
        delay = 2345
        received = np.zeros(reference.size + 5000)
        received[delay:delay + reference.size] = -.25 * reference
        result = self.module.align_reference(received, np.ones(received.size, bool), reference)
        self.assertEqual(result['offset_samples'], delay)
        self.assertGreater(result['raw']['absolute_correlation'], .999)
        self.assertLess(result['raw']['correlation'], -.999)
        self.assertTrue(result['gate_pass'])

    def test_absent_reception_does_not_become_recognition_failure_or_pass(self):
        reference = self.samples[:RATE * 3]
        result = self.module.align_reference(np.zeros(4 * RATE), np.ones(4 * RATE, bool), reference)
        self.assertFalse(result['gate_pass'])
        reconstruction = dict(samples=np.zeros(4 * RATE, dtype='<i2'), observed=np.ones(4 * RATE, bool),
                              windows=[row(1, RATE, class_id=0)], report={'capture_gap_states': 0})
        event = dict(event_index=0, start_sample=0, end_sample=3 * RATE, target_index=0)
        report = self.module.assess_events(reconstruction, reference, {'events': [event]}, result)[0]
        self.assertEqual(report['decision_status'], 'not_assessable')
        self.assertIsNone(report['expected_decision_windows'])

    def test_envelope_alignment_still_requires_each_event_waveform_under_clock_drift(self):
        reference = np.zeros(54 * RATE, dtype='<i2')
        received = np.zeros(60 * RATE, dtype='<i2')
        rng = np.random.default_rng(456)
        events, rows = [], []
        for i in range(10):
            start, end = (4 + 5 * i) * RATE, (7 + 5 * i) * RATE
            pcm = rng.integers(-1500 - i * 200, 1501 + i * 200, 3 * RATE, dtype=np.int16)
            reference[start:end] = pcm
            offset = 2 * RATE + i * 80
            received[start + offset:end + offset] = pcm
            events.append(dict(event_index=i, target_index=i % 5, start_sample=start, end_sample=end))
            rows.append(row(i + 1, start + offset + RATE, i % 5))
        observed = np.ones(received.size, bool)
        alignment = self.module.align_reference(received, observed, reference)
        self.assertTrue(alignment['gate_pass'])
        self.assertEqual(alignment['method'], 'envelope')
        reconstruction = dict(samples=received, observed=observed, windows=rows,
                              report={'capture_gap_states': 0})
        reports = self.module.assess_events(reconstruction, reference, {'events': events}, alignment)
        self.assertTrue(all(r['reception']['gate_pass'] for r in reports))
        self.assertTrue(all(r['decision_status'] == 'observed_match' for r in reports))

    def test_decision_timing_uses_full_one_second_window_and_processing_time(self):
        # Reference event 1..4 s is heard at capture 2..5 s. Ends at 2.5 and
        # 5.5 s straddle its boundaries; only windows ending 3 and 5 s qualify.
        reference = self.samples.copy()
        received = np.r_[np.zeros(RATE, dtype='<i2'), reference]
        rows = [row(1, 40000, 0), row(2, 48000, 0), row(3, 80000, 4), row(4, 88000, 0)]
        reconstruction = dict(samples=received, observed=np.ones(received.size, bool),
                              windows=rows, report={'capture_gap_states': 0})
        alignment = dict(offset_samples=RATE, gate_pass=True, method='raw', reasons=[])
        event = dict(event_index=0, start_sample=RATE, end_sample=4 * RATE, target_index=0)
        report = self.module.assess_events(reconstruction, reference, {'events': [event]}, alignment)[0]
        self.assertTrue(report['reception']['gate_pass'])
        self.assertEqual(report['decision_counts'], {'0': 1, '4': 1})
        self.assertEqual(report['boundary_window_count'], 2)
        self.assertEqual(report.get('boundary_decision_counts'), {'0': 2})
        self.assertEqual(report['decision_status'], 'mixed')
        self.assertAlmostEqual(report['first_expected_result_after_aligned_onset_ms'], 1125)
        self.assertAlmostEqual(report['first_expected_overlapping_result_after_aligned_onset_ms'], 625)
        self.assertAlmostEqual(report['first_expected_overlapping_window_start_from_onset_ms'], -500)

    def test_quality_gap_prevents_event_success_even_when_label_matches(self):
        reference = self.samples[:3 * RATE]
        observed = np.ones(reference.size, bool)
        observed[RATE:2 * RATE] = False
        reconstruction = dict(samples=reference, observed=observed, windows=[row(1, RATE)],
                              report={'capture_gap_states': 0})
        alignment = dict(offset_samples=0, gate_pass=True, method='raw', reasons=[])
        event = dict(event_index=0, start_sample=0, end_sample=3 * RATE, target_index=0)
        report = self.module.assess_events(reconstruction, reference, {'events': [event]}, alignment)[0]
        self.assertFalse(report['reception']['gate_pass'])
        self.assertEqual(report['decision_status'], 'not_assessable')
        self.assertIsNone(report['expected_decision_windows'])

    def test_report_latencies_and_background_violations_do_not_use_state_scores(self):
        rows = [row(1, RATE, 4, class_scores=[.1, .1, .1, .1, .6]),
                row(2, 32000, 0, end_to_end_ms=175)]
        clean = dict(class_contract='soundjie-alert4bg-v2', legacy_class_id=4,
                     classification_fresh=True, session_state='running', legacy_visible=True,
                     classification_rejected=False, main_score_visible=False,
                     main_display_confidence=None, legacy_class_scores=[], alert_active=False,
                     usb_ok=True, dpu_ok=True, paint_ms=55, framebuffer_ms=38,
                     monotonic_ms=11200, framebuffer_ok=True)
        dirty = dict(clean, main_score_visible=True, main_display_confidence=.6,
                     legacy_class_scores=[.1, .1, .1, .1, .6], alert_active=True)
        report = self.module.summarize_reports(rows, [clean, dirty], [])
        self.assertEqual(report['latency_ms']['receive_to_result']['p50'], 150)
        self.assertEqual(report['ui']['background_score_violation_count'], 1)
        self.assertEqual(report['ui']['background_alert_violation_count'], 1)
        self.assertEqual(report['ui']['exact_frame_age_ms']['count'], 0)
        self.assertTrue(report['health']['all_inference_windows_usb_dpu_ok'])

    def test_full_54_second_ten_event_cli_reports_receipt_and_never_overwrites(self):
        reference = np.zeros(54 * RATE, dtype='<i2')
        rng = np.random.default_rng(123)
        events = []
        for index in range(10):
            start, end = (4 + 5 * index) * RATE, (7 + 5 * index) * RATE
            reference[start:end] = rng.integers(-4000, 4001, 3 * RATE, dtype=np.int16)
            events.append(dict(event_index=index, target_index=index % 5,
                               start_sample=start, end_sample=end))
        offset = 36000
        received = np.zeros(61 * RATE, dtype='<i2')
        received[offset:offset + len(reference)] = reference
        rows = []
        for sequence in range(1, 242):
            end = RATE + (sequence - 1) * 4000
            target = next((e['target_index'] for e in events
                           if end - RATE >= e['start_sample'] + offset
                           and end <= e['end_sample'] + offset), 4)
            rows.append(row(sequence, end, target))
        with tempfile.TemporaryDirectory(prefix='alert4-live-test-') as folder:
            root = Path(folder)
            with wave.open(str(root / 'playback.wav'), 'wb') as writer:
                writer.setnchannels(1); writer.setsampwidth(2); writer.setframerate(RATE)
                writer.writeframes(reference.tobytes())
            schedule = dict(contract='soundjie-alert4bg-v2', rate=RATE,
                            independent_accuracy=False, total_samples=len(reference), events=events,
                            playback_sha256=hashlib.sha256((root / 'playback.wav').read_bytes()).hexdigest())
            (root / 'schedule.json').write_text(json.dumps(schedule))
            members = [('session/inference/timeline.jsonl',
                        ('\n'.join(json.dumps(r) for r in rows) + '\n').encode())]
            members += [('session/inference/' + name, data) for name, data in payloads(rows, received).items()]
            session = root / 'session.tar.gz'
            session.write_bytes(archive(members).getvalue())
            output = root / 'assessment.json'
            args = ['--session', str(session), '--playlist', str(root), '--output', str(output)]
            self.assertEqual(self.module.main(args), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report['alignment']['offset_samples'], offset)
            self.assertEqual(report['summary']['assessable_events'], 10)
            self.assertEqual(report['summary']['events_with_expected_decision'], 10)
            self.assertEqual(report['reconstruction']['unretained_windows'], 1)
            self.assertFalse(report['independent_accuracy'])
            self.assertEqual(report['missing_reports'], ['logs/ui.jsonl', 'state/doa.jsonl'])
            self.assertTrue(all(e['first_expected_result_after_aligned_onset_ms'] >= 1000 for e in report['events']))
            before = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.module.main(args), 2)
            self.assertEqual(output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
