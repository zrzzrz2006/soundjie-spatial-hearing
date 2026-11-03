import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from scripts.host.soundjie_alert6_replay import crop_reference, build_waveform, align_capture, verify_schedule

class ReplayTests(unittest.TestCase):
    def test_unreceived_event_is_quarantined_despite_valid_global_sync(self):
        from scripts.host.soundjie_alert6_replay import reception_check
        rng=np.random.default_rng(16)
        reference=rng.normal(0,.05,40000)
        self.assertFalse(reception_check(np.zeros(40000),reference)['passed'])
        self.assertFalse(reception_check(rng.normal(0,.05,40000),reference)['passed'])
        self.assertTrue(reception_check(reference*.2,reference)['passed'])

    def test_native_pcm16_only(self):
        import io
        import soundfile as sf
        from scripts.host.soundjie_alert6_replay import validate_native_capture
        for subtype,accepted in [('PCM_16',True),('PCM_24',False),('FLOAT',False)]:
            buffer=io.BytesIO()
            sf.write(buffer,np.zeros((16000,2)),16000,format='WAV',subtype=subtype)
            buffer.seek(0)
            if accepted:validate_native_capture(buffer,16000)
            else:
                with self.assertRaises(ValueError):validate_native_capture(buffer,16000)

    def test_inspect_quarantines_events_when_only_markers_arrive(self):
        import json,tempfile
        import soundfile as sf
        from scripts.host.soundjie_alert6_replay import inspect,write_wav,digest_file,ROOT
        rng=np.random.default_rng(101);reference=rng.normal(0,.05,40000)
        refs=[dict(samples=reference,recording_id=str(i),purpose='train',source_group=str(i),target_index=0)
              for i in range(22)]
        waveform,plan=build_waveform(refs)
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as name:
            root=Path(name);ref=root/'reference.wav';write_wav(ref,reference)
            for event in plan['events']:
                waveform[event['start_sample']:event['end_sample']]=0
                event.update(reference_wav=str(ref),reference_sha256=digest_file(ref))
            (root/'schedule.json').write_text(json.dumps(plan),encoding='utf-8')
            waveform=np.pad(waveform,(0,plan['capture_seconds']*16000-len(waveform)))
            capture=root/'capture.wav';sf.write(capture,np.stack([waveform,waveform],axis=1),16000,subtype='PCM_16')
            inspect(root,capture)
            result=json.loads((root/'alignment-report.json').read_text(encoding='utf-8'))
            self.assertEqual(result['quarantined_events'],22)
            self.assertFalse(any(e['adaptation_allowed'] for e in result['events']))

    def test_play_works_without_python314_sync_constant(self):
        import winsound
        from scripts.host.soundjie_alert6_replay import play
        # Python 3.13 lacks SND_SYNC; default playback must still block.
        with patch.object(Path,'read_text',return_value='{"playback_sha256":"x","duration_s":104}'), \
             patch('scripts.host.soundjie_alert6_replay.digest_file',return_value='x'), \
             patch('scripts.host.soundjie_alert6_replay.time.perf_counter',side_effect=[0,104]), \
             patch('scripts.host.soundjie_alert6_replay.write_json'), \
             patch.object(winsound,'PlaySound') as player:
            play('unused')
        self.assertEqual(player.call_count,1)
        self.assertEqual(player.call_args.args[1] & winsound.SND_ASYNC,0)

    def test_crop_is_fixed_length_and_preserves_event(self):
        a=np.zeros(96000);a[50000:51000]=.5
        x,start=crop_reference(a)
        self.assertEqual(x.shape,(40000,));self.assertGreater(x.max(),.4)
        self.assertTrue(start<=50000<start+40000)

    def test_timing_and_no_marker_as_event(self):
        refs=[dict(samples=np.ones(40000)*.1,recording_id='one',purpose='train',source_group='g',target_index=0)]
        x,plan=build_waveform(refs)
        self.assertEqual(len(plan['events']),1)
        event=plan['events'][0]
        self.assertGreater(event['start_sample'],plan['sync_start_sample']+plan['sync_samples'])
        self.assertEqual(event['end_sample']-event['start_sample'],40000)
        self.assertTrue(np.all(x[event['end_sample']:event['end_sample']+16000]==0))

    def test_cross_purpose_source_rejected(self):
        with self.assertRaises(ValueError):verify_schedule([
          dict(recording_id='a',source_group='same',purpose='train'),
          dict(recording_id='b',source_group='same',purpose='validation')])

    def test_sync_recovers_offset_and_rejects_missing_signal(self):
        from scripts.host.soundjie_alert6_replay import marker
        x=marker();recording=np.pad(x*.3,(3700,5000))+np.random.default_rng(3).normal(0,.0001,len(x)+8700)
        result=align_capture(recording,x,search_samples=len(recording))
        self.assertLessEqual(abs(result['sample']-3700),1)
        self.assertGreater(result['normalized_score'],.99)
        with self.assertRaises(ValueError):align_capture(np.zeros(len(recording)),x,search_samples=len(recording))

if __name__=='__main__':unittest.main()
