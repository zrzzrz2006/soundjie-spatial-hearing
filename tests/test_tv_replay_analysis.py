import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.host.analyze_soundjie_tv_replay import reconstruct, covered_slice, matched_origin


class TvReplayAnalysisTests(unittest.TestCase):
    def test_overlapping_windows_recover_exact_samples(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); x=np.arange(24000,dtype=np.int16)
            (p/'capture-1.s16').write_bytes(x[:16000].astype('<i2').tobytes())
            (p/'capture-2.s16').write_bytes(x[8000:].astype('<i2').tobytes())
            pcm,mask=reconstruct([dict(capture_file='capture-1.s16',window_end_sample=16000),dict(capture_file='capture-2.s16',window_end_sample=24000)],p)
            np.testing.assert_array_equal(covered_slice(pcm,mask,0,24000),x)

    def test_conflicting_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'capture-1.s16').write_bytes(np.zeros(16000,dtype='<i2').tobytes())
            (p/'capture-2.s16').write_bytes(np.ones(16000,dtype='<i2').tobytes())
            with self.assertRaises(ValueError):
                reconstruct([dict(capture_file='capture-1.s16',window_end_sample=16000),dict(capture_file='capture-2.s16',window_end_sample=24000)],p)

    def test_missing_audio_is_not_silently_filled_for_calibration(self):
        pcm=np.zeros(48000,dtype=np.int16); mask=np.ones(48000,dtype=bool);mask[20000]=False
        with self.assertRaises(ValueError):covered_slice(pcm,mask,0,48000)

    def test_capture_path_and_unbounded_sample_index_are_rejected(self):
        for name,end in [('../secret',16000),('capture-1.s16',16000*3600)]:
            with self.assertRaises(ValueError):reconstruct([dict(capture_file=name,window_end_sample=end)],Path('.'))

    def test_inconsistent_reference_times_cannot_define_clip_origin(self):
        rows=[dict(tv_matched=True,window_end_sample=int((20+i)*16000),tv_position_seconds=i) for i in range(1,10)]
        self.assertAlmostEqual(matched_origin(rows),20)
        for i,row in enumerate(rows):row['tv_position_seconds']+=i*.1
        with self.assertRaises(ValueError):matched_origin(rows)


if __name__=='__main__':unittest.main()
