"""Meaningful native tests of the same bounded C++ core intended for ARM."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def linux(p): return '/mnt/'+str(p)[0].lower()+str(p)[2:].replace('\\','/')

class KnownClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ready=(ROOT/'src/known_clip_cli.cpp').exists()
        if not cls.ready: return
        cls.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts')
        cls.work=Path(cls.tmp.name)
        cls.binary=cls.work/'matcher'
        r=subprocess.run(['wsl.exe','-d','Ubuntu','--','g++','-std=c++11','-O2','-Wall','-Wextra','-Werror',
            '-I'+linux(ROOT/'src'),linux(ROOT/'src/fft_radix2.cpp'),linux(ROOT/'src/known_clip_matcher.cpp'),
            linux(ROOT/'src/known_clip_cli.cpp'),'-o',linux(cls.binary)],capture_output=True)
        if r.returncode: raise AssertionError(r.stderr.decode('utf-8','replace'))
        rng=np.random.default_rng(14)
        # Nonperiodic harmonic sequence, independent of the production algorithm.
        ref=np.zeros(16000*9)
        for k in range(45):
            t=np.arange(3200)/16000
            f=rng.uniform(180,2600)
            ref[k*3200:(k+1)*3200]=.13*np.sin(2*np.pi*f*t)+.08*np.sin(2*np.pi*(f*1.83)*t)
        cls.ref=ref
        cls.raw=cls.work/'ref.raw';cls.write_pcm(cls.raw,ref)
        cls.feat=cls.work/'reference.bin'
        cls.invoke(['--make-reference',linux(cls.raw),'--output',linux(cls.feat)])
        cls.cues=cls.work/'cues.tsv'
        cls.cues.write_text('q1\t2.0\t5.0\tA\t打开“百度地图”\n',encoding='utf-8')

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'tmp'): cls.tmp.cleanup()

    @staticmethod
    def write_pcm(p,x): p.write_bytes(np.clip(np.rint(x*32768),-32768,32767).astype('<i2').tobytes())

    @classmethod
    def invoke(cls,args):
        r=subprocess.run(['wsl.exe','-d','Ubuntu','--',linux(cls.binary),*args],capture_output=True)
        if r.returncode: raise AssertionError(r.stderr.decode('utf-8','replace'))
        return [json.loads(x) for x in r.stdout.decode().splitlines() if x.startswith(b'{'.decode())]

    def setUp(self): self.assertTrue(self.ready,'known clip core is not implemented')

    def match(self,x):
        p=self.work/'query.raw';self.write_pcm(p,x)
        return self.invoke(['--reference',linux(self.feat),'--cues',linux(self.cues),'--pcm',linux(p)])

    def test_known_excerpt_and_gain_find_actual_position(self):
        rows=self.match(self.ref[16000:16000*6]*.4)
        matched=[r for r in rows if r['matched']]
        self.assertGreater(len(matched),5)
        self.assertLess(max(abs(r['position_seconds']-(r['input_seconds']+1)) for r in matched),.10)
        self.assertTrue(any(r['text']=='打开“百度地图”' for r in rows))

    def test_broadband_exact_replay_does_not_lose_frame_alignment(self):
        noise=np.random.default_rng(912).normal(0,.12,16000*5)
        raw=self.work/'broadband.raw';self.write_pcm(raw,noise)
        feat=self.work/'broadband.bin'
        self.invoke(['--make-reference',linux(raw),'--output',linux(feat)])
        cues=self.work/'empty.tsv';cues.write_text('',encoding='utf-8')
        rows=self.invoke(['--reference',linux(feat),'--cues',linux(cues),'--pcm',linux(raw)])
        after=[r for r in rows if r['input_seconds']>=2]
        self.assertTrue(all(r['matched'] and r['score']>.99 for r in after))

    def test_unknown_audio_is_rejected(self):
        self.assertFalse(any(r['matched'] for r in self.match(np.random.default_rng(33).normal(0,.15,16000*4))))

    def test_acoustic_evidence_can_continue_after_strong_audio_lock(self):
        # A frequency-dependent noisy reference segment models acoustic mismatch.
        # Its true alignment remains visible, but individual scores fall below .65.
        # Broadband content has a unique time fingerprint. The separate harmonic
        # fixture below deliberately tests ambiguous candidates instead.
        signal=np.random.default_rng(314).normal(0,.13,16000*9)
        raw=self.work/'continuity-reference.raw';self.write_pcm(raw,signal)
        features=self.work/'continuity-reference.bin'
        self.invoke(['--make-reference',linux(raw),'--output',linux(features)])
        payload=features.read_bytes()
        f=np.frombuffer(payload[16:],'<f4').reshape(-1,32).copy()
        region=f[600:1400]
        level=float(np.sqrt(np.mean((region-region.mean(0))**2)))
        region+=np.random.default_rng(315).normal(0,level*1.4,region.shape).astype('f4')
        reference=self.work/'acoustically-degraded.bin'
        reference.write_bytes(payload[:16]+f.astype('<f4').tobytes())
        rows=self.invoke(['--reference',linux(reference),'--cues',linux(self.cues),'--pcm',linux(raw)])
        weak=[r for r in rows if 4.8<=r['input_seconds']<=6.8]
        self.assertTrue(all(.45<=r['score']<.65 for r in weak), 'fixture must exercise low-score evidence')
        self.assertTrue(all(r['margin']>=.10 for r in weak), 'fixture must have unique alignment evidence')
        self.assertTrue(all(r['matched'] for r in weak), 'continuous matching audio must retain lock')
        self.assertLess(max(abs(r['position_seconds']-r['input_seconds']) for r in weak),.04)

        # The identical weak audio by itself cannot acquire a lock or manufacture
        # evidence from elapsed time. Keep the strong acquisition threshold.
        query=self.work/'weak-without-lock.raw';self.write_pcm(query,signal[16000*4:16000*6])
        alone=self.invoke(['--reference',linux(reference),'--cues',linux(self.cues),'--pcm',linux(query)])
        self.assertFalse(any(r['matched'] for r in alone))

        # After a real lock, unrelated room/noise must still clear the match.
        query=self.work/'unrelated-after-lock.raw'
        self.write_pcm(query,np.r_[signal[:16000*3],np.random.default_rng(451).normal(0,.15,16000*3)])
        changed=self.invoke(['--reference',linux(reference),'--cues',linux(self.cues),'--pcm',linux(query)])
        self.assertFalse(any(r['matched'] for r in changed if r['input_seconds']>=4.6))

    def test_ambiguous_acoustic_windows_break_an_existing_lock(self):
        payload=self.feat.read_bytes()
        f=np.frombuffer(payload[16:],'<f4').reshape(-1,32).copy()
        region=f[600:1400];level=float(np.sqrt(np.mean((region-region.mean(0))**2)))
        region+=np.random.default_rng(315).normal(0,level*1.4,region.shape).astype('f4')
        reference=self.work/'ambiguous-reference.bin';reference.write_bytes(payload[:16]+f.astype('<f4').tobytes())
        rows=self.invoke(['--reference',linux(reference),'--cues',linux(self.cues),'--pcm',linux(self.raw)])
        self.assertTrue(any(r['matched'] for r in rows if r['input_seconds']<=3.8))
        self.assertFalse(any(r['matched'] for r in rows if 4.2<=r['input_seconds']<=6.8),
                         'ambiguous audio must break lock, even if later weak windows align again')

    def test_broadband_replay_can_start_between_reference_frame_boundaries(self):
        noise=np.random.default_rng(712).normal(0,.12,16000*5)
        raw=self.work/'phase-reference.raw';self.write_pcm(raw,noise)
        feat=self.work/'phase-reference.bin'
        self.invoke(['--make-reference',linux(raw),'--output',linux(feat)])
        query=self.work/'phase-query.raw';self.write_pcm(query,noise[101:])
        cues=self.work/'phase-empty.tsv';cues.write_text('',encoding='utf-8')
        rows=self.invoke(['--reference',linux(feat),'--cues',linux(cues),'--pcm',linux(query)])
        after=[r for r in rows if r['input_seconds']>=2]
        self.assertTrue(all(r['matched'] and r['score']>.8 for r in after))

    def test_silence_stops_current_caption(self):
        rows=self.match(np.r_[self.ref[:16000*4],np.zeros(16000*2)])
        self.assertTrue(any(r['text'] for r in rows))
        self.assertTrue(all(not r['matched'] and not r['text'] for r in rows if r['input_seconds']>4.6))

    def test_seek_recovers_without_timer(self):
        rows=self.match(np.r_[self.ref[:16000*3],self.ref[16000*6:]])
        tail=[r for r in rows if r['input_seconds']>5 and r['matched']]
        self.assertTrue(tail)
        self.assertLess(max(abs(r['position_seconds']-(r['input_seconds']+3)) for r in tail),.10)

    def test_corrupt_reference_is_rejected(self):
        bad=self.work/'bad.bin';bad.write_bytes(b'SJTVF1\x00\x00'+b'\xff'*8)
        r=subprocess.run(['wsl.exe','-d','Ubuntu','--',linux(self.binary),'--reference',linux(bad),'--cues',linux(self.cues),'--pcm',linux(self.raw)],capture_output=True)
        self.assertNotEqual(r.returncode,0)

    def test_invalid_utf8_cue_is_rejected_before_audio(self):
        cues=self.work/'invalid-utf8.tsv';cues.write_bytes(b'bad\t1.0\t2.0\tA\t\xc3\n')
        r=subprocess.run(['wsl','-d','Ubuntu','--',linux(self.binary),'--reference',linux(self.feat),'--cues',linux(cues),'--pcm',linux(self.raw)],capture_output=True)
        self.assertNotEqual(r.returncode,0)

if __name__=='__main__': unittest.main()
