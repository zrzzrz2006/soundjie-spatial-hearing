import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import time
import numpy as np
import torch
import tests.test_soundjie_d4_runtime as legacy
from scripts.host.train_soundjie_traffic import export_model
ROOT=Path(__file__).resolve().parents[1]
lin=legacy.wsl_path

class TvRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');cls.work=Path(cls.tmp.name);cls.binary=cls.work/'runtime';cls.matcher=cls.work/'matcher'
        def build(sources,out):
            r=subprocess.run(['wsl','-d','Ubuntu','--','g++','-std=c++11','-O2','-Wall','-Wextra','-Werror','-I'+lin(ROOT/'src'),*[lin(s) for s in sources],'-o',lin(out)],capture_output=True)
            if r.returncode:raise AssertionError(r.stderr.decode('utf-8','replace'))
        build(legacy.SOURCES,cls.binary)
        build([ROOT/'src'/s for s in ('fft_radix2.cpp','known_clip_matcher.cpp','known_clip_cli.cpp')],cls.matcher)
        rng=np.random.default_rng(88);cls.pcm=rng.integers(-7000,7000,64000,dtype=np.int16)
        raw=cls.work/'ref.raw';raw.write_bytes(cls.pcm.astype('<i2').tobytes());cls.ref=cls.work/'ref.bin'
        subprocess.run(['wsl','-d','Ubuntu','--',lin(cls.matcher),'--make-reference',lin(raw),'--output',lin(cls.ref)],check=True,capture_output=True)
        cls.cues=cls.work/'cues.tsv';cls.cues.write_text('test\t1.2\t2.5\tA\t测试字幕\n',encoding='utf-8')
        cls.model=cls.work/'traffic.bin';m=torch.nn.Sequential(torch.nn.Linear(96,48),torch.nn.ReLU(),torch.nn.Linear(48,3))
        with torch.no_grad():
            for p in m.parameters():p.zero_()
            m[2].bias.fill_(3)
        export_model(cls.model,np.zeros(96),np.ones(96),m)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_real_pcm_drives_caption_and_terminal_clears_all_extensions(self):
        pcm=self.work/'stereo.raw';x=np.r_[self.pcm,np.zeros(32000,dtype=np.int16)].astype('<i2');pcm.write_bytes(np.repeat(x[:,None],2,axis=1).tobytes())
        out=self.work/'session';current=self.work/'current.json'
        r=subprocess.run(['wsl','-d','Ubuntu','--',lin(self.binary),'--input',lin(pcm),'--output-dir',lin(out),'--current',lin(current),'--mock-class-id','7','--hop-samples','3200','--tv-reference',lin(self.ref),'--tv-cues',lin(self.cues),'--traffic-model',lin(self.model)],capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)
        rows=[json.loads(s) for s in (out/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(any(s.get('caption_text')=='测试字幕' for s in rows))
        self.assertTrue(any(s.get('fine_valid') and s['ambulance_score']>.65 for s in rows))
        self.assertTrue(all(s['class_id']==7 for s in rows[:-1]),'must retain actual legacy DPU class')
        self.assertTrue(all(not s.get('fine_valid') and not s.get('caption_text') for s in rows if s['window_end_sample']>72000))
        self.assertFalse(rows[-1]['tv_matched']);self.assertEqual(rows[-1]['caption_text'],'')
    def test_incomplete_reference_arguments_fail_closed(self):
        r=subprocess.run(['wsl','-d','Ubuntu','--',lin(self.binary),'--input','/dev/null','--output-dir',lin(self.work/'bad'),'--current',lin(self.work/'bad.json'),'--mock-class-id','7','--tv-reference',lin(self.ref)],capture_output=True)
        self.assertNotEqual(r.returncode,0)
    def test_five_class_protocol_and_clip_caption_run_together(self):
        pcm=self.work/'siren-stereo.raw';pcm.write_bytes(np.repeat(self.pcm[:,None],2,axis=1).astype('<i2').tobytes())
        classifier=self.work/'six-classifier';classifier.write_text('#!/bin/sh\nprintf "2 .4 20 .27 .26 .4 .02 .02 .03\\n"\n',encoding='utf-8',newline='\n')
        out=self.work/'siren-session';current=self.work/'siren-current.json'
        r=subprocess.run(['wsl','-d','Ubuntu','--',lin(self.binary),'--input',lin(pcm),'--output-dir',lin(out),'--current',lin(current),
            '--classifier-command',lin(classifier),'--class-contract','soundjie-siren5-v1','--hop-samples','3200',
            '--tv-reference',lin(self.ref),'--tv-cues',lin(self.cues),'--traffic-model',lin(self.model)],capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)
        rows=[json.loads(s) for s in (out/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(any(s.get('caption_text')=='测试字幕' and s['class_id']==0 for s in rows))
        self.assertTrue(all(s['label_zh']=='警笛' and len(s['class_scores'])==5 for s in rows[:-1]))
        self.assertTrue(all(s['class_contract']=='soundjie-siren5-v1' for s in rows))

    def test_live_gap_cannot_reactivate_old_caption_with_tiny_new_chunk(self):
        out=self.work/'gap-session';current=self.work/'gap-current.json'
        cmd=['wsl','-d','Ubuntu','--',lin(self.binary),'--input','-','--output-dir',lin(out),'--current',lin(current),
            '--mock-class-id','7','--tv-reference',lin(self.ref),'--tv-cues',lin(self.cues),'--traffic-model',lin(self.model)]
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        try:
            proc.stdin.write(np.repeat(self.pcm[:32000,None],2,axis=1).astype('<i2').tobytes());proc.stdin.flush()
            deadline=time.monotonic()+4
            while time.monotonic()<deadline:
                if (out/'timeline.jsonl').exists():
                    observed=[json.loads(s) for s in (out/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
                    if any(s['caption_text'] for s in observed):break
                time.sleep(.03)
            self.assertTrue((out/'timeline.jsonl').exists())
            time.sleep(.15)
            observed=[json.loads(s) for s in (out/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
            last_end=observed[-1]['window_end_sample'];target=last_end+11900
            self.assertGreater(target,32000)
            proc.stdin.write(np.repeat(self.pcm[32000:target,None],2,axis=1).astype('<i2').tobytes());proc.stdin.flush()
            time.sleep(.1)
            time.sleep(2.2)
            proc.stdin.write(np.zeros((100,2),dtype='<i2').tobytes());proc.stdin.flush();time.sleep(.25)
            proc.stdin.close();proc.stdin=None
            _,err=proc.communicate(timeout=8);self.assertEqual(proc.returncode,0,err)
        finally:
            if proc.poll() is None:proc.kill();proc.communicate()
        rows=[json.loads(s) for s in (out/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(any(s['caption_text'] for s in rows),'fixture must establish a caption before gap')
        self.assertFalse(any(s['tv_matched'] and s['window_end_sample']>=target+100 for s in rows))
if __name__=='__main__':unittest.main()
