import json
import subprocess
import unittest
import numpy as np
from tests import test_tv_runtime as old

class TwoClipRuntimeTests(unittest.TestCase):
    setUpClass=classmethod(old.TvRuntimeTests.setUpClass.__func__)
    tearDownClass=classmethod(old.TvRuntimeTests.tearDownClass.__func__)
    def test_six_logits_two_clips_caption_and_traffic_isolation(self):
        pcm2=np.random.default_rng(123).integers(-7000,7000,64000,dtype=np.int16)
        raw=self.work/'dialogue.raw';raw.write_bytes(pcm2.astype('<i2').tobytes())
        ref=self.work/'dialogue.bin'
        subprocess.run(['wsl','-d','Ubuntu','--',old.lin(self.matcher),'--make-reference',old.lin(raw),'--output',old.lin(ref)],check=True,capture_output=True)
        cues=self.work/'dialogue.tsv';cues.write_text('line_b\t1.5\t3.8\tB\t我们俩都听不见。\n',encoding='utf-8')
        bank=self.work/'bank.tsv';bank.write_text('ambulance\tambulance_traffic\tref.bin\tcues.tsv\ndialogue\tdialogue\tdialogue.bin\tdialogue.tsv\n',encoding='utf-8')
        classifier=self.work/'speech-classifier';classifier.write_text('#!/bin/sh\nprintf "4 .9 20 .02 .02 .02 .02 .9 .02\\n"\n',encoding='utf-8',newline='\n')
        x=np.concatenate([self.pcm,np.zeros(32000,dtype=np.int16),pcm2,np.zeros(32000,dtype=np.int16)])
        stereo=self.work/'bank-stereo.raw';stereo.write_bytes(np.repeat(x[:,None],2,axis=1).astype('<i2').tobytes())
        out=self.work/'bank-session'
        r=subprocess.run(['wsl','-d','Ubuntu','--',old.lin(self.binary),'--input',old.lin(stereo),
            '--output-dir',old.lin(out),'--current',old.lin(self.work/'bank-current.json'),
            '--classifier-command',old.lin(classifier),'--class-contract','soundjie-alert5bg-v3',
            '--alert-thresholds','.5,.5,.5,.5,.5','--hop-samples','3200',
            '--tv-bank',old.lin(bank),'--traffic-model',old.lin(self.model),
            '--tv-dialogue-horn-start','4.8','--tv-dialogue-horn-end','6.8'],capture_output=True,timeout=40)
        self.assertEqual(r.returncode,0,r.stderr)
        rows=[json.loads(s) for s in (out/'timeline.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertTrue(any(s.get('tv_clip_id')=='ambulance' and s['fine_valid'] for s in rows))
        self.assertTrue(any(s.get('caption_text')=='我们俩都听不见。' and s['caption_speaker']=='B' for s in rows))
        dialogue=[s for s in rows if s.get('tv_clip_id')=='dialogue']
        self.assertTrue(dialogue)
        self.assertTrue(all(not s['fine_valid'] and s['road_score']==0 for s in dialogue))
        for s in rows[:-1]:
            self.assertEqual(s['class_id'],4)
            self.assertEqual(s['label_zh'],'普通说话声')
            self.assertEqual(s['class_scores'],[.02,.02,.02,.02,.9,.02])
        self.assertEqual(rows[-1]['tv_clip_id'],'')
        self.assertEqual(rows[-1]['caption_text'],'')
        self.assertTrue(all(not s.get('tv_clip_id') and not s.get('caption_text') for s in rows if not s['tv_matched']))
    def test_new_contract_rejects_four_thresholds_and_mock_bypass(self):
        for i,args in enumerate([['--alert-thresholds','.5,.5,.5,.5'],['--alert-thresholds','.5,.5,.5,.5,.5']]):
            r=subprocess.run(['wsl','-d','Ubuntu','--',old.lin(self.binary),'--input','/dev/null',
                '--output-dir',old.lin(self.work/('bad-new-'+str(i))),'--current',old.lin(self.work/('bad-new-'+str(i)+'.json')),
                '--mock-class-id','4','--class-contract','soundjie-alert5bg-v3',*args],capture_output=True)
            self.assertNotEqual(r.returncode,0)

if __name__=='__main__':unittest.main()
