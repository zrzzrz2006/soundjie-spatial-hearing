import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_soundjie_d4_runtime as legacy
from tests import test_soundjie_d4_classifier as adapter
wsl_path=legacy.wsl_path


class Siren5RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):legacy.SoundjieD4RuntimeTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls):legacy.SoundjieD4RuntimeTests.tearDownClass.__func__(cls)

    def run_protocol(self,scores,contract='soundjie-siren5-v1'):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pcm=root/'in.raw';pcm.write_bytes(b'\x00\x10'*32000)
            i=max(range(len(scores)),key=scores.__getitem__)
            protocol=' '.join(map(str,[i,scores[i],20]+scores))
            cmd=root/'classifier';cmd.write_text('#!/bin/sh\nprintf "%s\\n" "'+protocol+'"\n',encoding='utf-8',newline='\n')
            args=['wsl','-d','Ubuntu','--',wsl_path(self.binary),'--input',wsl_path(pcm),'--output-dir',wsl_path(root/'out'),
                  '--current',wsl_path(root/'state.json'),'--classifier-command',wsl_path(cmd)]
            if contract:args+=['--class-contract',contract]
            r=subprocess.run(args,capture_output=True)
            path=root/'out/timeline.jsonl'
            rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
            return r,rows

    def test_merge_can_change_winner(self):
        r,rows=self.run_protocol([.27,.26,.4,.02,.02,.03]);self.assertEqual(r.returncode,0,r.stderr)
        s=rows[0];self.assertEqual(s['class_id'],0);self.assertEqual(s['label_zh'],'警笛')
        self.assertAlmostEqual(s['confidence'],.53);self.assertEqual(len(s['class_scores']),5)
        self.assertEqual(s['class_contract'],'soundjie-siren5-v1');self.assertFalse(s['classification_rejected'])

    def test_low_confidence_horn_is_background_without_fabricated_score(self):
        r,rows=self.run_protocol([.1,.1,.6,.05,.05,.1]);self.assertEqual(r.returncode,0,r.stderr)
        s=rows[0];self.assertEqual(s['class_id'],4);self.assertAlmostEqual(s['confidence'],.1)
        self.assertTrue(s['classification_rejected']);self.assertEqual(s['raw_class_id'],1)
        self.assertEqual(s['top3'][0]['id'],1)

    def test_wrong_vector_and_unknown_contract_fail(self):
        for scores,contract in [([.125]*8,'soundjie-siren5-v1'),([1/6]*6,None),([.125]*8,'made-up')]:
            r,_=self.run_protocol(scores,contract);self.assertNotEqual(r.returncode,0)

    def test_legacy_eight_default_preserved(self):
        r,rows=self.run_protocol([.02,.8,.03,.03,.03,.03,.03,.03],None)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(rows[0]['class_id'],1)
        self.assertEqual(len(rows[0]['class_scores']),8)

class SixOutputAdapterTests(unittest.TestCase):
    outputs=6
    setUpClass=classmethod(adapter.D4ClassifierTests.setUpClass.__func__)
    setUp=adapter.D4ClassifierTests.setUp
    run_case=adapter.D4ClassifierTests.run_case
    def test_real_adapter_reads_six_and_refuses_runtime_mismatch(self):
        r=self.run_case();self.assertEqual(r.returncode,0,r.stderr)
        f=r.stdout.split();self.assertEqual(len(f),9)
        self.assertAlmostEqual(sum(map(float,f[3:])),1)
        r=self.run_case(SJ_TEST_FAIL='output_size');self.assertNotEqual(r.returncode,0)
        self.assertEqual(r.stdout,b'')

if __name__=='__main__':unittest.main()
