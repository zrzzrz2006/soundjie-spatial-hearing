import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from tests import test_soundjie_hdmi_ui as legacy


class Siren5UiTests(unittest.TestCase):
    setUpClass=classmethod(legacy.SoundjieHdmiUiTests.setUpClass.__func__)
    tearDownClass=classmethod(legacy.SoundjieHdmiUiTests.tearDownClass.__func__)
    def render_case(self,**changes):
        s=legacy.snapshot();s.update(class_contract='soundjie-siren5-v1',class_id=0,label_zh='警笛',confidence=.7,
            class_scores=[.7,.1,.05,.05,.1],classification_rejected=False,tv_enabled=True,tv_matched=True,fine_valid=True,
            tv_position_seconds=10,tv_score=.9,caption_text='打开百度地图',caption_speaker='A',ambulance_score=.9,road_score=.8,speech_score=.7)
        s.update(changes)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'snapshot.json';f.write_text(json.dumps(s,ensure_ascii=False),encoding='utf-8')
            args=['wsl','-d','Ubuntu','--',legacy.wsl_path(self.binary),'--snapshot',legacy.wsl_path(f),
                  '--atlas',legacy.wsl_path(legacy.ROOT/'assets/hdmi/soundjie_siren5_glyph_atlas.bin'),
                  '--offscreen',legacy.wsl_path(root/'ui.ppm'),'--report',legacy.wsl_path(root/'report.json')]
            r=subprocess.run(args,capture_output=True)
            return r,json.loads((root/'report.json').read_text(encoding='utf-8')) if r.returncode==0 else {}

    def test_five_rows_keep_caption_direction_and_clip(self):
        r,s=self.render_case();self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(s['class_contract'],'soundjie-siren5-v1');self.assertEqual(s['display_class_count'],5)
        self.assertEqual(s['display_label'],'警笛');self.assertTrue(s['caption_visible'])
        self.assertEqual(s['caption_text'],'打开百度地图');self.assertTrue(s['fine_events_visible'])
        self.assertIsNotNone(s['arrow_tip'])

    def test_rejection_is_visibly_uncertain(self):
        r,s=self.render_case(class_id=4,label_zh='背景',confidence=.1,class_scores=[.2,.6,.05,.05,.1],classification_rejected=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(s['display_label'],'声音待确认')

    def test_wrong_count_or_contract_refused(self):
        for changes in [dict(class_scores=[.125]*8),dict(class_contract='unknown'),dict(class_id=7)]:
            r,_=self.render_case(**changes);self.assertNotEqual(r.returncode,0)

    def test_live_stale_keeps_five_rows_but_clears_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'state.json';config=root/'runtime.tsv'
            config.write_text('key\tvalue\ndoa_stale_ms\t200\ndoa_ema_alpha\t0.5\n',encoding='utf-8')
            proc=subprocess.Popen(['wsl','-d','Ubuntu','--',legacy.wsl_path(self.binary),
                '--snapshot',legacy.wsl_path(state),'--atlas',legacy.wsl_path(legacy.ROOT/'assets/hdmi/soundjie_siren5_glyph_atlas.bin'),
                '--doa-jsonl',legacy.wsl_path(root/'doa.jsonl'),'--runtime-config',legacy.wsl_path(config),
                '--frames','10','--interval-ms','100','--snapshot-stale-ms','400','--width','640','--height','360',
                '--offscreen-dir',legacy.wsl_path(root/'frames')],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                first=json.loads(proc.stdout.readline());s=legacy.snapshot()
                s.update(timestamp_ms=first['monotonic_ms'],class_contract='soundjie-siren5-v1',classification_rejected=False,
                         class_id=0,label_zh='警笛',confidence=.7,class_scores=[.7,.1,.05,.05,.1])
                state.write_text(json.dumps(s,ensure_ascii=False),encoding='utf-8')
                out,err=proc.communicate(timeout=20)
            finally:
                if proc.poll() is None:proc.kill();proc.communicate()
            self.assertEqual(proc.returncode,0,err)
            rows=[json.loads(line) for line in out.splitlines()]
            self.assertTrue(any(r['classification_fresh'] and r['display_class_count']==5 for r in rows))
            self.assertFalse(rows[-1]['classification_fresh']);self.assertFalse(rows[-1]['legacy_visible'])
            self.assertEqual(rows[-1]['display_class_count'],5)
            self.assertEqual(rows[-1]['legacy_class_scores'],[])

if __name__=='__main__':unittest.main()
