"""New model contract must not reuse old glass labels, or alert on background."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from tests import test_soundjie_siren5_runtime as runtime
from tests import test_soundjie_siren5_ui as ui
from tests import test_soundjie_d4_classifier as adapter
from tests import test_soundjie_d4_supervisor as supervisor

class Alert4RuntimeTests(runtime.Siren5RuntimeTests):
    # Inherited cases also verify backward compatibility.
    def test_five_physical_outputs_have_new_labels(self):
        for index,label in enumerate(('警笛','车辆鸣笛','门铃','狗叫','背景')):
            scores=[.025]*5;scores[index]=.9
            r,rows=self.run_protocol(scores,'soundjie-alert4bg-v2')
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(rows[0]['label_zh'],label)
            self.assertEqual(rows[0]['class_scores'],scores)

    def test_old_six_outputs_rejected_by_new_contract(self):
        r,_=self.run_protocol([.1,.5,.1,.1,.1,.1],'soundjie-alert4bg-v2')
        self.assertNotEqual(r.returncode,0)

class Alert4UiTests(ui.Siren5UiTests):
    def render_case(self,**changes):
        if changes.get('class_contract')!='soundjie-alert4bg-v2':return super().render_case(**changes)
        s=ui.legacy.snapshot();s.update(classification_rejected=False,tv_enabled=True,tv_matched=True,fine_valid=True,
            tv_position_seconds=10,tv_score=.9,caption_text='打开百度地图',caption_speaker='A',ambulance_score=.9,road_score=.8,speech_score=.7)
        s.update(changes)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'snapshot.json';f.write_text(json.dumps(s,ensure_ascii=False),encoding='utf-8')
            args=['wsl','-d','Ubuntu','--',ui.legacy.wsl_path(self.binary),'--snapshot',ui.legacy.wsl_path(f),
                '--atlas',ui.legacy.wsl_path(ui.legacy.ROOT/'assets/hdmi/soundjie_unified_glyph_atlas.bin'),
                '--offscreen',ui.legacy.wsl_path(root/'ui.ppm'),'--report',ui.legacy.wsl_path(root/'report.json')]
            r=subprocess.run(args,capture_output=True)
            return r,json.loads((root/'report.json').read_text(encoding='utf-8')) if r.returncode==0 else {}

    def new_case(self,**changes):
        s=dict(class_contract='soundjie-alert4bg-v2',class_id=3,label_zh='狗叫',
               confidence=.9,class_scores=[.025,.025,.025,.9,.025])
        s.update(changes);return self.render_case(**s)

    def test_five_display_rows_and_existing_caption_direction(self):
        r,s=self.new_case();self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(s['classification_display_label'],'狗叫');self.assertEqual(s['display_class_count'],5)
        self.assertEqual(s['display_label'],'救护车警笛 + 车流声')
        self.assertTrue(s['alert_active']);self.assertTrue(s['main_score_visible'])
        self.assertTrue(s['caption_visible']);self.assertIsNotNone(s['arrow_tip'])

    def test_background_and_rejection_do_not_show_any_main_scores(self):
        for scores,rejected in [([.025,.025,.025,.025,.9],False),([.2,.3,.2,.2,.1],True)]:
            r,s=self.new_case(class_id=4,confidence=scores[4],class_scores=scores,classification_rejected=rejected,tv_matched=False)
            self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(s['display_label'],'背景')
            self.assertFalse(s['alert_active']);self.assertFalse(s['main_score_visible'])
            self.assertEqual(s['legacy_class_scores'],[])
            self.assertIsNone(s['main_display_confidence'])
            self.assertFalse(s['caption_visible']);self.assertTrue(s['caption_panel_visible'])

    def test_fault_is_not_quiet_background(self):
        r,s=self.new_case(usb_ok=False,dpu_ok=False,class_id=4,confidence=.9,class_scores=[.025]*4+[.9])
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(s['display_label'],'不可用')
        self.assertFalse(s['alert_active']);self.assertFalse(s['main_score_visible'])

    def test_atlas_covers_new_alert_labels(self):
        import struct
        b=(ui.legacy.ROOT/'assets/hdmi/soundjie_alert4_glyph_atlas.bin').read_bytes()
        codes={struct.unpack_from('<I',b,p)[0] for p in range(16,len(b),40)}
        self.assertTrue({ord(c) for c in '四类声音提醒警笛车辆鸣笛门铃狗叫背景'}.issubset(codes))

    def test_pinned_no_tv_startup_stale_fault_and_stop_never_alert(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/'state.json';config=root/'runtime.tsv'
            config.write_text('key\tvalue\ndoa_stale_ms\t200\ndoa_ema_alpha\t0.5\n',encoding='utf-8',newline='\n')
            command=['wsl','-d','Ubuntu','--','timeout','8',ui.legacy.wsl_path(self.binary),
                '--class-contract','soundjie-alert4bg-v2','--snapshot',ui.legacy.wsl_path(state),
                '--atlas',ui.legacy.wsl_path(ui.legacy.ROOT/'assets/hdmi/soundjie_alert4_glyph_atlas.bin'),
                '--doa-jsonl',ui.legacy.wsl_path(root/'doa.jsonl'),'--runtime-config',ui.legacy.wsl_path(config),
                '--frames','20','--interval-ms','100','--snapshot-stale-ms','500','--width','640','--height','360',
                '--offscreen-dir',ui.legacy.wsl_path(root/'frames')]
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            phase=0;seen=[];payload={}
            try:
                for line in proc.stdout:
                    r=json.loads(line);self.assertEqual(r['display_class_count'],5);self.assertEqual(r['class_contract'],'soundjie-alert4bg-v2')
                    if phase==0:
                        self.assertEqual(r['display_label'],'不可用');self.assertFalse(r['main_score_visible']);seen.append('startup')
                        payload=ui.legacy.snapshot();payload.update(timestamp_ms=r['monotonic_ms'],class_contract='soundjie-alert4bg-v2',class_id=3,
                            label_zh='狗叫',class_scores=[.025,.025,.025,.9,.025],confidence=.9,classification_rejected=False,tv_enabled=False)
                        state.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8');phase=1
                    elif phase==1 and r['alert_active']:
                        seen.append('alert');payload['timestamp_ms']=0;state.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8');phase=2
                    elif phase==2 and not r['classification_fresh']:
                        self.assertFalse(r['alert_active']);self.assertFalse(r['main_score_visible']);seen.append('stale')
                        state.write_text('{"session_state":"fault"}',encoding='utf-8');phase=3
                    elif phase==3 and r['session_state']=='fault':
                        self.assertEqual(r['display_label'],'不可用');self.assertFalse(r['main_score_visible']);seen.append('fault')
                        state.write_text('{"session_state":"stopped"}',encoding='utf-8');phase=4
                    elif phase==4 and r['session_state']=='stopped':
                        self.assertEqual(r['display_label'],'STOPPED');self.assertFalse(r['main_score_visible']);seen.append('stopped');phase=5
                _,err=proc.communicate(timeout=3)
                self.assertEqual(proc.returncode,0,err);self.assertEqual(seen,['startup','alert','stale','fault','stopped'])
            finally:
                if proc.poll() is None:proc.kill();proc.communicate()

class FiveOutputAdapterTests(runtime.SixOutputAdapterTests):
    outputs=5
    def test_real_adapter_reads_six_and_refuses_runtime_mismatch(self):
        r=self.run_case();self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(len(r.stdout.split()),8)
        r=self.run_case(SJ_TEST_FAIL='output_size');self.assertNotEqual(r.returncode,0)
        self.assertEqual(r.stdout,b'')

class FivePaddedOutputAdapterTests(FiveOutputAdapterTests):
    storage=6

    def test_compiler_padding_is_not_a_sixth_class(self):
        import numpy as np
        r=self.run_case();self.assertEqual(r.returncode,0,r.stderr)
        values=r.stdout.decode().split();self.assertEqual(len(values),8)
        expected=np.exp(np.asarray([-2,-1,0,4,1])*.5);expected/=expected.sum()
        np.testing.assert_allclose([float(v) for v in values[3:]],expected,rtol=1e-8)
        r=self.run_case(SJ_TEST_FAIL='padding')
        self.assertNotEqual(r.returncode,0);self.assertEqual(r.stdout,b'')

class Alert4SupervisorTests(supervisor.D4SupervisorTests):
    def test_new_policy_and_cadence_reach_runtime_and_terminal_state(self):
        (self.bundle/'config/class-contract.txt').write_text('soundjie-alert4bg-v2\n',newline='\n')
        (self.bundle/'config/alert-thresholds.txt').write_text('0.95,0.825,0.625,0.725\n',newline='\n')
        (self.bundle/'config/hop-samples.txt').write_text('4000\n',newline='\n')
        r=subprocess.run(self.command('ui'),capture_output=True,timeout=12)
        self.assertEqual(r.returncode,2,r.stderr)
        args=(self.root/'runtime.args').read_text().splitlines()
        for key,value in [('--alert-thresholds','0.95,0.825,0.625,0.725'),('--hop-samples','4000'),('--cadence-ms','250')]:
            self.assertEqual(args[args.index(key)+1],value)
        capture=(self.root/'capture.args').read_text().splitlines()
        self.assertIn('--period-size=800',capture)
        self.assertIn('--buffer-size=8000',capture)
        state=json.loads(self.read_session('state/current.json'))
        self.assertEqual(state['class_contract'],'soundjie-alert4bg-v2');self.assertEqual(state['class_id'],-1)

    def test_missing_new_policy_does_not_start_capture(self):
        (self.bundle/'config/class-contract.txt').write_text('soundjie-alert4bg-v2\n',newline='\n')
        r=subprocess.run(self.command(),capture_output=True,timeout=6)
        self.assertNotEqual(r.returncode,0);self.assertEqual(list(self.root.glob('*.started')),[])

if __name__=='__main__':unittest.main()
