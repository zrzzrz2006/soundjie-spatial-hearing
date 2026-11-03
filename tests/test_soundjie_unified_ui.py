"""Verify real scores, isolated traffic presentation, and state clearing."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from tests import test_soundjie_hdmi_ui as legacy


ROOT = legacy.ROOT
ATLAS = ROOT / 'assets/hdmi/soundjie_unified_glyph_atlas.bin'
NAMES = ['警笛', '车辆鸣笛', '门铃', '狗叫', '车流声']


def snapshot(**changes):
    s = legacy.snapshot('front_right')
    s.update(class_contract='soundjie-alert4bg-v2', class_id=0,
             label_zh='警笛', confidence=.96, class_scores=[.96,.01,.01,.01,.01],
             classification_rejected=False, tv_enabled=True, tv_matched=False,
             fine_valid=False, tv_position_seconds=14, tv_score=.94,
             caption_text='打开百度地图', caption_speaker='A',
             ambulance_score=.91, road_score=.83, speech_score=.7)
    s.update(changes)
    return s


class UnifiedUiTests(unittest.TestCase):
    setUpClass = classmethod(legacy.SoundjieHdmiUiTests.setUpClass.__func__)
    tearDownClass = classmethod(legacy.SoundjieHdmiUiTests.tearDownClass.__func__)

    def test_atlas_covers_every_unified_label_and_caption(self):
        data=ATLAS.read_bytes()
        codes={struct.unpack_from('<I',data,p)[0] for p in range(16,len(data),40)}
        required=(ROOT/'config/soundjie_unified_ui_text.txt').read_text(encoding='utf-8')
        self.assertTrue({ord(c) for c in required if ord(c)>=32}.issubset(codes))

    def test_shipped_atlas_preserves_every_previous_glyph(self):
        previous=(ROOT/'assets/hdmi/soundjie_alert4_glyph_atlas.bin').read_bytes()
        current=ATLAS.read_bytes()
        old={struct.unpack_from('<I',previous,p)[0]:previous[p:p+40] for p in range(16,len(previous),40)}
        new={struct.unpack_from('<I',current,p)[0]:current[p:p+40] for p in range(16,len(current),40)}
        self.assertTrue(set(old).issubset(new),'replacement atlas drops previous glyphs')
        self.assertEqual({cp:new[cp] for cp in old},old)

    def test_legacy_and_siren5_render_identically_with_the_shipped_atlas(self):
        old_atlas=ROOT/'assets/hdmi/soundjie_alert4_glyph_atlas.bin'
        for contract in ('soundjie-legacy8-v1','soundjie-siren5-v1'):
            s=snapshot(class_contract=contract)
            if contract=='soundjie-legacy8-v1':
                s.update(class_id=1,label_zh='应急车辆警笛',confidence=.87,
                         class_scores=[.01,.87,.02,.02,.02,.02,.02,.02])
            _,old=self.render(s,atlas=old_atlas,contract=contract)
            _,new=self.render(s,contract=contract)
            self.assertEqual(new,old)

    def render(self, payload, width=1280, height=720, atlas=ATLAS, contract='soundjie-alert4bg-v2'):
        root = Path(tempfile.mkdtemp(dir=self.temporary.name))
        state = root/'state.json'
        state.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
        original = hashlib.sha256(state.read_bytes()).hexdigest()
        r = subprocess.run(['wsl','-d','Ubuntu','--',legacy.wsl_path(self.binary),
            '--snapshot',legacy.wsl_path(state),'--atlas',legacy.wsl_path(atlas),
            '--class-contract',contract,
            '--width',str(width),'--height',str(height),
            '--offscreen',legacy.wsl_path(root/'screen.ppm'),
            '--report',legacy.wsl_path(root/'report.json')],capture_output=True,timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(original, hashlib.sha256(state.read_bytes()).hexdigest())
        return json.loads((root/'report.json').read_text(encoding='utf-8')), (root/'screen.ppm').read_bytes()

    def test_ordinary_state_has_five_rows_empty_caption_and_inactive_road(self):
        s, _ = self.render(snapshot())
        self.assertEqual(s['layout'], 'unified_live_caption_v2')
        self.assertEqual(s['display_class_count'], 5)
        self.assertEqual([r['label'] for r in s['sound_rows']], NAMES)
        self.assertEqual([r['score'] for r in s['sound_rows'][:4]], [.96,.01,.01,.01])
        self.assertIsNone(s['sound_rows'][4]['score'])
        self.assertEqual(s['sound_rows'][4]['score_text'], '')
        self.assertFalse(s['sound_rows'][4]['enabled'])
        self.assertTrue(s['caption_panel_visible'])
        self.assertFalse(s['caption_visible'])
        self.assertEqual(s['caption_text'], '')
        self.assertEqual(s['caption_speaker'], '')
        self.assertFalse(s['volume_numeric_visible'])

    def test_scene_keeps_real_four_scores_and_adds_independent_road(self):
        normal, _ = self.render(snapshot())
        s, _ = self.render(snapshot(tv_matched=True, fine_valid=True))
        self.assertEqual(s['display_class_count'], 5)
        self.assertEqual(s['display_label'], '救护车警笛 + 车流声')
        self.assertEqual([r['score'] for r in s['sound_rows'][:4]], [.96,.01,.01,.01])
        self.assertEqual(s['sound_rows'][4]['score'], .83)
        self.assertGreaterEqual(s['sound_rows'][4]['display_score'], .85)
        self.assertLessEqual(s['sound_rows'][4]['display_score'], 1)
        self.assertEqual(s['sound_rows'][4]['display_basis'], 'presentation_random_85_100')
        self.assertEqual(s['sound_rows'][4]['score_text'],
                         f"{s['sound_rows'][4]['display_score']*100:.0f}%")
        self.assertEqual([r['display_score'] for r in s['sound_rows'][:4]], [.96,.01,.01,.01])
        self.assertTrue(s['sound_rows'][4]['enabled'])
        self.assertEqual(s['caption_text'], '打开百度地图')
        self.assertEqual(s['caption_speaker'], 'A')
        self.assertEqual(s['caption_panel_bounds'], normal['caption_panel_bounds'])
        self.assertEqual(s['sound_panel_bounds'], normal['sound_panel_bounds'])

    def test_background_suppresses_alert_and_four_scores_but_keeps_five_rows(self):
        s, _ = self.render(snapshot(class_id=4, confidence=.9, class_scores=[.025]*4+[.9]))
        self.assertEqual(s['display_label'], '背景')
        self.assertFalse(s['alert_active'])
        self.assertFalse(s['main_score_visible'])
        self.assertEqual([r['score'] for r in s.get('sound_rows',[])], [None]*5)
        self.assertEqual([r['score_text'] for r in s['sound_rows']], ['']*5)
        self.assertTrue(s['caption_panel_visible'])

    def test_road_presentation_preserves_raw_scores_and_does_not_activate_nontraffic(self):
        for score in (0,.2,.649,.65,.848313749,.99,.999998808):
            with self.subTest(score=score):
                s, _ = self.render(snapshot(tv_matched=True,fine_valid=True,road_score=score))
                row=s['sound_rows'][4]
                self.assertTrue(row['enabled'])
                self.assertAlmostEqual(row['score'],score,places=6)
                if score>=.65:
                    self.assertGreaterEqual(row['display_score'],.85)
                    self.assertLessEqual(row['display_score'],1)
                    self.assertEqual(row['display_basis'],'presentation_random_85_100')
                    self.assertEqual(row['score_text'],f"{row['display_score']*100:.0f}%")
                else:
                    self.assertAlmostEqual(row['display_score'],score,places=6)
                    self.assertEqual(row['score_text'],f'{score*100:.0f}%' if score else '')
                    self.assertEqual(row['display_basis'],'model_output')
                    self.assertEqual(s['display_label'],'救护车警笛')

    def test_presentation_timing_bounds_seed_and_reset(self):
        root=Path(tempfile.mkdtemp(dir=self.temporary.name))
        code=root/'traffic_test.cpp';binary=root/'traffic_test'
        code.write_text(r'''
#include "soundjie_traffic_presentation.h"
#include <cassert>
#include <cmath>
#include <set>
int main() {
  soundjie::TrafficPresentation a(123),b(123);
  std::set<int> seen;
  for (uint64_t n=0;n<2000;++n) {
    const auto start=100+n*1200;
    const double value=a.update(true,start);
    assert(value==b.update(true,start));
    assert(value>=.85 && value<=1);
    seen.insert(static_cast<int>(std::lround(value*100)));
    assert(a.update(true,start+1199)==value);
  }
  assert(seen.size()==16);
  assert(a.update(false,2400101)==0);
  assert(a.update(false,2400102)==0);
  assert(a.update(true,2400103)>=.85);
  assert(a.update(true,1)>=.85); // Defensive monotonic clock reset.
}
''',encoding='utf-8')
        result=subprocess.run(['wsl','-d','Ubuntu','--','g++','-std=c++11',
            '-Wall','-Wextra','-Werror','-I'+legacy.wsl_path(ROOT/'src'),
            legacy.wsl_path(code),'-o',legacy.wsl_path(binary)],capture_output=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run(['wsl','-d','Ubuntu','--',legacy.wsl_path(binary)],capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_live_presentation_changes_while_raw_road_score_remains_fixed(self):
        root=Path(tempfile.mkdtemp(dir=self.temporary.name));state=root/'state.json'
        cmd=['wsl','-d','Ubuntu','--',legacy.wsl_path(self.binary),
             '--snapshot',legacy.wsl_path(state),'--atlas',legacy.wsl_path(ATLAS),
             '--class-contract','soundjie-alert4bg-v2','--traffic-display-seed','123',
             '--doa-jsonl',legacy.wsl_path(root/'doa.jsonl'),
             '--runtime-config',legacy.wsl_path(ROOT/'config/soundjie_d4_runtime.tsv'),
             '--offscreen-dir',legacy.wsl_path(root/'frames'),
             '--width','640','--height','360','--interval-ms','100','--frames','30',
             '--snapshot-stale-ms','500']
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        values=[]
        try:
            for line in proc.stdout:
                report=json.loads(line)
                payload=snapshot(timestamp_ms=report['monotonic_ms'],tv_matched=True,
                                 fine_valid=True,road_score=.999998808)
                state.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
                row=report['sound_rows'][4]
                if row['enabled']:
                    self.assertAlmostEqual(row['score'],.999998808,places=6)
                    self.assertGreaterEqual(row['display_score'],.85)
                    self.assertLessEqual(row['display_score'],1)
                    values.append(row['display_score'])
            _,err=proc.communicate(timeout=4)
            self.assertEqual(proc.returncode,0,err)
            self.assertGreater(len(values),20)
            self.assertGreaterEqual(len(set(values)),2)
            self.assertLessEqual(sum(a!=b for a,b in zip(values,values[1:])),4)
        finally:
            if proc.poll() is None:proc.kill();proc.communicate()

    def test_scene_does_not_replace_background_model_values_with_fine_siren(self):
        s, _ = self.render(snapshot(tv_matched=True, fine_valid=True, class_id=4,
                                    confidence=.9, class_scores=[.025]*4+[.9]))
        self.assertEqual([r['score'] for r in s.get('sound_rows',[])], [.025]*4+[.83])
        self.assertEqual(s['display_label'], '救护车警笛 + 车流声')
        self.assertFalse(s['alert_active'])

    def test_missing_caption_keeps_box_but_clears_speaker(self):
        s, blank = self.render(snapshot(tv_matched=True, fine_valid=True, caption_text=''))
        filled, image = self.render(snapshot(tv_matched=True, fine_valid=True))
        self.assertTrue(s.get('caption_panel_visible'))
        self.assertEqual(s['caption_speaker'], '')
        self.assertEqual(s['caption_text'], '')
        self.assertEqual(s['caption_panel_bounds'], filled['caption_panel_bounds'])
        self.assertNotEqual(blank, image)

    def test_fault_and_invalid_fine_results_do_not_leave_fake_scores(self):
        for changes in (dict(usb_ok=False), dict(dpu_ok=False), dict(session_state='fault')):
            with self.subTest(changes=changes):
                s, _ = self.render(snapshot(tv_matched=True, fine_valid=True, **changes))
                self.assertEqual(s['display_label'], '不可用')
                self.assertEqual([r['score'] for r in s.get('sound_rows',[])], [None]*5)
                self.assertEqual([r['display_score'] for r in s['sound_rows']], [None]*5)
                self.assertFalse(s['caption_visible'])
                self.assertFalse(s['fine_events_visible'])
        s, _ = self.render(snapshot(tv_matched=True, fine_valid=False))
        self.assertIsNone(s['sound_rows'][4]['score'])
        self.assertEqual(s['display_label'], '声音待确认')

    def test_direction_invalid_has_no_arrow_and_volume_has_no_numbers(self):
        payload = snapshot(doa_valid=False, doa_sector=None, doa_deg=None, doa_reason='no_speech')
        s, _ = self.render(payload)
        self.assertIsNone(s['arrow_tip'])
        self.assertEqual(s['doa_text'], '方向暂不可用')
        self.assertFalse(s['volume_numeric_visible'])
        s, _ = self.render(snapshot())
        self.assertEqual(s['doa_text'], '右前方')
        self.assertEqual(s['doa_angle_deg'], 45)

    def test_long_caption_is_clipped_inside_its_panel_at_both_sizes(self):
        for width,height in ((1280,720),(640,360)):
            s, ppm = self.render(snapshot(tv_matched=True,fine_valid=True,
                caption_text='打开百度地图'*10),width,height)
            self.assertEqual(s['layout'], 'unified_live_caption_v2')
            self.assertTrue(s['caption_truncated'])
            self.assertLessEqual(s['caption_lines'],2)
            self.assertLessEqual(s['caption_max_right'],s['caption_panel_right'])
            self.assertLessEqual(s['caption_max_bottom'],s['caption_panel_bounds'][1]+s['caption_panel_bounds'][3])
            self.assertTrue(ppm.startswith(f'P6\n{width} {height}\n255\n'.encode()))

    def test_live_exit_stale_and_stop_clear_data_without_changing_layout(self):
        root = Path(tempfile.mkdtemp(dir=self.temporary.name))
        state = root/'state.json'
        cmd=['wsl','-d','Ubuntu','--',legacy.wsl_path(self.binary),
             '--snapshot',legacy.wsl_path(state),'--atlas',legacy.wsl_path(ATLAS),
             '--class-contract','soundjie-alert4bg-v2',
             '--doa-jsonl',legacy.wsl_path(root/'doa.jsonl'),
             '--runtime-config',legacy.wsl_path(ROOT/'config/soundjie_d4_runtime.tsv'),
             '--offscreen-dir',legacy.wsl_path(root/'frames'),
             '--width','640','--height','360','--interval-ms','100','--frames','18',
             '--snapshot-stale-ms','500']
        proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        phase=0;seen=[]
        try:
            for line in proc.stdout:
                s=json.loads(line)
                self.assertEqual(s['layout'],'unified_live_caption_v2')
                self.assertEqual(s['display_class_count'],5)
                self.assertTrue(s['caption_panel_visible'])
                if phase==0:
                    seen.append('startup');phase=1
                    payload=snapshot(timestamp_ms=s['monotonic_ms'],tv_matched=True,fine_valid=True)
                    state.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
                elif phase==1 and s['caption_visible']:
                    seen.append('scene');phase=2
                    payload=snapshot(timestamp_ms=s['monotonic_ms'])
                    state.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
                elif phase==2 and s['classification_fresh'] and not s['tv_matched']:
                    self.assertIsNone(s['sound_rows'][4]['score'])
                    self.assertIsNone(s['sound_rows'][4]['display_score'])
                    self.assertEqual(s['sound_rows'][4]['score_text'],'')
                    self.assertEqual(s['caption_text'],'');seen.append('ordinary');phase=3
                    payload['timestamp_ms']=0
                    state.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf-8')
                elif phase==3 and not s['classification_fresh']:
                    self.assertEqual([r['score'] for r in s['sound_rows']],[None]*5)
                    seen.append('stale');phase=4
                    state.write_text('{"session_state":"stopped"}',encoding='utf-8')
                elif phase==4 and s['session_state']=='stopped':
                    self.assertEqual(s['display_label'],'STOPPED')
                    self.assertFalse(s['caption_visible']);seen.append('stopped');phase=5
            _,err=proc.communicate(timeout=4)
            self.assertEqual(proc.returncode,0,err)
            self.assertEqual(seen,['startup','scene','ordinary','stale','stopped'])
        finally:
            if proc.poll() is None:proc.kill();proc.communicate()


if __name__ == '__main__':
    unittest.main()
