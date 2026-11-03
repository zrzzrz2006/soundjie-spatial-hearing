"""Six visible rows and clip-local presentation must keep real classifier data."""
import unittest
from tests import test_soundjie_unified_ui as old

ATLAS = old.ROOT / 'assets/hdmi/soundjie_two_clip_glyph_atlas.bin'
NAMES = ['警笛','车辆鸣笛','门铃','狗叫','普通说话声','车流声']

def snapshot(**changes):
    s=old.snapshot(class_contract='soundjie-alert5bg-v3',class_id=4,
        confidence=.90,class_scores=[.02,.02,.02,.02,.90,.02],
        tv_clip_id='',tv_scene_profile='',scene_horn_start=4.8,scene_horn_end=6.8)
    s.update(changes)
    return s

class TwoClipUiTests(unittest.TestCase):
    setUpClass=classmethod(old.UnifiedUiTests.setUpClass.__func__)
    tearDownClass=classmethod(old.UnifiedUiTests.tearDownClass.__func__)
    def render(self,payload,width=1280,height=720):
        return old.UnifiedUiTests.render(self,payload,width,height,ATLAS,'soundjie-alert5bg-v3')
    def test_normal_speech_is_real_nonalert_and_traffic_blank(self):
        r,_=self.render(snapshot())
        self.assertEqual(r['layout'],'unified_live_caption_v3')
        self.assertEqual(r['display_class_count'],6)
        self.assertEqual([x['label'] for x in r['sound_rows']],NAMES)
        self.assertEqual(r['display_label'],'普通说话声')
        self.assertEqual([x['score'] for x in r['sound_rows']], [.02,.02,.02,.02,.90,None])
        self.assertFalse(r['alert_active'])
        self.assertFalse(r['caption_visible'])
        self.assertEqual(r['sound_rows'][-1]['score_text'],'')
    def test_dialogue_interval_and_subtitles_never_activate_road(self):
        for t,heading in [(2,'普通说话声'),(4.8,'车辆鸣笛'),(6.799,'车辆鸣笛'),(6.8,'普通说话声'),(50,'普通说话声')]:
            with self.subTest(t=t):
                r,_=self.render(snapshot(tv_matched=True,tv_clip_id='hearing_dialogue_20261101',
                    tv_scene_profile='dialogue',tv_position_seconds=t,fine_valid=True,road_score=1,
                    caption_text='我们俩都听不见。',caption_speaker='B'))
                self.assertEqual(r['display_label'],heading)
                self.assertEqual(r['caption_speaker'],'B')
                self.assertEqual(r['sound_rows'][4]['score'],.9)
                self.assertFalse(r['sound_rows'][5]['enabled'])
                self.assertEqual(r['sound_rows'][5]['score_text'],'')
                self.assertFalse(r['caption_truncated'])
    def test_ambulance_and_dialogue_share_bounds_with_real_five_scores(self):
        base,_=self.render(snapshot())
        r,_=self.render(snapshot(tv_matched=True,fine_valid=True,
            tv_clip_id='ambulance_escort_45_75',tv_scene_profile='ambulance_traffic'))
        self.assertEqual(r['display_label'],'救护车警笛 + 车流声')
        self.assertEqual([x['display_score'] for x in r['sound_rows'][:5]],[.02,.02,.02,.02,.9])
        road=r['sound_rows'][5]
        self.assertEqual(road['score'],.83)
        self.assertEqual(road['display_basis'],'presentation_random_85_100')
        self.assertGreaterEqual(road['display_score'],.85)
        self.assertLessEqual(road['display_score'],1)
        self.assertEqual(r['sound_panel_bounds'],base['sound_panel_bounds'])
        self.assertEqual(r['caption_panel_bounds'],base['caption_panel_bounds'])
    def test_background_and_fault_clear_all_six_scores(self):
        for changes in [dict(class_id=5,confidence=.9,class_scores=[.02]*5+[.9]),dict(usb_ok=False)]:
            r,_=self.render(snapshot(**changes))
            self.assertFalse(r['alert_active'])
            self.assertTrue(all(x['score'] is None for x in r['sound_rows']))
            self.assertFalse(r['caption_visible'])
    def test_all_new_captions_fit_both_resolutions(self):
        import json
        cfg=json.loads((old.ROOT/'config/soundjie_tv_dialogue_clip.json').read_text(encoding='utf-8'))
        for width,height in [(1280,720),(640,480)]:
            for cue in cfg['cues']:
                r,_=self.render(snapshot(tv_matched=True,tv_clip_id=cfg['clip_id'],tv_scene_profile='dialogue',
                    caption_text=cue['text'],caption_speaker=cue['speaker']),width,height)
                self.assertFalse(r['caption_truncated'],cue['id'])
                self.assertLessEqual(r['caption_max_right'],r['caption_panel_right'])

if __name__=='__main__':unittest.main()
