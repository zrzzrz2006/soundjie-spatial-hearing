import copy
import importlib
import unittest


class TVReferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.mod = importlib.import_module('scripts.host.prepare_soundjie_tv')
        except ModuleNotFoundError:
            cls.mod = None

    def setUp(self):
        self.assertIsNotNone(self.mod, 'reference preparation module is not implemented')
        self.config = dict(schema=1, clip_id='ambulance_45_75',
                           source_sha256='a'*64, start_seconds=45, end_seconds=75,
                           sample_rate=16000, cues=[dict(id='q1',start=12.9,end=14.6,
                           speaker='A',text='打开百度地图',status='confirmed_anchor')])

    def test_accepts_relative_cues_and_exact_sample_interval(self):
        self.assertEqual(self.mod.validate_config(self.config), 480000)

    def test_rejects_outside_and_overlapping_cues(self):
        for cue in [dict(id='q2',start=14,end=16,speaker='A',text='导航',status='candidate'),
                    dict(id='q2',start=29,end=31,speaker='A',text='导航',status='candidate')]:
            cfg=copy.deepcopy(self.config);cfg['cues'].append(cue)
            with self.assertRaises(ValueError): self.mod.validate_config(cfg)

    def test_rejects_duplicate_ids_and_empty_text(self):
        for change in [dict(text=''),dict(id='q1',start=17,end=18)]:
            cfg=copy.deepcopy(self.config)
            if 'start' in change:
                cue=dict(cfg['cues'][0]);cue.update(change);cfg['cues'].append(cue)
            else: cfg['cues'][0].update(change)
            with self.assertRaises(ValueError): self.mod.validate_config(cfg)

    def test_rejects_nan_and_wrong_rate(self):
        for key,value in [('end_seconds',float('nan')),('sample_rate',48000),('start_seconds',-1)]:
            cfg=copy.deepcopy(self.config);cfg[key]=value
            with self.assertRaises(ValueError): self.mod.validate_config(cfg)

    def test_text_serialization_rejects_control_characters(self):
        for value in ['a\tb','a\nb','a\x00b']:
            cfg=copy.deepcopy(self.config);cfg['cues'][0]['text']=value
            with self.assertRaises(ValueError): self.mod.validate_config(cfg)

    def test_candidate_text_is_not_an_enabled_caption(self):
        cfg=copy.deepcopy(self.config);cfg['cues'][0]['status']='candidate'
        self.assertEqual(self.mod.enabled_cues(cfg), [])

    def test_phrase_text_with_quotes_is_preserved(self):
        cfg=copy.deepcopy(self.config);cfg['cues'][0]['text']='他说“导航”'
        self.mod.validate_config(cfg)
        self.assertEqual(self.mod.enabled_cues(cfg)[0]['text'],'他说“导航”')


if __name__ == '__main__': unittest.main()
