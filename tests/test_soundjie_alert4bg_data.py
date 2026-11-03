"""Semantic, identity and physical-output regressions for alert4bg v2."""
import unittest
import numpy as np
from scripts.host import soundjie_alert4bg as a


def row(identity='a', purpose='train', **kw):
    return dict(recording_id=identity, purpose=purpose, source_group=identity,
                sha256=identity*64, target_index=4, target_id=a.LABELS[4], **kw)


class DataContractTests(unittest.TestCase):
    def test_explicit_bark_only(self):
        self.assertEqual(a.classify_fsd(['Animal','Dog','Domestic_animals_and_pets','Bark']), (3,'bark'))
        for labels in (['Dog'], ['Growling','Dog'], ['Bark','Growling'], ['Bark','Speech'],
                       ['Bark','Siren'], ['Bark','Doorbell'], []):
            self.assertIsNone(a.classify_fsd(labels), labels)

    def test_alerts_and_negatives(self):
        cases=[(['Alarm'],(0,'alarm')), (['Siren','Alarm'],(0,'siren')),
               (['Vehicle_horn_and_car_horn_and_honking','Alarm'],(1,'horn')),
               (['Doorbell','Alarm','Door'],(2,'doorbell')), (['Door','Knock'],(4,'knock')),
               (['Glass','Shatter'],(4,'glass')), (['Speech','Human_voice'],(4,'ordinary_speech')),
               (['Crying_and_sobbing','Human_voice'],(4,'crying_screaming'))]
        for labels,expected in cases:self.assertEqual(a.classify_fsd(labels),expected)
        for labels in (['Siren','Doorbell','Alarm'],['Doorbell','Knock','Door'],['Siren','Bark'],['Music']):
            self.assertIsNone(a.classify_fsd(labels))

    def test_legacy_door_family_needs_evidence(self):
        base=dict(source_id='esc50',source_group='esc50:123',target_index=4,target_id='door_knock_bell')
        self.assertEqual(a.map_existing(dict(base,original_labels=['door_wood_knock']))['target_index'],4)
        self.assertIsNone(a.map_existing(dict(base,original_labels=[])))
        self.assertIsNone(a.map_existing(dict(base,original_labels=['dog'])))
        glass=dict(base,original_labels=['glass_breaking'],target_index=3,target_id='glass_breaking')
        self.assertEqual(a.map_existing(glass)['target_id'],'background_unknown')

    def test_purposes_groups_hashes_and_originals_do_not_cross(self):
        a.validate_rows([row(), row('b','validation')], require_all=False)
        for key in ('source_group','sha256','sampling_identity','uploader_group'):
            left,right=row(),row('b','validation')
            left[key]=right[key]='same'
            with self.assertRaisesRegex(ValueError,'cross'):a.validate_rows([left,right],require_all=False)
        with self.assertRaises(ValueError):a.validate_rows([row(purpose='d5_holdout')],require_all=False)
        bad=row();bad['target_id']='glass_breaking'
        with self.assertRaises(ValueError):a.validate_rows([bad],require_all=False)

    def test_frozen_uploader_purpose_not_resplit(self):
        self.assertEqual(a.choose_purpose('known',{'known':{'validation'}},'train'),'validation')
        self.assertIsNone(a.choose_purpose('heldout',{'heldout':{'d5_holdout'}},'train'))
        self.assertIsNone(a.choose_purpose('mixed',{'mixed':{'train','validation'}},'train'))
        self.assertEqual(a.choose_purpose('new',{},'train'),'train')

    def test_plan_excludes_holdout_author_and_keeps_old_validation(self):
        from scripts.host.acquire_soundjie_alert4bg import plan_sources
        from scripts.host.acquire_soundjie_fsd50k import uploader_group
        existing=[dict(recording_id='fsd50k:11',source_recording_id='freesound:11',
            source_id='fsd50k',source_group=uploader_group('kept'),author='kept',
            purpose='validation',sha256='a'*64,original_labels=['Doorbell'],
            target_index=4,target_id='door_knock_bell',license='CC0-1.0')]
        historical=existing+[dict(recording_id='old-heldout',source_group='esc50:99',
            purpose='d5_holdout',sha256='b'*64)]
        gt=[dict(fname=str(i),labels=['Bark'],mids=['/bark'],split='train') for i in (12,13,14,15)]
        meta={str(i):dict(uploader=name,license=license) for i,name,license in (
            (99,'blocked','https://creativecommons.org/publicdomain/zero/1.0/'),
            (12,'blocked','https://creativecommons.org/publicdomain/zero/1.0/'),
            (13,'kept','https://creativecommons.org/publicdomain/zero/1.0/'),
            (14,'new','https://creativecommons.org/licenses/by-nc/3.0/'),
            (15,'new2','https://creativecommons.org/publicdomain/zero/1.0/'))}
        cfg=dict(bark_limit=320,max_bark_per_uploader=8)
        fsd_cfg=dict(allowed_licenses={'https://creativecommons.org/publicdomain/zero/1.0/':'CC0-1.0'},
                     split_seed='fixed',validation_fraction=.2)
        old,new,rejected,stats,_=plan_sources(existing,historical,gt,meta,cfg,fsd_cfg)
        self.assertEqual(old[0]['purpose'],'validation')
        self.assertEqual(old[0]['target_index'],2)
        self.assertEqual({r['freesound_id'] for r in new},{'13','15'})
        self.assertEqual(next(r['purpose'] for r in new if r['freesound_id']=='13'),'validation')
        self.assertEqual(stats['excluded_historical_uploader_purpose'],1)
        self.assertEqual(stats['excluded_license'],1)
        reversed_plan=plan_sources(existing,historical,list(reversed(gt)),meta,cfg,fsd_cfg)
        self.assertEqual(new,reversed_plan[1])

    def test_physical_five_and_order(self):
        import torch
        torch.set_num_threads(4)
        model=a.make_model().eval()
        with torch.no_grad():actual=model(torch.zeros(2,1,64,96))
        self.assertEqual(tuple(actual.shape),(2,5))
        self.assertEqual(tuple(model.fc8.weight.shape),(5,2304))
        self.assertEqual(a.LABELS,('siren','car_horn','doorbell','dog_bark','background_unknown'))
        self.assertEqual(a.output_contract()['contract'],'soundjie-alert4bg-v2')

    def test_thresholds_do_not_promote_runner_up(self):
        scores=np.array([[.3,.1,.1,.1,.4],[.1,.1,.1,.6,.1]])
        np.testing.assert_array_equal(a.apply_thresholds(scores,[.2,.5,.5,.7]),[4,4])
        for bad in (np.ones((2,6))/6,np.ones((2,5))):
            with self.assertRaises(ValueError):a.apply_thresholds(bad,[.5]*4)

    def test_balancing_counts_sources_not_windows(self):
        y=np.array([0]*100+[0,1,2,3,4]);ids=np.array(['many']*100+['one','b','c','d','e'])
        idx=a.balanced_source_indices(y,ids,1000,7)
        self.assertEqual(np.bincount(y[idx]).tolist(),[1000]*5)
        fraction=np.mean(ids[idx[y[idx]==0]]=='one')
        self.assertTrue(.4<fraction<.6)


if __name__=='__main__':unittest.main()
