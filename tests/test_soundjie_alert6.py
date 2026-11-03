import unittest
import numpy as np

from scripts.host.soundjie_alert6 import (LABELS, map_legacy, classify_fsd,
    validate_rows, metrics, balanced_source_indices, make_model)


def row(identity, target, purpose='train', group=None, sha=None):
    return dict(recording_id=identity, target_index=target, target_id=LABELS[target],
                purpose=purpose, source_group=group or identity, sha256=sha or identity,
                path='unused', license='CC0', original_url='https://example.com/'+identity)


class ContractTests(unittest.TestCase):
    def test_voice_outputs_are_background_not_runner_up(self):
        self.assertEqual(len(LABELS), 6)
        for old in (5, 6, 7):
            original = dict(target_index=old, target_id='legacy', purpose='validation')
            mapped = map_legacy(original)
            self.assertEqual(mapped['target_index'], 5)
            self.assertEqual(original['target_index'], old)
        self.assertEqual(map_legacy(dict(target_index=0, target_id='fire_smoke_alarm'))['target_id'], 'alarm_sound')

    def test_competing_events_not_accepted_as_single_label(self):
        self.assertIsNone(classify_fsd(['Alarm','Siren','Vehicle_horn_and_car_horn_and_honking']))
        self.assertEqual(classify_fsd(['Alarm','Siren'])[0], 1)
        self.assertEqual(classify_fsd(['Glass','Shatter'])[0], 3)
        self.assertEqual(classify_fsd(['Alarm','Door','Doorbell','Domestic_sounds_and_home_sounds'])[0], 4)
        self.assertEqual(classify_fsd(['Speech','Human_voice'])[0], 5)
        self.assertEqual(classify_fsd(['Screaming','Human_voice'])[0], 5)
        self.assertIsNone(classify_fsd(['Alarm','Speech']))

    def test_split_conflicts_and_removed_indices_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows([row('a',0,group='same'), row('b',1,'validation',group='same')], require_all=False)
        with self.assertRaises(ValueError):
            validate_rows([row('a',0,sha='same'), row('b',1,'validation',sha='same')], require_all=False)
        bad=row('a',0);bad['target_index']=6
        with self.assertRaises(ValueError):validate_rows([bad], require_all=False)
        with self.assertRaises(ValueError):validate_rows([row('a',0,'d5_holdout')], require_all=False)

    def test_metrics_expose_negative_false_alerts_and_each_target(self):
        result=metrics(np.array([0,1,2,3,4,5,5]), np.array([0,1,2,3,4,5,1]))
        self.assertEqual(result['confusion_matrix'][5][1],1)
        self.assertEqual(result['negative_false_alert_rate'],0.5)
        self.assertEqual(len(result['per_class']),6)

    def test_sampling_balances_sources_not_number_of_augmentations(self):
        y=np.array([0]*100+[0]+list(range(1,6)))
        groups=np.array(['a']*100+['b','c','d','e','f','g'])
        indices=balanced_source_indices(y, groups, per_class=1000, seed=1)
        selected=groups[indices[y[indices]==0]]
        self.assertLess(abs(np.sum(selected=='a')-np.sum(selected=='b')),100)

    def test_new_model_really_has_six_outputs(self):
        import torch
        torch.set_num_threads(2)
        model=make_model().eval()
        with torch.no_grad():out=model(torch.zeros(2,1,64,96))
        self.assertEqual(tuple(out.shape),(2,6))
        self.assertEqual(tuple(model.fc8.weight.shape),(6,2304))

    def test_validation_has_no_augmentation_or_adjacent_training_windows(self):
        from scripts.host.train_soundjie_alert6 import source_windows
        signal=np.zeros(48000);signal[20000:22000]=np.sin(np.arange(2000)*.3)
        self.assertEqual(len(source_windows(signal, {'target_index':2,'purpose':'validation'}, 1)), 1)
        self.assertGreater(len(source_windows(signal, {'target_index':2,'purpose':'train'}, 1)), 1)
        with self.assertRaises(ValueError):
            source_windows(signal, {'target_index':2,'purpose':'d5_holdout'}, 1)

    def test_threshold_does_not_promote_second_choice(self):
        from scripts.host.train_soundjie_alert6 import apply_thresholds
        scores=np.array([[.2,.1,.1,.1,.1,.4],[.3,.4,.1,.05,.05,.1]])
        np.testing.assert_array_equal(apply_thresholds(scores,[.2,.8,.5,.5,.5]),[5,5])

    def test_export_fold_keeps_six_logits_numerically_equivalent(self):
        import torch
        from scripts.host.evaluate_soundjie_alert6 import fold_and_check
        torch.set_num_threads(2);torch.manual_seed(23)
        model=make_model().eval()
        data=np.random.default_rng(1).normal(size=(3,1,64,96)).astype(np.float32)
        arrays,report=fold_and_check(model,data)
        self.assertEqual(arrays['fc8.weight'].shape,(6,2304))
        self.assertLess(report['max_absolute_error'],1e-4)

    def test_acquisition_excludes_old_heldout_id_and_known_uploader(self):
        from scripts.host.acquire_soundjie_alert6 import plan_fsd,select_legacy
        old=[dict(recording_id='old',target_index=2,target_id='car_horn',
                  purpose='d5_holdout',source_group='esc50:123',sha256='a')]
        gt=[dict(fname=i,labels=['Glass','Shatter'],mids=['a','b'],split='train') for i in ('123','124','125')]
        metadata={i:dict(uploader='old_author' if i!='125' else 'new_author',
                  license='https://creativecommons.org/publicdomain/zero/1.0/') for i in ('123','124','125')}
        config=dict(allowed_licenses={'https://creativecommons.org/publicdomain/zero/1.0/':'CC0-1.0'},
                    split_seed='test',validation_fraction=.2)
        selected,_,_=plan_fsd(gt,metadata,old,config)
        self.assertEqual([r['freesound_id'] for r in selected],['125'])
        self.assertEqual(select_legacy(old),[])

    def test_shifted_event_must_retain_energy_relative_to_positive_center(self):
        from scripts.host.train_soundjie_alert6 import source_windows
        signal=np.random.default_rng(13).normal(0,.001,32000)
        signal[16000:16064]+=.8*np.sin(np.arange(64)*.8)
        windows=source_windows(signal,dict(target_index=3,purpose='train'),1)
        center_rms=np.sqrt(np.mean(windows[0].samples**2))
        self.assertTrue(all(np.sqrt(np.mean(w.samples**2))>=.15*center_rms for w in windows))

    def test_excerpt_identity_is_separate_from_uploader_or_split_identity(self):
        from scripts.host.soundjie_alert6 import sampling_identity
        a=dict(source_id='esc50',source_group='esc50:180156',recording_id='excerpt-a')
        b=dict(a,recording_id='excerpt-b')
        self.assertEqual(sampling_identity(a),sampling_identity(b))
        c=dict(source_id='fsd50k',source_group='freesound-uploader:same',
               source_recording_id='freesound:123',recording_id='fsd50k:123')
        d=dict(c,source_recording_id='freesound:124',recording_id='fsd50k:124')
        self.assertNotEqual(sampling_identity(c),sampling_identity(d))


if __name__=='__main__':unittest.main()
