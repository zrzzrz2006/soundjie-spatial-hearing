"""Contract, source mapping and leakage regressions for the speech sidecar."""
import copy
import hashlib
import unittest
import numpy as np
import torch
from scripts.host import soundjie_alert5bg as sj
from scripts.host.export_soundjie_alert5bg import calibration_indices
from scripts.host.train_soundjie_alert5bg import validate_cache_arrays


def source(index=4, subtype='ordinary_speech', purpose='train', rid='speech1'):
    return dict(recording_id=rid, target_index=index,
                target_id=['siren','car_horn','doorbell','dog_bark','background_unknown'][index],
                negative_subtype=subtype, class_contract='soundjie-alert4bg-v2',
                purpose=purpose, source_group=rid, sampling_identity=rid,
                sha256=rid, source_id='librispeech_dev_clean', original_labels=['speech'])


class Alert5bgTests(unittest.TestCase):
    def test_ordinary_speech_becomes_positive_without_split_change(self):
        r=source(); before=copy.deepcopy(r); mapped=sj.map_existing(r)
        self.assertEqual(r,before)
        self.assertEqual((mapped['target_index'],mapped['target_id']), (4,'speech'))
        for k in ('purpose','source_group','sampling_identity','sha256','recording_id'):
            self.assertEqual(mapped[k],r[k])
        self.assertEqual(mapped['class_contract'],'soundjie-alert5bg-v3')

    def test_crying_never_speech_or_noise(self):
        r=sj.map_existing(source(subtype='crying_screaming'))
        self.assertEqual(r['target_index'],5)
        self.assertFalse(sj.noise_eligible(r))

    def test_old_alerts_keep_positions(self):
        for c in range(4):
            r=sj.map_existing(source(c,['siren','horn','doorbell','bark'][c]))
            self.assertEqual(r['target_index'],c)

    def test_upstream_contract_is_required(self):
        r=source(); r['class_contract']='old-six-unrelated'
        with self.assertRaises(ValueError): sj.map_existing(r)

    def test_voice_is_never_noise(self):
        for subtype in ('ordinary_speech','crying_screaming'):
            self.assertFalse(sj.noise_eligible(sj.map_existing(source(subtype=subtype))))
        r=sj.map_existing(source(subtype='ambient'))
        self.assertTrue(sj.noise_eligible(r))
        r['purpose']='validation'; self.assertFalse(sj.noise_eligible(r))

    def test_shape_and_order(self):
        torch.set_num_threads(4)
        self.assertEqual(sj.LABELS,('siren','car_horn','doorbell','dog_bark','speech','background_unknown'))
        self.assertEqual(sj.output_contract()['background_index'],5)
        self.assertEqual(tuple(sj.make_model().eval()(torch.zeros(2,1,64,96)).shape),(2,6))

    def test_reject_winner_without_promoting_runner_up(self):
        scores=np.array([[.05,.05,.05,.05,.5,.3],[.4,.35,.05,.05,.05,.1]])
        np.testing.assert_array_equal(sj.apply_thresholds(scores,[.5,.2,.5,.5,.6]),[5,5])
        with self.assertRaises(ValueError): sj.apply_thresholds(scores,[.5]*4)

    def test_speech_metrics_not_counted_as_background(self):
        m=sj.metrics(np.array([4,5,5]),np.array([4,4,5]))
        self.assertEqual(m['per_class']['speech']['recall'],1.)
        self.assertEqual(m['negative_false_alert_rate'],.5)

    def test_speech_split_overlap_rejected(self):
        a=sj.map_existing(source()); b=copy.deepcopy(a)
        b.update(recording_id='different-file',purpose='validation')
        with self.assertRaises(ValueError): sj.validate_rows([a,b],require_all=False)

    def test_balanced_sampling_six_classes(self):
        y=np.arange(6); ix=sj.balanced_source_indices(y,np.array(['s'+str(i) for i in y]),5,7)
        np.testing.assert_array_equal(np.bincount(y[ix]),[5]*6)

    def test_calibration_distinct_unaugmented_training_only(self):
        rows=[dict(purpose='train',target_index=c,derivation=None,sampling_identity=str(c)+'-'+str(j)) for c in range(6) for j in range(32)]
        self.assertEqual(len(calibration_indices(rows)),192)
        rows[-1]['purpose']='validation'
        with self.assertRaises(ValueError): calibration_indices(rows)

    def test_cache_rejects_speech_noise_even_when_train(self):
        x=np.zeros((2,1,64,96),np.float32); y=np.array([4,4]); purposes=np.array(['train','validation'])
        rows=[]
        for i,p in enumerate(purposes):
            rows.append(dict(recording_id=str(i),source_group=str(i),sampling_identity=str(i),parent_sha256=str(i),
                purpose=str(p),target_index=4,negative_subtype='ordinary_speech',feature_sha256=hashlib.sha256(x[i].tobytes()).hexdigest(),
                derivation=dict(background_purpose='train',background_target_index=4,background_negative_subtype='ordinary_speech') if i==0 else None))
        with self.assertRaises(ValueError): validate_cache_arrays(x,y,purposes,rows)


if __name__=='__main__': unittest.main()
