"""Training boundary checks; no D5 files or device actions."""
import unittest
import numpy as np
from scripts.host import train_soundjie_alert4bg as t
from scripts.host import export_soundjie_alert4bg as e
from scripts.host import soundjie_alert4bg as a


class TrainingTests(unittest.TestCase):
    def test_sub_lsb_augmented_train_audio_is_excluded_not_fabricated(self):
        from scripts.host.logmel_reference import load_frontend_config
        frontend=load_frontend_config(a.ROOT/'config/logmel_frontend_v1.json')
        quiet=np.sin(np.arange(16000)*.1)*8e-6
        self.assertIsNone(t.quantized_feature(quiet,frontend,{'seed':7},'train'))
        with self.assertRaises(ValueError):t.quantized_feature(quiet,frontend,None,'validation')
        with self.assertRaises(ValueError):t.quantized_feature(quiet,frontend,{'seed':7},'validation')
        audible=t.quantized_feature(quiet*10000,frontend,{'seed':7},'train')
        self.assertEqual(audible.shape,(1,64,96))

    def test_validation_is_one_unaugmented_window_and_holdout_rejected(self):
        audio=np.sin(np.arange(40000)*.1)
        self.assertEqual(len(t.source_windows(audio,dict(purpose='validation',target_index=3),7)),1)
        with self.assertRaises(ValueError):t.source_windows(audio,dict(purpose='d5_holdout',target_index=3),7)

    def test_feature_array_validation_catches_replay_leakage(self):
        import hashlib
        x=np.zeros((2,1,64,96),dtype=np.float32);y=np.array([3,4]);p=np.array(['train','validation'])
        rows=[dict(recording_id=str(i),source_group=str(i),sampling_identity=str(i),
            parent_sha256=str(i),purpose=str(p[i]),target_index=int(y[i]),uploader_group=str(i),
            derivation=None,feature_sha256=hashlib.sha256(x[i].tobytes()).hexdigest()) for i in range(2)]
        t.validate_cache_arrays(x,y,p,rows)
        rows[1]['sampling_identity']=rows[0]['sampling_identity']
        with self.assertRaisesRegex(ValueError,'cross'):t.validate_cache_arrays(x,y,p,rows)
        rows[1]['sampling_identity']='1';rows[1]['derivation']={'seed':7}
        with self.assertRaisesRegex(ValueError,'validation'):t.validate_cache_arrays(x,y,p,rows)

    def test_calibration_train_only_distinct_originals(self):
        rows=[]
        for c in range(5):
            for i in range(4):
                rows.append(dict(purpose='train' if i<3 else 'validation',target_index=c,
                                 sampling_identity=f'{c}/{i//2}',derivation=None))
        idx=e.calibration_indices(rows,2)
        self.assertEqual(len(idx),10)
        self.assertTrue(all(rows[i]['purpose']=='train' for i in idx))
        self.assertEqual(len({rows[i]['sampling_identity'] for i in idx}),10)
        with self.assertRaises(ValueError):e.calibration_indices(rows,3)

    def test_folding_preserves_real_model_logits(self):
        import torch
        torch.set_num_threads(4);torch.manual_seed(1)
        model=a.make_model().eval();x=np.random.default_rng(2).normal(size=(3,1,64,96)).astype('float32')
        arrays,report=e.fold_and_check(model,x)
        self.assertEqual(arrays['fc8.weight'].shape,(5,2304))
        self.assertLess(report['max_absolute_error'],1e-4)
        self.assertTrue(report['top1_identical'])

    def test_metrics_include_dog_and_glass_background_fp(self):
        y=np.array([0,1,2,3,4,4]);p=np.array([0,1,2,4,3,4])
        rows=[dict(target_index=int(c),negative_subtype='glass',sampling_identity=str(i),source_group=str(i)) for i,c in enumerate(y)]
        m=a.metrics(y,p,rows)
        self.assertEqual(m['dog_recall'],0.)
        self.assertEqual(m['negative_false_alert_rate'],.5)
        self.assertEqual(m['negative_subtypes']['glass']['false_alerts'],1)


if __name__=='__main__':unittest.main()
