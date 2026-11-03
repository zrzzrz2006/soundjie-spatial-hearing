import unittest
import numpy as np
import torch
from scripts.host.soundjie_siren_merge import paired_models
from scripts.host.assess_soundjie_siren_merge import replay_summary
from scripts.host.export_soundjie_siren_merge import calibration_indices,fold_and_check


class SirenAssessmentTests(unittest.TestCase):
    def test_six_physical_outputs_remain_distinct_from_five_display_classes(self):
        from scripts.host.export_soundjie_siren_merge import output_contract
        c=output_contract(6)
        self.assertEqual(c['physical_output_count'],6)
        self.assertEqual(len(c['physical_labels']),6)
        self.assertEqual(c['probability_merge_groups'],[[0,1],[2],[3],[4],[5]])
        self.assertEqual(len(c['display_labels']),5)
        self.assertEqual(c['merge_stage'],'after_softmax_never_sum_logits')
        with self.assertRaises(ValueError):output_contract(8)

    def test_replay_keeps_glass_separates_sources_and_uses_merged_truth(self):
        def row(p,t,pred):
            value=dict(top1=pred,thresholded_windows=[pred])
            return dict(purpose=p,target_index=t,digital=value,acoustic=value)
        rows=[row('train',0,0),row('validation',1,0),row('validation',3,2),row('validation',5,4)]
        report=replay_summary(rows)
        self.assertEqual(report['train']['acoustic']['clips'],1)
        self.assertEqual(report['validation']['acoustic']['clips'],3)
        self.assertEqual(report['validation']['acoustic']['matches'],3)
        self.assertEqual(report['validation']['acoustic']['positive_clips'],2)

    def test_calibration_uses_train_only_unique_originals(self):
        rows=[]
        for c in range(6):
            rows.append(dict(purpose='validation',target_index=c,derivation=None,sampling_identity='heldout'+str(c)))
            for n in range(33):
                rows.append(dict(purpose='train',target_index=c,derivation=None,sampling_identity=f'{c}/{n}'))
                rows.append(dict(purpose='train',target_index=c,derivation=None,sampling_identity=f'{c}/{n}'))
        indices=calibration_indices(rows)
        self.assertEqual(len(indices),192)
        self.assertEqual(len({rows[i]['sampling_identity'] for i in indices}),192)
        self.assertTrue(all(rows[i]['purpose']=='train' for i in indices))
        np.testing.assert_array_equal(np.bincount([rows[i]['target_index'] for i in indices]),[32]*6)

    def test_five_output_folding_preserves_logits(self):
        _,model=paired_models(15);torch.set_num_threads(2)
        x=np.random.default_rng(4).normal(size=(3,1,64,96)).astype(np.float32)
        arrays,report=fold_and_check(model,x)
        self.assertEqual(arrays['fc8.weight'].shape,(5,2304))
        self.assertTrue(report['top1_identical'])
        self.assertLess(report['max_absolute_error'],1e-4)


if __name__=='__main__':unittest.main()
