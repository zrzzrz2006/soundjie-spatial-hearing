import unittest
import numpy as np
import torch
from scripts.host.soundjie_siren_merge import (
    map_targets,merge_probabilities,paired_models,metrics,apply_thresholds,promotion_decision,
)


class SirenMergeTests(unittest.TestCase):
    def test_alarm_siren_merge_retains_glass_and_background(self):
        self.assertEqual(map_targets(np.arange(6)).tolist(),[0,0,1,2,3,4])
        for values in [np.array([-1]),np.array([6]),np.array([1.5])]:
            with self.assertRaises(ValueError):map_targets(values)

    def test_grouped_probability_conserves_mass_and_can_differ_from_rename(self):
        scores=np.array([[.30,.29,.02,.02,.01,.36]])
        merged=merge_probabilities(scores)
        np.testing.assert_allclose(merged,[[.59,.02,.02,.01,.36]])
        np.testing.assert_allclose(merged.sum(1),1)
        self.assertEqual(int(merged.argmax(1)[0]),0)
        self.assertEqual(map_targets(scores.argmax(1)).tolist(),[4])
        for bad in [np.array([[.5]*6]),np.full((1,6),np.nan),np.array([[-.1,.1,.2,.2,.2,.4]])]:
            with self.assertRaises(ValueError):merge_probabilities(bad)

    def test_counts_show_old_siren_failure_despite_merged_majority(self):
        y=np.array([0]*52+[1]*20+[2]*13+[3]*35+[4]*43+[5]*193)
        pred=map_targets(y);pred[52:72]=4
        m=metrics(y,pred)
        self.assertEqual(m['support'],356)
        self.assertEqual(m['per_class']['siren']['support'],72)
        self.assertEqual(m['per_class']['glass_breaking']['support'],35)
        self.assertAlmostEqual(m['per_class']['siren']['recall'],52/72)
        self.assertEqual(m['original_family']['emergency_siren']['recall'],0)
        self.assertEqual(m['negative_false_alert_rate'],0)

    def test_rejection_never_promotes_runnerup(self):
        scores=np.array([[.45,.40,.05,.05,.05],[.10,.10,.10,.10,.60]])
        self.assertEqual(apply_thresholds(scores,[.5,.3,.3,.3]).tolist(),[4,4])

    def test_shared_initial_trunk_and_retained_head_rows(self):
        six,five=paired_models(73)
        for key,value in six.state_dict().items():
            expected=value[[0,2,3,4,5]] if key.startswith('fc8.') else value
            torch.testing.assert_close(expected,five.state_dict()[key],rtol=0,atol=0)
        self.assertEqual(five.fc8.out_features,5)

    def test_guard_inclusive_rate_boundary_and_siren_family_regression(self):
        guards=dict(minimum_each_macro_gain=-.01,minimum_mean_macro_gain=.01,
                    max_background_fpr_increase=.02,max_positive_recall_decrease=.1,
                    max_siren_recall_decrease=.05,max_voice_extra_false_alerts=1)
        def arm(macro,bg,siren):
            return dict(metrics={'macro_recall':macro,'negative_false_alert_rate':bg,
                'per_class':{k:{'recall':.8} for k in ['siren','car_horn','glass_breaking','door_knock_bell']},
                'original_family':{'emergency_siren':{'recall':siren}},
                'negative_subtypes':{'ordinary_speech':{'false_alerts':1},'crying_screaming':{'false_alerts':4}}},
                replay={'validation':{'acoustic':{'matches':5}}})
        rows=[dict(seed=1,control6=arm(.70,.3,16/20),merged5=arm(.72,.32,15/20))]
        self.assertTrue(promotion_decision(rows,guards)['eligible_for_deployment_preparation'])
        rows[0]['merged5']['metrics']['original_family']['emergency_siren']['recall']=14/20
        self.assertFalse(promotion_decision(rows,guards)['eligible_for_deployment_preparation'])


if __name__=='__main__':unittest.main()
