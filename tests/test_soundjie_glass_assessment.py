import unittest
from scripts.host.assess_soundjie_glass_ablation import replay_summary,promotion_decision
from scripts.host.ablate_soundjie_glass import LABELS5


class AssessmentTests(unittest.TestCase):
    def test_inclusive_guard_accepts_exact_count_boundary_not_real_regression(self):
        guards=dict(mean_macro_gain=.02,max_background_fpr_increase=.02,
                    max_positive_class_recall_decrease=.1,at_least_one_seed_replay_gain=1)
        def arm(macro,fpr,siren,replay):
            per={name:dict(recall=.7) for name in LABELS5}
            per['emergency_siren']['recall']=siren
            return dict(metrics=dict(macro_recall=macro,negative_false_alert_rate=fpr,per_class=per),
                        replay={'validation':{'acoustic':{'matches':replay}}})
        rows=[dict(seed=1,retain6=arm(.70,.3,16/20,3),drop5=arm(.73,.32,14/20,4))]
        self.assertTrue(promotion_decision(rows,guards)['eligible_for_deployment_preparation'])
        rows[0]['drop5']['metrics']['per_class']['emergency_siren']['recall']=13/20
        self.assertFalse(promotion_decision(rows,guards)['eligible_for_deployment_preparation'])

    def test_glass_and_train_do_not_inflate_replay_validation(self):
        clips=[dict(purpose=p,target_index=t,acoustic=dict(mapped_top1=v),digital=dict(mapped_top1=v))
               for p,t,v in [('train',0,0),('validation',0,4),('validation',5,4),('validation',3,4)]]
        report=replay_summary(clips)
        self.assertEqual(report['validation']['acoustic']['clips'],2)
        self.assertEqual(report['validation']['acoustic']['matches'],1)
        self.assertEqual(report['validation']['acoustic']['background_clips'],1)
        self.assertEqual(report['glass']['validation']['acoustic']['clips'],1)

    def test_one_good_seed_cannot_mask_regression_or_replay_loss(self):
        guards=dict(mean_macro_gain=.02,max_background_fpr_increase=.02,
                    max_positive_class_recall_decrease=.1,at_least_one_seed_replay_gain=1)
        def arm(macro,fpr,replay):
            return dict(metrics=dict(macro_recall=macro,negative_false_alert_rate=fpr,
                        per_class={name:dict(recall=.7) for name in LABELS5}),
                        replay={'validation':{'acoustic':{'matches':replay}}})
        rows=[dict(seed=1,retain6=arm(.70,.1,3),drop5=arm(.73,.09,4)),
              dict(seed=2,retain6=arm(.72,.1,3),drop5=arm(.75,.09,3))]
        self.assertTrue(promotion_decision(rows,guards)['eligible_for_deployment_preparation'])
        rows[1]['drop5']['replay']['validation']['acoustic']['matches']=2
        self.assertFalse(promotion_decision(rows,guards)['eligible_for_deployment_preparation'])
        rows[1]['drop5']=arm(.71,.09,4)
        self.assertFalse(promotion_decision(rows,guards)['eligible_for_deployment_preparation'])


if __name__=='__main__':unittest.main()
