import importlib
import unittest
import numpy as np

class TrafficModelTests(unittest.TestCase):
    def setUp(self):
        self.m=importlib.import_module('scripts.host.train_soundjie_traffic')

    def test_features_are_bounded_and_level_independent(self):
        x=np.random.default_rng(51).normal(0,.1,16000)
        a=self.m.feature(x);b=self.m.feature(x*.2)
        self.assertEqual(a.shape,(96,))
        self.assertTrue(np.isfinite(self.m.feature(np.zeros(16000))).all())
        np.testing.assert_allclose(a,b,atol=.03)

    def test_masked_unknown_labels_do_not_contribute_to_loss(self):
        import torch
        logits=torch.tensor([[0.,20.,-20.]])
        y=torch.tensor([[1.,0.,1.]]);mask=torch.tensor([[1.,0.,0.]])
        self.assertAlmostEqual(float(self.m.masked_loss(logits,y,mask)),np.log(2),places=6)

    def test_demo_cannot_be_counted_as_unseen_validation(self):
        rows=[dict(source_group='a',purpose='train'),dict(source_group='a',purpose='validation')]
        with self.assertRaises(ValueError):self.m.audit_groups(rows)
        with self.assertRaises(ValueError):self.m.audit_groups([dict(source_group='demo:clip',purpose='validation')])

    def test_demo_labels_explicitly_preserve_user_scene_uncertainty(self):
        policy=self.m.demo_label_policy()
        self.assertEqual(policy['basis'],'user_scene_weak_not_source_verified')
        self.assertFalse(policy['siren_source_independently_verified'])
        self.assertFalse(policy['framewise_audition_performed'])

if __name__=='__main__':unittest.main()
