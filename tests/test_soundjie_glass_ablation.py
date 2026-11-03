import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from scripts.host.ablate_soundjie_glass import (
    map_without_glass, common_metrics, paired_models, verify_pins, selection_key,
)


class GlassAblationTests(unittest.TestCase):
    def test_removed_top1_is_background_without_promoting_runner_up(self):
        scores=np.array([[.05,.3,.1,.4,.1,.05]])
        self.assertEqual(map_without_glass(scores.argmax(1)).tolist(),[4])
        self.assertEqual(map_without_glass(np.arange(6)).tolist(),[0,1,2,4,3,4])
        for bad in [np.array([6]),np.array([-1]),np.array([1.5])]:
            with self.assertRaises(ValueError):map_without_glass(bad)

    def test_glass_exclusion_does_not_change_original_background_denominator(self):
        y=np.array([0,1,2,4,5,5,3,3,3])
        pred=np.array([0,1,2,4,5,3,3,0,5])
        report=common_metrics(y,pred,6)
        self.assertEqual(report['support'],6)
        self.assertEqual(report['per_class']['background_unknown']['support'],2)
        self.assertEqual(report['negative_false_alert_rate'],0)
        self.assertEqual(report['glass_as_negative']['support'],3)
        self.assertAlmostEqual(report['glass_as_negative']['false_alert_rate'],1/3)
        mapped=common_metrics(y,map_without_glass(pred),5)
        self.assertEqual(report,mapped)

    def test_paired_initial_parameters_identical_except_removed_head_row(self):
        six,five=paired_models(12)
        keep=[0,1,2,4,5]
        for key,value in six.state_dict().items():
            expected=value[keep] if key.startswith('fc8.') else value
            torch.testing.assert_close(five.state_dict()[key],expected,atol=0,rtol=0)
        self.assertEqual(five.fc8.out_features,5)

    def test_hash_drift_fails_closed(self):
        import hashlib
        with tempfile.TemporaryDirectory(dir=Path('artifacts')) as folder:
            path=Path(folder)/'source';path.write_bytes(b'approved')
            pins={str(path):hashlib.sha256(b'approved').hexdigest()}
            verify_pins(pins)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_pins(pins)

    def test_selection_ties_prefer_lower_false_alert_then_earlier_epoch(self):
        def r(e,macro,fpr):return dict(epoch=e,metrics=dict(macro_recall=macro,negative_false_alert_rate=fpr))
        options=[r(1,.7,.1),r(2,.7,.08),r(3,.7,.08),r(4,.69,0)]
        self.assertEqual(max(options,key=selection_key)['epoch'],2)


if __name__=='__main__':unittest.main()
