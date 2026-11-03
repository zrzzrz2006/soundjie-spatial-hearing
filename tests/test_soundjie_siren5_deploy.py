import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.host import deploy_soundjie_siren5 as d


class Siren5DeploymentTests(unittest.TestCase):
    def test_six_parity_contract_and_nonfinite(self):
        x = np.arange(72, dtype=np.float32).reshape(12, 6)
        self.assertEqual(d.parity(x, x)['top1_matches'], 12)
        for bad in (np.zeros((12, 5)), np.full((12, 6), np.nan), x + 0.1):
            with self.assertRaises(ValueError): d.parity(x, bad)

    def test_calibration_is_192_distinct_train_originals(self):
        x = np.zeros((192, 1, 64, 96), np.float32)
        rows = [dict(purpose='train', derivation=None, sampling_identity=str(i),
                     target_index=i//32, feature_sha256=d.sha_bytes(x[i].tobytes())) for i in range(192)]
        d.check_calibration(x, rows)
        for change in ({'purpose':'validation'}, {'sampling_identity':'1'}, {'target_index':5}, {'feature_sha256':'bad'}):
            broken = copy.deepcopy(rows); broken[0].update(change)
            with self.assertRaises(ValueError): d.check_calibration(x, broken)

    def test_hash_and_paths_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'a').write_bytes(b'ok')
            d.check_files(root, {'a':d.sha_bytes(b'ok')})
            for pins in ({'a':'bad'}, {'../a':d.sha_bytes(b'ok')}):
                with self.assertRaises(ValueError): d.check_files(root,pins)

    def test_merge_after_softmax_and_thresholds(self):
        logits=np.log([[.27,.26,.4,.02,.02,.03],[.1,.1,.6,.05,.05,.1]])
        p=d.five_scores(logits)
        np.testing.assert_allclose(p[0],[.53,.4,.02,.02,.03])
        np.testing.assert_array_equal(d.decide(p),[0,4])
        with self.assertRaises(ValueError): d.five_scores(np.zeros((2,8)))

    def test_shapes_reject_wrong_fc_and_nan(self):
        a={k:np.zeros(s,np.float32) for k,s in d.SHAPES.items()}
        d.check_weights(a)
        a['fc8.weight']=np.zeros((8,2304),np.float32)
        with self.assertRaises(ValueError): d.check_weights(a)

if __name__=='__main__': unittest.main()
