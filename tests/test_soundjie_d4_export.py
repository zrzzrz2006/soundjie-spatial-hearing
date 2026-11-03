import hashlib
import importlib
import unittest

import numpy as np


class D4ExportTests(unittest.TestCase):
    def module(self):
        from importlib.util import find_spec
        self.assertIsNotNone(find_spec('scripts.host.export_soundjie_d4'),
                             'D4 export adapter must exist')
        return importlib.import_module('scripts.host.export_soundjie_d4')

    def fixture(self):
        x = np.zeros((24, 1, 64, 96), dtype=np.float32)
        y = np.tile(np.arange(8, dtype=np.int64), 3)
        purpose = np.asarray(['train'] * 8 + ['validation'] * 16)
        provenance = []
        for i in range(24):
            x[i].fill(i)
            provenance.append(dict(purpose=str(purpose[i]), target_index=int(y[i]),
                recording_id=str(i), parent_sha256=str(i), source_group=str(i),
                feature_sha256=hashlib.sha256(x[i].tobytes()).hexdigest()))
        return x, y, purpose, provenance

    def test_selects_two_validation_sources_per_class_not_training(self):
        mod = self.module()
        x, y, purpose, provenance = self.fixture()
        self.assertEqual(mod.select_parity(x, y, purpose, provenance),
                         [8, 16, 9, 17, 10, 18, 11, 19, 12, 20, 13, 21, 14, 22, 15, 23])

    def test_rejects_holdout_and_feature_or_label_drift(self):
        mod = self.module()
        for mutation in ('holdout', 'feature', 'target', 'nan', 'missing'):
            x, y, purpose, provenance = self.fixture()
            if mutation == 'holdout':
                purpose[0] = 'd5_holdout'
            elif mutation == 'feature':
                x[8, 0, 0, 0] += 1
            elif mutation == 'target':
                provenance[8]['target_index'] = 7
            elif mutation == 'nan':
                x[8, 0, 0, 0] = np.nan
            else:
                provenance.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                mod.select_parity(x, y, purpose, provenance)

    def test_rejects_repeated_validation_source_or_train_leak(self):
        mod = self.module()
        for first in (0, 8):
            x, y, purpose, provenance = self.fixture()
            provenance[16]['source_group'] = provenance[first]['source_group']
            with self.subTest(first=first), self.assertRaises(ValueError):
                mod.select_parity(x, y, purpose, provenance)

    def test_source_binding_fails_closed(self):
        import tempfile
        from pathlib import Path
        mod = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input'
            path.write_bytes(b'abc')
            mod.check_hash(path, hashlib.sha256(b'abc').hexdigest())
            with self.assertRaises(ValueError):
                mod.check_hash(path, '0' * 64)


if __name__ == '__main__':
    unittest.main()
