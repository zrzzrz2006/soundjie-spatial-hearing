import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.host.calibrate_soundjie_tv_acoustic import (
    blended_reference, load_network, read_reference,
)
from scripts.host.train_soundjie_traffic import export_model


class TvAcousticCalibrationTests(unittest.TestCase):
    def test_reference_rejects_wrong_shape_truncation_and_nonfinite_values(self):
        valid = b'SJTVF2\0\0' + struct.pack('<II', 297, 32)
        payload = np.zeros((297, 32), dtype='<f4')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reference.features'
            for data in (valid, b'SJTVF2\0\0' + struct.pack('<II', 297, 31) + payload.tobytes()):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read_reference(path)
            payload[0, 0] = np.nan
            path.write_bytes(valid + payload.tobytes())
            with self.assertRaises(ValueError):
                read_reference(path)

    def test_enrollment_rejects_unaligned_or_flat_reference(self):
        x = np.random.default_rng(7).normal(size=(297, 32))
        with self.assertRaises(ValueError):
            blended_reference(x, x[:-1])
        with self.assertRaises(ValueError):
            blended_reference(x, np.zeros_like(x))

    def test_enrollment_tolerates_channel_bias_and_positive_gain(self):
        x = np.random.default_rng(17).normal(size=(297, 32))
        y = 0.3 * x + np.linspace(-3, 3, 32)
        np.testing.assert_allclose(blended_reference(x, y), x - x.mean(0), atol=2e-7)

    def test_binary_loader_preserves_model_outputs(self):
        torch.manual_seed(31)
        model = torch.nn.Sequential(torch.nn.Linear(96, 48), torch.nn.ReLU(), torch.nn.Linear(48, 3))
        mean = np.linspace(-1, 1, 96).astype('f4')
        scale = np.linspace(.3, 2, 96).astype('f4')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'traffic.bin'
            export_model(path, mean, scale, model)
            got_mean, got_scale, got_model = load_network(path)
            np.testing.assert_array_equal(got_mean, mean)
            np.testing.assert_array_equal(got_scale, scale)
            x = torch.randn(8, 96)
            with torch.no_grad():
                torch.testing.assert_close(got_model(x), model(x), rtol=0, atol=0)

    def test_binary_loader_rejects_invalid_scale_and_truncated_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'traffic.bin'
            for data in (b'SJTF1\0\0\0', b'SJTF1\0\0\0' + struct.pack('<III', 96, 48, 3) + bytes(4995 * 4)):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    load_network(path)


if __name__ == '__main__':
    unittest.main()
