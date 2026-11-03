import importlib
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
DEPLOY = REPO / "models" / "soundjie_d3" / "deploy.prototxt"
PARITY_RUNNER = REPO / "scripts" / "guest" / "run_soundjie_d3_caffe_parity.sh"


def load_exporter():
    sys.path.insert(0, str(REPO))
    try:
        return importlib.import_module("scripts.guest.export_soundjie_d3_caffe")
    finally:
        sys.path.pop(0)


class SoundjieD3CaffeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exporter = load_exporter()

    def test_deploy_prototxt_has_only_supported_layers(self):
        text = DEPLOY.read_text(encoding="utf-8")
        self.assertIn('input: "data"', text)
        self.assertIn("dim: 64", text)
        self.assertIn("dim: 96", text)
        self.assertEqual(text.count('type: "Convolution"'), 4)
        self.assertEqual(text.count('type: "ReLU"'), 4)
        self.assertEqual(text.count('type: "Pooling"'), 4)
        self.assertEqual(text.count('type: "InnerProduct"'), 1)
        for forbidden in ("BatchNorm", 'type: "Scale"', "Softmax", "Python"):
            self.assertNotIn(forbidden, text)

    def test_container_parity_runner_uses_vitis_ai_caffe_environment(self):
        text = PARITY_RUNNER.read_text(encoding="utf-8")
        self.assertIn("conda activate vitis-ai-caffe", text)
        self.assertIn("export_soundjie_d3_caffe.py", text)
        self.assertIn("--output-dir", text)
        self.assertNotIn("bash -lc", text)

    def test_layer_names_map_one_to_one(self):
        self.assertEqual(
            self.exporter.EXPORTED_LAYERS,
            ("conv1", "conv2", "conv3", "conv4", "fc8"),
        )
        text = DEPLOY.read_text(encoding="utf-8")
        for name in self.exporter.EXPORTED_LAYERS:
            self.assertEqual(text.count(f'name: "{name}"'), 1)

    def test_rejects_missing_extra_or_wrong_shaped_weight(self):
        good = {
            key: np.zeros(shape, dtype=np.float32)
            for key, shape in self.exporter.EXPECTED_ARRAY_SHAPES.items()
        }
        self.exporter.validate_fused_arrays(good)
        for mutation in ("missing", "extra", "shape"):
            changed = dict(good)
            if mutation == "missing":
                changed.pop("conv1.weight")
            elif mutation == "extra":
                changed["surprise"] = np.zeros(1, dtype=np.float32)
            else:
                changed["fc8.weight"] = np.zeros((8, 1), dtype=np.float32)
            with self.assertRaisesRegex(self.exporter.ExportError, "weight contract"):
                self.exporter.validate_fused_arrays(changed)

    def _state_dict(self):
        channels = (16, 32, 64, 96)
        inputs = (1, 16, 32, 64)
        state = {}
        for index, (input_channels, output_channels) in enumerate(zip(inputs, channels), 1):
            state["conv{}.weight".format(index)] = torch.randn(
                output_channels, input_channels, 3, 3, dtype=torch.float32
            )
            prefix = "bn{}.".format(index)
            state[prefix + "weight"] = torch.randn(output_channels, dtype=torch.float32)
            state[prefix + "bias"] = torch.randn(output_channels, dtype=torch.float32)
            state[prefix + "running_mean"] = torch.randn(output_channels, dtype=torch.float32)
            state[prefix + "running_var"] = torch.rand(output_channels, dtype=torch.float32) + 0.5
            state[prefix + "num_batches_tracked"] = torch.tensor(7, dtype=torch.int64)
        state["fc8.weight"] = torch.randn(8, 2304, dtype=torch.float32)
        state["fc8.bias"] = torch.randn(8, dtype=torch.float32)
        return state

    def test_fuse_state_dict_rejects_missing_extra_and_wrong_shape(self):
        good = self._state_dict()
        self.exporter.fuse_state_dict(good)
        for mutation in ("missing", "extra", "shape"):
            changed = dict(good)
            if mutation == "missing":
                changed.pop("bn2.running_var")
            elif mutation == "extra":
                changed["unused.weight"] = torch.zeros(1)
            else:
                changed["conv3.weight"] = torch.zeros((1, 1, 1, 1))
            with self.assertRaisesRegex(self.exporter.ExportError, "state dict contract"):
                self.exporter.fuse_state_dict(changed)

    def test_batch_norm_fusion_matches_eval_equation(self):
        state = self._state_dict()
        fused = self.exporter.fuse_state_dict(state)
        index = 1
        x = torch.randn(2, 1, 9, 11)
        conv = torch.nn.Conv2d(1, 16, 3, padding=1, bias=False)
        bn = torch.nn.BatchNorm2d(16)
        conv.weight.data.copy_(state["conv1.weight"])
        bn.weight.data.copy_(state["bn1.weight"])
        bn.bias.data.copy_(state["bn1.bias"])
        bn.running_mean.copy_(state["bn1.running_mean"])
        bn.running_var.copy_(state["bn1.running_var"])
        bn.num_batches_tracked.copy_(state["bn1.num_batches_tracked"])
        conv.eval()
        bn.eval()
        fused_conv = torch.nn.Conv2d(1, 16, 3, padding=1, bias=True)
        fused_conv.weight.data.copy_(torch.from_numpy(fused["conv1.weight"]))
        fused_conv.bias.data.copy_(torch.from_numpy(fused["conv1.bias"]))
        with torch.no_grad():
            expected = bn(conv(x)).numpy()
            actual = fused_conv(x).numpy()
        self.assertLess(float(np.max(np.abs(expected - actual))), 2e-5)

    def test_export_weights_maps_each_blob_and_rejects_blob_drift(self):
        arrays = {
            key: np.full(shape, index + 1, dtype=np.float32)
            for index, (key, shape) in enumerate(self.exporter.EXPECTED_ARRAY_SHAPES.items())
        }

        class Blob:
            def __init__(self, shape):
                self.data = np.zeros(shape, dtype=np.float32)

        class Net:
            pass

        net = Net()
        net.params = {
            layer: [Blob(arrays[layer + ".weight"].shape), Blob(arrays[layer + ".bias"].shape)]
            for layer in self.exporter.EXPORTED_LAYERS
        }
        self.exporter.export_weights(arrays, net)
        for layer in self.exporter.EXPORTED_LAYERS:
            np.testing.assert_array_equal(net.params[layer][0].data, arrays[layer + ".weight"])
            np.testing.assert_array_equal(net.params[layer][1].data, arrays[layer + ".bias"])
        net.params["conv1"][0] = Blob((1,))
        with self.assertRaisesRegex(self.exporter.ExportError, "blob shape drift"):
            self.exporter.export_weights(arrays, net)

    def test_source_files_are_bound_to_checkpoint_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_config = root / "model.json"
            prepared = root / "prepared.jsonl"
            labels = root / "labels.txt"
            model_config.write_text("model\n", encoding="utf-8")
            prepared.write_text("prepared\n", encoding="utf-8")
            labels.write_text("a\nb\n", encoding="utf-8")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            upstream = {
                "model_config_sha256": digest(model_config),
                "prepared_manifest_sha256": digest(prepared),
                "labels_sha256": digest(labels),
                "class_map_sha256": "a" * 64,
                "frontend_sha256": "b" * 64,
            }
            loaded = SimpleNamespace(class_map_sha256="a" * 64, frontend_sha256="b" * 64)
            self.exporter.validate_source_files(
                upstream, model_config, prepared, labels, loaded
            )
            prepared.write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(self.exporter.ExportError, "prepared manifest SHA drift"):
                self.exporter.validate_source_files(
                    upstream, model_config, prepared, labels, loaded
                )

    def test_create_only_refuses_to_replace_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "artifact.json"
            destination.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(self.exporter.ExportError, "artifact exists"):
                self.exporter._write_json(destination, {"replacement": True})
            self.assertEqual(destination.read_text(encoding="utf-8"), "original")

    def test_caffe_command_hashes_published_logits(self):
        exporter = self.exporter
        arrays = {
            key: np.zeros(shape, dtype=np.float32)
            for key, shape in self.exporter.EXPECTED_ARRAY_SHAPES.items()
        }

        class Blob:
            def __init__(self, shape):
                self.data = np.zeros(shape, dtype=np.float32)

        class FakeNet:
            def __init__(self, _deploy, _phase):
                self.params = {
                    layer: [Blob(arrays[layer + ".weight"].shape), Blob(arrays[layer + ".bias"].shape)]
                    for layer in exporter.EXPORTED_LAYERS
                }
                self.blobs = {"data": Blob((1, 1, 64, 96)), "fc8": Blob((1, 8))}

            def forward(self):
                self.blobs["fc8"].data[...] = 0.0

            def save(self, path):
                Path(path).write_bytes(b"fake-caffemodel")

        fake_caffe = SimpleNamespace(TEST=object(), set_mode_cpu=lambda: None, Net=FakeNet)
        previous = sys.modules.get("caffe")
        sys.modules["caffe"] = fake_caffe
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                weights = root / "weights.npz"
                inputs = root / "inputs.npy"
                reference = root / "reference.npy"
                deploy = root / "deploy.prototxt"
                output = root / "output"
                np.savez(weights, **arrays)
                np.save(inputs, np.zeros((16, 1, 64, 96), dtype=np.float32))
                np.save(reference, np.zeros((16, 8), dtype=np.float32))
                deploy.write_text("fake", encoding="utf-8")
                self.exporter.command_caffe(SimpleNamespace(
                    weights=weights, deploy=deploy, inputs=inputs,
                    reference_logits=reference, output_dir=output,
                ))
                manifest = json.loads(
                    (output / "caffe-export-manifest.json").read_text(encoding="utf-8")
                )
                digest = hashlib.sha256((output / "caffe-logits.npy").read_bytes()).hexdigest()
                self.assertEqual(manifest["caffe_logits_sha256"], digest)
                self.assertEqual(len(manifest["per_sample_max_abs_error"]), 16)
                self.assertEqual(manifest["argmax_matches"], 16)
        finally:
            if previous is None:
                sys.modules.pop("caffe", None)
            else:
                sys.modules["caffe"] = previous

    def test_rejects_checkpoint_contract_drift(self):
        upstream = {"class_map_sha256": "a" * 64, "frontend_sha256": "b" * 64}
        checkpoint = {"schema": 1, "labels": [str(i) for i in range(8)], "upstream": upstream}
        manifest = {"schema": 1, "upstream": dict(upstream)}
        self.exporter.validate_checkpoint_contract(checkpoint, manifest, [str(i) for i in range(8)])
        manifest["upstream"]["frontend_sha256"] = "c" * 64
        with self.assertRaisesRegex(self.exporter.ExportError, "contract drift"):
            self.exporter.validate_checkpoint_contract(checkpoint, manifest, [str(i) for i in range(8)])

    def test_parity_threshold_is_strictly_below_1e_4(self):
        reference = np.zeros((16, 8), dtype=np.float32)
        below = reference.copy()
        below[0, 0] = np.nextafter(np.float32(1e-4), np.float32(0.0))
        report = self.exporter.assess_parity(reference, below)
        self.assertLess(report["max_abs_error"], 1e-4)
        at_limit = reference.copy()
        at_limit[0, 0] = np.float32(1e-4)
        with self.assertRaisesRegex(self.exporter.ExportError, "parity"):
            self.exporter.assess_parity(reference, at_limit)
        argmax_drift = reference.copy()
        argmax_drift[:, 0] = np.float32(1e-6)
        argmax_drift[0, 1] = np.float32(2e-6)
        with self.assertRaisesRegex(self.exporter.ExportError, "argmax=15/16"):
            self.exporter.assess_parity(reference, argmax_drift)


if __name__ == "__main__":
    unittest.main()
