import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "sound_shape_probe.json"
DEPLOY = ROOT / "models" / "sound_shape_probe" / "deploy.prototxt"
FLOAT = ROOT / "models" / "sound_shape_probe" / "float.prototxt"
LABELS = ROOT / "models" / "sound_shape_probe" / "labels.txt"
AUDITOR = ROOT / "scripts" / "host" / "audit_sound_shape_compile.py"
GENERATOR = ROOT / "scripts" / "guest" / "generate_sound_shape_probe.py"
BUILDER = ROOT / "scripts" / "guest" / "build_sound_shape_probe.sh"

EXPECTED_LABELS = [
    "fire_smoke_alarm",
    "emergency_siren",
    "car_horn",
    "glass_break",
    "knock_doorbell",
    "distress_scream",
    "speech",
    "background_unknown",
]

_AUDITOR_MODULE = None


def load_auditor():
    global _AUDITOR_MODULE
    if _AUDITOR_MODULE is not None:
        return _AUDITOR_MODULE
    if not AUDITOR.is_file():
        raise AssertionError(f"missing compile auditor: {AUDITOR}")
    spec = importlib.util.spec_from_file_location("audit_sound_shape_compile", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _AUDITOR_MODULE = module
    return _AUDITOR_MODULE


class SoundShapeModelContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_schema_shape_nodes_target_and_labels(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(config["schema"], 1)
        self.assertEqual(config["kernel"], "sound_shape_probe")
        self.assertEqual(
            config["input"],
            {"name": "conv1", "blob": "data", "shape": [1, 1, 64, 96]},
        )
        self.assertEqual(config["output"], {"name": "logits", "count": 8})
        self.assertEqual(
            config["compiler"],
            {
                "dpu": "1152FA",
                "cpu_arch": "arm32",
                "abi": 1,
                "dpu_target": "v1.4.0",
                "accepted_dnnc_versions": ["v2.04", "v2.05"],
            },
        )
        self.assertEqual(config["labels"], EXPECTED_LABELS)
        self.assertEqual(LABELS.read_text(encoding="utf-8").splitlines(), EXPECTED_LABELS)

    def test_deploy_has_four_minimal_supported_blocks_and_fc8_without_softmax(self):
        text = DEPLOY.read_text(encoding="utf-8")
        types = []
        for chunk in text.split("layer {")[1:]:
            marker = 'type: "'
            start = chunk.index(marker) + len(marker)
            types.append(chunk[start:chunk.index('"', start)])
        expected = []
        for _ in range(4):
            expected += ["Convolution", "ReLU", "Pooling"]
        expected += ["InnerProduct"]
        self.assertEqual(types, expected)
        self.assertIn('top: "logits"', text)
        self.assertIn("num_output: 8", text)
        self.assertNotIn("Softmax", text)
        self.assertNotIn("BatchNorm", text)
        self.assertNotIn('type: "Scale"', text)
        self.assertNotIn('type: "Data"', text)

    def test_float_and_deploy_reject_forbidden_or_dynamic_layers(self):
        forbidden = (
            'type: "LSTM"', 'type: "Recurrent"', 'type: "Attention"',
            'type: "Python"', "group:", "depthwise", "input_dim: -1",
        )
        for path in (FLOAT, DEPLOY):
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, f"{path.name}: {marker}")

    def test_calibration_input_is_available_in_quantizer_train_phase(self):
        text = FLOAT.read_text(encoding="utf-8")
        self.assertIn('type: "ImageData"', text)
        self.assertNotIn("include { phase:", text)

    def test_generator_produces_repeatable_calibration_without_caffe(self):
        out_a = self.root / "a"
        out_b = self.root / "b"
        for out in (out_a, out_b):
            subprocess.run(
                ["py", "-3", str(GENERATOR), "--output-dir", str(out),
                 "--seed", "20261005", "--calibration-only"],
                check=True, capture_output=True, text=True,
            )
        manifest_a = json.loads((out_a / "calibration-manifest.json").read_text())
        manifest_b = json.loads((out_b / "calibration-manifest.json").read_text())
        self.assertEqual(manifest_a, manifest_b)
        self.assertEqual(manifest_a["shape"], [1, 1, 64, 96])
        self.assertEqual(manifest_a["dtype"], "float32-le")
        self.assertEqual(manifest_a["seed"], 20261005)
        self.assertGreaterEqual(manifest_a["sample_count"], 8)
        self.assertEqual(manifest_a["samples"][0]["kind"], "zero")
        self.assertEqual(manifest_a["samples"][1]["kind"], "pattern")
        self.assertTrue(all(len(x["sha256"]) == 64 for x in manifest_a["samples"]))

    def test_generator_keeps_zero_input_output_observable(self):
        source = GENERATOR.read_text(encoding="utf-8")
        self.assertIn('layer_name == "logits"', source)
        self.assertIn("np.linspace(-0.35, 0.35", source)

    def test_generator_avoids_all_zero_convolution_biases(self):
        source = GENERATOR.read_text(encoding="utf-8")
        self.assertIn('layer_name.startswith("conv")', source)
        self.assertIn("rng.uniform(-0.01, 0.01", source)

    def valid_log(self, **changes):
        values = {
            "target": "1152FA", "cpu": "arm32", "kernels": "sound_shape_probe",
            "input": "conv1", "output": "logits", "subgraphs": "1", "fallback": "none",
        }
        values.update(changes)
        return "\n".join(f"{k.upper()}={v}" for k, v in values.items()) + "\n"

    def contract(self):
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def kernel_info(self, **changes):
        values = {
            "kernel": "sound_shape_probe", "type": "DPUKernel",
            "input": "conv1", "output": "logits",
        }
        values.update(changes)
        return f'''Kernel Name : {values["kernel"]}
Kernel Type : {values["type"]}
Input Node(s)   (H*W*C)
  {values["input"]}(0) : 64*96*1
Output Node(s)   (H*W*C)
  {values["output"]}(0) : 1*1*8
'''

    def compatible_elf(self):
        return (
            b"\x7fELF\x00dnnc version v2.05\x00DPU Target : v1.4.0\x00"
            b"DPU_1152FA\x00"
        )

    def audit(self, log_text=None, elf=None, kernel_info_text=None):
        log = self.root / "dnnc.log"
        model = self.root / "dpu_sound_shape_probe.elf"
        kernel_info = self.root / "sound_shape_probe_kernel.info"
        if log_text is not None:
            log.write_text(log_text, encoding="utf-8")
        if elf is not False:
            model.write_bytes(self.compatible_elf() if elf is None else elf)
        kernel_info.write_text(
            kernel_info_text or self.kernel_info(), encoding="utf-8"
        )
        return load_auditor().audit_compile(self.contract(), log, model, kernel_info)

    def assert_rejected(self, reason, **kwargs):
        auditor = load_auditor()
        with self.assertRaisesRegex(auditor.CompileAuditError, reason):
            self.audit(**kwargs)

    def test_compile_auditor_accepts_one_exact_dpu_kernel(self):
        result = self.audit(self.valid_log())
        self.assertTrue(result.ok)
        self.assertEqual(result.kernel, "sound_shape_probe")

    def test_compile_auditor_rejects_missing_or_empty_elf(self):
        self.assert_rejected("missing_elf", log_text=self.valid_log(), elf=False)
        self.assert_rejected("empty_elf", log_text=self.valid_log(), elf=b"")

    def test_compile_auditor_rejects_unproven_embedded_compiler_contract(self):
        incompatible = {
            "wrong_dnnc_version": (
                b"\x7fELF\x00dnnc version v3.00\x00DPU Target : v1.4.0\x00DPU_1152FA\x00"
            ),
            "wrong_dpu_target": (
                b"\x7fELF\x00dnnc version v2.05\x00DPU Target : v2.0.0\x00DPU_1152FA\x00"
            ),
            "wrong_embedded_dpu": (
                b"\x7fELF\x00dnnc version v2.05\x00DPU Target : v1.4.0\x00DPU_4096FA\x00"
            ),
            "missing_compiler_metadata": b"\x7fELF\x00sound_shape_probe\x00",
        }
        for reason, elf in incompatible.items():
            with self.subTest(reason=reason):
                self.assert_rejected(reason, log_text=self.valid_log(), elf=elf)

    def test_compile_auditor_rejects_multiple_kernels(self):
        self.assert_rejected("kernel_count", log_text=self.valid_log(kernels="sound_shape_probe,other"))

    def test_compile_auditor_rejects_core_cpu_fallback(self):
        for layer in ("Convolution:conv2", "Pooling:pool1", "InnerProduct:logits"):
            self.assert_rejected("core_cpu_fallback", log_text=self.valid_log(fallback=layer))

    def test_compile_auditor_rejects_wrong_nodes_or_target(self):
        self.assert_rejected("wrong_input", log_text=self.valid_log(input="data"))
        self.assert_rejected("wrong_output", log_text=self.valid_log(output="prob"))
        self.assert_rejected("wrong_target", log_text=self.valid_log(target="4096FA"))
        self.assert_rejected("wrong_cpu_arch", log_text=self.valid_log(cpu="aarch64"))

    def test_compile_auditor_rejects_unsupported_diagnostics(self):
        self.assert_rejected(
            "unsupported_layer",
            log_text=self.valid_log() + "ERROR: unsupported layer Python(foo)\n",
        )

    def test_compile_auditor_rejects_non_dpu_or_wrong_runtime_nodes(self):
        self.assert_rejected("wrong_kernel_type", kernel_info_text=self.kernel_info(type="CPUKernel"), log_text=self.valid_log())
        self.assert_rejected("wrong_runtime_input", kernel_info_text=self.kernel_info(input="data"), log_text=self.valid_log())
        self.assert_rejected("wrong_runtime_output", kernel_info_text=self.kernel_info(output="prob"), log_text=self.valid_log())

    def test_builder_is_create_only_and_pins_compiler_target(self):
        text = BUILDER.read_text(encoding="utf-8")
        self.assertIn("--dpu 1152FA", text)
        self.assertIn("--cpu_arch arm32", text)
        self.assertIn("--abi 1", text)
        self.assertIn("--kernel-info", text)
        self.assertIn("decent_q", text)
        self.assertIn("dnnc", text)
        self.assertIn("audit_sound_shape_compile.py", text)
        self.assertIn("output directory already exists", text)


if __name__ == "__main__":
    unittest.main()
