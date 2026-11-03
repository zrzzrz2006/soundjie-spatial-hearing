import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "guest" / "build_soundjie_d3_model.sh"
MODEL_CONFIG = ROOT / "config" / "soundjie_d3_compile.json"


def load_auditor():
    return importlib.import_module("scripts.host.audit_soundjie_d3_model")


class SoundjieD3CompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.auditor = load_auditor()
        cls.contract = cls.auditor.load_compiler_contract(MODEL_CONFIG)

    def test_calibration_rejects_demo_overlap(self):
        calibration = [{"recording_id": "a", "source_group": "g1", "split": "train"}]
        demo = [{"recording_id": "a", "source_group": "g2", "split": "demo"}]
        with self.assertRaisesRegex(self.auditor.ModelAuditError, "demo overlap"):
            self.auditor.validate_calibration_records(calibration, demo)
        calibration[0]["recording_id"] = "b"
        calibration[0]["source_group"] = "g2"
        with self.assertRaisesRegex(self.auditor.ModelAuditError, "demo overlap"):
            self.auditor.validate_calibration_records(calibration, demo)

    def test_compile_contract_rejects_wrong_compiler_target_and_abi(self):
        good = self.auditor.expected_compile_evidence(self.contract)
        self.auditor.validate_compile_evidence(self.contract, good)
        for key, value in (
            ("dnnc_version", "v3.00"),
            ("dpu_target", "v1.3.0"),
            ("abi", 0),
        ):
            changed = dict(good)
            changed[key] = value
            with self.assertRaises(self.auditor.ModelAuditError):
                self.auditor.validate_compile_evidence(self.contract, changed)

    def test_compile_contract_rejects_multiple_or_cpu_kernels(self):
        good = self.auditor.expected_compile_evidence(self.contract)
        for mutation in (
            {"kernels": ["soundjie_d3", "second"]},
            {"kernel_type": "CPUKernel"},
            {"unsupported_layers": ["fc8"]},
        ):
            changed = dict(good)
            changed.update(mutation)
            with self.assertRaises(self.auditor.ModelAuditError):
                self.auditor.validate_compile_evidence(self.contract, changed)

    def test_compile_contract_rejects_non_eight_output(self):
        evidence = self.auditor.expected_compile_evidence(self.contract)
        evidence["output_count"] = 7
        with self.assertRaisesRegex(self.auditor.ModelAuditError, "output count"):
            self.auditor.validate_compile_evidence(self.contract, evidence)

    def test_upstream_hash_drift_is_rejected(self):
        expected = {"checkpoint_sha256": "a" * 64, "caffemodel_sha256": "b" * 64}
        self.auditor.validate_upstream_hashes(expected, dict(expected))
        actual = dict(expected)
        actual["caffemodel_sha256"] = "c" * 64
        with self.assertRaisesRegex(self.auditor.ModelAuditError, "upstream hash drift"):
            self.auditor.validate_upstream_hashes(expected, actual)

    def test_model_hash_chain_binds_float_export_calibration_and_quantization(self):
        fused = {
            "weights_sha256": "a" * 64,
            "deploy_sha256": "b" * 64,
            "parity_inputs_sha256": "c" * 64,
            "pytorch_logits_sha256": "d" * 64,
            "upstream": {"prepared_manifest_sha256": "e" * 64},
        }
        caffe_export = {
            "weights_sha256": "a" * 64,
            "deploy_sha256": "b" * 64,
            "inputs_sha256": "c" * 64,
            "reference_logits_sha256": "d" * 64,
            "caffemodel_sha256": "9" * 64,
        }
        calibration = {"prepared_manifest_sha256": "e" * 64}
        metrics = {
            "inputs_sha256": "c" * 64,
            "reference_logits_sha256": "d" * 64,
            "quantized_caffemodel_sha256": "f" * 64,
        }
        actual = {
            "float_caffemodel_sha256": "9" * 64,
            "float_deploy_sha256": "b" * 64,
            "quantized_caffemodel_sha256": "f" * 64,
        }
        self.auditor.validate_model_hash_chain(
            fused, caffe_export, calibration, metrics, actual
        )
        for key in actual:
            changed = dict(actual)
            changed[key] = "0" * 64
            with self.assertRaisesRegex(self.auditor.ModelAuditError, "upstream hash drift"):
                self.auditor.validate_model_hash_chain(
                    fused, caffe_export, calibration, metrics, changed
                )
        for key in caffe_export:
            changed_export = dict(caffe_export)
            changed_export[key] = "0" * 64
            with self.assertRaisesRegex(self.auditor.ModelAuditError, "upstream hash drift"):
                self.auditor.validate_model_hash_chain(
                    fused, changed_export, calibration, metrics, actual
                )

    def test_manifest_publish_and_output_directory_are_create_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "manifest.json"
            self.auditor.publish_manifest(destination, {"schema": 1})
            with self.assertRaisesRegex(self.auditor.ModelAuditError, "already exists"):
                self.auditor.publish_manifest(destination, {"schema": 2})
        text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("output directory already exists", text)

    def test_manifest_publish_does_not_require_hardlink_on_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.json"
            with mock.patch.object(self.auditor.os, "link", side_effect=OSError("unsupported")):
                self.auditor.publish_manifest(destination, {"schema": 1})
            self.assertTrue(destination.is_file())

    def test_manifest_publish_uses_cross_filesystem_exclusive_create(self):
        source = (ROOT / "scripts" / "host" / "audit_soundjie_d3_model.py").read_text(encoding="utf-8")
        self.assertNotIn("os.link(", source)
        self.assertIn('path.open("x"', source)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "manifest.json"
            with mock.patch.object(self.auditor.os, "link", side_effect=AssertionError("hardlink used")):
                self.auditor.publish_manifest(destination, {"schema": 1})
            self.assertEqual(destination.read_text(encoding="utf-8"), '{\n  "schema": 1\n}\n')

    def test_build_script_pins_quantizer_and_validated_legacy_dnnc(self):
        text = BUILD_SCRIPT.read_text(encoding="utf-8")
        for token in (
            "vai_q_caffe quantize", "--dnnc-bin", "--dnnc-lib-dir",
            '"$DNNC_BIN" --version', "--dpu 1152FA", "--cpu_arch arm32",
            "--abi 1", "dnnc-v2.05.log", "ABI=1", "CPUKernel",
            "unsupported", "soundjie_d3",
            "quantize-command.txt", "compile-command.txt",
            "calibration-manifest.json", "demo-manifest.json",
            "conda activate vitis-ai-caffe",
        ):
            self.assertIn(token, text)
        self.assertNotIn("vai_c_caffe", text)

    def test_build_script_defines_ps1_before_sourcing_legacy_conda(self):
        text = BUILD_SCRIPT.read_text(encoding="utf-8")
        ps1_guard = ': "${PS1:=}"'
        self.assertIn(ps1_guard, text)
        self.assertLess(text.index(ps1_guard), text.index("source /etc/profile.d/conda.sh"))

    def test_build_script_preserves_generated_diagnostics_on_failure(self):
        text = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('FAILED_DIR="${OUTPUT_DIR}.failed"', text)
        self.assertIn('mv "$TMP_DIR" "$FAILED_DIR"', text)
        self.assertIn("preserved failed build diagnostics", text)

    def test_auditor_cli_is_compatible_with_vitis_ai_python36_argparse(self):
        source = (ROOT / "scripts" / "host" / "audit_soundjie_d3_model.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn('add_subparsers(dest="command", required=True)', source)
        self.assertIn("if args.command is None:", source)

    def test_quantizer_model_replaces_static_input_with_hdf5_data(self):
        deploy = 'name: "n"\ninput: "data"\ninput_shape { dim: 1 dim: 1 dim: 64 dim: 96 }\nlayer { name: "fc8" type: "InnerProduct" bottom: "data" top: "fc8" }\n'
        quantizer = self.auditor.make_quantizer_prototxt(deploy, "/tmp/calibration-list.txt")
        self.assertIn('type: "HDF5Data"', quantizer)
        self.assertIn('source: "/tmp/calibration-list.txt"', quantizer)
        self.assertNotIn("include { phase:", quantizer)
        self.assertNotIn('input_shape', quantizer)
        runtime = self.auditor.make_runtime_prototxt(quantizer)
        self.assertTrue(runtime.startswith('input: "data"\ninput_shape'))
        self.assertIn('input: "data"', runtime)
        self.assertIn('dim: 64', runtime)
        self.assertNotIn('HDF5Data', runtime)

    def test_runtime_reuses_quantizer_generated_test_input_without_duplicate_data(self):
        merged = '''layer {
  name: "calibration_input"
  type: "HDF5Data"
  top: "data"
}
layer {
  name: "calibration_input"
  type: "Input"
  top: "data"
  input_param { shape { dim: 1 dim: 1 dim: 64 dim: 96 } }
}
layer { name: "fc8" type: "InnerProduct" bottom: "data" top: "fc8" }
'''
        runtime = self.auditor.make_runtime_prototxt(merged)
        self.assertFalse(runtime.startswith('input: "data"'))
        self.assertEqual(runtime.count('type: "Input"'), 1)
        self.assertNotIn('type: "HDF5Data"', runtime)

    def test_calibration_selection_is_balanced_train_only_and_deterministic(self):
        records = []
        for target in range(8):
            for index in range(3):
                records.append({
                    "recording_id": f"r-{target}-{index}",
                    "source_group": f"g-{target}-{index}",
                    "split": "train",
                    "target_index": target,
                    "feature_path": f"f-{target}-{index}.npy",
                })
        records.append({"recording_id": "demo", "source_group": "demo", "split": "demo", "target_index": 0, "feature_path": "demo.npy"})
        selected = self.auditor.select_calibration_records(records, per_class=2)
        self.assertEqual(len(selected), 16)
        self.assertEqual([item["target_index"] for item in selected], [i for i in range(8) for _ in range(2)])
        self.assertTrue(all(item["split"] == "train" for item in selected))

    def test_classification_metrics_cover_eight_classes_and_detect_collapse(self):
        targets = np.arange(8, dtype=np.int64).repeat(2)
        reference = np.full((16, 8), -5.0, dtype=np.float32)
        quantized = reference.copy()
        reference[np.arange(16), targets] = 5.0
        quantized[np.arange(16), targets] = 4.0
        result = self.auditor.classification_metrics(reference, quantized, targets)
        self.assertEqual(result["class_count"], 8)
        self.assertEqual(result["quantized_unique_predictions"], 8)
        self.assertAlmostEqual(result["reference_macro_f1"], 1.0)
        self.assertAlmostEqual(result["quantized_macro_f1"], 1.0)
        collapsed = np.zeros((16, 8), dtype=np.float32)
        with self.assertRaisesRegex(self.auditor.ModelAuditError, "class collapse"):
            self.auditor.classification_metrics(reference, collapsed, targets)

    def test_quantization_audit_compares_coverage_to_float_reference(self):
        targets = np.arange(8, dtype=np.int64).repeat(2)
        reference = np.full((16, 8), -5.0, dtype=np.float32)
        reference_predictions = np.array([3, 1, 2, 1, 7, 2, 3, 3, 4, 4, 5, 5, 7, 3, 7, 7])
        reference[np.arange(16), reference_predictions] = 5.0
        result = self.auditor.classification_metrics(reference, reference.copy(), targets)
        self.assertEqual(result["reference_unique_predictions"], 6)
        self.assertEqual(result["quantized_unique_predictions"], 6)
        self.assertEqual(result["argmax_match_count"], 16)
        self.assertAlmostEqual(result["argmax_match_ratio"], 1.0)

    def test_real_kernel_info_and_elf_metadata_are_parsed_fail_closed(self):
        kernel_info = '''
Kernel ID : Name
0 : soundjie_d3
Kernel Name : soundjie_d3
Kernel Type : DPUKernel
Boundary Output Tensor(s)   (H*W*C)
fc8:0(0) : 1*1*8
Input Node(s)   (H*W*C)
conv1(0) : 64*96*1
Output Node(s)   (H*W*C)
fc8(0) : 1*1*8
'''
        binary = b"prefix dnnc version v2.05 suffix DPU Target : v1.4.0 DPU_1152FA"
        evidence = self.auditor.extract_compile_evidence(
            self.contract, kernel_info, binary, "compile completed\n"
        )
        self.auditor.validate_compile_evidence(self.contract, evidence)
        self.assertEqual(evidence["output_count"], 8)
        self.assertEqual(evidence["kernels"], ["soundjie_d3"])
        with self.assertRaisesRegex(self.auditor.ModelAuditError, "unsupported"):
            self.auditor.extract_compile_evidence(
                self.contract, kernel_info, binary, "unsupported operator found"
            )

    def test_legacy_dnnc_log_uses_compiled_output_layer_for_eight_outputs(self):
        legacy_log = '''
DNNC Kernel Information

kernel numbers  : 1
kernel id       : 0
kernel name       : soundjie_d3
type              : DPUKernel
input node(s)   : conv1(0)
output node(s)  : fc8(0)
'''
        network = '''
layer {
  name: "fc8"
  type: "InnerProduct"
  bottom: "pool4"
  top: "fc8"
  inner_product_param { num_output: 8 }
}
'''
        binary = b"dnnc version v2.05 DPU Target : v1.4.0 DPU_1152FA"
        evidence = self.auditor.extract_compile_evidence(
            self.contract, legacy_log, binary, legacy_log, network
        )
        self.auditor.validate_compile_evidence(self.contract, evidence)
        self.assertEqual(evidence["output_count"], 8)


if __name__ == "__main__":
    unittest.main()
