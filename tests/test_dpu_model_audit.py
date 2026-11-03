import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR_PATH = ROOT / "scripts" / "host" / "audit_dpu_model.py"
REAL_MODEL = Path(
    r"F:\MLK\DPU_ZYNQ7000\references\Zynq-7000-DPU-TRD_guest"
    r"\reference-files\sdcard\yolo_prune\model\dpu_yolo.elf"
)


def load_auditor():
    if not AUDITOR_PATH.is_file():
        raise AssertionError(f"auditor implementation is missing: {AUDITOR_PATH}")
    spec = importlib.util.spec_from_file_location("audit_dpu_model", AUDITOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_arm_elf(markers: list[str]) -> bytes:
    header = bytearray(52)
    header[0:4] = b"\x7fELF"
    header[4] = 1  # ELFCLASS32
    header[5] = 1  # little-endian
    header[6] = 1  # current ELF version
    header[16:18] = (1).to_bytes(2, "little")  # ET_REL
    header[18:20] = (40).to_bytes(2, "little")  # EM_ARM
    header[20:24] = (1).to_bytes(4, "little")
    header[36:40] = (0x05000000).to_bytes(4, "little")  # EABI5
    return (
        bytes(header)
        + b"\x00"
        + b"\x00".join(x.encode("ascii") for x in markers)
        + b"\x00"
    )


class DpuModelAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.markers = [
            "DPU Target : v1.4.0",
            "DPU_1152FA",
            "_dpu_yolo_layer0_conv_code",
            "layer0_conv",
            "layer81_conv",
            "layer93_conv",
            "layer105_conv",
            "layer117_conv",
        ]
        self.model = self.root / "dpu_yolo.elf"
        self.model.write_bytes(synthetic_arm_elf(self.markers))

    def tearDown(self):
        self.temp.cleanup()

    def contract(self, **overrides):
        auditor = load_auditor()
        data = self.model.read_bytes()
        values = {
            "expected_size": len(data),
            "expected_sha256": hashlib.sha256(data).hexdigest(),
            "target": "v1.4.0",
            "architecture": "DPU_1152FA",
            "kernel": "yolo",
            "input_node": "layer0_conv",
            "output_nodes": (
                "layer81_conv",
                "layer93_conv",
                "layer105_conv",
                "layer117_conv",
            ),
        }
        values.update(overrides)
        return auditor.ModelContract(**values)

    def assert_rejected(self, reason: str, contract=None):
        auditor = load_auditor()
        with self.assertRaisesRegex(auditor.ModelAuditError, reason):
            auditor.audit_model(self.model, contract or self.contract())

    def test_accepts_exact_arm_eabi5_model_contract(self):
        auditor = load_auditor()
        result = auditor.audit_model(self.model, self.contract())
        self.assertTrue(result.ok)
        self.assertEqual(result.elf_class, "ELF32")
        self.assertEqual(result.machine, "ARM")
        self.assertEqual(result.eabi, "EABI5")
        self.assertEqual(result.kernel, "yolo")

    def test_rejects_size_mismatch(self):
        self.assert_rejected("size_mismatch", self.contract(expected_size=1))

    def test_rejects_sha256_mismatch(self):
        self.assert_rejected("sha256_mismatch", self.contract(expected_sha256="0" * 64))

    def test_rejects_non_arm_elf_machine(self):
        data = bytearray(self.model.read_bytes())
        data[18:20] = (62).to_bytes(2, "little")  # EM_X86_64
        self.model.write_bytes(data)
        self.assert_rejected("wrong_elf_machine", self.contract())

    def test_rejects_missing_target_marker(self):
        self.model.write_bytes(synthetic_arm_elf([m for m in self.markers if "Target" not in m]))
        self.assert_rejected("missing_target_marker", self.contract())

    def test_rejects_missing_architecture_marker(self):
        self.model.write_bytes(synthetic_arm_elf([m for m in self.markers if m != "DPU_1152FA"]))
        self.assert_rejected("missing_architecture_marker", self.contract())

    def test_rejects_missing_kernel_symbol(self):
        self.model.write_bytes(synthetic_arm_elf([m for m in self.markers if "_dpu_yolo_" not in m]))
        self.assert_rejected("missing_kernel_symbol", self.contract())

    def test_rejects_missing_input_node(self):
        self.model.write_bytes(synthetic_arm_elf([m for m in self.markers if m != "layer0_conv"]))
        self.assert_rejected("missing_input_node", self.contract())

    def test_rejects_any_missing_output_node(self):
        self.model.write_bytes(synthetic_arm_elf([m for m in self.markers if m != "layer105_conv"]))
        self.assert_rejected("missing_output_node:layer105_conv", self.contract())

    @unittest.skipUnless(REAL_MODEL.is_file(), "real Level A model is not present")
    def test_real_model_matches_pinned_identity(self):
        auditor = load_auditor()
        contract = auditor.ModelContract(
            expected_size=2_755_220,
            expected_sha256="cf73502ebfdec418bf4a93df9e2957d50658a5fd465dd56ae39a07b0335fb19e",
            target="v1.4.0",
            architecture="DPU_1152FA",
            kernel="yolo",
            input_node="layer0_conv",
            output_nodes=("layer81_conv", "layer93_conv", "layer105_conv", "layer117_conv"),
        )
        result = auditor.audit_model(REAL_MODEL, contract)
        self.assertTrue(result.ok)
        self.assertEqual(result.size, 2_755_220)


if __name__ == "__main__":
    unittest.main()
