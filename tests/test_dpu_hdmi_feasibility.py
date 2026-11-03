import importlib.util
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts" / "host" / "audit_dpu_hdmi_hdf.py"
INDEXER = ROOT / "scripts" / "host" / "index_mz7100_display_sources.ps1"
VIVADO_REPORTER = ROOT / "scripts" / "vivado" / "report_dpu_hdmi_feasibility.tcl"
REAL_HDF = Path(
    r"F:\MLK\DPU_ZYNQ7000\artifacts\mz7100_dpu_2017_probe_v1"
    r"\mz7100_dpu_system_wrapper.hdf"
)


def load_auditor():
    if not AUDITOR.is_file():
        raise AssertionError(f"missing auditor: {AUDITOR}")
    spec = importlib.util.spec_from_file_location("audit_dpu_hdmi_hdf", AUDITOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_hdf(path: Path, text: str):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("system_bd.tcl", text)
        archive.writestr("system.hwh", text)
        archive.writestr("sysdef.xml", text)


class DpuHdmiFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def audit(self, text: str):
        fixture = self.root / "fixture.hdf"
        write_hdf(fixture, text)
        return load_auditor().audit_hdf(fixture)

    def test_dpu_only_requires_hardware_merge(self):
        result = self.audit(
            "set dpu [create_bd_cell -vlnv xilinx.com:ip:dpu:1.4 DPU_0]\n"
            "connect_bd_intf_net DPU_0/M_AXI_DATA0 processing_system7_0/S_AXI_HP1\n"
        )
        self.assertTrue(result.dpu_present)
        self.assertFalse(result.vdma_present)
        self.assertEqual(result.display_hp_ports, ())
        self.assertEqual(result.action, "hardware_merge_required")

    def test_display_only_reports_missing_dpu(self):
        result = self.audit(
            "create_bd_cell -vlnv xilinx.com:ip:axi_vdma:6.3 axi_vdma_lcd\n"
            "create_bd_cell -vlnv topic:display:vdma-fb:1.0 vdma_fb\n"
            "connect_bd_intf_net axi_vdma_lcd/M_AXI_MM2S processing_system7_0/S_AXI_HP0\n"
        )
        self.assertFalse(result.dpu_present)
        self.assertTrue(result.vdma_present)
        self.assertEqual(result.action, "dpu_hardware_missing")

    def test_merged_valid_is_ready_for_vivado_report(self):
        result = self.audit(
            "xilinx.com:ip:dpu:1.4 DPU_0\n"
            "xilinx.com:ip:axi_vdma:6.3 axi_vdma_lcd\n"
            "topic:display:vdma-fb:1.0 vdma_fb\n"
            "DPU_0/M_AXI_DATA0 processing_system7_0/S_AXI_HP1\n"
            "axi_vdma_lcd/M_AXI_MM2S processing_system7_0/S_AXI_HP0\n"
        )
        self.assertEqual(result.action, "vivado_feasibility_report_required")
        self.assertFalse(result.hp_conflicts)

    def test_merged_conflicting_hp_port_is_rejected(self):
        result = self.audit(
            "xilinx.com:ip:dpu:1.4 DPU_0\n"
            "xilinx.com:ip:axi_vdma:6.3 axi_vdma_lcd\n"
            "topic:display:vdma-fb:1.0 vdma_fb\n"
            "DPU_0/M_AXI_DATA0 processing_system7_0/S_AXI_HP0\n"
            "axi_vdma_lcd/M_AXI_MM2S processing_system7_0/S_AXI_HP0\n"
        )
        self.assertEqual(result.action, "resolve_hardware_conflicts")
        self.assertIn("HP0", result.hp_conflicts)

    def test_rejects_non_zip_hdf(self):
        fixture = self.root / "bad.hdf"
        fixture.write_bytes(b"not an hdf")
        auditor = load_auditor()
        with self.assertRaisesRegex(auditor.HdfAuditError, "not_zip_hdf"):
            auditor.audit_hdf(fixture)

    def test_indexer_is_read_only_and_records_hashes(self):
        vendor = self.root / "vendor"
        vendor.mkdir()
        project = vendor / "demo.xpr"
        project.write_text("project", encoding="utf-8")
        archive = vendor / "HDMI_demo.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("source/system.bd", "axi_vdma")
        before = sorted(str(p.relative_to(vendor)) for p in vendor.rglob("*"))
        output = self.root / "index.json"
        completed = subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(INDEXER), "-Root", str(vendor),
                "-OutputManifest", str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        after = sorted(str(p.relative_to(vendor)) for p in vendor.rglob("*"))
        self.assertEqual(before, after)
        manifest = json.loads(output.read_text(encoding="utf-8-sig"))
        self.assertEqual(manifest["schema"], 1)
        self.assertEqual(manifest["result"], "pass")
        self.assertEqual(len(manifest["candidates"]), 2)
        self.assertTrue(all(len(x["sha256"]) == 64 for x in manifest["candidates"]))
        zip_entry = next(x for x in manifest["candidates"] if x["extension"] == ".zip")
        self.assertIn("source/system.bd", zip_entry["members"])

    def test_vivado_report_contains_required_contract_keys(self):
        text = VIVADO_REPORTER.read_text(encoding="utf-8")
        for key in (
            "DPU_CELL", "VDMA_CELL", "DPU_HP_PORTS", "DISPLAY_HP_PORTS",
            "ADDRESS_CONFLICTS", "IRQ_CONFLICTS", "CLOCK_CONFLICTS",
            "LUT_PERCENT", "BRAM_PERCENT", "DSP_PERCENT", "RESULT",
        ):
            self.assertIn(key, text)
        self.assertNotIn("create_bd_cell", text)
        self.assertNotIn("delete_bd", text)

    @unittest.skipUnless(REAL_HDF.is_file(), "real DPU HDF is not present")
    def test_real_current_hdf_requires_hardware_merge(self):
        result = load_auditor().audit_hdf(REAL_HDF)
        self.assertTrue(result.dpu_present)
        self.assertFalse(result.vdma_present)
        self.assertEqual(result.action, "hardware_merge_required")


if __name__ == "__main__":
    unittest.main()
