import hashlib
import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "soundjie_d4_pl_sources.json"
STAGER = ROOT / "scripts" / "host" / "stage_soundjie_d4_pl_sources.ps1"
BUILD_TCL = ROOT / "scripts" / "vivado" / "build_soundjie_d4_combined.tcl"
REPORT_TCL = ROOT / "scripts" / "vivado" / "report_soundjie_d4_combined.tcl"
CDC_XDC = ROOT / "constraints" / "soundjie_d4_axis2video_cdc.xdc"
COMBINED_DTSI = ROOT / "config" / "soundjie_d4_combined_pl.dtsi"


class SoundjieD4PlContractTests(unittest.TestCase):
    def load_config(self):
        self.assertTrue(CONFIG.is_file(), f"missing PL source manifest: {CONFIG}")
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def read_required(self, path):
        self.assertTrue(path.is_file(), f"missing required file: {path}")
        return path.read_text(encoding="utf-8")

    def test_source_hashes_and_vivado_version_are_pinned(self):
        cfg = self.load_config()
        self.assertEqual(cfg["vivado"]["version"], "2017.4")
        self.assertEqual(cfg["vivado"]["part"], "xc7z100ffg900-2")
        self.assertEqual(
            cfg["sources"]["ch24_archive"]["original_archive_sha256"],
            "473b0600e9f51a75fd1e383827f7db082f093b7ac278a4efce6dd34d755938df",
        )
        self.assertEqual(
            cfg["sources"]["dpu_hdf"]["sha256"],
            "a98bfc2a9563c2b0ba6e1ac617e6ba29737faa1e7d9b4ea1d6924e958d403a2b",
        )
        for source_name in ("ch24_archive", "dpu_hdf"):
            source = Path(cfg["sources"][source_name]["path"])
            self.assertTrue(source.is_file(), source)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(digest, cfg["sources"][source_name]["sha256"])
        self.assertTrue(Path(cfg["sources"]["dpu_ip_repo"]["path"]).is_dir())

    def test_stager_never_writes_vendor_source_or_existing_work_dir(self):
        script = self.read_required(STAGER)
        self.assertNotIn("Remove-Item", script)
        self.assertNotIn("/MIR", script.upper())
        for rejected in (ROOT, Path(r"F:\MLK\MZ7100FA\forbidden-output")):
            completed = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(STAGER),
                    "-ConfigPath",
                    str(CONFIG),
                    "-WorkDir",
                    str(rejected),
                    "-ValidateOnly",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0, completed.stdout)
            self.assertIn("refuse", (completed.stdout + completed.stderr).lower())

    def test_stager_accepts_only_approved_e_rebuild_prefix(self):
        prefix = "E:\\Soundjie_Work\\d4\\artifacts\\pl_rebuild_"
        self.assertEqual(self.load_config()["staging"]["work_dir_prefix"], prefix)
        for target, expected in (
            (prefix + "validation_only", 0),
            (r"F:\MLK\DPU_ZYNQ7000\work\soundjie_d4_pl_forbidden", 1),
            (prefix + r"escape\..\..\outside", 1),
        ):
            completed = subprocess.run(
                ["pwsh.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", str(STAGER), "-ConfigPath", str(CONFIG),
                 "-WorkDir", target, "-ValidateOnly"],
                capture_output=True, text=True, encoding="utf-8", check=False,
            )
            if expected == 0:
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("D4_PL_STAGE_VALIDATION=pass", completed.stdout)
                self.assertFalse(Path(target).exists())
            else:
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("refuse", (completed.stdout + completed.stderr).lower())

    def test_display_uses_hp0_and_dpu_uses_hp1_hp2_hp3(self):
        cfg = self.load_config()
        ports = cfg["integration"]["hp_ports"]
        self.assertEqual(ports["display_vdma"], "S_AXI_HP0")
        self.assertEqual(ports["dpu_data0"], "S_AXI_HP1")
        self.assertEqual(ports["dpu_data1"], "S_AXI_HP2")
        self.assertEqual(ports["dpu_instr"], "S_AXI_HP3")
        self.assertEqual(len(set(ports.values())), 4)
        build = self.read_required(BUILD_TCL)
        for logical, physical in ports.items():
            self.assertIn(logical, build)
            self.assertIn(physical, build)

    def test_address_irq_clock_domains_are_distinct(self):
        cfg = self.load_config()["integration"]
        address_values = list(cfg["control_addresses"].values())
        irq_values = list(cfg["interrupts"].values())
        clock_values = list(cfg["clock_domains"].values())
        self.assertEqual(len(address_values), len(set(address_values)))
        self.assertEqual(len(irq_values), len(set(irq_values)))
        self.assertEqual(len(clock_values), len(set(clock_values)))

    def test_tcl_refuses_ip_upgrade_or_missing_xc7z100_part(self):
        build = self.read_required(BUILD_TCL)
        self.assertIn("xc7z100ffg900-2", build)
        self.assertIn("REFUSE_IP_UPGRADE", build)
        self.assertIn("get_ipdefs", build)
        self.assertNotIn("upgrade_ip", build)
        self.assertIn("PART_MISMATCH", build)

    def test_report_requires_dpu_vdma_framebuffer_and_timing_pass(self):
        report = self.read_required(REPORT_TCL)
        for required in (
            "dpu_eu",
            "axi_vdma",
            "framebuffer",
            "report_timing_summary",
            "WNS",
            "FAIL_CLOSED",
        ):
            self.assertIn(required, report)
        for implemented_flag in (
            "DPU_IMPLEMENTED",
            "VDMA_IMPLEMENTED",
            "FRAMEBUFFER_IMPLEMENTED",
        ):
            self.assertIn(implemented_flag, report)
        self.assertNotIn("audit_emit $fp dpu_eu [llength", report)

    def test_hdf_tar_timestamp_warning_requires_complete_payload(self):
        script = self.read_required(STAGER)
        for member in ("sysdef.xml", "system.hwh", "system_bd.tcl", "system_wrapper.bit"):
            self.assertIn(member, script)
        self.assertIn("HDF_PAYLOAD_VALIDATED=pass", script)
        self.assertNotIn(
            'if ($LASTEXITCODE -ne 0) { throw "DPU HDF extraction failed', script
        )

    def test_interrupt_extends_existing_concat_without_disconnecting_display(self):
        build = self.read_required(BUILD_TCL)
        self.assertIn("get_bd_cells -quiet xlconcat_0", build)
        self.assertIn("CONFIG.NUM_PORTS {3}", build)
        self.assertIn("$irq_concat/In2", build)
        self.assertNotIn("disconnect_bd_net", build)

    def test_axi_automation_owns_dpu_control_clock_and_reset(self):
        build = self.read_required(BUILD_TCL)
        self.assertIn("apply_bd_automation -rule xilinx.com:bd_rule:axi4", build)
        self.assertNotIn(
            "connect_bd_net [get_bd_pins $ps/FCLK_CLK0] [get_bd_pins $dpu/s_axi_aclk]",
            build,
        )

    def test_dpu_instr_width_and_clock_input_follow_ch24_contract(self):
        cfg = self.load_config()["integration"]
        self.assertEqual(cfg["hp_data_width_bits"]["dpu_instr"], 32)
        build = self.read_required(BUILD_TCL)
        self.assertIn("CONFIG.PCW_S_AXI_HP3_DATA_WIDTH {32}", build)
        self.assertIn("get_property CONFIG.FREQ_HZ [get_bd_pins $ps/FCLK_CLK0]", build)
        self.assertNotIn("CONFIG.PRIM_IN_FREQ {100.000}", build)

    def test_build_resets_copied_runs_and_resume_requires_existing_dpu(self):
        build = self.read_required(BUILD_TCL)
        self.assertIn('mode eq "resume"', build)
        self.assertIn('resume requires an existing DPU', build)
        self.assertIn("reset_run synth_1", build)
        self.assertIn("reset_run impl_1", build)

    def test_axis2video_vsync_synchronizer_is_narrowly_constrained(self):
        xdc = self.read_required(CDC_XDC)
        self.assertIn("ASYNC_REG TRUE", xdc)
        self.assertIn("axis_vsync_reg", xdc)
        self.assertIn(r"axis_vsync_reg\[0\]/D", xdc)
        self.assertIn("set_false_path -to", xdc)
        self.assertNotIn("set_clock_groups", xdc)
        self.assertNotIn("if {", xdc)
        build = self.read_required(BUILD_TCL)
        self.assertIn("soundjie_d4_axis2video_cdc.xdc", build)
        self.assertIn("add_files -fileset constrs_1", build)
        report = self.read_required(REPORT_TCL)
        self.assertIn("CDC_AXIS_VSYNC_REGS", report)
        self.assertIn("CDC_AXIS_VSYNC_STAGE0_PINS", report)
        self.assertIn("string equal -nocase", report)

    def test_report_mode_reuses_implemented_run_without_resetting_it(self):
        build = self.read_required(BUILD_TCL)
        report_pos = build.index('mode eq "report"')
        reset_pos = build.index("reset_run impl_1")
        self.assertLess(report_pos, reset_pos)
        self.assertIn("open_run impl_1", build[report_pos:reset_pos])
        self.assertIn("SOUNDJIE_D4_REPORT_ONLY", build[report_pos:reset_pos])

    def test_export_mode_requires_passing_report_before_bit_and_hdf(self):
        build = self.read_required(BUILD_TCL)
        export_pos = build.index('mode eq "export"')
        report_pos = build.index("report_soundjie_d4_combined.tcl", export_pos)
        bit_pos = build.index("write_bitstream -force", export_pos)
        hdf_pos = build.index("write_hwdef -force", export_pos)
        self.assertLess(report_pos, bit_pos)
        self.assertLess(report_pos, hdf_pos)
        self.assertIn("SOUNDJIE_D4_EXPORT=pass", build[export_pos:hdf_pos + 200])

    def test_combined_device_tree_contains_dpu_vdma_and_framebuffer(self):
        dtsi = self.read_required(COMBINED_DTSI)
        for required in (
            'compatible = "deephi,dpu"',
            'compatible = "xlnx,axi-vdma-1.00.a"',
            'compatible = "topic,vdma-fb"',
            'reg = <0x4f000000 0x1000000>',
            'reg = <0x43000000 0x10000>',
            'interrupts = <0 31 4>',
            'interrupts = <0 30 4>',
        ):
            self.assertIn(required, dtsi)
        self.assertIn('dmas = <&axi_vdma_0 0>', dtsi)


if __name__ == "__main__":
    unittest.main()
