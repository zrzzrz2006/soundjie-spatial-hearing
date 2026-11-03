"""Reject missing/disabled EMIO SDIO1 before considering a board candidate."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/host/audit_emmc_handoff.py'


def handoff(disabled=False, omit_port=None):
    parameters = {
        'PCW_SD1_PERIPHERAL_ENABLE': '0' if disabled else '1',
        'PCW_EN_SDIO1': '1', 'PCW_EN_EMIO_SDIO1': '1',
        'PCW_SD1_SD1_IO': 'EMIO',
        'PCW_SD1_GRP_CD_ENABLE': '1', 'PCW_SD1_GRP_CD_IO': 'EMIO',
        'PCW_SD1_GRP_WP_ENABLE': '1', 'PCW_SD1_GRP_WP_IO': 'EMIO',
    }
    ports = {'CLK': 'O', 'CLK_FB': 'I', 'CMD_I': 'I', 'CMD_O': 'O',
             'CMD_T': 'O', 'DATA_I': 'I', 'DATA_O': 'O', 'DATA_T': 'O',
             'CDN': 'I', 'WP': 'I'}
    return ('<SYSTEM><MODULES><MODULE MODTYPE="processing_system7"><PARAMETERS>'
            + ''.join(f'<PARAMETER NAME="{k}" VALUE="{v}"/>' for k, v in parameters.items())
            + '</PARAMETERS><PORTS>'
            + ''.join(f'<PORT NAME="SDIO1_{k}" DIR="{v}"/>' for k, v in ports.items()
                      if k != omit_port)
            + '</PORTS></MODULE></MODULES></SYSTEM>')


class EmmcHandoffAuditTests(unittest.TestCase):
    def run_audit(self, xml, archive=False, duplicate=False):
        self.assertTrue(SCRIPT.is_file(), 'eMMC handoff audit has not been implemented')
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as tmp:
            path = Path(tmp) / ('sample.hdf' if archive else 'sample.hwh')
            if archive:
                with zipfile.ZipFile(path, 'w') as z:
                    z.writestr('system.hwh', xml)
                    if duplicate:
                        z.writestr('another.hwh', xml)
            else:
                path.write_text(xml, encoding='utf-8')
            before = path.read_bytes()
            result = subprocess.run([sys.executable, '-B', str(SCRIPT), str(path)],
                                    capture_output=True, text=True)
            self.assertEqual(path.read_bytes(), before, 'audit must be read-only')
            return result.returncode, json.loads(result.stdout)

    def test_enabled_hwh_reports_configuration_only(self):
        code, report = self.run_audit(handoff())
        self.assertEqual(code, 0)
        self.assertEqual(report['status'], 'configuration_present_not_board_validated')
        self.assertFalse(report['board_ready'])

    def test_hdf_uses_embedded_hwh(self):
        code, _ = self.run_audit(handoff(), archive=True)
        self.assertEqual(code, 0)

    def test_disabled_sdio_cannot_pass(self):
        code, report = self.run_audit(handoff(disabled=True))
        self.assertEqual(code, 2)
        self.assertTrue(any('PCW_SD1_PERIPHERAL_ENABLE' in e for e in report['errors']))

    def test_missing_feedback_port_cannot_pass(self):
        code, report = self.run_audit(handoff(omit_port='CLK_FB'))
        self.assertEqual(code, 2)
        self.assertTrue(any('SDIO1_CLK_FB' in e for e in report['errors']))

    def test_absent_ps_cannot_pass(self):
        code, _ = self.run_audit('<SYSTEM/>')
        self.assertEqual(code, 2)

    def test_ambiguous_archive_cannot_pass(self):
        code, _ = self.run_audit(handoff(), archive=True, duplicate=True)
        self.assertEqual(code, 2)

    def test_malformed_input_fails_closed(self):
        code, _ = self.run_audit('<broken')
        self.assertEqual(code, 2)


if __name__ == '__main__':
    unittest.main()
