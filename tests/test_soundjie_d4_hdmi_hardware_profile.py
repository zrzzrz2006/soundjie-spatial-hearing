"""Actual exported hardware: accept the audited pair, never mixed revisions."""
import importlib.util
from pathlib import Path
import unittest
import tempfile

ROOT = Path(__file__).resolve().parents[1]
EXPORTED = ROOT / 'artifacts/pl_rebuild_20261022_hdmi/result_hdmi_c_20261023'
LEGACY = ROOT / 'artifacts/development-bundle-20261022c'


class HdmiHardwareProfileTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('profile_audit', ROOT / 'scripts/host/audit_soundjie_d4_package.py')
        self.audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.audit)
        self.assertTrue(callable(getattr(self.audit, 'hardware_profile', None)), 'paired hardware profile validation is missing')
        self.old = {name: (LEGACY / name).read_bytes() for name in
                    ('SJD4PL.BIT', 'SJD4PL.DTB', 'hardware/soundjie_d4_combined.hdf')}
        self.new = dict(self.old)
        self.new['SJD4PL.BIT'] = (EXPORTED / 'soundjie_d4_combined.bit').read_bytes()
        self.new['hardware/soundjie_d4_combined.hdf'] = (EXPORTED / 'soundjie_d4_combined.hdf').read_bytes()

    def test_exact_hdmi_pair_selected_and_legacy_remains_distinct(self):
        self.assertEqual(self.audit.hardware_profile(self.new), 'hdmi_20261023')
        self.assertEqual(self.audit.hardware_profile(self.old), 'legacy_lcd')

    def test_mixed_bit_and_hdf_is_rejected_in_both_directions(self):
        for name in ('SJD4PL.BIT', 'hardware/soundjie_d4_combined.hdf'):
            mixed = dict(self.new)
            mixed[name] = self.old[name]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'hardware SHA'):
                self.audit.hardware_profile(mixed)

    def test_corrupt_hdmi_bit_is_rejected(self):
        self.new['SJD4PL.BIT'] = self.new['SJD4PL.BIT'][:-1] + b'X'
        with self.assertRaisesRegex(ValueError, 'hardware SHA'):
            self.audit.hardware_profile(self.new)

    def test_verified_hdmi_is_not_mistaken_for_retained_lcd_ports(self):
        tree = self.audit.read_dtb(self.new['SJD4PL.DTB'])
        report = self.audit.hardware_semantics(tree, self.new['hardware/soundjie_d4_combined.hdf'])
        self.assertTrue(report['hdmi_output_verified'])
        self.assertEqual(report['display_output'], 'HDMI_TMDS_VIDEO')
        self.assertFalse(self.audit.hardware_semantics(tree, self.old['hardware/soundjie_d4_combined.hdf'])['hdmi_output_verified'])

    def test_new_candidate_is_reproducible_audited_and_not_board_accepted(self):
        config = ROOT / 'config/soundjie_d4_hdmi_bundle_sources.json'
        self.assertTrue(config.is_file(), 'new HDMI candidate input manifest is missing')
        spec = importlib.util.spec_from_file_location('hdmi_builder', ROOT / 'scripts/host/build_soundjie_d4_bundle.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / 'a', Path(folder) / 'b'
            builder.build_local_candidate(config, a)
            builder.build_local_candidate(config, b)
            self.assertEqual((a / 'SJD4A.SHA').read_bytes(), (b / 'SJD4A.SHA').read_bytes())
            self.assertEqual((a / 'SJD4A.TGZ').read_bytes(), (b / 'SJD4A.TGZ').read_bytes())
            report = self.audit.audit_directory(a)
            self.assertFalse(report['board_tested'])
            self.assertTrue(report['hardware']['hdmi_output_verified'])
            self.assertEqual(report['hardware_profile'], 'hdmi_20261023')


if __name__ == '__main__':
    unittest.main()
