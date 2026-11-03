"""Audit the real development archive; mutations must fail closed."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import shutil
import unittest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'artifacts/development-bundle-20261022c'


class PackageAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tarfile.open(PACKAGE / 'SJD4DEV.TGZ') as archive:
            cls.members = {m.name: archive.extractfile(m).read() for m in archive}
        cls.hardware = {n: (PACKAGE / n).read_bytes() for n in
                        ('SJD4PL.BIT', 'SJD4PL.DTB', 'hardware/soundjie_d4_combined.hdf')}

    def setUp(self):
        path = ROOT / 'scripts/host/audit_soundjie_d4_package.py'
        self.assertTrue(path.is_file(), 'package-byte auditor is missing')
        spec = importlib.util.spec_from_file_location('d4audit', path)
        self.audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.audit)

    def test_actual_package_has_matching_model_and_hardware_without_board_claim(self):
        report = self.audit.audit_payloads(self.members, self.hardware)
        self.assertEqual(report['status'], 'HOST_BYTES_CHECKED_BOARD_PENDING')
        self.assertFalse(report['board_tested'])
        self.assertEqual(report['model_sections_matched'], 6)
        self.assertEqual(report['hardware']['vdma_frame_stores'], 3)
        self.assertEqual(report['hardware']['dpu_spi'], 31)
        self.assertEqual(report['hardware'].get('display_output'), 'LCD_RGB_NO_HDMI')
        self.assertEqual(report['elf']['runtime/soundjie_d4_runtime']['max_glibc'], '2.17')

    def test_missing_launcher_rejected_even_if_manifest_were_regenerated(self):
        members = dict(self.members)
        del members['probe/run_soundjie_d4_session.sh']
        with self.assertRaisesRegex(ValueError, 'required'):
            self.audit.audit_payloads(members, self.hardware)

    def test_arm_header_rejects_x86_and_soft_float(self):
        original = self.members['runtime/soundjie_d4_runtime']
        for offset, replacement in ((18, b'\x03\x00'), (36, b'\x00\x00\x00\x05')):
            payload = bytearray(original)
            payload[offset:offset + len(replacement)] = replacement
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'ARM|float'):
                self.audit.elf_contract(bytes(payload), executable=True)

    def test_glibc_version_too_new_is_rejected(self):
        payload = self.members['runtime/soundjie_d4_runtime']
        self.assertIn(b'GLIBC_2.17', payload)
        with self.assertRaisesRegex(ValueError, 'GLIBC'):
            self.audit.elf_contract(payload.replace(b'GLIBC_2.17', b'GLIBC_2.99'), executable=True)

    def test_changed_linked_dpu_section_rejected(self):
        model = self.members['model/dpu_soundjie_d3.elf']
        classifier = bytearray(self.members['runtime/soundjie_d4_classifier'])
        sections = self.audit.elf_sections(bytes(classifier))
        section = sections['.deephi.code.soundjie_d3']
        classifier[section['offset']] ^= 1
        with self.assertRaisesRegex(ValueError, 'DPU section'):
            self.audit.compare_model_sections(model, bytes(classifier))

    def test_hardware_pair_change_rejected(self):
        hardware = dict(self.hardware)
        hardware['SJD4PL.BIT'] = hardware['SJD4PL.BIT'][:-1] + b'X'
        with self.assertRaisesRegex(ValueError, 'hardware SHA'):
            self.audit.audit_payloads(self.members, hardware)

    def test_semantic_device_tree_checks_actual_properties(self):
        tree = self.audit.read_dtb(self.hardware['SJD4PL.DTB'])
        self.assertEqual(tree['/amba_pl/dma@43000000']['xlnx,num-fstores'], b'\x00\x00\x00\x03')
        tree['/amba_pl/dma@43000000']['xlnx,num-fstores'] = b'\x00\x00\x00\x01'
        with self.assertRaisesRegex(ValueError, 'frame stores'):
            self.audit.hardware_semantics(tree, self.hardware['hardware/soundjie_d4_combined.hdf'])

    def test_changed_payload_cannot_pass_old_manifest(self):
        members = dict(self.members)
        members['model/labels.txt'] += b'fake\n'
        with self.assertRaisesRegex(ValueError, 'manifest|checksum'):
            self.audit.audit_payloads(members, self.hardware)

    def test_inner_checksum_file_is_verified(self):
        members = dict(self.members)
        members['manifest/SHA256SUMS'] = b'0' * 64 + b'  model/labels.txt\n'
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.audit.audit_payloads(members, self.hardware)

    def test_executable_mode_is_required(self):
        members = dict(self.members)
        manifest = json.loads(members['manifest/development.json'])
        next(m for m in manifest['members'] if m['name'] == 'runtime/soundjie_d4_runtime')['mode'] = '0644'
        members['manifest/development.json'] = json.dumps(manifest).encode()
        sums = members['manifest/SHA256SUMS'].decode().splitlines()
        members['manifest/SHA256SUMS'] = ('\n'.join(
            hashlib.sha256(members['manifest/development.json']).hexdigest() + '  manifest/development.json'
            if line.endswith('  manifest/development.json') else line for line in sums) + '\n').encode()
        with self.assertRaisesRegex(ValueError, 'executable mode'):
            self.audit.audit_payloads(members, self.hardware)

    def test_local_candidate_blocks_current_lcd_only_hardware_before_output(self):
        spec = importlib.util.spec_from_file_location('d4pack_candidate', ROOT / 'scripts/host/build_soundjie_d4_bundle.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        self.assertTrue(hasattr(builder, 'build_local_candidate'), 'local candidate gate is missing')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with self.assertRaisesRegex(ValueError, 'HDMI'):
                builder.build_local_candidate(ROOT / 'config/soundjie_d4_bundle_sources.json', folder / 'a')
            self.assertFalse((folder / 'a').exists())

    def test_outer_checksum_is_verified_before_opening_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            shutil.copyfile(PACKAGE / 'SJD4DEV.TGZ', folder / 'SJD4DEV.TGZ')
            (folder / 'SJD4DEV.SHA').write_text('0' * 64 + '  SJD4DEV.TGZ\n', encoding='ascii')
            with self.assertRaisesRegex(ValueError, 'outer checksum'):
                self.audit.audit_directory(folder)


if __name__ == '__main__':
    unittest.main()
