"""Development-package tests; deliberately not board/ABI acceptance tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DevelopmentBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        module = ROOT / 'scripts/host/build_soundjie_d4_bundle.py'
        self.assertTrue(module.is_file(), 'D4 development packager is missing')
        spec = importlib.util.spec_from_file_location('d4pack', module)
        self.pack = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pack)
        self.source = self.root / 'launch.sh'
        self.source.write_bytes(b'#!/bin/sh\r\necho ready\r\n')
        self.config = self.root / 'sources.json'
        self.data = {'schema': 'soundjie-d4-bundle-inputs-v1',
                     'status': 'preflight_only_not_packaged',
                     'source_commit': 'a' * 40,
                     'pending_gates': ['physical test', 'license audit'],
                     'archive_members': [self.entry('probe/start.sh', self.source, True)],
                     'external_hardware': []}

    def entry(self, name, source, normalize=False):
        content = source.read_bytes()
        return {'name': name, 'source': str(source), 'size': len(content),
                'sha256': hashlib.sha256(content).hexdigest(), 'mode': '0755',
                'normalize_lf': normalize}

    def build(self, target='out'):
        self.config.write_text(json.dumps(self.data), encoding='utf-8')
        return self.pack.build_development_bundle(self.config, self.root / target)

    def test_deterministic_package_retains_pending_gates_and_normalizes_shell(self):
        self.build('one')
        self.build('two')
        a = self.root / 'one/SJD4DEV.TGZ'
        self.assertEqual(a.read_bytes(), (self.root / 'two/SJD4DEV.TGZ').read_bytes())
        with tarfile.open(a) as archive:
            self.assertEqual(archive.extractfile('probe/start.sh').read(), b'#!/bin/sh\necho ready\n')
            manifest = json.load(archive.extractfile('manifest/development.json'))
            self.assertFalse(manifest['board_tested'])
            self.assertEqual(manifest['status'], 'DEVELOPMENT_ONLY_NOT_RELEASED')
            self.assertEqual(manifest['pending_gates'], ['physical test', 'license audit'])
            self.assertNotEqual(manifest['members'][0]['source_sha256'], manifest['members'][0]['sha256'])
        self.assertFalse((self.root / 'one/SJD4A.TGZ').exists())

    def test_refuses_existing_output_without_changes(self):
        self.build()
        before = (self.root / 'out/SJD4DEV.TGZ').read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(before, (self.root / 'out/SJD4DEV.TGZ').read_bytes())

    def test_corrupt_input_fails_before_creating_output(self):
        self.source.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.root / 'out').exists())

    def test_rejects_audio_traversal_duplicates_and_reserved_metadata(self):
        for name in ('train/a.wav', 'audio/recording.flac', 'audio/training.aiff',
                     'assets/training.wav.gz', 'runtime/SJD4PL.BIT', '../escape', '/absolute',
                     'probe/../escape', 'probe\\escape', 'manifest/development.json'):
            with self.subTest(name=name):
                self.data['archive_members'][0]['name'] = name
                self.data['archive_members'][0]['normalize_lf'] = False
                target = 'bad-' + hashlib.sha256(name.encode()).hexdigest()[:12]
                with self.assertRaises(ValueError):
                    self.build(target)
                self.assertFalse((self.root / target).exists())
        self.data['archive_members'][0]['name'] = 'probe/start.sh'
        self.data['archive_members'].append(dict(self.data['archive_members'][0]))
        with self.assertRaises(ValueError):
            self.build()

    def test_rejects_regular_file_as_parent_of_another_member(self):
        self.data['archive_members'] = [self.entry('probe/tool.sh', self.source),
                                       self.entry('probe/tool.sh/start.sh', self.source)]
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.root / 'out').exists())

    def test_hardware_is_separate_and_hash_bound(self):
        hw = self.root / 'hardware.bin'
        hw.write_bytes(b'hardware-fixture')
        self.data['external_hardware'] = [self.entry('SJD4PL.BIT', hw)]
        self.build()
        self.assertEqual((self.root / 'out/SJD4PL.BIT').read_bytes(), hw.read_bytes())
        with tarfile.open(self.root / 'out/SJD4DEV.TGZ') as archive:
            self.assertNotIn('SJD4PL.BIT', archive.getnames())
            metadata = json.load(archive.extractfile('manifest/development.json'))
            self.assertEqual(metadata['external_hardware'][0]['sha256'], hashlib.sha256(hw.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
