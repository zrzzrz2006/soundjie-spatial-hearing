import importlib.util
from pathlib import Path
import tempfile
import tkinter
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CandidateGuardTests(unittest.TestCase):
    def guard(self, project, part='xc7z100ffg900-2', cells='processing_system7_0'):
        interp = tkinter.Tcl()
        interp.call('source', str(ROOT / 'scripts/vivado/emmc_candidate_guard.tcl'))
        expected = '/approved/candidate/system.xpr'
        return interp.call('emmc_candidate_guard', project, expected, part, cells)

    def test_accepts_exact_project_part_and_single_ps(self):
        self.assertEqual(self.guard('/approved/candidate/system.xpr'), 'candidate_scope_verified')

    def test_rejects_old_project_wrong_part_and_multiple_ps(self):
        for project, part, cells in (
            ('/old/system.xpr', 'xc7z100ffg900-2', 'ps0'),
            ('/approved/candidate/system.xpr', 'xc7z020clg400-1', 'ps0'),
            ('/approved/candidate/system.xpr', 'xc7z100ffg900-2', 'ps0 ps1'),
            ('/approved/candidate/system.xpr', 'xc7z100ffg900-2', ''),
        ):
            with self.subTest(project=project, part=part, cells=cells):
                with self.assertRaisesRegex(tkinter.TclError, 'CANDIDATE_SCOPE'):
                    self.guard(project, part, cells)


class SourceBindingTests(unittest.TestCase):
    def test_binding_rejects_mutation_missing_file_and_existing_manifest(self):
        spec = importlib.util.spec_from_file_location('bind', ROOT / 'scripts/host/bind_emmc_sources.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.v'
            source.write_bytes(b'original')
            manifest = root / 'binding.json'
            module.create(manifest, [source])
            self.assertEqual(module.verify(manifest), 1)
            with self.assertRaises(FileExistsError):
                module.create(manifest, [source])
            source.write_bytes(b'mutated!')
            with self.assertRaisesRegex(ValueError, 'source drift'):
                module.verify(manifest)
            source.unlink()  # only this test's own temporary fixture
            with self.assertRaises(OSError):
                module.verify(manifest)
