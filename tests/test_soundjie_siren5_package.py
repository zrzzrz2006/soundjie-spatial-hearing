import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from scripts.host import package_soundjie_siren5 as p


class SirenPackageTests(unittest.TestCase):
    def test_archive_roundtrip_and_create_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'patch.tgz';payload={'model/model.elf':b'bytes','config/class-contract.txt':b'soundjie-siren5-v1\n'}
            p.archive(payload,out)
            old=out.read_bytes()
            with tarfile.open(out) as f:
                self.assertEqual(f.extractfile('sj-siren5/config/class-contract.txt').read(),payload['config/class-contract.txt'])
            with self.assertRaises(FileExistsError):p.archive(payload,out)
            self.assertEqual(old,out.read_bytes())

    def test_archive_rejects_unsafe_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('../escape','/absolute','C:/bad','foo\\bar','a/../../b'):
                with self.assertRaises(ValueError):p.archive({name:b'bad'},Path(tmp)/'bad.tgz')

    def test_class_contract_metadata_matches_physical_not_display_count(self):
        report='PHYSICAL_OUTPUTS=8\nMODEL_SHA256=abc\n'
        with self.assertRaises(ValueError):p.check_classifier_contract(report,'abc')
        with self.assertRaises(ValueError):p.check_classifier_contract(report.replace('=8','=6'),'other')
        p.check_classifier_contract(report.replace('=8','=6'),'abc')

if __name__=='__main__':unittest.main()
