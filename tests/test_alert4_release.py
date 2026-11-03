import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from scripts.host.assemble_soundjie_alert4_release import read_archive
from scripts.host import assemble_soundjie_alert4_release as release

class ReleaseArchiveTests(unittest.TestCase):
    def archive(self,entries):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz') as t:
            for name,data,kind in entries:
                m=tarfile.TarInfo(name);m.type=kind;m.size=len(data) if kind==tarfile.REGTYPE else 0
                t.addfile(m,io.BytesIO(data) if kind==tarfile.REGTYPE else None)
        return stream.getvalue()

    def read(self,data,prefix=''):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.tgz';p.write_bytes(data)
            return read_archive(p,hashlib.sha256(data).hexdigest(),prefix)

    def test_release_tar_preserves_diagnostic_and_program_execution(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'release.tgz'
            payload={n:b'data' for n in ['dnndk/pkgs/bin/dexplorer','runtime/soundjie_d4_runtime',
                'audio/bin/arecord','audio/load-usb-audio.sh','dnndk/pkgs/lib/libn2cube.so']}
            release.write_archive(p,payload)
            with tarfile.open(p) as t:
                for name in payload:
                    expected=0o644 if '/lib/' in name else 0o755
                    self.assertEqual(t.getmember('sj-alert4-release/'+name).mode,expected)

    def test_normal_file_and_prefix(self):
        data=self.archive([('pkg/audio/bin/arecord',b'file',tarfile.REGTYPE)])
        self.assertEqual(self.read(data,'pkg/'),{'audio/bin/arecord':b'file'})

    def test_escape_link_device_and_duplicate_rejected(self):
        for name,kind in [('pkg/../user',tarfile.REGTYPE),('/user',tarfile.REGTYPE),
                          ('pkg/link',tarfile.SYMTYPE),('pkg/device',tarfile.CHRTYPE),
                          ('pkg/back\\slash',tarfile.REGTYPE)]:
            with self.subTest(name=name),self.assertRaises(ValueError):
                self.read(self.archive([(name,b'',kind)]),'pkg/')
        with self.assertRaises(ValueError):
            self.read(self.archive([('a',b'1',tarfile.REGTYPE),('a',b'2',tarfile.REGTYPE)]))

    def test_hash_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a';p.write_bytes(self.archive([('a',b'b',tarfile.REGTYPE)]))
            with self.assertRaises(ValueError):read_archive(p,'0'*64)
