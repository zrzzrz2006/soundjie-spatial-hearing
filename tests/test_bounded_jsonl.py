import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def linux(path):
    p=Path(path).resolve()
    return '/mnt/'+p.drive[0].lower()+p.as_posix().split(':',1)[1]

class BoundedJsonlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.superpowers',prefix='bounded-')
        cls.binary=Path(cls.tmp.name)/'driver'
        sources=['tests/fixtures/bounded_jsonl_driver.cpp','src/soundjie_d4_doa_stream.cpp',
                 'src/soundjie_d4_state.cpp','src/xvf3800_control_protocol.cpp']
        r=subprocess.run(['wsl','-d','Ubuntu','--','g++','-std=c++11','-O2','-Wall','-Wextra','-Werror',
                          '-I'+linux(ROOT/'src'),*[linux(ROOT/s) for s in sources],'-o',linux(cls.binary)],capture_output=True)
        if r.returncode:
            cls.tmp.cleanup()
            raise AssertionError(r.stderr.decode(errors='replace'))

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def run_case(self,mode,cap,existing=False):
        with tempfile.TemporaryDirectory(dir=self.tmp.name) as d:
            p=Path(d)/'log.jsonl'
            if existing:p.write_text('ORIGINAL')
            r=subprocess.run(['wsl','-d','Ubuntu','--',linux(self.binary),linux(p),mode,str(cap)],capture_output=True,timeout=60)
            files={f.name:f.read_bytes() for f in Path(d).iterdir()}
            return r,files

    def test_rotations_keep_complete_latest_rows_and_doa_fresh_then_stale(self):
        r,files=self.run_case('follow',4096)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertGreater(json.loads(r.stdout)['rotations'],200)
        self.assertEqual(set(files),{'log.jsonl'})
        self.assertLessEqual(len(files['log.jsonl']),4096)
        rows=[json.loads(s) for s in files['log.jsonl'].splitlines()]
        self.assertEqual(rows[-1]['monotonic_ms'],500900)

    def test_unlimited_retains_all_diagnostic_rows(self):
        r,files=self.run_case('write',0)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(json.loads(r.stdout)['rotations'],0)
        self.assertEqual(len(files['log.jsonl'].splitlines()),5000)

    def test_existing_output_is_preserved(self):
        r,files=self.run_case('write',4096,True)
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(files,{'log.jsonl':b'ORIGINAL'})

    def test_oversize_row_fails_without_growth(self):
        r,files=self.run_case('oversize',4096)
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(files,{'log.jsonl':b''})

    def test_externally_replaced_output_is_not_overwritten(self):
        r,files=self.run_case('replace',4096)
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(files['log.jsonl'],b'ORIGINAL')
        self.assertEqual(files['log.jsonl.old'],b'{}\n')

    def test_foreign_rotation_temporary_is_not_deleted(self):
        r,files=self.run_case('temp-conflict',4096)
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(files['log.jsonl'],b'{}\n')
        self.assertEqual(len(files),2)
        self.assertIn(b'ORIGINAL',files.values())
