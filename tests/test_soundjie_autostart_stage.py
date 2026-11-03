import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/host'))


class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('auto_stage',ROOT/'scripts/host/stage_soundjie_autostart.py')
        cls.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.m)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.sd=self.root/'card';self.sd.mkdir()
        self.src=self.root/'release';self.src.mkdir();self.backup=self.root/'backup'
        self.old=b'original environment\r\n';self.new=b'candidate environment\n'
        (self.sd/'uEnv.txt').write_bytes(self.old)
        (self.sd/'BOOT.bin').write_bytes(b'original boot')
        (self.sd/'user-audio.wav').write_bytes(b'user original audio')
        for n,b in [('uEnv.txt',self.new),('SJAUTO1.SH',b'candidate script'),('SJUBAK1.ENV',self.old)]:
            (self.src/n).write_bytes(b)
        h=lambda b:hashlib.sha256(b).hexdigest()
        self.manifest=dict(required_files={'uEnv.txt':h(self.old),'BOOT.bin':h(b'original boot')},files={n:dict(bytes=p.stat().st_size,sha256=h(p.read_bytes())) for n in ['uEnv.txt','SJAUTO1.SH','SJUBAK1.ENV'] for p in [self.src/n]})
        self.loader=patch.object(self.m,'load_release',return_value=self.manifest)
        self.loader.start();self.addCleanup(self.loader.stop)
        self.identity=lambda p:dict(disk_serial='fixture',volume='fixture')

    def run_stage(self,**kwargs):
        return self.m.stage_release(self.src,self.sd,self.backup,identity_check=kwargs.pop('identity_check',self.identity),**kwargs)

    def test_success_backs_up_then_changes_only_environment(self):
        r=self.run_stage()
        self.assertEqual((self.sd/'uEnv.txt').read_bytes(),self.new)
        self.assertEqual((self.sd/'SJUBAK1.ENV').read_bytes(),self.old)
        self.assertEqual((self.backup/'uEnv.txt').read_bytes(),self.old)
        self.assertEqual((self.sd/'BOOT.bin').read_bytes(),b'original boot')
        self.assertEqual((self.sd/'user-audio.wav').read_bytes(),b'user original audio')
        self.assertEqual(r['unchanged_original_files'],2)
        self.assertFalse((self.sd/'SJAUTO1.NEW').exists())

    def test_wrong_identity_makes_no_changes(self):
        def reject(p):raise ValueError('wrong card')
        with self.assertRaisesRegex(ValueError,'wrong card'):self.run_stage(identity_check=reject)
        self.assertEqual(sorted(p.name for p in self.sd.iterdir()),['BOOT.bin','uEnv.txt','user-audio.wav'])

    def test_existing_backup_is_never_overwritten(self):
        self.backup.mkdir();(self.backup/'keep').write_bytes(b'keep')
        with self.assertRaises(FileExistsError):self.run_stage()
        self.assertEqual((self.backup/'keep').read_bytes(),b'keep')
        self.assertEqual((self.sd/'uEnv.txt').read_bytes(),self.old)

    def test_destination_collision_is_preserved(self):
        (self.sd/'SJAUTO1.SH').write_bytes(b'previous work')
        with self.assertRaises(FileExistsError):self.run_stage()
        self.assertEqual((self.sd/'SJAUTO1.SH').read_bytes(),b'previous work')
        self.assertEqual((self.sd/'uEnv.txt').read_bytes(),self.old)

    def test_original_hash_mismatch_rejected(self):
        (self.sd/'BOOT.bin').write_bytes(b'different boot')
        with self.assertRaisesRegex(ValueError,'original'):self.run_stage()
        self.assertEqual((self.sd/'uEnv.txt').read_bytes(),self.old)

    def test_pre_activation_failure_leaves_original_and_backup(self):
        with patch.object(self.m.os,'replace',side_effect=OSError('interrupted activation')):
            with self.assertRaisesRegex(OSError,'interrupted activation'):self.run_stage()
        self.assertEqual((self.sd/'uEnv.txt').read_bytes(),self.old)
        self.assertEqual((self.backup/'uEnv.txt').read_bytes(),self.old)
        self.assertEqual((self.sd/'SJAUTO1.NEW').read_bytes(),self.new)

    def test_script_write_failure_never_activates_environment(self):
        real=self.m.write_new_verified
        def fail_script(path,data):
            if path.name=='SJAUTO1.SH':raise OSError('disk full')
            return real(path,data)
        with patch.object(self.m,'write_new_verified',side_effect=fail_script):
            with self.assertRaisesRegex(OSError,'disk full'):self.run_stage()
        self.assertEqual((self.sd/'uEnv.txt').read_bytes(),self.old)
        self.assertEqual((self.backup/'uEnv.txt').read_bytes(),self.old)

    def test_backup_inside_card_rejected(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            self.m.stage_release(self.src,self.sd,self.sd/'backup',identity_check=self.identity)
        self.assertEqual((self.sd/'uEnv.txt').read_bytes(),self.old)


if __name__=='__main__':unittest.main()
