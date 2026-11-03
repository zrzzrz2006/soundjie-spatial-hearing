import subprocess
import unittest
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


class DiagnosticOverlayTests(unittest.TestCase):
    def test_symlink_parent_cannot_escape_ram_before_hash_checks(self):
        source=ROOT/'scripts/board/prepare_soundjie_diagnostic_overlay.sh'
        linux='/mnt/'+source.drive[0].lower()+source.as_posix().split(':',1)[1]
        probe=r'''
import pathlib,subprocess,sys,tempfile
with tempfile.TemporaryDirectory(prefix='sj-overlay-test-',dir='/run') as name:
 root=pathlib.Path(name)
 (root/'base').mkdir();(root/'patch').mkdir()
 (root/'patch/SHA256SUMS').write_text('')
 (root/'redirect').symlink_to('/tmp',target_is_directory=True)
 result=subprocess.run(['sh',sys.argv[1],str(root/'base'),str(root/'patch'),str(root/'redirect/never-created-sj')],capture_output=True)
 print(result.stderr.decode())
 assert result.returncode!=0
 assert b'overlay parent must be real /run on tmpfs' in result.stderr
 assert not pathlib.Path('/tmp/never-created-sj').exists()
'''
        result=subprocess.run(['wsl.exe','-d','Ubuntu','-u','root','--','python3','-c',probe,linux],capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout.decode(errors='replace')+result.stderr.decode(errors='replace'))


if __name__=='__main__':unittest.main()
