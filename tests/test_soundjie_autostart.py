import hashlib
import importlib.util
import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'scripts/host/build_soundjie_autostart.py'


class AutoStartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('autostart_build', MODULE)
        cls.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.m)

    def test_bootstrap_is_one_init_command_with_pinned_script_and_rescue(self):
        script_hash = 'a' * 64
        args = self.m.make_bootargs(script_hash)
        self.assertLess(len(args.encode('ascii')), 1024)
        # The generated string intentionally has no shell escape or nested quote
        # syntax: Linux's double-quote tokenizer must see one -c argument.
        tail = args.split(' -- ', 1)[1]
        parsed = shlex.split(tail)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0], '-c')
        command = parsed[1]
        self.assertIn(script_hash + '  /run/sj-init', command)
        self.assertIn('sha256sum -c - && exec /bin/sh /run/sj-init', command)
        self.assertIn('while true; do /bin/sh -i; sleep 1; done', command)
        self.assertIn('ro rootfstype=ext4 rootflags=noload', args)
        self.assertNotIn('$', command)
        self.assertNotIn('\\', command)

    def test_bootstrap_rejects_injected_or_invalid_hash(self):
        for value in ['x'*64, 'a'*63, 'a'*64+';reboot', 'a'*64+'\n']:
            with self.assertRaises(ValueError):
                self.m.make_bootargs(value)

    def test_environment_blocks_legacy_fallthrough_and_preserves_pl_order(self):
        text = self.m.make_environment('a'*64).decode('ascii')
        env = dict(x.split('=', 1) for x in text.splitlines())
        for hook in ['loadkernel','loaddtb','fpga_loadbit','check_ramdisk','baseargs','tryboot']:
            self.assertEqual(env[hook], 'false')
        self.assertIn('sleep 2', env['uenvcmd'])
        self.assertIn('run sj_loadpl && run sj_loadlinux', env['sj_start'])
        pl = env['sj_loadpl']
        self.assertLess(pl.index('fpga loadb'), pl.index('0xf800b000'))
        self.assertIn('test ${filesize} = 109c10a', pl)
        self.assertNotIn('md.l', text)
        for forbidden in ['saveenv','fatwrite','mmc write','sf write','reset','reboot']:
            self.assertNotIn(forbidden, text)

    def test_shell_syntax_on_host(self):
        bash = Path('C:/Program Files/Git/bin/bash.exe')
        if not bash.exists():
            self.skipTest('Git Bash unavailable; board dash check still required')
        script = ROOT / 'scripts/board/soundjie_autostart.sh'
        result = subprocess.run([str(bash), '-n', str(script)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        command = shlex.split(self.m.make_bootargs('a'*64).split(' -- ',1)[1])[1]
        result = subprocess.run([str(bash), '-n', '-c', command], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_startup_refuses_non_pid1_and_corrupt_bundle(self):
        bash = Path('C:/Program Files/Git/bin/bash.exe')
        if not bash.exists():
            self.skipTest('Git Bash unavailable')
        script = ROOT / 'scripts/board/soundjie_autostart.sh'
        for args in [[], ['--prepare'], ['--unknown']]:
            result = subprocess.run([str(bash), str(script), *args], capture_output=True)
            self.assertEqual(result.returncode, 64, result.stderr.decode())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); (p/'data').write_bytes(b'changed')
            (p/'SHA256SUMS').write_text('0'*64+'  data\n',encoding='ascii')
            result = subprocess.run([str(bash),str(script),'--check-bundle',p.as_posix()],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertNotIn(b'SJ_AUTOSTART_BUNDLE_CHECK_OK',result.stdout)
            self.assertEqual((p/'data').read_bytes(),b'changed')

    def test_kernel_414_tokenizer_keeps_embedded_equals_and_spaces(self):
        # Independent restricted model of next_arg+repair_env_string from
        # Xilinx 4.14 kernel/params.c and init/main.c; not a shell tokenizer.
        def kernel_words(text):
            values=[]
            while text.strip():
                text=text.lstrip(); quoted=text.startswith('"')
                if quoted: text=text[1:]
                inside=quoted; end=0
                for end,ch in enumerate(text):
                    if ch.isspace() and not inside: break
                    if ch=='"': inside=not inside
                else: end=len(text)
                word=text[:end]
                if quoted and word.endswith('"'): word=word[:-1]
                # Kernel temporarily splits at first =, then set_init_arg
                # repairs it. The resulting argument must preserve both sides.
                values.append(word)
                text=text[end:]
            return values
        args=self.m.make_bootargs('b'*64).split(' -- ',1)[1]
        words=kernel_words(args)
        self.assertEqual(words,shlex.split(args))
        self.assertIn('size=256m',words[1])

    def test_tampered_package_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); app = root/'bad.tgz'; app.write_bytes(b'wrong model')
            out = root/'release'
            with self.assertRaisesRegex(ValueError, 'application'):
                self.m.build_release(out, app, ROOT/'artifacts/sd-emmc-stage-20261024a/uEnv.txt')
            self.assertFalse(out.exists())

    def test_output_collision_does_not_touch_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'keep').write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                self.m.build_release(root, root/'missing', root/'missing')
            self.assertEqual((root/'keep').read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
