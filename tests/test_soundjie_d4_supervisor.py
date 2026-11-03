import json
import os
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/board/run_soundjie_d4_session.sh'


def linux(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


class D4SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.superpowers')
        self.root = Path(self.temp.name)
        # DrvFS cannot represent a Linux FIFO. Board /run and Linux /tmp can.
        self.session = '/tmp/sj-supervisor-' + uuid.uuid4().hex
        self.bundle = self.root / 'bundle with spaces'
        for folder in ('runtime', 'audio/bin', 'config', 'assets/hdmi'):
            (self.bundle / folder).mkdir(parents=True)
        for name in ('config/soundjie_d4_runtime.tsv', 'assets/hdmi/soundjie_glyph_atlas.bin'):
            (self.bundle / name).write_bytes(b'fixture')
        (self.bundle/'config/log-policy.txt').write_bytes(b'bounded-jsonl-v1\n')
        paths = {'capture': 'audio/bin/arecord', 'runtime': 'runtime/soundjie_d4_runtime',
                 'doa': 'runtime/xvf3800_doa', 'ui': 'runtime/soundjie_hdmi_ui',
                 'classifier': 'runtime/soundjie_d4_classifier'}
        for role, relative in paths.items():
            body = f'''#!/bin/sh
role={role}
printf '%s\\n' "$@" > "$TEST_ROOT/$role.args"
if [ "$role" = ui ]; then
  case " $* " in *" --frames 0 "*) :;; *) printf 'cleared\\n' > "$TEST_ROOT/cleared"; exit 0;; esac
  printf '%s\\n' "$@" > "$TEST_ROOT/ui.live.args"
fi
printf '%s\\n' "$$" > "$TEST_ROOT/$role.started"
trap 'printf "stopped\\n" > "$TEST_ROOT/$role.stopped"; exit 0' TERM INT
if [ "$FAIL_ROLE" = "$role" ]; then sleep .3; exit 42; fi
while :; do
  if [ "$role" = capture ]; then printf '\\000\\000\\000\\000'; fi
  sleep .05
done
'''
            path = self.bundle / relative
            path.write_bytes(body.encode())
            path.chmod(0o755)

    def tearDown(self):
        # Only this test's UUID-owned Linux directory; never a user directory.
        if self.session.startswith('/tmp/sj-supervisor-'):
            cleanup = '''import os, pathlib, shutil, signal, sys, time
p = sys.argv[1]
assert p.startswith('/tmp/sj-supervisor-') and len(p) == 51
for entry in pathlib.Path('/proc').iterdir():
    if not entry.name.isdigit(): continue
    try:
        args = (entry / 'cmdline').read_bytes().split(b'\\0')
        pid = int(entry.name)
        if b'd4-child' in args and any(a.startswith((p+'/logs/').encode()) for a in args) and os.getpgid(pid) == pid:
            os.killpg(pid, signal.SIGKILL)
    except (FileNotFoundError, ProcessLookupError, PermissionError): pass
time.sleep(.1)
shutil.rmtree(p, ignore_errors=True)
'''
            subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'python3', '-c',
                            cleanup, self.session], capture_output=True, timeout=5, check=True)
        self.temp.cleanup()

    def read_session(self, name):
        result = subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'cat', self.session + '/' + name], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        return result.stdout.decode()

    def command(self, failed=''):
        return ['wsl.exe', '-d', 'Ubuntu', '--', 'env', 'TEST_ROOT=' + linux(self.root),
                'FAIL_ROLE=' + failed, 'sh', linux(SCRIPT), linux(self.bundle),
                self.session, '/dev/fb0']

    def test_any_child_exit_fails_session_stops_peers_and_clears_screen(self):
        self.check_child_failure('capture')

    def test_runtime_exit_stops_session(self):
        self.check_child_failure('runtime')

    def test_doa_exit_stops_session(self):
        self.check_child_failure('doa')

    def test_ui_exit_stops_session(self):
        self.check_child_failure('ui')

    def check_child_failure(self, failed):
        result = subprocess.run(self.command(failed), capture_output=True, timeout=12)
        self.assertEqual(result.returncode, 2, result.stderr.decode(errors='replace'))
        status = json.loads(self.read_session('status.json'))
        # Runtime EOF can also terminate capture with SIGPIPE before the poll.
        # status names the first observed exit, not a claimed root cause.
        allowed = {failed + '_exited'}
        if failed == 'runtime':
            allowed.add('capture_exited')
        self.assertIn(status['reason'], allowed)
        observed = status['reason'].removesuffix('_exited')
        self.assertEqual(status['child_exit'], int(self.read_session('logs/' + observed + '.exit')))
        self.assertEqual(int(self.read_session('logs/' + failed + '.exit')), 42)
        self.assertEqual(status['clear_exit'], 0)
        self.assertTrue((self.root / 'cleared').is_file())
        for role in ('capture', 'runtime', 'doa', 'ui'):
            if role == failed:
                continue
            if failed == 'runtime' and role == 'capture':
                continue  # Broken FIFO may deliver SIGPIPE, not TERM.
            self.assertTrue((self.root / (role + '.stopped')).is_file(), role)
        state = json.loads(self.read_session('state/current.json'))
        self.assertFalse(state['usb_ok'])
        self.assertFalse(state['dpu_ok'])
        self.assertFalse(state['doa_valid'])
        self.assertEqual(state['class_id'], -1)
        self.assertIn('--classifier-timeout-ms', (self.root / 'runtime.args').read_text())
        for role in ('runtime','doa'):
            args=(self.root/(role+'.args')).read_text().splitlines()
            self.assertEqual(args[args.index('--log-max-bytes')+1],'1048576')
        args=(self.root/'ui.live.args').read_text().splitlines()
        self.assertEqual(args[args.index('--log-max-bytes')+1],'1048576')
        self.assertEqual(args[args.index('--jsonl')+1],self.session+'/logs/ui.jsonl')
        self.assertNotIn('--jsonl',(self.root/'ui.args').read_text())

    def test_legacy_bundle_does_not_receive_new_log_arguments(self):
        (self.bundle/'config/log-policy.txt').unlink()
        r=subprocess.run(self.command('capture'),capture_output=True,timeout=30)
        self.assertEqual(r.returncode,2,r.stderr)
        for role in ('runtime','doa','ui.live'):
            args=(self.root/(role+'.args')).read_text().splitlines()
            self.assertNotIn('--log-max-bytes',args)
            if role=='ui.live':self.assertNotIn('--jsonl',args)

    def test_existing_session_is_never_reused(self):
        session = self.root / 'session'
        session.mkdir()
        self.session = linux(session)
        (session / 'sentinel').write_bytes(b'keep')
        result = subprocess.run(self.command(), capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('already exists', result.stderr.decode(errors='replace'))
        self.assertEqual((session / 'sentinel').read_bytes(), b'keep')
        self.assertEqual(list(self.root.glob('*.started')), [])

    def test_timeout_group_signal_is_requested_stop_not_supervisor_error(self):
        command = self.command()
        at = command.index('sh')
        command[at:at] = ['timeout', '-k', '3', '2']
        result = subprocess.run(command, capture_output=True, timeout=12)
        self.assertEqual(result.returncode, 124, result.stderr.decode(errors='replace'))
        status = json.loads(self.read_session('status.json'))
        self.assertEqual(status['reason'], 'requested_stop')
        self.assertEqual(status['exit_code'], 0)
        state = json.loads(self.read_session('state/current.json'))
        self.assertEqual(state['session_state'], 'stopped')
        self.assertEqual(state['class_id'], -1)
        self.assertFalse(state['doa_valid'])

    def test_missing_dependency_fails_before_starting_children(self):
        (self.bundle / 'runtime/xvf3800_doa').rename(self.bundle / 'runtime/xvf3800_doa.absent')
        result = subprocess.run(self.command(), capture_output=True, timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing executable', result.stderr.decode(errors='replace'))
        self.assertEqual(subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'test', '-e', self.session], capture_output=True).returncode, 1)
        self.assertEqual(list(self.root.glob('*.started')), [])

    def test_low_space_stops_owned_children_and_clears_screen(self):
        # The injected df reports one healthy check, then low space. No disk
        # filling and no deletion of user data is involved in this test.
        bindir = self.root / 'fake tools'
        bindir.mkdir()
        df = bindir / 'df'
        df.write_bytes(b'''#!/bin/sh
if [ -e "$TEST_ROOT/df-first" ]; then free=100; else
  free=999999; printf 'checked\\n' > "$TEST_ROOT/df-first"
fi
printf 'Filesystem 1024-blocks Used Available Capacity Mounted\\n'
printf 'test 1000000 1 %s 1%% /test\\n' "$free"
''')
        df.chmod(0o755)
        command = self.command()
        command.insert(command.index('sh'),
                       'PATH=' + linux(bindir) + ':/usr/bin:/bin')
        result = subprocess.run(command, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 2, result.stderr.decode(errors='replace'))
        status = json.loads(self.read_session('status.json'))
        self.assertEqual(status['reason'], 'storage_low')
        self.assertEqual(status['clear_exit'], 0)
        for role in ('capture', 'runtime', 'doa', 'ui'):
            self.assertTrue((self.root / (role + '.stopped')).is_file(), role)
        state = json.loads(self.read_session('state/current.json'))
        self.assertEqual(state['session_state'], 'fault')
        self.assertEqual(state['class_id'], -1)
        self.assertFalse(state['dpu_ok'])

    def test_missing_space_evidence_refuses_to_start(self):
        bindir = self.root / 'fake tools'
        bindir.mkdir()
        df = bindir / 'df'
        df.write_bytes(b'#!/bin/sh\nexit 1\n')
        df.chmod(0o755)
        command = self.command()
        command.insert(command.index('sh'),
                       'PATH=' + linux(bindir) + ':/usr/bin:/bin')
        result = subprocess.run(command, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 66)
        self.assertEqual(list(self.root.glob('*.started')), [])
        self.assertFalse((self.root / 'cleared').exists())

    def test_tv_mode_requires_assets_before_starting_capture(self):
        command=self.command();command.insert(command.index('sh'),'SOUNDJIE_TV_ENABLE=1')
        result=subprocess.run(command,capture_output=True,timeout=5)
        self.assertNotEqual(result.returncode,0)
        self.assertIn(b'missing TV file',result.stderr)
        self.assertEqual(list(self.root.glob('*.started')),[])

    def test_tv_mode_forwards_quoted_reference_paths_to_single_runtime(self):
        (self.bundle/'tv').mkdir()
        for name in ('reference.features','cues.tsv','traffic.bin'):(self.bundle/'tv'/name).write_bytes(b'fixture')
        command=self.command('ui');command.insert(command.index('sh'),'SOUNDJIE_TV_ENABLE=1')
        result=subprocess.run(command,capture_output=True,timeout=12)
        self.assertEqual(result.returncode,2)
        args=(self.root/'runtime.args').read_text().splitlines()
        for option,name in (('--tv-reference','reference.features'),('--tv-cues','cues.tsv'),('--traffic-model','traffic.bin')):
            self.assertEqual(args[args.index(option)+1],linux(self.bundle/'tv'/name))

    def test_explicit_stop_reaps_owned_processes_without_hardware_shutdown(self):
        proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 6
            while not (self.root / 'ui.started').exists() and time.monotonic() < deadline and proc.poll() is None:
                time.sleep(.05)
            if not (self.root / 'ui.started').exists():
                out, err = proc.communicate(timeout=2)
                self.fail(err.decode(errors='replace'))
            pid = self.read_session('supervisor.pid').strip()
            signal = subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'kill', '-TERM', pid], capture_output=True)
            self.assertEqual(signal.returncode, 0, signal.stderr)
            _, err = proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, 0, err.decode(errors='replace'))
            status = json.loads(self.read_session('status.json'))
            self.assertEqual(status['reason'], 'requested_stop')
            for role in ('capture', 'runtime', 'doa', 'ui'):
                self.assertTrue((self.root / (role + '.stopped')).is_file(), role)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()


if __name__ == '__main__':
    unittest.main()
