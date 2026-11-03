import json
import struct
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "src" / "pcm_ring_buffer.cpp",
    ROOT / "src" / "soundjie_d4_state.cpp",
    ROOT / "src" / "soundjie_d4_runtime.cpp",
    ROOT / "src" / "fft_radix2.cpp",
    ROOT / "src" / "known_clip_matcher.cpp",
    ROOT / "src" / "traffic_fine_model.cpp",
]


def wsl_path(path: Path) -> str:
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


class SoundjieD4RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temporary.name) / "soundjie_d4_runtime"
        result = subprocess.run(
            ["wsl.exe", "g++", "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
             f"-I{wsl_path(ROOT / 'src')}", *(wsl_path(p) for p in SOURCES),
             "-o", wsl_path(cls.binary)],
            text=True, encoding="utf-8", errors="replace", capture_output=True,
        )
        if result.returncode:
            raise AssertionError(f"runtime build failed\n{result.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_json(self, *args):
        result = subprocess.run(
            ["wsl.exe", wsl_path(self.binary), *map(str, args)],
            text=True, encoding="utf-8", errors="replace", capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_bounded_timeline_retains_eof_and_current_state(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pcm=root/'input.raw';out=root/'session';current=root/'current.json'
            pcm.write_bytes(struct.pack('<hh',1000,1000)*204)
            result=self.run_json('--fixture-pcm',wsl_path(pcm),'--output-dir',wsl_path(out),
                '--current',wsl_path(current),'--window-samples','4','--hop-samples','2',
                '--cadence-ms','1','--mock-class-id','3','--log-max-bytes','4096')
            timeline=out/'timeline.jsonl'
            self.assertEqual(result['dpu_calls'],101)
            self.assertLessEqual(timeline.stat().st_size,4096)
            rows=[json.loads(s) for s in timeline.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[-1],json.loads(current.read_text(encoding='utf-8')))
            self.assertEqual(rows[-1]['sequence'],102)
            self.assertFalse(rows[-1]['usb_ok'])

    def test_ring_wrap_preserves_latest_one_second(self):
        payload = self.run_json("--ring-test", "0,1,2,3,4,5,6,7,8,9,10,11", "--capacity", "8")
        self.assertEqual(payload["latest"], [4, 5, 6, 7, 8, 9, 10, 11])

    def test_stereo_antiphase_uses_existing_equal_mix_contract(self):
        payload = self.run_json("--mix-test", "1000,-1000,32767,-32767,-1234,1234")
        self.assertEqual(payload["mono"], [0, 0, 0])

    def test_fixture_runtime_is_serial_atomic_append_only_and_eof_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pcm = root / "input.raw"
            frames = [(1000, 1000)] * 12
            pcm.write_bytes(b"".join(struct.pack("<hh", *frame) for frame in frames))
            output = root / "session"
            current = root / "current.json"
            payload = self.run_json(
                "--fixture-pcm", wsl_path(pcm), "--output-dir", wsl_path(output),
                "--current", wsl_path(current), "--window-samples", "4",
                "--hop-samples", "2", "--cadence-ms", "1", "--mock-delay-ms", "3",
                "--mock-class-id", "3", "--expected-class", "7",
            )
            self.assertEqual(payload["dpu_calls"], 5)
            self.assertEqual(payload["max_concurrent_inference"], 1)
            self.assertGreater(payload["deadline_misses"], 0)
            self.assertFalse(payload["usb_ok"])
            self.assertEqual(payload["predicted_class"], 3)
            self.assertFalse((root / "current.json.tmp").exists())
            final = json.loads(current.read_text(encoding="utf-8"))
            self.assertFalse(final["usb_ok"])
            rows = [json.loads(row) for row in (output / "timeline.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), payload["dpu_calls"] + 1)
            self.assertEqual([row["sequence"] for row in rows], list(range(1, len(rows) + 1)))
            self.assertEqual([row["class_id"] for row in rows[:-1]], [3] * payload["dpu_calls"])

            second = subprocess.run(
                ["wsl.exe", wsl_path(self.binary), "--fixture-pcm", wsl_path(pcm),
                 "--output-dir", wsl_path(output), "--current", wsl_path(current),
                 "--window-samples", "4", "--hop-samples", "2", "--mock-class-id", "1"],
                text=True, encoding="utf-8", errors="replace", capture_output=True,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("already exists", second.stderr)

    def test_board_launcher_is_continuous_and_has_no_prediction_override(self):
        script = (ROOT / "scripts" / "board" / "run_soundjie_d4_runtime.sh").read_text(
            encoding="utf-8")
        self.assertEqual(script.count('exec "$arecord_bin"'), 1)
        self.assertIn("-t raw", script)
        self.assertIn("S16_LE", script)
        self.assertNotIn("expected-class", script)
        self.assertNotIn("--mock-class-id", script)

    def test_hdmi_supervisor_passes_bounded_retention_to_actual_runtime(self):
        script = (ROOT/'scripts/board/run_soundjie_d4_session.sh').read_text(encoding='utf-8')
        self.assertIn('--retain-windows "${SOUNDJIE_RETAIN_WINDOWS:-0}"', script)

    def test_full_scores_and_bounded_exact_pcm_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pcm = root / 'input.raw'
            mono = [100, 200, 300, 400, 500, 600, 700, 800]
            pcm.write_bytes(b''.join(struct.pack('<hh', x, x) for x in mono))
            classifier = root / 'classifier.sh'
            classifier.write_bytes(b"#!/bin/sh\nprintf '3 .37 1 .10 .06 .07 .37 .08 .20 .06 .06\\n'\n")
            classifier.chmod(0o755)
            result = self.run_json('--fixture-pcm', wsl_path(pcm), '--output-dir', wsl_path(root/'session'),
                '--current', wsl_path(root/'current.json'), '--window-samples', '4', '--hop-samples', '2',
                '--classifier-command', wsl_path(classifier), '--retain-windows', '2')
            self.assertEqual(result['dpu_calls'], 3)
            rows = [json.loads(x) for x in (root/'session/timeline.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[0]['class_scores'], [.10, .06, .07, .37, .08, .20, .06, .06])
            self.assertAlmostEqual(rows[0]['top2_margin'], .17)
            self.assertEqual([x['id'] for x in rows[0]['top3']], [3, 5, 0])
            self.assertEqual(rows[0]['class_id'], 3)  # Diagnostics never fake an improved prediction.
            self.assertEqual(rows[0]['capture_file'], 'capture-1.s16')
            self.assertEqual(rows[0]['window_end_sample'], 4)
            self.assertEqual(rows[1]['window_end_sample'], 6)
            self.assertIsNone(rows[2]['capture_file'])
            self.assertEqual(len(list((root/'session').glob('capture-*.s16'))), 2)
            self.assertEqual((root/'session/capture-1.s16').read_bytes(), struct.pack('<hhhh', *mono[:4]))
            self.assertEqual((root/'session/capture-2.s16').read_bytes(), struct.pack('<hhhh', *mono[2:6]))
            self.assertEqual(rows[-1]['class_scores'], [])

    def test_malformed_score_vectors_fail_without_publishing_prediction(self):
        invalid = ['.1 .1', '.1 .1 .1 .1 .1 .1 .1 .1',
                   '.8 .01 .01 .05 .01 .01 .01 .1',
                   '.1 .1 .1 .4 .1 .1 .1 0', '.1 .1 .1 .3 .1 .1 .1 nan']
        for scores in invalid:
            with self.subTest(scores=scores), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root/'input.raw').write_bytes(struct.pack('<hh', 100, 100)*4)
                clf = root/'classifier.sh'
                clf.write_text("#!/bin/sh\nprintf '3 .3 1 " + scores + "\\n'\n", encoding='ascii')
                clf.chmod(0o755)
                result = subprocess.run(['wsl.exe', wsl_path(self.binary), '--input', wsl_path(root/'input.raw'),
                    '--output-dir', wsl_path(root/'session'), '--current', wsl_path(root/'current.json'),
                    '--window-samples', '4', '--classifier-command', wsl_path(clf)], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                final = json.loads((root/'current.json').read_text(encoding='utf-8'))
                self.assertEqual(final['class_id'], -1)
                self.assertFalse(final['dpu_ok'])

    def test_sixty_second_fixture_soak_streams_with_bounded_ring(self):
        source = (ROOT / "src" / "soundjie_d4_runtime.cpp").read_text(encoding="utf-8")
        self.assertNotIn("read_stereo_raw", source)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pcm = root / "sixty-seconds.raw"
            pcm.write_bytes(struct.pack("<hh", 400, 400) * (16000 * 60))
            payload = self.run_json(
                "--fixture-pcm", wsl_path(pcm), "--output-dir", wsl_path(root / "session"),
                "--current", wsl_path(root / "current.json"), "--window-samples", "16000",
                "--hop-samples", "12000", "--mock-class-id", "2",
            )
            self.assertEqual(payload["dpu_calls"], 79)
            self.assertEqual(payload["max_concurrent_inference"], 1)
            self.assertEqual(
                len((root / "session" / "timeline.jsonl").read_text(
                    encoding="utf-8").splitlines()), 80)

    def test_arm_build_contract_includes_runtime_state_and_ring(self):
        script = (ROOT / "scripts" / "guest" / "build_soundjie_d4_runtime.sh").read_text(
            encoding="utf-8")
        self.assertIn("arm-linux-gnueabihf-g++", script)
        self.assertIn("pcm_ring_buffer.cpp", script)
        self.assertIn("soundjie_d4_state.cpp", script)
        self.assertIn("soundjie_d4_runtime.cpp", script)
        self.assertIn("MAX_GLIBC=", script)
        self.assertIn("DT_NEEDED=", script)

    def test_external_classifier_protocol_drives_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pcm = root / "input.raw"
            pcm.write_bytes(struct.pack("<hh", 700, 700) * 6)
            classifier = root / "classifier.sh"
            classifier.write_bytes(
                b"#!/bin/sh\n[ -f \"$1\" ] || exit 2\nprintf '5 0.75 12.5\\n'\n")
            classifier.chmod(0o755)
            payload = self.run_json(
                "--fixture-pcm", wsl_path(pcm), "--output-dir", wsl_path(root / "session"),
                "--current", wsl_path(root / "current.json"), "--window-samples", "4",
                "--hop-samples", "2", "--classifier-command", wsl_path(classifier),
            )
            self.assertEqual(payload["dpu_calls"], 2)
            self.assertEqual(payload["predicted_class"], 5)
            rows = [json.loads(row) for row in (root / "session" / "timeline.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["confidence"], 0.75)
            self.assertEqual(rows[0]["inference_ms"], 12.5)
            self.assertEqual(rows[0]["label_zh"], "紧急人声")
            self.assertGreater(rows[0]["timestamp_ms"], 1000)

    def test_log_rotation_failure_still_publishes_fault_snapshot(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.superpowers') as d:
            root=Path(d);pcm=root/'in.raw';classifier=root/'classifier.sh'
            pcm.write_bytes(struct.pack('<hh',700,700)*30)
            classifier.write_bytes(b'''#!/bin/sh
case "$1" in *window-2.s16)
  d=$(dirname "$1")
  mv "$d/timeline.jsonl" "$d/previous.jsonl"
  printf FOREIGN > "$d/timeline.jsonl"
;; esac
printf '5 0.75 1\\n'
''')
            classifier.chmod(0o755)
            r=subprocess.run(['wsl','-d','Ubuntu','--',wsl_path(self.binary),
                '--fixture-pcm',wsl_path(pcm),'--output-dir',wsl_path(root/'session'),
                '--current',wsl_path(root/'current.json'),'--window-samples','4','--hop-samples','2',
                '--classifier-command',wsl_path(classifier),'--log-max-bytes','4096'],
                capture_output=True,timeout=15)
            self.assertNotEqual(r.returncode,0)
            self.assertIn(b'JSONL output identity changed',r.stderr)
            current=json.loads((root/'current.json').read_text(encoding='utf-8'))
            self.assertEqual(current['class_id'],-1)
            self.assertFalse(current['dpu_ok'])
            self.assertEqual((root/'session/timeline.jsonl').read_bytes(),b'FOREIGN')

    def test_classifier_faults_are_bounded_and_clear_previous_prediction(self):
        cases = [
            ("sleep 2", "classifier deadline exceeded"),
            ("exec 1>&-; sleep 2", "classifier deadline exceeded"),
            ("head -c 8192 /dev/zero", "classifier output limit exceeded"),
            ("printf '3 0.9 1\\n'; (sleep 2) & exit 0", "classifier deadline exceeded"),
        ]
        for body, reason in cases:
            with self.subTest(body=body), tempfile.TemporaryDirectory(dir=ROOT / '.superpowers') as folder:
                root = Path(folder)
                pcm, classifier = root / 'input.raw', root / 'classifier.sh'
                pcm.write_bytes(struct.pack('<hh', 700, 700) * 6)
                classifier.write_bytes(("#!/bin/sh\ncase \"$1\" in *window-1.s16) printf '5 0.75 1\\n'; exit 0;; esac\n" + body + "\n").encode())
                classifier.chmod(0o755)
                started = time.monotonic()
                result = subprocess.run([
                    'wsl.exe', '-d', 'Ubuntu', '--', wsl_path(self.binary),
                    '--fixture-pcm', wsl_path(pcm), '--output-dir', wsl_path(root / 'session'),
                    '--current', wsl_path(root / 'current.json'), '--window-samples', '4',
                    '--hop-samples', '2', '--classifier-command', wsl_path(classifier),
                    '--classifier-timeout-ms', '100'], capture_output=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(reason, result.stderr.decode(errors='replace'))
                self.assertLess(time.monotonic() - started, 1.8)
                rows = [json.loads(line) for line in (root / 'session/timeline.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertEqual(rows[0]['class_id'], 5)
                final = json.loads((root / 'current.json').read_text(encoding='utf-8'))
                self.assertEqual(final['class_id'], -1)
                self.assertFalse(final['dpu_ok'])
                self.assertFalse(final['usb_ok'])
                self.assertEqual(final['rms'], 0)
                self.assertEqual(final, rows[-1])
                self.assertEqual(list((root / 'session').glob('window-*.s16')), [])

    def test_launcher_exports_bundle_and_audio_library_paths(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = root / 'bundle with spaces'
            (bundle / 'audio/bin').mkdir(parents=True)
            (bundle / 'runtime').mkdir()
            for path, data in (
                (bundle / 'audio/bin/arecord', b'#!/bin/sh\nexit 0\n'),
                (bundle / 'runtime/soundjie_d4_classifier', b'#!/bin/sh\nexit 0\n'),
                (bundle / 'runtime/soundjie_d4_runtime', b'#!/bin/sh\nprintf "%s\\n" "$SOUNDJIE_BUNDLE_DIR" "$LD_LIBRARY_PATH" "$ALSA_CONFIG_PATH"\n'),
            ):
                path.write_bytes(data)
                path.chmod(0o755)
            script = ROOT / 'scripts/board/run_soundjie_d4_runtime.sh'
            result = subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'sh', wsl_path(script),
                                     wsl_path(bundle), wsl_path(root / 'session'), wsl_path(root / 'current.json')],
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.decode().splitlines()
            self.assertEqual(lines[0], wsl_path(bundle))
            self.assertIn(wsl_path(bundle / 'dnndk/pkgs/lib'), lines[1])
            self.assertIn(wsl_path(bundle / 'audio/lib'), lines[1])
            self.assertEqual(lines[2], wsl_path(bundle / 'audio/share/alsa/alsa.conf'))

    def test_live_input_drains_during_slow_inference_and_uses_latest_window(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            classifier = root / 'classifier.sh'
            classifier.write_bytes(b'''#!/bin/sh
case "$1" in *window-1.s16) touch "$1.started"; sleep 1;; esac
value=$(od -An -td2 -N2 "$1")
if [ "$value" -eq 500 ]; then printf '2 0.8 1\\n'; else printf '1 0.8 1\\n'; fi
''')
            classifier.chmod(0o755)
            driver = r'''
import json,os,pathlib,struct,subprocess,sys,time
binary,root,clf=sys.argv[1:]
root=pathlib.Path(root)
p=subprocess.Popen([binary,'--input','-','--output-dir',str(root/'session'),
 '--current',str(root/'current.json'),'--window-samples','4','--hop-samples','4',
 '--classifier-command',clf],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
try:
 p.stdin.write(struct.pack('<hh',100,100)*4); p.stdin.flush()
 deadline=time.monotonic()+5
 while not (root/'session/window-1.s16.started').exists():
  if p.poll() is not None or time.monotonic()>deadline: raise RuntimeError('classifier not started')
  time.sleep(.005)
 started=time.monotonic()
 data=struct.pack('<hh',200,200)*32768+struct.pack('<hh',500,500)*8
 os.set_blocking(p.stdin.fileno(),False)
 written=0
 while written<len(data) and time.monotonic()-started<2:
  try: written+=os.write(p.stdin.fileno(),data[written:])
  except BlockingIOError: time.sleep(.005)
 elapsed=time.monotonic()-started
 # Later partial frame must not make the already-complete window look fresh.
 time.sleep(.8)
 os.write(p.stdin.fileno(),b'\x01')
 deadline=time.monotonic()+4
 rows=[]
 while time.monotonic()<deadline:
  path=root/'session/timeline.jsonl'
  if path.exists(): rows=[json.loads(x) for x in path.read_text().splitlines()]
  if len(rows)>=2: break
  time.sleep(.01)
 print(json.dumps({'write_seconds':elapsed,'rows':rows[:2],'written':written,'total':len(data)}))
finally:
 p.terminate()
 try: p.communicate(timeout=3)
 except subprocess.TimeoutExpired: p.kill(); p.communicate(timeout=3)
'''
            result = subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'python3', '-c', driver,
                                     wsl_path(self.binary), wsl_path(root), wsl_path(classifier)],
                                    capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
            report = json.loads(result.stdout)
            self.assertEqual(report['written'], report['total'])
            self.assertEqual(len(report['rows']), 2)
            self.assertEqual(report['rows'][0]['class_id'], 1)
            self.assertEqual(report['rows'][1]['class_id'], 2)
            self.assertLess(report['write_seconds'], .7)
            self.assertAlmostEqual(report['rows'][0]['rms'], 100 / 32768, places=5)
            self.assertAlmostEqual(report['rows'][1]['rms'], 500 / 32768, places=5)
            self.assertGreaterEqual(report['rows'][0]['end_to_end_ms'], 900)
            self.assertLess(report['rows'][1]['timestamp_ms'] - report['rows'][0]['timestamp_ms'], 700)

    def test_live_truncated_frame_clears_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', wsl_path(self.binary),
                                     '--input', '-', '--output-dir', wsl_path(root / 'session'),
                                     '--current', wsl_path(root / 'current.json'), '--mock-class-id', '1'],
                                    input=b'\x00\x00\x00', capture_output=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('truncated stereo PCM', result.stderr.decode(errors='replace'))
            state = json.loads((root / 'current.json').read_text(encoding='utf-8'))
            self.assertEqual(state['class_id'], -1)
            self.assertFalse(state['usb_ok'])
            self.assertEqual(state['rms'], 0)


if __name__ == "__main__":
    unittest.main()
