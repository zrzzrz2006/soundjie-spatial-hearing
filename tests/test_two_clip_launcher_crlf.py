"""Run the real POSIX launcher and inspect exact argv at stub process boundaries.

No board, serial, audio, framebuffer, or classifier is accessed. All fixtures
are retained under SOUNDJIE_LAUNCHER_TEST_ROOT (F: by default). The mkfifo
dependency is substituted with exclusive regular-file creation because DrvFS
does not support FIFOs; this test concerns shell arguments, not FIFO transport.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/board/run_soundjie_d4_session.sh"
TV_SCRIPT = ROOT / "scripts/board/run_soundjie_tv_session.sh"
ARTIFACTS = Path(os.environ.get(
    "SOUNDJIE_LAUNCHER_TEST_ROOT",
    "F:/MLK/DPU_ZYNQ7000/artifacts/two-clip-launcher-crlf-20261101a" if os.name == "nt"
    else "/mnt/f/MLK/DPU_ZYNQ7000/artifacts/two-clip-launcher-crlf-20261101a"))


def posix(path):
    path = Path(path).resolve()
    if os.name == "nt":
        return "/mnt/"+path.drive[0].lower()+path.as_posix().split(":",1)[1]
    return str(path)


STUB = r'''#!/bin/sh
set -eu
role=ROLE
if [ "$role" = ui ]; then
  case " $* " in
    *" --frames 0 "*) :;;
    *) printf '%s\000' "$@" > "$TEST_ROOT/ui.final.argv"; exit 0;;
  esac
fi
printf '%s\000' "$@" > "$TEST_ROOT/$role.argv"
if [ "$role" = runtime ]; then
  # Let every sibling capture its arguments before ending this synthetic session.
  attempts=0
  while [ ! -s "$TEST_ROOT/capture.argv" ] || [ ! -s "$TEST_ROOT/doa.argv" ] || [ ! -s "$TEST_ROOT/ui.argv" ]; do
    attempts=$((attempts + 1))
    [ "$attempts" -lt 100 ] || exit 87
    sleep .02
  done
  exit 0
fi
trap 'exit 0' TERM INT
while :; do sleep .05; done
'''


class TwoClipLauncherCrlfTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True,exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix="launcher-tests-",dir=ARTIFACTS))
        print("Retained POSIX launcher fixtures:",cls.work,flush=True)

    def fixture(self, ending):
        root = Path(tempfile.mkdtemp(prefix=self._testMethodName+"-",dir=self.work))
        bundle = root / "bundle with spaces"
        paths = {
            "config/class-contract.txt": b"soundjie-alert5bg-v3\n",
            "config/alert-thresholds.txt": b"0.525,0.9,0.575,0.6,0.675\n",
            "config/hop-samples.txt": b"6400\n",
            "config/log-policy.txt": b"bounded-jsonl-v1\n",
            "config/soundjie_d4_runtime.tsv": b"unused stub config\n",
            "assets/hdmi/soundjie_glyph_atlas.bin": b"unused stub atlas",
            "tv/bank.tsv": b"unused stub bank\n",
            "tv/traffic.bin": b"unused stub traffic model",
            "tv/dialogue-horn-start.txt": b"4.8"+ending,
            "tv/dialogue-horn-end.txt": b"6.8"+ending,
            "board/run_soundjie_d4_session.sh": SCRIPT.read_bytes().replace(b"\r\n",b"\n"),
            "board/run_soundjie_tv_session.sh": TV_SCRIPT.read_bytes().replace(b"\r\n",b"\n"),
        }
        for role, name in (("capture","audio/bin/arecord"),("runtime","runtime/soundjie_d4_runtime"),
                           ("classifier","runtime/soundjie_d4_classifier"),("doa","runtime/xvf3800_doa"),
                           ("ui","runtime/soundjie_hdmi_ui")):
            paths[name] = STUB.replace("ROLE",role).encode("ascii")
        for name, data in paths.items():
            path = bundle/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(data)
            if name.startswith(("runtime/","audio/bin/")) or name.endswith(".sh"):
                path.chmod(0o755)
        tools = root/"stub tools"
        tools.mkdir()
        fifo = tools/"mkfifo"
        fifo.write_bytes(b'#!/bin/sh\nset -euC\n[ "$#" -eq 1 ]\n: > "$1"\n')
        fifo.chmod(0o755)
        return root,bundle,tools

    def exercise(self, ending, overrides):
        root,bundle,tools = self.fixture(ending)
        session = root/"new session"
        framebuffer = root/"unused framebuffer"
        # Isolate from the developer's environment, including any real device or
        # LD_LIBRARY_PATH setting. All launched programs below are fixture stubs.
        env = {
            "PATH":posix(tools)+":/usr/bin:/bin", "LC_ALL":"C",
            "TEST_ROOT":posix(root), "SOUNDJIE_PCM_DEVICE":"unused PCM with spaces",
        }
        env.update(overrides)
        prefix = ["wsl.exe","-d","Ubuntu","--"] if os.name == "nt" else []
        command = prefix+["env","-i"]+[key+"="+value for key,value in env.items()]+[
            "timeout","-k","3","8","sh",posix(bundle/"board/run_soundjie_tv_session.sh"),
            posix(bundle),posix(session),posix(framebuffer)]
        result = subprocess.run(command,capture_output=True,timeout=20)
        (root/"launcher.stdout").write_bytes(result.stdout)
        (root/"launcher.stderr").write_bytes(result.stderr)
        # A deliberate clean EOF from the stub runtime triggers the real
        # supervisor's normal fail-stop path and reaps its own child groups.
        self.assertEqual(result.returncode,2,result.stderr.decode("utf-8","replace"))
        status = json.loads((session/"status.json").read_bytes())
        self.assertEqual(status["reason"],"runtime_exited")
        self.assertEqual(status["child_exit"],0)
        self.assertEqual(status["clear_exit"],0)

        def argv(role):
            raw = (root/(role+".argv")).read_bytes()
            self.assertTrue(raw.endswith(b"\0"))
            return [value.decode("utf-8") for value in raw[:-1].split(b"\0")]

        hop = overrides.get("SOUNDJIE_HOP_SAMPLES","6400")
        logs = overrides.get("SOUNDJIE_LOG_MAX_BYTES","1048576")
        retention = overrides.get("SOUNDJIE_RETAIN_WINDOWS","120")
        deadline = overrides.get("SOUNDJIE_CLASSIFIER_TIMEOUT_MS","5000")
        expected = [
            "--input","-","--output-dir",posix(session/"inference"),
            "--current",posix(session/"state/current.json"),"--window-samples","16000",
            "--hop-samples",hop,"--cadence-ms",str(int(hop)//16),
            "--classifier-command",posix(bundle/"runtime/soundjie_d4_classifier"),
            "--classifier-timeout-ms",deadline,"--retain-windows",retention,
            "--tv-bank",posix(bundle/"tv/bank.tsv"),"--traffic-model",posix(bundle/"tv/traffic.bin"),
            "--tv-dialogue-horn-start","4.8","--tv-dialogue-horn-end","6.8",
            "--alert-thresholds","0.525,0.9,0.575,0.6,0.675",
            "--class-contract","soundjie-alert5bg-v3","--log-max-bytes",logs,
        ]
        actual = argv("runtime")
        self.assertEqual(actual,expected)
        self.assertFalse(any("\r" in value for value in actual))
        self.assertEqual(argv("capture"),[
            "-D","unused PCM with spaces","-t","raw","-f","S16_LE","-c","2","-r","16000",
            "--period-size=800","--buffer-size=8000"])
        self.assertEqual(argv("doa"),[
            "--jsonl",posix(session/"state/doa.jsonl"),"--interval-ms","100","--log-max-bytes",logs])
        self.assertEqual(argv("ui"),[
            "--log-max-bytes",logs,"--jsonl",posix(session/"logs/ui.jsonl"),
            "--class-contract","soundjie-alert5bg-v3","--snapshot",posix(session/"state/current.json"),
            "--doa-jsonl",posix(session/"state/doa.jsonl"),"--runtime-config",posix(bundle/"config/soundjie_d4_runtime.tsv"),
            "--atlas",posix(bundle/"assets/hdmi/soundjie_glyph_atlas.bin"),"--framebuffer",posix(framebuffer),
            "--frames","0","--interval-ms","100","--snapshot-stale-ms","2000"])
        self.assertEqual((bundle/"tv/dialogue-horn-start.txt").read_bytes(),b"4.8"+ending)
        self.assertEqual((bundle/"tv/dialogue-horn-end.txt").read_bytes(),b"6.8"+ending)
        self.assertFalse((root/"classifier.argv").exists(),"no real or stub inference should be invoked")

    def test_crlf_horn_times_reach_runtime_without_cr_and_preserve_overrides(self):
        self.exercise(b"\r\n",{
            "SOUNDJIE_HOP_SAMPLES":"8000","SOUNDJIE_RETAIN_WINDOWS":"0",
            "SOUNDJIE_LOG_MAX_BYTES":"2097152","SOUNDJIE_CLASSIFIER_TIMEOUT_MS":"7300"})

    def test_lf_horn_times_preserve_default_options(self):
        self.exercise(b"\n",{})


if __name__ == "__main__":
    unittest.main()
