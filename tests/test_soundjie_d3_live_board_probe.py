import hashlib
import os
import shutil
import struct
import subprocess
import tempfile
import textwrap
import unittest
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "scripts" / "board" / "run_soundjie_d3_live_once.sh"
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
CLASS_IDS = (
    "fire_smoke_alarm", "emergency_siren", "car_horn", "glass_breaking",
    "door_knock_bell", "distress_voice", "normal_speech", "background_unknown",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bash_path(path):
    path = Path(path).resolve()
    return f"/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


class SoundjieD3LiveBoardProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "bundle"
        for folder in (
            "runtime", "model", "config", "audio/board-tools", "audio/bin",
            "audio/lib", "audio/share/alsa",
            "dnndk/pkgs/bin", "dnndk/pkgs/lib", "manifest",
        ):
            (self.bundle / folder).mkdir(parents=True, exist_ok=True)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.proc = self.root / "proc"
        self.asound = self.proc / "asound"
        self.card = self.asound / "card2"
        self.card.mkdir(parents=True)
        self.dev = self.root / "dev"
        self.dev.mkdir()
        (self.dev / "dpu").write_bytes(b"")
        self.fdt = self.root / "fdt"
        self.fdt.write_bytes(b"deephi,dpu\0")
        self.modules = self.proc / "modules"
        self.modules.write_text("dpu 40960 0 - Live 0x0\n", encoding="ascii")
        self.dmesg = self.root / "dmesg"
        self.dmesg.write_text("old: dpu timeout already investigated\nboot complete\n", encoding="ascii")
        self.counter = self.root / "counter"
        self.counter.write_text("0", encoding="ascii")
        self.order = self.root / "order"
        self.mount_log = self.root / "mount-log"
        self.audio_loaded = self.root / "audio-loaded"
        self.lib_target = self.root / "lib-target"
        self.lib_target.mkdir()
        self.capture_source = self.root / "capture-source.wav"
        self._write_capture(self.capture_source)
        self._write_platform_fixture()
        self._write_bundle()
        self.write_sums()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _exe(path, content):
        path.write_text(textwrap.dedent(content), encoding="utf-8", newline="\n")
        path.chmod(0o755)

    @staticmethod
    def _write_capture(path):
        frames = bytearray()
        for index in range(96000):
            sample = 5000 if (index // 80) % 2 == 0 else -5000
            frames.extend(struct.pack("<hh", sample, sample))
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(bytes(frames))

    def _write_platform_fixture(self):
        (self.asound / "cards").write_text(
            " 2 [XVF3800]: USB-Audio - ReSpeaker XVF3800\n"
            "                 Seeed Studio ReSpeaker XVF3800\n", encoding="ascii"
        )
        (self.card / "id").write_text("XVF3800\n", encoding="ascii")
        (self.card / "usbid").write_text("2886:001a\n", encoding="ascii")
        (self.card / "stream0").write_text("ReSpeaker XVF3800 USB Audio\n", encoding="ascii")
        self._exe(self.tools / "uname", """#!/bin/sh
            case "$1" in -m) echo armv7l ;; -r) echo 4.14.0-xilinx ;; *) exit 64 ;; esac
        """)
        self._exe(self.tools / "modinfo", "#!/bin/sh\necho '4.14.0-xilinx SMP ARMv7'\n")
        self._exe(self.tools / "insmod", "#!/bin/sh\nexit 0\n")
        self._exe(self.tools / "mount", """#!/bin/sh
            echo "mount:$*" >> "$LIVE_TEST_MOUNT_LOG"
            exit 0
        """)
        self._exe(self.tools / "umount", """#!/bin/sh
            echo "umount:$*" >> "$LIVE_TEST_MOUNT_LOG"
            exit 0
        """)
        self._exe(self.tools / "dmesg", "#!/bin/sh\ncat \"$LIVE_TEST_DMESG\"\n")

    def _write_bundle(self):
        for name, content in (
            ("runtime/dpu-4.14.0-xilinx.ko", b"module"),
            ("model/dpu_soundjie_d3.elf", b"verified-model"),
            ("dnndk/pkgs/lib/libn2cube.so", b"n2cube"),
            ("dnndk/pkgs/lib/libhineon.so", b"hineon"),
            ("audio/lib/libasound.so.2", b"asound"),
            ("audio/share/alsa/alsa.conf", b"# fixture\n"),
            ("config/logmel_frontend_v1.json", b"{}\n"),
            ("config/soundjie_live_v1.tsv", b"capture_seconds\t6\n"),
        ):
            (self.bundle / name).write_bytes(content)
        (self.bundle / "model/labels.txt").write_text("\n".join(CLASS_IDS) + "\n", encoding="ascii")
        (self.bundle / "model/display_labels_zh.txt").write_text(
            "火灾报警\n紧急警报\n汽车鸣笛\n玻璃破碎\n敲门门铃\n求救人声\n正常人声\n背景未知\n",
            encoding="utf-8", newline="\n",
        )
        shutil.copy2(ROOT / "scripts/board/xvf3800_probe.sh", self.bundle / "audio/board-tools/xvf3800_probe.sh")
        shutil.copy2(ROOT / "scripts/board/xvf3800_record.sh", self.bundle / "audio/board-tools/xvf3800_record.sh")
        os.chmod(self.bundle / "audio/board-tools/xvf3800_probe.sh", 0o755)
        os.chmod(self.bundle / "audio/board-tools/xvf3800_record.sh", 0o755)
        self._exe(self.bundle / "audio/load-usb-audio.sh", """#!/bin/sh
            : > "$LIVE_TEST_AUDIO_LOADED"
        """)
        self._exe(self.bundle / "audio/bin/arecord", """#!/bin/sh
            case " $* " in
              *" -l "*) echo 'card 2: XVF3800 [ReSpeaker XVF3800], device 0: USB Audio [USB Audio]' ;;
              *" --dump-hw-params "*)
                echo 'FORMAT: S16_LE'; echo 'CHANNELS: 2'; echo 'RATE: 16000' ;;
              *)
                [ -f "$LIVE_TEST_AUDIO_LOADED" ] || exit 9
                for output in "$@"; do :; done
                echo capture >> "$LIVE_TEST_ORDER"
                cp "$LIVE_TEST_CAPTURE_SOURCE" "$output" ;;
            esac
        """)
        self._exe(self.bundle / "runtime/wav_probe", """#!/bin/sh
            echo CHANNELS=2
            echo SAMPLE_RATE=16000
            echo BITS_PER_SAMPLE=16
            echo "SIGNAL=${LIVE_TEST_WAV_SIGNAL:-present}"
        """)
        self._exe(self.bundle / "runtime/live_window_selector", """#!/bin/sh
            metrics= selected=
            while [ $# -gt 0 ]; do
              case "$1" in --metrics) metrics=$2; shift 2;; --selected) selected=$2; shift 2;; *) shift 2;; esac
            done
            echo selector >> "$LIVE_TEST_ORDER"
            printf 'start_sample\tstart_ms\trms\tpeak\tclipping_rate\taccepted\trejection\tselected_rank\n' > "$metrics"
            printf 'rank\tstart_sample\trms\tpeak\tclipping_rate\n' > "$selected"
            [ "${LIVE_TEST_SELECTOR_EMPTY:-0}" != 1 ] || { echo REASON=audio_signal_absent; exit 2; }
            printf '0\t0\t0.2\t0.3\t0\t1\t\t1\n8000\t500\t0.18\t0.3\t0\t1\t\t2\n' >> "$metrics"
            printf '1\t0\t0.2\t0.3\t0\n2\t8000\t0.18\t0.3\t0\n' >> "$selected"
        """)
        self._exe(self.bundle / "runtime/logmel_cli", """#!/bin/sh
            float= tensor= report= start=
            while [ $# -gt 0 ]; do
              case "$1" in
                --start-sample) start=$2; shift 2;; --float-output) float=$2; shift 2;;
                --int8-output) tensor=$2; shift 2;; --report) report=$2; shift 2;; *) shift 2;;
              esac
            done
            echo "logmel:$start" >> "$LIVE_TEST_ORDER"
            dd if=/dev/zero of="$float" bs=24576 count=1 2>/dev/null
            dd if=/dev/zero of="$tensor" bs=6144 count=1 2>/dev/null
            printf 'LOGMEL_STATUS=pass\nINT8_BYTES=6144\nINPUT_SCALE=128\n' > "$report"
        """)
        self._exe(self.bundle / "runtime/sound_shape_runner", """#!/bin/sh
            for arg in "$@"; do [ "$arg" = --describe-tensor ] && describe=1; done
            if [ "${describe:-0}" = 1 ]; then
              cat <<EOF
DPU_TENSOR_STATUS=pass
INPUT_NODE=conv1
OUTPUT_NODE=fc8
INPUT_SIZE=6144
INPUT_SCALE=128
OUTPUT_SIZE=8
OUTPUT_SCALE=0.0625
DPU_RUN_CALLS=0
EOF
              exit 0
            fi
            input= model= dump= report=
            while [ $# -gt 0 ]; do
              case "$1" in
                --input-file) input=$2; shift 2;; --model-file) model=$2; shift 2;;
                --dump) dump=$2; shift 2;; --report) report=$2; shift 2;; *) shift 2;;
              esac
            done
            echo runner >> "$LIVE_TEST_ORDER"
            [ "${LIVE_TEST_RUNNER_FAIL:-0}" != 1 ] || exit 7
            printf '%b' '\\012\\000\\000\\000\\000\\000\\000\\000' > "$dump"
            count=$(cat "$LIVE_TEST_COUNTER"); echo $((count + 1)) > "$LIVE_TEST_COUNTER"
            [ -z "${LIVE_TEST_NEW_DMESG:-}" ] || { echo "$LIVE_TEST_NEW_DMESG" >> "$LIVE_TEST_DMESG"; unset LIVE_TEST_NEW_DMESG; }
            cat > "$report" <<EOF
DPU_SOUND_STATUS=pass
INPUT_SCALE=128
OUTPUT_SCALE=0.0625
MODEL_SHA256=$(sha256sum "$model" | awk '{print $1}')
INPUT_SHA256=$(sha256sum "$input" | awk '{print $1}')
OUTPUT_SHA256=$(sha256sum "$dump" | awk '{print $1}')
OUTPUT_VALUES=10,0,0,0,0,0,0,0
DPU_RUN_CALLS=1
RUN_TOTAL_US=601
EOF
        """)
        self._exe(self.bundle / "runtime/live_result_fuser", """#!/bin/sh
            json= display= expected=
            while [ $# -gt 0 ]; do
              case "$1" in
                --session-result) json=$2; shift 2;; --display-status) display=$2; shift 2;;
                --expected-class) expected=$2; shift 2;; *) shift 2;;
              esac
            done
            echo fuser >> "$LIVE_TEST_ORDER"
            match=not_requested; [ -z "$expected" ] || { match=no; [ "$expected" = fire_smoke_alarm ] && match=yes; }
            printf '{"LIVE_STATUS":"pass","DPU_RUN_CALLS":2,"PREDICTION_MATCH":"%s","top1":{"id":"fire_smoke_alarm"}}\n' "$match" > "$json"
            printf 'LIVE_STATUS=pass\nCLASS_ID=fire_smoke_alarm\nPREDICTION_MATCH=%s\nERROR=none\n' "$match" > "$display"
            echo "LIVE_STATUS=pass"
            echo "DPU_RUN_CALLS=2"
            echo "PREDICTION_MATCH=$match"
        """)
        self._exe(self.bundle / "dnndk/pkgs/bin/dexplorer", """#!/bin/sh
            case "$1" in
              -w) echo 'DPU Arch : B1152F' ;;
              -s) value=$(cat "$LIVE_TEST_COUNTER"); [ "${LIVE_TEST_STATIC_COUNTER:-0}" = 1 ] && value=0; echo "Core 0 schedule : $value"; echo "Core 0 interrupt : $value" ;;
              *) exit 64 ;;
            esac
        """)
        model_sha = sha256(self.bundle / "model/dpu_soundjie_d3.elf")
        runner_sha = sha256(self.bundle / "runtime/sound_shape_runner")
        (self.bundle / "manifest/contract.tsv").write_text(
            "SCHEMA\t1\nCLASS_COUNT\t8\nMODEL_KERNEL\tsoundjie_d3\n"
            "INPUT_NODE\tconv1\nOUTPUT_NODE\tfc8\nINPUT_SHAPE\t1,1,64,96\n"
            "INPUT_SIZE\t6144\nINPUT_SCALE\t128\nOUTPUT_SIZE\t8\nOUTPUT_SCALE\t0.0625\n"
            "AUDIO_USB_ID\t2886:001a\nAUDIO_FORMAT\tS16_LE\nAUDIO_CHANNELS\t2\n"
            "AUDIO_RATE\t16000\nCAPTURE_SECONDS\t6\n"
            f"MODEL_SHA256\t{model_sha}\nRUNNER_SHA256\t{runner_sha}\n",
            encoding="ascii", newline="\n",
        )

    def write_sums(self):
        names = [
            path.relative_to(self.bundle).as_posix()
            for path in sorted(self.bundle.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        ]
        (self.bundle / "manifest/SHA256SUMS").write_text(
            "".join(f"{sha256(self.bundle / name)}  {name}\n" for name in names),
            encoding="ascii", newline="\n",
        )

    def run_live(self, name, expected=None, **overrides):
        env = os.environ.copy()
        env.update({
            "PATH": f"{bash_path(self.tools)}:/usr/bin:/bin",
            "SJD3_LIVE_FDT": bash_path(self.fdt),
            "SJD3_LIVE_PROC_MODULES": bash_path(self.modules),
            "SJD3_LIVE_PROC_ROOT": bash_path(self.proc),
            "SJD3_LIVE_DEV_ROOT": bash_path(self.dev),
            "SJD3_LIVE_LIB_TARGET": bash_path(self.lib_target),
            "LIVE_TEST_COUNTER": bash_path(self.counter),
            "LIVE_TEST_DMESG": bash_path(self.dmesg),
            "LIVE_TEST_ORDER": bash_path(self.order),
            "LIVE_TEST_MOUNT_LOG": bash_path(self.mount_log),
            "LIVE_TEST_AUDIO_LOADED": bash_path(self.audio_loaded),
            "LIVE_TEST_CAPTURE_SOURCE": bash_path(self.capture_source),
        })
        env.update({key: str(value) for key, value in overrides.items()})
        command = [
            str(GIT_BASH), "-c", 'PATH="$1:$PATH"; export PATH; shift; exec sh "$@"',
            "board-test", bash_path(self.tools), bash_path(BOARD),
            "--bundle-dir", bash_path(self.bundle),
            "--output-dir", bash_path(self.root / name),
        ]
        if expected is not None:
            command += ["--expected-class", expected]
        return subprocess.run(
            command, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True
        )

    def assert_pre_dpu(self, result, reason):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"REASON={reason}", result.stdout)
        self.assertIn("DPU_RUN_CALLS=0", result.stdout)
        self.assertEqual(self.counter.read_text(encoding="ascii").strip(), "0")

    def test_success_runs_capture_selector_logmel_runner_and_fuser_in_order(self):
        result = self.run_live("success", expected="fire_smoke_alarm")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("LIVE_STATUS=pass", result.stdout)
        self.assertIn("DPU_RUN_CALLS=2", result.stdout)
        self.assertEqual(self.order.read_text(encoding="ascii").splitlines(), [
            "capture", "selector", "logmel:0", "runner", "logmel:8000", "runner", "fuser",
        ])
        output = self.root / "success"
        for name in (
            "capture.wav", "capture-report.txt", "window-metrics.tsv",
            "selected-windows.tsv", "window-predictions.tsv", "session-result.json",
            "display-status.txt", "dexplorer-before.txt", "dexplorer-after.txt", "kernel-new.txt",
        ):
            self.assertTrue((output / name).is_file(), name)

    def test_rejects_bad_hash_wrong_usb_silence_and_empty_selection_before_dpu(self):
        model = self.bundle / "model/dpu_soundjie_d3.elf"
        model.write_bytes(model.read_bytes() + b"tamper")
        self.assert_pre_dpu(self.run_live("bad-hash"), "bundle_hash_mismatch")

        self.tearDown(); self.setUp()
        (self.card / "usbid").write_text("2886:0018\n", encoding="ascii")
        self.assert_pre_dpu(self.run_live("wrong-usb"), "wrong_audio_usb_id")

        self.tearDown(); self.setUp()
        self.assert_pre_dpu(self.run_live("silent", LIVE_TEST_WAV_SIGNAL="absent"), "audio_signal_absent")

        self.tearDown(); self.setUp()
        self.assert_pre_dpu(self.run_live("empty", LIVE_TEST_SELECTOR_EMPTY="1"), "audio_signal_absent")

    def test_requires_dpu_delta_equal_selected_window_count(self):
        result = self.run_live("static", LIVE_TEST_STATIC_COUNTER="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REASON=counter_delta_mismatch", result.stdout)
        self.assertIn("DPU_RUN_CALLS=2", result.stdout)

    def test_prediction_mismatch_still_reports_live_pass(self):
        result = self.run_live("mismatch", expected="glass_breaking")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("LIVE_STATUS=pass", result.stdout)
        self.assertIn("PREDICTION_MATCH=no", result.stdout)

    def test_failure_unmounts_and_does_not_publish_success(self):
        result = self.run_live("runner-fail", LIVE_TEST_RUNNER_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REASON=runner_failed", result.stdout)
        self.assertIn("umount:", self.mount_log.read_text(encoding="ascii"))
        self.assertFalse((self.root / "runner-fail/session-result.json").exists())

    def test_scans_only_new_kernel_interval_for_dpu_usb_mmc_ext4_faults(self):
        old_only = self.run_live("old-only")
        self.assertEqual(old_only.returncode, 0, old_only.stdout + old_only.stderr)

        self.tearDown(); self.setUp()
        result = self.run_live("new-fault", LIVE_TEST_NEW_DMESG="usb 1-1: reset high-speed USB device")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REASON=new_kernel_fault", result.stdout)

    def test_rejects_relative_or_existing_output_directory(self):
        relative = subprocess.run(
            [str(GIT_BASH), bash_path(BOARD), "--bundle-dir", bash_path(self.bundle),
             "--output-dir", "relative"],
            text=True, encoding="utf-8", errors="replace", capture_output=True,
        )
        self.assertNotEqual(relative.returncode, 0)
        self.assertIn("REASON=path_must_be_absolute", relative.stdout)
        existing = self.root / "existing"
        existing.mkdir()
        result = self.run_live("existing")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("REASON=output_already_exists", result.stdout)

    def test_live_capture_uses_bundled_alsa_runtime(self):
        script = BOARD.read_text(encoding="utf-8")
        self.assertIn('alsa_lib="$bundle_dir/audio/lib"', script)
        self.assertIn('alsa_config="$bundle_dir/audio/share/alsa/alsa.conf"', script)
        self.assertIn('export ALSA_CONFIG_PATH="$alsa_config"', script)
        self.assertIn('$libs:$alsa_lib', script)


if __name__ == "__main__":
    unittest.main()
