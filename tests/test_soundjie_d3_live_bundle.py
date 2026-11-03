import copy
import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "host" / "build_soundjie_d3_live_bundle.py"
CLASSES = (
    "fire_smoke_alarm", "emergency_siren", "car_horn", "glass_breaking",
    "door_knock_bell", "distress_voice", "normal_speech", "background_unknown",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("soundjie_d3_live_bundle", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LiveBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.members = []
        self.demos = []

        fixed = (
            "runtime/dpu-4.14.0-xilinx.ko", "runtime/sound_shape_runner",
            "runtime/wav_probe", "runtime/logmel_cli", "runtime/live_window_selector",
            "runtime/live_result_fuser", "model/dpu_soundjie_d3.elf",
            "model/labels.txt", "model/display_labels_zh.txt",
            "dnndk/pkgs/bin/dexplorer", "dnndk/pkgs/lib/libn2cube.so",
            "dnndk/pkgs/lib/libhineon.so", "config/logmel_frontend_v1.json",
            "config/soundjie_d3_model.json", "config/soundjie_live_v1.tsv",
            "manifest/soundjie-d3-model-manifest.json", "manifest/prepared-manifest.jsonl",
            "manifest/demo.jsonl", "manifest/calibration-manifest.json",
            "licenses/ESC-50-LICENSE", "licenses/LibriSpeech-LICENSE.TXT",
            "probe/run_soundjie_d3_live_once.sh", "probe/run_soundjie_d3_live_wav.sh",
            "audio/load-usb-audio.sh", "audio/board-tools/xvf3800_probe.sh",
            "audio/board-tools/xvf3800_record.sh", "audio/bin/arecord",
            "audio/lib/libasound.so.2", "audio/share/alsa/alsa.conf",
            "audio/modules/snd-hwdep.ko", "audio/modules/snd-rawmidi.ko",
            "audio/modules/snd-usbmidi-lib.ko", "audio/modules/snd-usb-audio.ko",
            "README.md",
        )
        for name in fixed:
            payload = (name + "\n").encode()
            if name == "model/labels.txt":
                payload = ("\n".join(CLASSES) + "\n").encode("ascii")
            self.add_member(name, payload, "0755" if name.endswith(".sh") or name.endswith("runner") or name.endswith("arecord") or "/bin/" in name else "0644")

        for index, class_id in enumerate(CLASSES):
            for ordinal in range(2):
                case_id = f"{index:02d}-{class_id}-demo-{ordinal:02d}"
                wav = f"demo/{case_id}.wav"
                tensor = f"inputs/{case_id}.i8"
                direct = f"direct/{case_id}.logits.i8"
                self.add_member(wav, b"RIFF" + bytes([index, ordinal]) * 40)
                self.add_member(tensor, bytes([(index + ordinal + 1) % 127]) * 6144)
                self.add_member(direct, bytes(range(8)))
                self.demos.append({
                    "id": case_id, "wav_member": wav, "tensor_member": tensor,
                    "direct_logits_member": direct, "target_index": index,
                    "target_id": class_id, "source_id": f"recording-{case_id}",
                    "license": "CC-BY-4.0", "creator": f"creator-{case_id}",
                    "source_url": f"https://example.invalid/recording/{case_id}",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "modification_notice": "resampled to mono 16 kHz WAV",
                })
        self.config = self.write_config()

    def tearDown(self):
        self.tmp.cleanup()

    def add_member(self, name, payload, mode="0644"):
        path = self.sources / f"f-{len(self.members):03d}"
        path.write_bytes(payload)
        self.members.append({"name": name, "source": str(path), "size": len(payload),
                             "sha256": digest(path), "mode": mode})

    def write_config(self, mutate=None, name="bundle.json"):
        data = {"schema": 1, "classes": list(CLASSES),
                "archive_members": copy.deepcopy(self.members),
                "demos": copy.deepcopy(self.demos)}
        if mutate:
            mutate(data)
        path = self.root / name
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def test_bundle_is_deterministic_create_only_and_fully_allowlisted(self):
        builder = load_builder()
        first = builder.build_bundle(self.config, self.root / "a")
        second = builder.build_bundle(self.config, self.root / "b")
        self.assertEqual(first["SJD3L1.TGZ"].sha256, second["SJD3L1.TGZ"].sha256)
        self.assertEqual(first["SJD3L1.SHA"].sha256, second["SJD3L1.SHA"].sha256)
        with tarfile.open(first["SJD3L1.TGZ"].path, "r:gz") as archive:
            members = archive.getmembers()
            names = [item.name for item in members]
            self.assertEqual(names[-3:], ["manifest/contract.tsv", "manifest/demo-playback.tsv", "manifest/SHA256SUMS"])
            self.assertEqual(len(names), len(set(names)))
            self.assertTrue(all(item.mtime == 0 and item.uid == 0 and item.gid == 0 for item in members))
            self.assertTrue(all(item.isfile() for item in members))
        with self.assertRaises(FileExistsError):
            builder.build_bundle(self.config, self.root / "a")

    def test_bundle_binds_audio_modules_frontend_model_tools_and_licenses(self):
        builder = load_builder()
        result = builder.build_bundle(self.config, self.root / "bound")
        with tarfile.open(result["SJD3L1.TGZ"].path, "r:gz") as archive:
            contract = archive.extractfile("manifest/contract.tsv").read().decode("ascii")
            playback = archive.extractfile("manifest/demo-playback.tsv").read().decode("utf-8")
            sums = archive.extractfile("manifest/SHA256SUMS").read().decode("ascii")
        for token in ("MODEL_SHA256", "RUNNER_SHA256", "FRONTEND_SHA256", "AUDIO_USB_ID\t2886:001a"):
            self.assertIn(token, contract)
        for name in ("snd-usb-audio.ko", "ESC-50-LICENSE", "LibriSpeech-LICENSE", "live_window_selector", "live_result_fuser"):
            self.assertIn(name, sums)
        self.assertEqual(len(playback.splitlines()), 17)

    def test_rejects_unlicensed_or_unlisted_demo(self):
        builder = load_builder()
        def unlicensed(data):
            data["demos"][0]["license"] = ""
        with self.assertRaisesRegex(ValueError, "license|attribution"):
            builder.build_bundle(self.write_config(unlicensed, "unlicensed.json"), self.root / "u")

        def unlisted(data):
            data["demos"][0]["wav_member"] = "demo/not-listed.wav"
        with self.assertRaisesRegex(ValueError, "demo|member"):
            builder.build_bundle(self.write_config(unlisted, "unlisted.json"), self.root / "x")


if __name__ == "__main__":
    unittest.main()
