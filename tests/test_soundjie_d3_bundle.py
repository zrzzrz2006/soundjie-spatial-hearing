import copy
import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "host" / "build_soundjie_d3_bundle.py"
CLASS_IDS = (
    "fire_smoke_alarm", "emergency_siren", "car_horn", "glass_breaking",
    "door_knock_bell", "distress_voice", "normal_speech", "background_unknown",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("soundjie_d3_bundle", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SoundjieD3BundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.base_records = []
        for index, (name, mode) in enumerate((
            ("runtime/dpu-4.14.0-xilinx.ko", "0644"),
            ("runtime/sound_shape_runner", "0755"),
            ("model/dpu_soundjie_d3.elf", "0644"),
            ("dnndk/pkgs/bin/dexplorer", "0755"),
            ("dnndk/pkgs/lib/libn2cube.so", "0644"),
            ("dnndk/pkgs/lib/libhineon.so", "0644"),
            ("config/logmel_frontend_v1.json", "0644"),
            ("config/soundjie_d3_model.json", "0644"),
            ("model/labels.txt", "0644"),
            ("manifest/soundjie-d3-model-manifest.json", "0644"),
            ("manifest/prepared-manifest.jsonl", "0644"),
            ("manifest/demo.jsonl", "0644"),
            ("manifest/calibration-manifest.json", "0644"),
            ("licenses/ESC-50-LICENSE", "0644"),
            ("licenses/LibriSpeech-LICENSE.TXT", "0644"),
            ("probe/run_soundjie_d3.sh", "0755"),
        )):
            path = self.sources / f"base-{index}"
            if name == "model/labels.txt":
                path.write_text("\n".join(CLASS_IDS) + "\n", encoding="ascii")
            elif name == "manifest/soundjie-d3-model-manifest.json":
                path.write_text(json.dumps({
                    "schema": 1, "result": "pass",
                    "compile_evidence": {
                        "kernels": ["soundjie_d3"], "input_node": "conv1",
                        "output_node": "fc8", "output_count": 8, "abi": 1,
                        "cpu_arch": "arm32", "dpu": "1152FA", "dpu_target": "v1.4.0",
                        "dnnc_version": "v2.05", "kernel_type": "DPUKernel",
                        "unsupported_layers": [],
                    },
                    "artifacts": {}, "upstream": {}, "quantization_metrics": {},
                }, sort_keys=True), encoding="utf-8")
            elif name == "manifest/prepared-manifest.jsonl":
                path.write_text("", encoding="utf-8")
            elif name == "manifest/demo.jsonl":
                path.write_text("", encoding="utf-8")
            elif name == "manifest/calibration-manifest.json":
                path.write_text(json.dumps({"records": []}), encoding="utf-8")
            else:
                path.write_bytes((name + f"-{index}").encode("ascii"))
            self.base_records.append({
                "name": name, "source": str(path), "size": path.stat().st_size,
                "sha256": sha256(path), "mode": mode,
            })

        self.cases = []
        prepared = []
        demo = []
        for class_index, class_id in enumerate(CLASS_IDS):
            for split, ordinal in (("test", 0), ("demo", 0), ("demo", 1)):
                case_id = f"{split}-{class_index}-{ordinal}"
                tensor = self.sources / f"{case_id}.i8"
                values = ((np.arange(6144) * (class_index + ordinal + 1)
                           + class_index * 7 + ordinal) % 241 - 120).astype(np.int8)
                tensor.write_bytes(values.tobytes())
                wav_path = None
                wav_hash = None
                if split == "demo":
                    wav_path = self.sources / f"{case_id}.wav"
                    with wave.open(str(wav_path), "wb") as stream:
                        stream.setnchannels(1)
                        stream.setsampwidth(2)
                        stream.setframerate(16000)
                        stream.writeframes((np.arange(16000, dtype=np.int16) + class_index).tobytes())
                    wav_hash = sha256(wav_path)
                recording_id = f"recording:{case_id}"
                source_group = f"group:{case_id}"
                source_hash = hashlib.sha256(recording_id.encode("ascii")).hexdigest()
                record = {
                    "id": case_id, "split": split, "target_id": class_id,
                    "target_index": class_index, "recording_id": recording_id,
                    "source_group": source_group, "source_sha256": source_hash,
                    "tensor_source": str(tensor), "tensor_sha256": sha256(tensor),
                }
                if split == "demo":
                    record.update({
                        "wav_source": str(wav_path), "wav_sha256": wav_hash,
                        "attribution": {
                            "source_id": "fixture", "license": "CC-BY-4.0",
                            "creator": f"creator-{class_index}",
                            "source_url": f"https://example.invalid/{case_id}",
                            "license_url": "https://creativecommons.org/licenses/by/4.0/",
                            "modification_notice": "resampled to mono 16 kHz WAV",
                            "source_citation": "Fixture dataset, recording-specific entry",
                        },
                    })
                    demo.append({
                        "recording_id": recording_id, "source_group": source_group,
                        "sha256": source_hash, "target_id": class_id,
                        "target_index": class_index, "license": "CC-BY-4.0",
                    })
                self.cases.append(record)
                prepared.append({
                    "recording_id": recording_id, "source_group": source_group,
                    "source_sha256": source_hash, "split": split,
                    "target_id": class_id, "target_index": class_index,
                })

        by_name = {item["name"]: item for item in self.base_records}
        prepared_path = Path(by_name["manifest/prepared-manifest.jsonl"]["source"])
        prepared_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in prepared),
                                 encoding="utf-8", newline="\n")
        demo_path = Path(by_name["manifest/demo.jsonl"]["source"])
        demo_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in demo),
                             encoding="utf-8", newline="\n")
        for item in self.base_records:
            path = Path(item["source"])
            item["size"], item["sha256"] = path.stat().st_size, sha256(path)

        by_name = {item["name"]: item for item in self.base_records}
        model_manifest_path = Path(
            by_name["manifest/soundjie-d3-model-manifest.json"]["source"]
        )
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        model_manifest["artifacts"] = {
            "elf_sha256": by_name["model/dpu_soundjie_d3.elf"]["sha256"]
        }
        model_manifest["upstream"] = {
            "frontend_sha256": by_name["config/logmel_frontend_v1.json"]["sha256"],
            "model_config_sha256": by_name["config/soundjie_d3_model.json"]["sha256"],
            "labels_sha256": by_name["model/labels.txt"]["sha256"],
            "prepared_manifest_sha256": by_name["manifest/prepared-manifest.jsonl"]["sha256"],
            "calibration_manifest_sha256": by_name["manifest/calibration-manifest.json"]["sha256"],
            "demo_manifest_sha256": by_name["manifest/demo.jsonl"]["sha256"],
            "git_commit": "a" * 40,
        }
        model_manifest_path.write_text(
            json.dumps(model_manifest, sort_keys=True), encoding="utf-8"
        )
        by_name["manifest/soundjie-d3-model-manifest.json"]["size"] = (
            model_manifest_path.stat().st_size
        )
        by_name["manifest/soundjie-d3-model-manifest.json"]["sha256"] = sha256(
            model_manifest_path
        )

        self.config = self.root / "bundle.json"
        self.write_config()

    def tearDown(self):
        self.temporary.cleanup()

    def write_config(self, mutate=None, name="bundle.json"):
        data = {
            "schema": 1,
            "git_commit": "a" * 40,
            "classes": list(CLASS_IDS),
            "model": {
                "architecture": "DPU_1152FA", "cpu_arch": "arm32", "abi": 1,
                "kernel": "soundjie_d3", "input_node": "conv1", "output_node": "fc8",
                "input_shape": [1, 1, 64, 96], "input_size": 6144,
                "input_scale": 128.0, "output_size": 8, "output_scale": 0.0625,
            },
            "archive_members": copy.deepcopy(self.base_records),
            "cases": copy.deepcopy(self.cases),
        }
        if mutate:
            mutate(data)
        path = self.root / name
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def test_bundle_is_deterministic_create_only_and_allowlisted(self):
        builder = load_builder()
        first = builder.build_bundle(self.config, self.root / "a")
        second = builder.build_bundle(self.config, self.root / "b")
        self.assertEqual(first["SJD3.TGZ"].sha256, second["SJD3.TGZ"].sha256)
        with tarfile.open(first["SJD3.TGZ"].path, "r:gz") as archive:
            members = archive.getmembers()
            names = [item.name for item in members]
            self.assertEqual(names, list(builder.expected_members(self.cases)))
            self.assertTrue(all(item.mtime == 0 and item.uid == 0 and item.gid == 0
                                for item in members))
            self.assertIn("manifest/ATTRIBUTION.md", names)
            self.assertEqual(sum(name.startswith("inputs/") for name in names), 24)
            self.assertEqual(sum(name.startswith("demo/") for name in names), 16)
        with self.assertRaises(FileExistsError):
            builder.build_bundle(self.config, self.root / "a")

    def test_rejects_demo_overlap_or_missing_attribution(self):
        builder = load_builder()

        def overlap(data):
            test = next(case for case in data["cases"] if case["split"] == "test")
            demo = next(case for case in data["cases"] if case["split"] == "demo")
            demo["source_group"] = test["source_group"]

        with self.assertRaisesRegex(ValueError, "demo overlap"):
            builder.build_bundle(self.write_config(overlap, "overlap.json"), self.root / "overlap")

        def missing(data):
            demo = next(case for case in data["cases"] if case["split"] == "demo")
            del demo["attribution"]["creator"]

        with self.assertRaisesRegex(ValueError, "attribution"):
            builder.build_bundle(self.write_config(missing, "missing.json"), self.root / "missing")

        def generic_source_page(data):
            demo = next(case for case in data["cases"] if case["split"] == "demo")
            demo["attribution"]["source_url"] = "https://example.invalid/dataset"
            demo["attribution"]["creator"] = demo["source_group"].split(":")[-1]

        with self.assertRaisesRegex(ValueError, "recording-specific attribution"):
            builder.build_bundle(
                self.write_config(generic_source_page, "generic.json"), self.root / "generic"
            )

    def test_model_manifest_must_bind_every_bundled_upstream(self):
        builder = load_builder()
        by_name = {item["name"]: item for item in self.base_records}
        manifest_path = Path(by_name["manifest/soundjie-d3-model-manifest.json"]["source"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertIn(
            "SJD3.TGZ",
            builder.build_bundle(self.write_config(name="bound.json"), self.root / "bound"),
        )

        for key in (
            "frontend_sha256", "model_config_sha256", "labels_sha256",
            "prepared_manifest_sha256", "calibration_manifest_sha256",
            "demo_manifest_sha256", "git_commit",
        ):
            with self.subTest(key=key):
                changed = json.loads(manifest_path.read_text(encoding="utf-8"))
                changed["upstream"][key] = "b" * (40 if key == "git_commit" else 64)
                manifest_path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
                by_name["manifest/soundjie-d3-model-manifest.json"]["size"] = manifest_path.stat().st_size
                by_name["manifest/soundjie-d3-model-manifest.json"]["sha256"] = sha256(manifest_path)
                with self.assertRaisesRegex(ValueError, "upstream hash chain"):
                    builder.build_bundle(
                        self.write_config(name=f"drift-{key}.json"), self.root / f"drift-{key}"
                    )
                changed["upstream"][key] = manifest["upstream"][key]
                manifest_path.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
                by_name["manifest/soundjie-d3-model-manifest.json"]["size"] = manifest_path.stat().st_size
                by_name["manifest/soundjie-d3-model-manifest.json"]["sha256"] = sha256(manifest_path)

    def test_rejects_hash_contract_and_allowlist_drift(self):
        builder = load_builder()
        mutations = (
            ("hash", lambda data: data["archive_members"][0].update(sha256="0" * 64), "SHA-256"),
            ("shape", lambda data: data["model"].update(input_shape=[1, 96, 64, 1]), "model contract"),
            ("node", lambda data: data["model"].update(output_node="wrong"), "model contract"),
            ("extra", lambda data: data["archive_members"].append(dict(data["archive_members"][0], name="extra")), "allowlist"),
        )
        for name, mutate, reason in mutations:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, reason):
                    builder.build_bundle(self.write_config(mutate, f"{name}.json"),
                                         self.root / f"out-{name}")


if __name__ == "__main__":
    unittest.main()
