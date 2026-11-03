import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "host" / "build_sound_wav_bundle.py"
CASES = (
    "xvf3800-start-0000000000",
    "xvf3800-start-0000072000",
    "xvf3800-start-0000144000",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("sound_wav_bundle", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SoundWavBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        names = [
            ("runtime/dpu-4.14.0-xilinx.ko", "dpu.ko", "0644"),
            ("runtime/sound_shape_runner", "runner", "0755"),
            ("model/dpu_sound_shape_probe.elf", "model", "0644"),
            ("dnndk/pkgs/bin/dexplorer", "dexplorer", "0755"),
            ("dnndk/pkgs/lib/libn2cube.so", "n2cube", "0644"),
            ("dnndk/pkgs/lib/libhineon.so", "hineon", "0644"),
            ("frontend/logmel_cli", "logmel", "0755"),
            ("config/logmel_frontend_v1.json", "config", "0644"),
            ("manifest/dpu-input-contract.txt", "contract", "0644"),
        ]
        for case in CASES:
            names.extend((
                (f"inputs/{case}.wav", f"{case}.wav", "0644"),
                (f"golden/{case}.f32le", f"{case}.f32", "0644"),
                (f"golden/{case}.i8", f"{case}.i8", "0644"),
            ))
        names.append(("probe/run_sound_wav_dpu.sh", "probe", "0755"))
        records = []
        for index, (target, source, mode) in enumerate(names):
            path = self.sources / source
            size = 6144 if target.endswith(".i8") else index + 11
            if target == "config/logmel_frontend_v1.json":
                path.write_bytes((ROOT / "config/logmel_frontend_v1.json").read_bytes())
                size = path.stat().st_size
            elif target.endswith(".i8"):
                case_index = int(target.split("start-")[1].split(".")[0]) // 72000
                values = ((np.arange(6144) * (case_index * 12 + 1) + case_index)
                          % 121 - 60).astype(np.int8)
                path.write_bytes(values.tobytes())
            else:
                path.write_bytes(bytes([(index % 251) + 1]) * size)
            records.append({"name": target, "source": str(path), "size": size,
                            "sha256": sha256(path), "mode": mode})
        self.config = self.root / "bundle.json"
        self.config.write_text(json.dumps({
            "schema": 2,
            "input_scale": 128.0,
            "normalization_profile": "bounded-log-power-64-v1",
            "config_sha256": next(item["sha256"] for item in records
                                  if item["name"] == "config/logmel_frontend_v1.json"),
            "cases": list(CASES),
            "model": {
                "architecture": "DPU_1152FA", "cpu_arch": "arm32",
                "kernel": "sound_shape_probe", "input_node": "conv1",
                "output_node": "logits", "input_shape": [1, 1, 64, 96],
                "class_count": 8,
            },
            "archive_members": records,
        }), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def test_deterministic_exact_allowlist_and_manifest(self):
        builder = load_builder()
        first = builder.build_bundle(self.config, self.root / "a")
        second = builder.build_bundle(self.config, self.root / "b")
        self.assertEqual(first["SJD2.TGZ"].sha256, second["SJD2.TGZ"].sha256)
        with tarfile.open(first["SJD2.TGZ"].path, "r:gz") as archive:
            members = archive.getmembers()
            self.assertTrue(all(item.mtime == 0 and item.uid == 0 and item.gid == 0
                                for item in members))
            names = [item.name for item in members]
            self.assertEqual(names[-3:], ["manifest/cases.tsv", "manifest/contract.tsv",
                                         "manifest/SHA256SUMS"])
            contract = archive.extractfile("manifest/contract.tsv").read().decode("ascii")
            self.assertIn("INPUT_SCALE\t128\n", contract)
            self.assertIn("INPUT_SIZE\t6144\n", contract)
            self.assertIn("MODEL_KERNEL\tsound_shape_probe\n", contract)
            self.assertIn("NORMALIZATION_PROFILE\tbounded-log-power-64-v1\n", contract)

    def test_rejects_hash_size_scale_model_traversal_and_overwrite(self):
        builder = load_builder()
        mutations = (
            ("hash", lambda data: data["archive_members"][0].update(sha256="0" * 64), "SHA-256"),
            ("size", lambda data: data["archive_members"][0].update(size=999), "wrong-size"),
            ("scale", lambda data: data.update(input_scale=8.0), "input scale"),
            ("model", lambda data: data["model"].update(kernel="wrong"), "model contract"),
            ("path", lambda data: data["archive_members"][0].update(name="../escape"), "allowlist"),
        )
        for name, mutate, reason in mutations:
            with self.subTest(name=name):
                data = json.loads(self.config.read_text(encoding="utf-8"))
                mutate(data)
                path = self.root / f"{name}.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, reason):
                    builder.build_bundle(path, self.root / f"out-{name}")
        builder.build_bundle(self.config, self.root / "existing")
        with self.assertRaises(FileExistsError):
            builder.build_bundle(self.config, self.root / "existing")

    def test_validated_source_snapshot_survives_later_mutation(self):
        builder = load_builder()
        data = json.loads(self.config.read_text(encoding="utf-8"))
        target = Path(data["archive_members"][1]["source"])
        expected = target.read_bytes()
        original = builder.tempfile.TemporaryDirectory

        class MutatingTemporaryDirectory:
            def __init__(self, *args, **kwargs):
                self.inner = original(*args, **kwargs)

            def __enter__(self):
                value = self.inner.__enter__()
                target.write_bytes(b"changed-after-validation")
                return value

            def __exit__(self, *args):
                return self.inner.__exit__(*args)

        builder.tempfile.TemporaryDirectory = MutatingTemporaryDirectory
        try:
            result = builder.build_bundle(self.config, self.root / "snapshot")
        finally:
            builder.tempfile.TemporaryDirectory = original
        with tarfile.open(result["SJD2.TGZ"].path, "r:gz") as archive:
            self.assertEqual(archive.extractfile("runtime/sound_shape_runner").read(), expected)

    def test_publishes_create_only_when_hard_links_are_unsupported(self):
        builder = load_builder()
        original = builder.os.link

        def unsupported_link(*_args, **_kwargs):
            raise OSError("hard links are unsupported on this filesystem")

        builder.os.link = unsupported_link
        try:
            result = builder.build_bundle(self.config, self.root / "no-hard-links")
        finally:
            builder.os.link = original
        self.assertTrue(result["SJD2.TGZ"].path.is_file())
        self.assertTrue(result["SJD2.SHA256"].path.is_file())
        with self.assertRaises(FileExistsError):
            builder.build_bundle(self.config, self.root / "no-hard-links")

    def test_rejects_old_profile_and_identical_real_tensors(self):
        builder = load_builder()
        data = json.loads(self.config.read_text(encoding="utf-8"))
        data["normalization_profile"] = "identity-64-v1"
        bad_profile = self.root / "bad-profile.json"
        bad_profile.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            builder.build_bundle(bad_profile, self.root / "bad-profile")

        data = json.loads(self.config.read_text(encoding="utf-8"))
        for item in data["archive_members"]:
            if item["name"].endswith(".i8"):
                path = Path(item["source"])
                path.write_bytes(bytes(range(64)) * 96)
                item["sha256"] = sha256(path)
        same = self.root / "same.json"
        same.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            builder.build_bundle(same, self.root / "same")


if __name__ == "__main__":
    unittest.main()
