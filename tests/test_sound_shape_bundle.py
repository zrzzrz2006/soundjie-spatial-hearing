import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "host" / "build_sound_shape_bundle.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("sound_shape_bundle", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SoundShapeBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        payloads = {
            "dpu.ko": b"module", "runner": b"runner", "model.elf": b"elf",
            "dexplorer": b"dexplorer", "libn2cube.so": b"n2cube",
            "libhineon.so": b"hineon", "probe.sh": b"#!/bin/sh\n",
        }
        for name, payload in payloads.items():
            (self.sources / name).write_bytes(payload)
        names = (
            ("runtime/dpu-4.14.0-xilinx.ko", "dpu.ko", "0644"),
            ("runtime/sound_shape_probe", "runner", "0755"),
            ("model/dpu_sound_shape_probe.elf", "model.elf", "0644"),
            ("dnndk/pkgs/bin/dexplorer", "dexplorer", "0755"),
            ("dnndk/pkgs/lib/libn2cube.so", "libn2cube.so", "0644"),
            ("dnndk/pkgs/lib/libhineon.so", "libhineon.so", "0644"),
            ("probe/run_sound_shape_probe.sh", "probe.sh", "0755"),
        )
        self.names = [name for name, _, _ in names]
        self.config = self.root / "bundle.json"
        self.config.write_text(json.dumps({
            "schema": 1,
            "model": {
                "architecture": "DPU_1152FA", "cpu_arch": "arm32",
                "kernel": "sound_shape_probe", "input_node": "conv1",
                "output_node": "logits", "input_shape": [1, 1, 64, 96],
                "class_count": 8,
            },
            "bundle": {"archive_members": [
                {"name": target, "source": str(self.sources / source),
                 "size": (self.sources / source).stat().st_size,
                 "sha256": digest(self.sources / source), "mode": mode}
                for target, source, mode in names
            ]},
        }), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_bundle_is_deterministic_and_exactly_allowlisted(self):
        builder = load_builder()
        a = builder.build_bundle(self.config, self.root / "a")
        b = builder.build_bundle(self.config, self.root / "b")
        self.assertEqual(a["SJD0.TGZ"].sha256, b["SJD0.TGZ"].sha256)
        with tarfile.open(self.root / "a/SJD0.TGZ", "r:gz") as archive:
            members = archive.getmembers()
            self.assertEqual([item.name for item in members[:7]], self.names)
            self.assertTrue(all(item.mtime == 0 for item in members))

    def test_rejects_hash_mismatch_and_traversal(self):
        builder = load_builder()
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["bundle"]["archive_members"][0]["sha256"] = "0" * 64
        bad_hash = self.root / "bad-hash.json"
        bad_hash.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            builder.build_bundle(bad_hash, self.root / "bad-a")
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["bundle"]["archive_members"][0]["name"] = "../escape"
        bad_path = self.root / "bad-path.json"
        bad_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            builder.build_bundle(bad_path, self.root / "bad-b")


if __name__ == "__main__":
    unittest.main()
