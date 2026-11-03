import gzip
import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "host" / "build_dpu_inference_bundle.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_builder():
    if not BUILDER_PATH.is_file():
        raise AssertionError(f"missing builder: {BUILDER_PATH}")
    spec = importlib.util.spec_from_file_location("dpu_inference_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DpuInferenceBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        payloads = {
            "dpu.ko": b"module",
            "runner": b"runner",
            "dexplorer": b"dexplorer",
            "libn2cube.so": b"n2cube",
            "libhineon.so": b"hineon",
            "probe.sh": b"#!/bin/sh\necho probe\n",
        }
        for name, payload in payloads.items():
            (self.sources / name).write_bytes(payload)
        members = [
            ("runtime/dpu-4.14.0-xilinx.ko", "dpu.ko", "0644"),
            ("runtime/dpu_yolo_probe", "runner", "0755"),
            ("dnndk/pkgs/bin/dexplorer", "dexplorer", "0755"),
            ("dnndk/pkgs/lib/libn2cube.so", "libn2cube.so", "0644"),
            ("dnndk/pkgs/lib/libhineon.so", "libhineon.so", "0644"),
            ("probe/run_real_dpu_inference.sh", "probe.sh", "0755"),
        ]
        config = {
            "schema": 1,
            "model": {
                "architecture": "DPU_1152FA",
                "target": "v1.4.0",
                "kernel": "yolo",
                "input_node": "layer0_conv",
                "output_nodes": [
                    "layer81_conv", "layer93_conv", "layer105_conv", "layer117_conv"
                ],
            },
            "bundle": {
                "archive_members": [
                    {"name": target, **self.record(source), "mode": mode}
                    for target, source, mode in members
                ]
            },
        }
        self.member_names = [target for target, _, _ in members]
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps(config), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def record(self, name: str):
        path = self.sources / name
        return {"source": str(path), "size": path.stat().st_size, "sha256": sha256(path)}

    def test_rejects_pinned_source_hash_mismatch(self):
        builder = load_builder()
        config = json.loads(self.config.read_text(encoding="utf-8"))
        config["bundle"]["archive_members"][0]["sha256"] = "0" * 64
        self.config.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            builder.build_bundle(self.config, self.root / "out")

    def test_rejects_absolute_or_traversal_member(self):
        builder = load_builder()
        for unsafe in ("/absolute", "../escape", "safe/../../escape"):
            config = json.loads(self.config.read_text(encoding="utf-8"))
            config["bundle"]["archive_members"][0]["name"] = unsafe
            candidate = self.root / (unsafe.replace("/", "_") + ".json")
            candidate.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "archive member"):
                builder.build_bundle(candidate, self.root / ("out-" + candidate.stem))

    def test_exact_allowlist_modes_and_manifests(self):
        builder = load_builder()
        result = builder.build_bundle(self.config, self.root / "out")
        self.assertEqual(set(result), {"DPUY1.TGZ", "DPUY1.SHA"})
        with tarfile.open(self.root / "out" / "DPUY1.TGZ", "r:gz") as archive:
            members = archive.getmembers()
            expected = self.member_names + [
                "manifest/inference-contract.json", "manifest/SHA256SUMS"
            ]
            self.assertEqual([item.name for item in members], expected)
            self.assertTrue(all(item.mtime == 0 for item in members))
            self.assertTrue(all(item.uid == 0 and item.gid == 0 for item in members))
            self.assertTrue(all(item.uname == "" and item.gname == "" for item in members))
            modes = {item.name: item.mode for item in members}
            self.assertEqual(modes["runtime/dpu_yolo_probe"], 0o755)
            self.assertEqual(modes["probe/run_real_dpu_inference.sh"], 0o755)
            contract = json.loads(
                archive.extractfile("manifest/inference-contract.json").read().decode("utf-8")
            )
            self.assertEqual(contract["kernel"], "yolo")
            self.assertEqual(len(contract["output_nodes"]), 4)

    def test_two_builds_are_byte_identical_and_sha_file_is_correct(self):
        builder = load_builder()
        first = builder.build_bundle(self.config, self.root / "out-a")
        second = builder.build_bundle(self.config, self.root / "out-b")
        for name in ("DPUY1.TGZ", "DPUY1.SHA"):
            self.assertEqual(first[name].sha256, second[name].sha256)
            self.assertEqual(first[name].size, second[name].size)
        with gzip.open(self.root / "out-a" / "DPUY1.TGZ", "rb") as archive:
            self.assertTrue(archive.read())
        expected = f"{first['DPUY1.TGZ'].sha256}  DPUY1.TGZ\n"
        self.assertEqual((self.root / "out-a" / "DPUY1.SHA").read_text("ascii"), expected)


if __name__ == "__main__":
    unittest.main()
