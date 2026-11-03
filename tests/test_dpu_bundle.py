import gzip
import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts" / "host" / "build_dpu_probe_bundle.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_builder():
    if not BUILDER_PATH.is_file():
        raise AssertionError(f"missing builder: {BUILDER_PATH}")
    spec = importlib.util.spec_from_file_location("dpu_bundle_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DpuBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        payloads = {
            "bit.bin": b"bitstream-source",
            "dpu.dtb": b"device-tree-source",
            "dpu.ko": b"kernel-module-source",
            "dpu_smoke": b"smoke-source",
            "dexplorer": b"dexplorer-source",
            "libn2cube.so": b"n2cube-source",
            "libhineon.so": b"hineon-source",
            "probe.sh": b"#!/bin/sh\necho probe\n",
        }
        for name, payload in payloads.items():
            (self.sources / name).write_bytes(payload)

        self.archive_members = [
            ("runtime/dpu-4.14.0-xilinx.ko", "dpu.ko", "0644"),
            ("runtime/dpu_smoke", "dpu_smoke", "0755"),
            ("dnndk/pkgs/bin/dexplorer", "dexplorer", "0755"),
            ("dnndk/pkgs/lib/libn2cube.so", "libn2cube.so", "0644"),
            ("dnndk/pkgs/lib/libhineon.so", "libhineon.so", "0644"),
            ("probe/run_ephemeral_dpu_probe.sh", "probe.sh", "0755"),
        ]
        config = {
            "schema": 1,
            "outputs": {
                "DPUV1.BIT": self.record("bit.bin"),
                "DPUV1.DTB": self.record("dpu.dtb"),
            },
            "archive_members": [
                {
                    "name": name,
                    **self.record(source),
                    "mode": mode,
                }
                for name, source, mode in self.archive_members
            ],
        }
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(config), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def record(self, name: str):
        path = self.sources / name
        return {
            "source": str(path),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }

    def test_rejects_source_size_mismatch(self):
        builder = load_builder()
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["outputs"]["DPUV1.BIT"]["size"] += 1
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            builder.build_bundle(self.config_path, self.root / "out")

    def test_rejects_source_hash_mismatch(self):
        builder = load_builder()
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["archive_members"][0]["sha256"] = "0" * 64
        self.config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            builder.build_bundle(self.config_path, self.root / "out")

    def test_bundle_contains_only_allowlisted_members(self):
        builder = load_builder()
        builder.build_bundle(self.config_path, self.root / "out")
        archive = self.root / "out" / "DPUV1.TGZ"
        expected = [name for name, _, _ in self.archive_members] + [
            "manifest/SHA256SUMS"
        ]
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            self.assertEqual([member.name for member in members], expected)
            self.assertTrue(all(member.mtime == 0 for member in members))
            self.assertTrue(all(member.uid == 0 and member.gid == 0 for member in members))
            self.assertTrue(all(member.uname == "" and member.gname == "" for member in members))
            modes = {member.name: member.mode for member in members}
            self.assertEqual(modes["runtime/dpu_smoke"], 0o755)
            self.assertEqual(modes["dnndk/pkgs/bin/dexplorer"], 0o755)
            self.assertEqual(modes["probe/run_ephemeral_dpu_probe.sh"], 0o755)
            manifest = bundle.extractfile("manifest/SHA256SUMS").read().decode("ascii")
            for name, source, _ in self.archive_members:
                self.assertIn(f"{sha256(self.sources / source)}  {name}\n", manifest)

    def test_two_builds_have_identical_sha256(self):
        builder = load_builder()
        first = builder.build_bundle(self.config_path, self.root / "out-a")
        second = builder.build_bundle(self.config_path, self.root / "out-b")
        self.assertEqual(set(first), {"DPUV1.BIT", "DPUV1.DTB", "DPUV1.TGZ", "DPUV1.SHA"})
        for name in first:
            self.assertEqual(first[name].sha256, second[name].sha256)
            self.assertEqual(first[name].size, second[name].size)
        with gzip.open(self.root / "out-a" / "DPUV1.TGZ", "rb") as archive:
            self.assertTrue(archive.read())
        sha_manifest = (self.root / "out-a" / "DPUV1.SHA").read_text(encoding="ascii")
        for name in ("DPUV1.BIT", "DPUV1.DTB", "DPUV1.TGZ"):
            self.assertIn(f"{first[name].sha256}  {name}\n", sha_manifest)


if __name__ == "__main__":
    unittest.main()
