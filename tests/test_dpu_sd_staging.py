import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
STAGER_PATH = ROOT / "scripts" / "host" / "stage_dpu_sd.py"
BOOT_NAMES = ("BOOT.bin", "uImage", "devicetree.dtb", "uEnv.txt")
DPU_NAMES = ("DPUV1.BIT", "DPUV1.DTB", "DPUV1.TGZ", "DPUV1.SHA")


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def load_stager():
    if not STAGER_PATH.is_file():
        raise AssertionError(f"missing stager: {STAGER_PATH}")
    spec = importlib.util.spec_from_file_location("dpu_sd_stager", STAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DpuSdStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sd = self.root / "sd"
        self.source = self.root / "prepared"
        self.sd.mkdir()
        self.source.mkdir()
        for index, name in enumerate(BOOT_NAMES):
            (self.sd / name).write_bytes(f"boot-{index}".encode("ascii"))
        for index, name in enumerate(DPU_NAMES[:3]):
            (self.source / name).write_bytes((f"dpu-{index}-" * 20).encode("ascii"))
        manifest = "".join(
            f"{sha256(self.source / name)}  {name}\n" for name in DPU_NAMES[:3]
        )
        (self.source / "DPUV1.SHA").write_text(manifest, encoding="ascii", newline="\n")
        self.boot_hashes = {name: sha256(self.sd / name) for name in BOOT_NAMES}

    def tearDown(self):
        self.temp.cleanup()

    def preflight(self, stager, **kwargs):
        with mock.patch.object(stager, "read_volume_serial", return_value="54D7-D258"):
            return stager.preflight(
                self.sd,
                self.source,
                expected_volume_serial="54D7-D258",
                expected_boot_hashes=self.boot_hashes,
                **kwargs,
            )

    def test_refuses_wrong_volume_serial(self):
        stager = load_stager()
        with mock.patch.object(stager, "read_volume_serial", return_value="1111-2222"):
            with self.assertRaisesRegex(ValueError, "volume serial mismatch"):
                stager.preflight(
                    self.sd,
                    self.source,
                    expected_volume_serial="54D7-D258",
                    expected_boot_hashes=self.boot_hashes,
                )

    def test_refuses_changed_original_boot_hash(self):
        stager = load_stager()
        (self.sd / "uImage").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "original boot hash mismatch"):
            self.preflight(stager)

    def test_refuses_any_existing_dpu_name_before_copy(self):
        stager = load_stager()
        (self.sd / "DPUV1.TGZ").write_bytes(b"pre-existing")
        with self.assertRaisesRegex(FileExistsError, "DPUV1.TGZ"):
            self.preflight(stager)
        self.assertFalse((self.sd / "DPUV1.BIT").exists())

    def test_refuses_insufficient_free_space(self):
        stager = load_stager()
        Usage = namedtuple("Usage", "total used free")
        with mock.patch.object(stager.shutil, "disk_usage", return_value=Usage(100, 99, 1)):
            with self.assertRaisesRegex(OSError, "insufficient free space"):
                self.preflight(stager)

    def test_dry_run_writes_nothing(self):
        stager = load_stager()
        plan = self.preflight(stager)
        report = stager.stage(plan, dry_run=True)
        self.assertTrue(report["dry_run"])
        self.assertTrue(all(not (self.sd / name).exists() for name in DPU_NAMES))

    def test_copy_is_create_only(self):
        stager = load_stager()
        plan = self.preflight(stager)
        (self.sd / "DPUV1.DTB").write_bytes(b"race-created")
        with self.assertRaisesRegex(FileExistsError, "DPUV1.DTB"):
            stager.stage(plan)
        self.assertFalse((self.sd / "DPUV1.BIT").exists())
        self.assertEqual((self.sd / "DPUV1.DTB").read_bytes(), b"race-created")

    def test_readback_hashes_match_manifest(self):
        stager = load_stager()
        plan = self.preflight(stager)
        report = stager.stage(plan)
        self.assertFalse(report["dry_run"])
        self.assertEqual(set(report["files"]), set(DPU_NAMES))
        for name in DPU_NAMES:
            self.assertEqual(report["files"][name]["source_sha256"], sha256(self.source / name))
            self.assertEqual(report["files"][name]["readback_sha256"], sha256(self.sd / name))
            self.assertEqual((self.sd / name).read_bytes(), (self.source / name).read_bytes())
        for name, expected in self.boot_hashes.items():
            self.assertEqual(sha256(self.sd / name), expected)


if __name__ == "__main__":
    unittest.main()
