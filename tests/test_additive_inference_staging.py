import hashlib
import importlib.util
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
STAGER_PATH = ROOT / "scripts" / "host" / "stage_additive_inference_sd.py"
BOOT_NAMES = ("BOOT.bin", "uImage", "devicetree.dtb", "uEnv.txt")
DESTINATIONS = ("DPUY1.TGZ", "DPUY1.SHA")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stager():
    if not STAGER_PATH.is_file():
        raise AssertionError(f"missing stager: {STAGER_PATH}")
    spec = importlib.util.spec_from_file_location("additive_stager", STAGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AdditiveInferenceStagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sd = self.root / "sd"
        self.source = self.root / "prepared"
        self.sd.mkdir()
        self.source.mkdir()
        for index, name in enumerate(BOOT_NAMES):
            (self.sd / name).write_bytes(f"boot-{index}".encode())
        (self.source / "DPUY1.TGZ").write_bytes(b"deterministic-real-dpu-bundle")
        (self.source / "DPUY1.SHA").write_text(
            f"{sha256(self.source / 'DPUY1.TGZ')}  DPUY1.TGZ\n", encoding="ascii"
        )
        self.boot_hashes = {name: sha256(self.sd / name) for name in BOOT_NAMES}

    def tearDown(self):
        self.temp.cleanup()

    def preflight(self, stager, *, clean=True):
        with mock.patch.object(stager, "read_volume_serial", return_value="54D7-D258"), mock.patch.object(
            stager, "read_filesystem_state", return_value={"clean": clean, "detail": "test"}
        ):
            return stager.preflight(
                self.sd, self.source, self.source / "DPUY1.SHA", DESTINATIONS,
                expected_volume_serial="54D7-D258", expected_boot_hashes=self.boot_hashes,
            )

    def test_rejects_wrong_volume_and_dirty_filesystem(self):
        stager = load_stager()
        with mock.patch.object(stager, "read_volume_serial", return_value="1111-2222"), mock.patch.object(
            stager, "read_filesystem_state", return_value={"clean": True, "detail": "test"}
        ):
            with self.assertRaisesRegex(ValueError, "volume serial mismatch"):
                stager.preflight(self.sd, self.source, self.source / "DPUY1.SHA", DESTINATIONS,
                                 expected_volume_serial="54D7-D258", expected_boot_hashes=self.boot_hashes)
        with self.assertRaisesRegex(OSError, "filesystem is not clean"):
            self.preflight(stager, clean=False)

    def test_reads_filesystem_state_from_unlettered_volume_path(self):
        stager = load_stager()
        volume_root = Path("\\\\?\\Volume{11111111-2222-3333-4444-555555555555}\\")
        completed = stager.subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=(
                '{"HealthStatus":"Healthy","OperationalStatus":"OK",'
                '"FileSystem":"FAT32"}'
            ),
            stderr="",
        )
        with mock.patch.object(stager.subprocess, "run", return_value=completed) as run:
            state = stager.read_filesystem_state(volume_root)

        self.assertTrue(state["clean"])
        command = run.call_args.args[0][-1]
        self.assertIn("Get-Volume -UniqueId", command)
        self.assertIn(str(volume_root), command)

    def test_rejects_changed_boot_existing_destination_and_low_space(self):
        stager = load_stager()
        (self.sd / "uImage").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "original boot hash mismatch"):
            self.preflight(stager)
        (self.sd / "uImage").write_bytes(b"boot-1")
        (self.sd / "DPUY1.TGZ").write_bytes(b"keep")
        with self.assertRaisesRegex(FileExistsError, "DPUY1.TGZ"):
            self.preflight(stager)
        self.assertEqual((self.sd / "DPUY1.TGZ").read_bytes(), b"keep")
        (self.sd / "DPUY1.TGZ").unlink()
        Usage = namedtuple("Usage", "total used free")
        with mock.patch.object(stager.shutil, "disk_usage", return_value=Usage(10, 9, 1)):
            with self.assertRaisesRegex(OSError, "insufficient free space"):
                self.preflight(stager)

    def test_rejects_partial_or_wrong_source_manifest(self):
        stager = load_stager()
        (self.source / "DPUY1.SHA").write_text("", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "manifest entries"):
            self.preflight(stager)
        (self.source / "DPUY1.SHA").write_text(f"{'0' * 64}  DPUY1.TGZ\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "prepared artifact hash mismatch"):
            self.preflight(stager)

    def test_dry_run_writes_nothing_and_copy_is_create_only(self):
        stager = load_stager()
        plan = self.preflight(stager)
        report = stager.stage(plan, dry_run=True)
        self.assertTrue(report["dry_run"])
        self.assertTrue(all(not (self.sd / name).exists() for name in DESTINATIONS))
        (self.sd / "DPUY1.SHA").write_bytes(b"race")
        with self.assertRaisesRegex(FileExistsError, "DPUY1.SHA"):
            stager.stage(plan)
        self.assertFalse((self.sd / "DPUY1.TGZ").exists())
        self.assertEqual((self.sd / "DPUY1.SHA").read_bytes(), b"race")

    def test_readback_mismatch_stops_without_cleanup(self):
        stager = load_stager()
        plan = self.preflight(stager)
        real_hash = stager.file_sha256

        def mismatching_hash(path):
            value = real_hash(path)
            if Path(path) == self.sd / "DPUY1.TGZ":
                return "f" * 64
            return value

        with mock.patch.object(stager, "file_sha256", side_effect=mismatching_hash):
            with self.assertRaisesRegex(OSError, "read-back verification failed"):
                stager.stage(plan)
        self.assertTrue((self.sd / "DPUY1.TGZ").exists())
        self.assertFalse((self.sd / "DPUY1.SHA").exists())

    def test_success_preserves_originals_and_reports_exact_readback(self):
        stager = load_stager()
        plan = self.preflight(stager)
        report = stager.stage(plan)
        self.assertEqual(set(report["files"]), set(DESTINATIONS))
        for name in DESTINATIONS:
            self.assertEqual(report["files"][name]["source_sha256"], sha256(self.source / name))
            self.assertEqual(report["files"][name]["readback_sha256"], sha256(self.sd / name))
        for name, digest in self.boot_hashes.items():
            self.assertEqual(sha256(self.sd / name), digest)


if __name__ == "__main__":
    unittest.main()
