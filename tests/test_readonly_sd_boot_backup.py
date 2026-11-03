import importlib.util
import io
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/host/backup_sd_boot_readonly.py"


def load_backup():
    if not SCRIPT.is_file():
        raise AssertionError("read-only boot backup has not been implemented")
    spec = importlib.util.spec_from_file_location("readonly_sd_backup", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReadOnlySdBackupTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_backup()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.img"
        self.source.write_bytes(b"0123456789abcdef")
        self.dest = self.root / "boot.img"

    def identity(self):
        return {
            "Number": 2, "SerialNumber": "121220160204",
            "Size": 15728640000, "BusType": "USB", "IsBoot": False,
            "IsSystem": False, "IsOffline": False,
            "Partitions": [
                {"PartitionNumber": 1, "Offset": 1048576, "Size": 103809024},
                {"PartitionNumber": 2, "Offset": 104857600, "Size": 15623782400},
            ],
        }

    def test_copies_only_selected_bytes_and_verifies_both_reads(self):
        with self.source.open("rb", buffering=0) as stream:
            result = self.mod.backup_range(stream, self.dest, 10, 6, chunk_size=2)
        self.assertEqual(self.dest.read_bytes(), b"abcdef")
        self.assertEqual(self.source.read_bytes(), b"0123456789abcdef")
        self.assertEqual(result["bytes"], 6)
        for name in ("source_sha256", "image_sha256", "source_repeat_sha256"):
            self.assertEqual(result[name], "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721")

    def test_existing_backup_is_never_overwritten(self):
        self.dest.write_bytes(b"keep")
        with self.source.open("rb", buffering=0) as stream:
            with self.assertRaises(FileExistsError):
                self.mod.backup_range(stream, self.dest, 0, 8)
        self.assertEqual(self.dest.read_bytes(), b"keep")

    def test_invalid_ranges_do_not_create_output(self):
        for offset, size, chunk in ((-1, 3, 2), (0, 0, 2), (0, -2, 2), (0, 3, 0)):
            with self.subTest(offset=offset, size=size, chunk=chunk):
                with self.source.open("rb", buffering=0) as stream:
                    with self.assertRaises(ValueError):
                        self.mod.backup_range(stream, self.dest, offset, size, chunk)
                self.assertFalse(self.dest.exists())

    def test_short_read_fails_and_preserves_partial_evidence(self):
        with self.source.open("rb", buffering=0) as stream:
            with self.assertRaisesRegex(OSError, "short read"):
                self.mod.backup_range(stream, self.dest, 12, 8, chunk_size=2)
        self.assertEqual(self.dest.read_bytes(), b"cdef")

    def test_source_changing_between_passes_is_rejected(self):
        class ChangingSource(io.BytesIO):
            passes = 0

            def seek(self, offset, whence=0):
                if offset == 0 and whence == 0:
                    self.passes += 1
                    if self.passes == 2:
                        super().seek(0)
                        self.write(b"Z")
                return super().seek(offset, whence)

        with ChangingSource(b"abcdef") as stream:
            with self.assertRaisesRegex(OSError, "source changed"):
                self.mod.backup_range(stream, self.dest, 0, 6)
        self.assertEqual(self.dest.read_bytes(), b"abcdef")

    def test_identity_accepts_only_exact_non_system_card_layout(self):
        self.mod.validate_identity(self.identity())
        for field, wrong in (("Number", 1), ("SerialNumber", "other"),
                             ("Size", 100), ("BusType", "SATA"),
                             ("IsBoot", True), ("IsSystem", True),
                             ("IsOffline", True), ("IsSystem", None)):
            with self.subTest(field=field):
                record = self.identity()
                record[field] = wrong
                with self.assertRaises(ValueError):
                    self.mod.validate_identity(record)
        for field in ("Offset", "Size", "PartitionNumber"):
            with self.subTest(partition_field=field):
                record = self.identity()
                record["Partitions"][0][field] += 1
                with self.assertRaises(ValueError):
                    self.mod.validate_identity(record)
        record = self.identity()
        record["Partitions"].pop()
        with self.assertRaises(ValueError):
            self.mod.validate_identity(record)


if __name__ == "__main__":
    unittest.main()
