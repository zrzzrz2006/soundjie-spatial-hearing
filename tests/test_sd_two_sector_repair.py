import importlib.util
import io
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/host/repair_sd_boot_two_sectors.py"


def load_repair():
    if not SCRIPT.is_file():
        raise AssertionError("guarded two-sector repair has not been implemented")
    spec = importlib.util.spec_from_file_location("sd_two_sector_repair", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(SCRIPT.parent)] + sys.path):
        spec.loader.exec_module(module)
    return module


class FileVolume:
    """In-memory I/O boundary; never opens a Windows volume."""
    def __init__(self, data, fail=None):
        self.stream = io.BytesIO(data)
        self.fail = fail
        self.events = []
        self.writes = []

    def validate_extent(self):
        self.events.append("extent")
        if self.fail == "extent":
            raise OSError("wrong extent")

    def lock(self):
        self.events.append("lock")
        if self.fail == "lock":
            raise OSError("lock refused")

    def dismount(self):
        self.events.append("dismount")
        if self.fail == "dismount":
            raise OSError("dismount refused")

    def read_at(self, offset, size):
        self.events.append("read")
        self.stream.seek(offset)
        data = self.stream.read(size)
        if self.fail == "short_read":
            return data[:-1]
        if self.fail == "readback" and self.writes:
            return b"!" + data[1:]
        return data

    def write_at(self, offset, data):
        self.events.append("write")
        self.writes.append((offset, data))
        self.stream.seek(offset)
        if self.fail == "partial_write":
            return self.stream.write(data[:256])
        return self.stream.write(data)

    def flush(self):
        self.events.append("flush")
        if self.fail == "flush" or (self.fail == "post_flush" and self.writes):
            raise OSError("flush failed")

    def close(self):
        self.events.append("close")


class SdTwoSectorRepairTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_repair()
        self.before = b"A" * 512 + b"B" * 512 + b"C" * 512 + b"D" * 512
        self.after = b"X" * 512 + b"B" * 512 + b"Y" * 512 + b"D" * 512
        self.offsets = (0, 1024)

    def run_patch(self, volume, before=None, after=None, audit=None):
        return self.mod.execute_patch(
            volume, self.before if before is None else before,
            self.after if after is None else after, self.offsets,
            audit if audit is not None else lambda event, **fields: None,
        )

    def test_writes_exact_two_sectors_only_after_lock_and_complete_compare(self):
        volume = FileVolume(self.before)
        result = self.run_patch(volume)
        self.assertEqual(result, self.after)
        self.assertEqual(volume.stream.getvalue(), self.after)
        self.assertEqual(volume.writes, [(0, b"X" * 512), (1024, b"Y" * 512)])
        self.assertEqual(volume.events[:4], ["extent", "lock", "flush", "read"])
        self.assertEqual(volume.events[-1], "close")
        self.assertIn("flush", volume.events)
        self.assertGreater(volume.events.index("dismount"),
                           max(i for i, e in enumerate(volume.events) if e == "flush"))

    def test_changed_source_prevents_every_write(self):
        volume = FileVolume(b"Z" + self.before[1:])
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            self.run_patch(volume)
        self.assertFalse(volume.writes)
        self.assertEqual(volume.events[-1], "close")

    def test_short_source_prevents_every_write(self):
        volume = FileVolume(self.before, "short_read")
        with self.assertRaises(OSError):
            self.run_patch(volume)
        self.assertFalse(volume.writes)

    def test_extent_lock_or_preflight_flush_failure_never_writes(self):
        for fail in ("extent", "lock", "flush"):
            with self.subTest(fail=fail):
                volume = FileVolume(self.before, fail)
                with self.assertRaises(OSError):
                    self.run_patch(volume)
                self.assertFalse(volume.writes)
                self.assertEqual(volume.events[-1], "close")
                if fail == "lock":
                    self.assertNotIn("dismount", volume.events)

    def test_difference_outside_allowlist_rejected_before_lock(self):
        volume = FileVolume(self.before)
        with self.assertRaises(ValueError):
            self.run_patch(volume, after=self.after[:512] + b"!" + self.after[513:])
        self.assertFalse(volume.writes)
        self.assertNotIn("lock", volume.events)

    def test_malformed_offsets_and_image_lengths_are_rejected(self):
        for offsets in ((0, 0), (1, 1024), (-512, 1024), (0, 4096), (0,)):
            with self.subTest(offsets=offsets):
                with self.assertRaises(ValueError):
                    self.mod.validate_diff(self.before, self.after, offsets)
        with self.assertRaises(ValueError):
            self.mod.validate_diff(self.before, self.after[:-1], self.offsets)

    def test_partial_write_stops_without_second_write_or_automatic_rollback(self):
        volume = FileVolume(self.before, "partial_write")
        with self.assertRaisesRegex(OSError, "short write"):
            self.run_patch(volume)
        self.assertEqual(len(volume.writes), 1)
        self.assertEqual(volume.events[-1], "close")

    def test_readback_or_flush_failure_stops_before_second_sector(self):
        for fail in ("readback", "post_flush"):
            with self.subTest(fail=fail):
                volume = FileVolume(self.before, fail)
                with self.assertRaises(OSError):
                    self.run_patch(volume)
                self.assertEqual(len(volume.writes), 1)

    def test_dismount_failure_after_verified_writes_cannot_report_success(self):
        volume = FileVolume(self.before, "dismount")
        with self.assertRaises(OSError):
            self.run_patch(volume)
        self.assertEqual(len(volume.writes), 2)
        self.assertEqual(volume.events[-1], "close")

    def test_real_dismounted_volume_flush_error_is_avoided_not_ignored(self):
        class DismountAwareVolume(FileVolume):
            def flush(self):
                if "dismount" in self.events:
                    raise OSError("STATUS_VOLUME_DISMOUNTED")
                return super().flush()
        volume = DismountAwareVolume(self.before)
        self.assertEqual(self.run_patch(volume), self.after)
        self.assertIn("dismount", volume.events)

    def test_explicit_remaining_sector_plan_writes_only_remaining_sector(self):
        partial = self.after[:512] + self.before[512:]
        volume = FileVolume(partial)
        result = self.mod.execute_patch(volume, partial, self.after, (1024,),
                                        lambda event, **fields: None)
        self.assertEqual(result, self.after)
        self.assertEqual(volume.writes, [(1024, b"Y" * 512)])

    def test_unexpected_change_during_dismount_cannot_pass(self):
        class MutatingDismount(FileVolume):
            def dismount(self):
                super().dismount()
                if self.writes:
                    self.stream.seek(512)
                    self.stream.write(b"!")
        with self.assertRaises(OSError):
            self.run_patch(MutatingDismount(self.before))

    def test_audit_failure_before_write_does_not_write(self):
        volume = FileVolume(self.before)
        def audit(event, **fields):
            if event == "write_attempt":
                raise OSError("evidence storage failed")
        with self.assertRaises(OSError):
            self.run_patch(volume, audit=audit)
        self.assertFalse(volume.writes)

    def test_extent_decoder_rejects_other_disk_or_other_partition(self):
        valid = struct.pack("<I4xI4xqq", 1, 2, 1048576, 103809024)
        self.mod.validate_extent_bytes(valid)
        for raw in (
            valid[:24], struct.pack("<I4xI4xqq", 2, 2, 1048576, 103809024),
            struct.pack("<I4xI4xqq", 1, 1, 1048576, 103809024),
            struct.pack("<I4xI4xqq", 1, 2, 104857600, 15623782400),
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    self.mod.validate_extent_bytes(raw)

    def test_main_without_apply_flag_never_queries_or_opens_volume(self):
        with patch.object(sys, "argv", [str(SCRIPT), "--run-name", "sd-boot-repair-test"]):
            with patch.object(self.mod, "read_identity") as identity:
                with patch.object(self.mod, "WindowsVolume") as volume:
                    with self.assertRaises(SystemExit) as context:
                        self.mod.main()
                    self.assertEqual(context.exception.code, 2)
                    identity.assert_not_called()
                    volume.assert_not_called()


if __name__ == "__main__":
    unittest.main()
