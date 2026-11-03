"""One-use flush diagnosis on the pinned SD. No WriteFile/write_at calls.

Requires administrator rights. It locks/dismounts the known boot volume and
compares flush results before/after dismount. Never performs a repair/retry.
"""
import ctypes
from ctypes import wintypes
import datetime
import json
from pathlib import Path
import traceback

from repair_sd_boot_two_sectors import (
    WindowsVolume, read_identity, validate_sector_geometry, read_all,
    IMAGE_BYTES, sha, save_exclusive,
)


def main():
    output = Path("E:/Soundjie_Work/d4/artifacts/sd-flush-diagnostic-20261024")
    output.mkdir(exist_ok=False)
    report = {"data_write_calls": 0, "flush_results": []}
    volume = None
    try:
        report["identity"] = read_identity()
        report["geometry"] = validate_sector_geometry()
        volume = WindowsVolume()
        volume.validate_extent()

        def test_flush(stage):
            try:
                # Probe the native behavior directly, not the repaired wrapper's
                # intentional guard against post-dismount filesystem flush.
                if not volume.k.FlushFileBuffers(volume.handle):
                    raise ctypes.WinError(ctypes.get_last_error())
                report["flush_results"].append({"stage": stage, "success": True})
            except OSError as error:
                report["flush_results"].append({
                    "stage": stage, "success": False,
                    "winerror": error.winerror, "message": str(error),
                })

        test_flush("opened_before_lock")
        volume.lock()
        test_flush("locked_before_dismount")
        volume.dismount()
        test_flush("locked_after_dismount")
        # Capture the underlying NTSTATUS to distinguish a dismounted filesystem
        # from a physical reader/card not-ready condition. No sector write here.
        class StatusUnion(ctypes.Union):
            _fields_ = [("Status", wintypes.LONG), ("Pointer", ctypes.c_void_p)]
        class IOStatus(ctypes.Structure):
            _fields_ = [("u", StatusUnion), ("Information", ctypes.c_size_t)]
        ntdll = ctypes.WinDLL("ntdll")
        ntdll.NtFlushBuffersFile.argtypes = [wintypes.HANDLE, ctypes.POINTER(IOStatus)]
        ntdll.NtFlushBuffersFile.restype = wintypes.LONG
        iosb = IOStatus()
        status = ntdll.NtFlushBuffersFile(volume.handle, ctypes.byref(iosb))
        report["after_dismount_ntstatus"] = "0x{:08x}".format(status & 0xffffffff)
        data = read_all(volume, IMAGE_BYTES)
        report["readback_sha256"] = sha(data)
        if sha(data) != "795299cc2ff145d116ceca3121e508b84e7bfacbdd0e297fedc1f06fc3135cea":
            raise ValueError("unexpected SD content change during flush-only diagnosis")
        report["data_unchanged"] = True
        report["status"] = "DIAGNOSTIC_COMPLETE_NO_SECTOR_WRITES"
    except Exception:
        report["status"] = "DIAGNOSTIC_FAILED"
        report["error"] = traceback.format_exc()
    finally:
        if volume is not None:
            try:
                volume.close()
            except Exception:
                report["status"] = "CLOSE_FAILED"
                report["close_error"] = traceback.format_exc()
        report["completed_at"] = datetime.datetime.now().astimezone().isoformat()
        save_exclusive(output / "result.json", json.dumps(report, indent=2).encode("utf-8"))
    return 0 if report["status"] == "DIAGNOSTIC_COMPLETE_NO_SECTOR_WRITES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
