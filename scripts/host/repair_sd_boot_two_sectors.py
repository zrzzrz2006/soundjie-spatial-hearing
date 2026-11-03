"""One-use, explicitly authorized repair of two known MZ7100 boot directory sectors.

Never formats, writes a disk image, changes partitions, or repairs Linux p2.
Artifacts and identity are pinned; a mismatch stops before any sector write.
Failures after a write attempt are NOT automatically retried or rolled back.
"""

import argparse
import ctypes
from ctypes import wintypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import traceback

from backup_sd_boot_readonly import read_identity


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/sd-boot-readonly-20261023-235618/boot-partition.img"
CANDIDATE = ROOT / "artifacts/sd-boot-trial-20261024a/boot-clean-candidate.img"
SOURCE_SHA = "7f0abdd710c30ae8cfb067baf6fbf00caf362e5d6b6e26ff98b8ede4a0e002be"
CANDIDATE_SHA = "4d7bb050c6a8402f2f50aafcfde353ae38b5cb2ef62e5cd62b6a6b55172e52d1"
PARTIAL_SOURCE = ROOT / "artifacts/sd-boot-readonly-20261024-after-flush-error/boot-partition.img"
PARTIAL_SHA = "795299cc2ff145d116ceca3121e508b84e7bfacbdd0e297fedc1f06fc3135cea"
VOLUME = r"\\?\Volume{9b6c8f5a-bf04-11f1-9fa3-005056c00008}"
IMAGE_BYTES = 103809024
OFFSETS = (4760576, 31046144)
SECTOR = 512
CHUNK = 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_diff(before, after, offsets):
    if not before or len(before) != len(after) or len(before) % SECTOR:
        raise ValueError("image lengths must match and be sector aligned")
    if (len(offsets) not in (1, 2) or len(set(offsets)) != len(offsets) or
            any(type(o) is not int or o < 0 or o % SECTOR or
                o + SECTOR > len(before) for o in offsets)):
        raise ValueError("one or two distinct, valid sector offsets required")
    changed = {
        o for o in range(0, len(before), SECTOR)
        if before[o:o + SECTOR] != after[o:o + SECTOR]
    }
    if changed != set(offsets):
        raise ValueError("image differences do not exactly match the allowed sectors")


def validate_extent_bytes(raw):
    # Windows VOLUME_DISK_EXTENTS: DWORD count; pad; DISK_EXTENT (DWORD, pad, q, q).
    if len(raw) < 32:
        raise ValueError("short volume extent response")
    if struct.unpack_from("<I4xI4xqq", raw) != (1, 2, 1048576, IMAGE_BYTES):
        raise ValueError("volume is not the exact authorized Disk 2 boot partition")


def read_all(volume, length):
    blocks = []
    for offset in range(0, length, CHUNK):
        size = min(CHUNK, length - offset)
        block = volume.read_at(offset, size)
        if len(block) != size:
            raise OSError("short read at {}".format(offset))
        blocks.append(block)
    return b"".join(blocks)


def execute_patch(volume, before, after, offsets, audit):
    try:
        validate_diff(before, after, offsets)
        volume.validate_extent()
        volume.lock()
        # A locked volume permits direct I/O and flushes successfully. On this
        # FAT driver, flushing an already dismounted volume returns 0xc000026e.
        # Test the real flush path BEFORE writing, and keep the lock throughout.
        volume.flush()
        current = read_all(volume, len(before))
        if current != before:
            raise ValueError("source mismatch: live SD is not the preserved original")
        audit("before_verified", bytes=len(current), sha256=sha(current))
        for offset in offsets:
            expected = after[offset:offset + SECTOR]
            old = before[offset:offset + SECTOR]
            if volume.read_at(offset, SECTOR) != old:
                raise ValueError("source mismatch before sector write")
            # Persist intent BEFORE the irreversible operation. A logging error aborts.
            audit("write_attempt", partition_offset=offset, bytes=SECTOR,
                  before_sha256=sha(old), after_sha256=sha(expected))
            written = volume.write_at(offset, expected)
            if written != SECTOR:
                raise OSError("short write at {}: {}".format(offset, written))
            volume.flush()
            if volume.read_at(offset, SECTOR) != expected:
                raise OSError("sector readback mismatch at {}".format(offset))
            audit("sector_verified", partition_offset=offset, sha256=sha(expected))
        volume.flush()
        final = read_all(volume, len(after))
        if final != after:
            raise OSError("full partition readback mismatch")
        volume.dismount()
        # Invalidate filesystem state only after writes/flushes; retain the lock
        # and check that dismount did not restore any stale cached metadata.
        final = read_all(volume, len(after))
        if final != after:
            raise OSError("full partition mismatch after dismount")
        audit("full_verified", bytes=len(final), sha256=sha(final))
        return final
    finally:
        volume.close()


class WindowsVolume:
    """Synchronous, aligned, unbuffered I/O, restricted to a single pinned volume."""

    def __init__(self):
        self.handle = None
        self.buffer = None
        self.locked = False
        self.dismounted = False
        self.extent_valid = False
        self.k = ctypes.WinDLL("kernel32", use_last_error=True)
        definitions = {
            "CreateFileW": ([wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                              ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                              wintypes.HANDLE], wintypes.HANDLE),
            "DeviceIoControl": ([wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p,
                                  wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                  ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p], wintypes.BOOL),
            "SetFilePointerEx": ([wintypes.HANDLE, ctypes.c_longlong,
                                   ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD], wintypes.BOOL),
            "ReadFile": ([wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                           ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p], wintypes.BOOL),
            "WriteFile": ([wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                            ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p], wintypes.BOOL),
            "FlushFileBuffers": ([wintypes.HANDLE], wintypes.BOOL),
            "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
            "VirtualAlloc": ([ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD,
                               wintypes.DWORD], ctypes.c_void_p),
            "VirtualFree": ([ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD], wintypes.BOOL),
        }
        for name, (args, result) in definitions.items():
            function = getattr(self.k, name)
            function.argtypes, function.restype = args, result
        try:
            handle = self.k.CreateFileW(VOLUME, 0xC0000000, 3, None, 3, 0xA0000000, None)
            if handle == ctypes.c_void_p(-1).value or handle is None:
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = handle
            self.buffer = self.k.VirtualAlloc(None, CHUNK, 0x3000, 4)
            if not self.buffer:
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.close()
            raise

    def ioctl(self, code, output=None):
        returned = wintypes.DWORD()
        if not self.k.DeviceIoControl(self.handle, code, None, 0, output,
                                      ctypes.sizeof(output) if output is not None else 0,
                                      ctypes.byref(returned), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return returned.value

    def validate_extent(self):
        output = ctypes.create_string_buffer(64)
        count = self.ioctl(0x00560000, output)
        validate_extent_bytes(output.raw[:count])
        self.extent_valid = True

    def lock(self):
        if not self.extent_valid:
            raise ValueError("extent not validated")
        self.ioctl(0x00090018)  # FSCTL_LOCK_VOLUME; never force or retry on refusal.
        self.locked = True
        # Read the 18 padding sectors past FAT's BPB size but INSIDE p1.
        # Windows still enforces partition bounds; our own limits are stricter.
        self.ioctl(0x00090083)  # FSCTL_ALLOW_EXTENDED_DASD_IO

    def dismount(self):
        if not self.locked:
            raise ValueError("refusing dismount without exclusive volume lock")
        self.ioctl(0x00090020)  # FSCTL_DISMOUNT_VOLUME
        self.dismounted = True

    def position(self, offset, size):
        if not (self.locked and self.extent_valid):
            raise ValueError("volume is not locked and extent-validated")
        if (type(offset) is not int or type(size) is not int or offset < 0 or
                not 0 < size <= CHUNK or offset % SECTOR or size % SECTOR or
                offset + size > IMAGE_BYTES):
            raise ValueError("I/O outside boot partition or unaligned")
        actual = ctypes.c_longlong()
        if not self.k.SetFilePointerEx(self.handle, offset, ctypes.byref(actual), 0):
            raise ctypes.WinError(ctypes.get_last_error())
        if actual.value != offset:
            raise OSError("file pointer mismatch")

    def read_at(self, offset, size):
        self.position(offset, size)
        count = wintypes.DWORD()
        if not self.k.ReadFile(self.handle, self.buffer, size, ctypes.byref(count), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if count.value != size:
            raise OSError("short native read")
        return ctypes.string_at(self.buffer, size)

    def write_at(self, offset, data):
        if self.dismounted:
            raise ValueError("refusing write after dismount: flush would not be available")
        if offset not in OFFSETS or len(data) != SECTOR:
            raise ValueError("native write is not one of the two authorized sectors")
        self.position(offset, len(data))
        ctypes.memmove(self.buffer, data, len(data))
        count = wintypes.DWORD()
        if not self.k.WriteFile(self.handle, self.buffer, len(data), ctypes.byref(count), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return count.value

    def flush(self):
        if self.dismounted:
            raise ValueError("refusing filesystem flush after dismount")
        if not self.k.FlushFileBuffers(self.handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        errors = []
        if self.handle is not None:
            if not self.k.CloseHandle(self.handle):
                errors.append(ctypes.WinError(ctypes.get_last_error()))
            self.handle = None
        if self.buffer:
            if not self.k.VirtualFree(self.buffer, 0, 0x8000):
                errors.append(ctypes.WinError(ctypes.get_last_error()))
            self.buffer = None
        self.locked = self.dismounted = self.extent_valid = False
        if errors:
            raise errors[0]


def validate_sector_geometry():
    command = ("$ErrorActionPreference='Stop'; Get-Disk -Number 2 | "
               "Select-Object LogicalSectorSize,PhysicalSectorSize | ConvertTo-Json -Compress")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    record = json.loads(result.stdout.decode("utf-8-sig"))
    if record != {"LogicalSectorSize": 512, "PhysicalSectorSize": 512}:
        raise ValueError("unexpected SD sector geometry: {!r}".format(record))
    return record


def save_exclusive(path, data):
    with path.open("xb") as out:
        if out.write(data) != len(data):
            raise OSError("short evidence write")
        out.flush()
        os.fsync(out.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--apply-confirmed", action="store_true")
    parser.add_argument("--resume-second", action="store_true",
                        help="accept ONLY pinned first-sector-applied state; write only second sector")
    args = parser.parse_args()
    if not args.apply_confirmed:
        parser.error("explicit --apply-confirmed is required")
    if (not args.run_name.startswith("sd-boot-repair-") or
            any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.run_name)):
        parser.error("invalid repair run name")
    if ROOT.resolve() != Path("E:/Soundjie_Work/d4").resolve():
        parser.error("refusing relocated repair script")
    output = ROOT / "artifacts" / args.run_name
    output.mkdir(exist_ok=False)
    attempts = []
    source_path = PARTIAL_SOURCE if args.resume_second else SOURCE
    source_sha = PARTIAL_SHA if args.resume_second else SOURCE_SHA
    offsets = (OFFSETS[1],) if args.resume_second else OFFSETS
    report = {
        "status": "NOT_WRITTEN", "volume": VOLUME,
        "source_sha256": source_sha, "candidate_sha256": CANDIDATE_SHA,
        "mode": "remaining_second_sector" if args.resume_second else "both_sectors",
        "authorized_scope": "User confirmed exactly two directory sectors, 2026-10-24",
        "write_attempts": attempts, "whole_disk_restore": False,
        "linux_partition_written": False,
    }
    with (output / "audit.jsonl").open("x", encoding="utf-8") as log:
        def audit(event, **fields):
            row = dict(event=event, time=datetime.datetime.now().astimezone().isoformat(), **fields)
            log.write(json.dumps(row) + "\n")
            log.flush()
            os.fsync(log.fileno())
            if event == "write_attempt":
                attempts.append(fields["partition_offset"])

        try:
            if shutil.disk_usage(output).free < 250 * 1024 * 1024:
                raise OSError("insufficient E drive evidence space")
            before, after = source_path.read_bytes(), CANDIDATE.read_bytes()
            if len(before) != IMAGE_BYTES or sha(before) != source_sha or sha(after) != CANDIDATE_SHA:
                raise ValueError("pinned source/candidate artifact mismatch")
            validate_diff(before, after, offsets)
            report["identity_before"] = read_identity()
            report["geometry"] = validate_sector_geometry()
            audit("preflight_verified", identity=report["identity_before"], geometry=report["geometry"])
            final = execute_patch(WindowsVolume(), before, after, offsets, audit)
            report["status"] = "SD_FULL_READBACK_VERIFIED_EVIDENCE_PENDING"
            capture = output / "boot-readback.img"
            save_exclusive(capture, final)
            if sha(capture.read_bytes()) != CANDIDATE_SHA:
                raise OSError("saved readback image verification failed")
            report["readback_path"] = str(capture)
            report["readback_sha256"] = sha(final)
            report["readback_bytes"] = len(final)
            report["identity_after"] = read_identity()
            report["status"] = "VERIFIED_TWO_SECTOR_REPAIR"
            audit("complete", sha256=sha(final), bytes=len(final))
        except Exception:
            report["failure_stage_status"] = report["status"]
            report["status"] = "FAILED_POSSIBLE_PARTIAL_WRITE" if attempts else "FAILED_BEFORE_SECTOR_WRITE"
            report["error"] = traceback.format_exc()
        report["completed_at"] = datetime.datetime.now().astimezone().isoformat()
        save_exclusive(output / "result.json", json.dumps(report, indent=2).encode("utf-8"))
    return 0 if report["status"] == "VERIFIED_TWO_SECTOR_REPAIR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
