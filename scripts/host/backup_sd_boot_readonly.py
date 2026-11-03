"""Preserve ONLY the known MZ7100 SD prefix/boot partition; never write SD.

Run elevated on Windows. No mount, offline, repair, format or eject operation.
Creates a new evidence directory under this repository's artifacts directory.
An unsuccessful capture is retained, never silently deleted or reused.
"""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback


DEVICE = r"\\.\PhysicalDrive2"
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def validate_identity(record):
    expected = {
        "Number": 2, "SerialNumber": "121220160204", "Size": 15728640000,
        "BusType": "USB", "IsBoot": False, "IsSystem": False,
        "IsOffline": False,
    }
    for key, value in expected.items():
        actual = record.get(key)
        if isinstance(value, bool):
            matches = actual is value
        else:
            matches = actual == value
        if not matches:
            raise ValueError("SD identity mismatch: {}={!r}".format(key, actual))
    parts = sorted(record.get("Partitions", []), key=lambda p: p["PartitionNumber"])
    layout = [(p["PartitionNumber"], p["Offset"], p["Size"]) for p in parts]
    if layout != [(1, 1048576, 103809024), (2, 104857600, 15623782400)]:
        raise ValueError("SD partition layout mismatch: {!r}".format(layout))


def read_identity():
    # Select only the exact disk number, then fail closed on identity/layout.
    command = r"""
    $ErrorActionPreference='Stop'
    $d=Get-Disk -Number 2
    $p=@(Get-Partition -DiskNumber 2 | Select-Object PartitionNumber,Offset,Size)
    [pscustomobject]@{
      Number=[int]$d.Number; SerialNumber=$d.SerialNumber.Trim();
      Size=[long]$d.Size; BusType=[string]$d.BusType;
      IsBoot=[bool]$d.IsBoot; IsSystem=[bool]$d.IsSystem;
      IsOffline=[bool]$d.IsOffline; Partitions=$p
    } | ConvertTo-Json -Depth 4 -Compress
    """
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=60,
    )
    record = json.loads(result.stdout.decode("utf-8-sig"))
    validate_identity(record)
    return record


def backup_range(source, destination, offset, length, chunk_size=1024 * 1024):
    if offset < 0 or length <= 0 or chunk_size <= 0:
        raise ValueError("invalid backup range or chunk size")
    digest = hashlib.sha256()
    source.seek(offset)
    # Exclusive creation protects both previous evidence and existing files.
    with Path(destination).open("xb") as target:
        remaining = length
        while remaining:
            block = source.read(min(chunk_size, remaining))
            if not block:
                raise OSError("short read while preserving SD range")
            target.write(block)
            digest.update(block)
            remaining -= len(block)
        target.flush()
        os.fsync(target.fileno())
    image_digest = hashlib.sha256()
    with Path(destination).open("rb") as target:
        while True:
            block = target.read(chunk_size)
            if not block:
                break
            image_digest.update(block)
    if Path(destination).stat().st_size != length or image_digest.digest() != digest.digest():
        raise OSError("image readback mismatch")
    repeat = hashlib.sha256()
    source.seek(offset)
    remaining = length
    while remaining:
        block = source.read(min(chunk_size, remaining))
        if not block:
            raise OSError("short read during source repeat")
        repeat.update(block)
        remaining -= len(block)
    if repeat.digest() != digest.digest():
        raise OSError("source changed between read passes")
    return {
        "offset": offset, "bytes": length, "path": str(destination),
        "source_sha256": digest.hexdigest(),
        "image_sha256": image_digest.hexdigest(),
        "source_repeat_sha256": repeat.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-name", required=True)
    args = parser.parse_args()
    name = args.capture_name
    if not name.startswith("sd-boot-readonly-") or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name):
        parser.error("capture name must be sd-boot-readonly- plus lowercase letters/digits/hyphens")
    # Do not accidentally run a relocated copy against another output drive.
    if ARTIFACTS.resolve() != Path("E:/Soundjie_Work/d4/artifacts").resolve():
        parser.error("expected E:/Soundjie_Work/d4/artifacts output root")
    output = ARTIFACTS / name
    output.mkdir(exist_ok=False)
    with (output / "capture.log").open("x", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                before = read_identity()
                print("Validated card:", json.dumps(before), flush=True)
                if shutil.disk_usage(output).free < 250 * 1024 * 1024:
                    raise OSError("insufficient E drive free space")
                captures = []
                # rb requests read access only. Buffer size/read lengths aligned to 512 bytes.
                with open(DEVICE, "rb", buffering=0) as source:
                    for filename, offset, size in (
                        ("disk-prefix.img", 0, 1048576),
                        ("boot-partition.img", 1048576, 103809024),
                    ):
                        print("Reading", filename, offset, size, flush=True)
                        captures.append(backup_range(source, output / filename, offset, size))
                        print("Verified", json.dumps(captures[-1]), flush=True)
                after = read_identity()
                report = {
                    "status": "VERIFIED_READ_ONLY_CAPTURE", "device": DEVICE,
                    "completed_at": datetime.datetime.now().astimezone().isoformat(),
                    "identity_before": before, "identity_after": after,
                    "captures": captures,
                    "scope": "Prefix and FAT boot partition only; NOT full SD/root filesystem backup",
                    "sd_operations": "read only; no repair/offline/mount/format/eject",
                }
                with (output / "verified.json").open("x", encoding="utf-8") as out:
                    json.dump(report, out, indent=2)
                print("VERIFIED_READ_ONLY_CAPTURE", flush=True)
                return 0
            except Exception:
                traceback.print_exc()
                print("FAILED: partial evidence retained; no SD writes requested", flush=True)
                return 1


if __name__ == "__main__":
    raise SystemExit(main())
