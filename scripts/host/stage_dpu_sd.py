#!/usr/bin/env python3
import argparse
import ctypes
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import NamedTuple


BOOT_NAMES = ("BOOT.bin", "uImage", "devicetree.dtb", "uEnv.txt")
DPU_NAMES = ("DPUV1.BIT", "DPUV1.DTB", "DPUV1.TGZ", "DPUV1.SHA")
ORIGINAL_BOOT_HASHES = {
    "BOOT.bin": "992f696208b3cefdecb091dd0f1fb1053e26505199c38e841685d61d8cd53983",
    "uImage": "ed95081d1cbe16d9b35dcdd5b0e1b0db4063b384c889f138dc5a037193677398",
    "devicetree.dtb": "84a613762b49e32d07223e425f1c2526c872ae832a98986bd814a082286bc701",
    "uEnv.txt": "900ee46eb0f86c3c90f4f40b0d5c2d05e46df1e06be1f483192084cfcde6bbb3",
}
FREE_SPACE_MARGIN = 1024 * 1024


class CopyRecord(NamedTuple):
    name: str
    source: Path
    destination: Path
    size: int
    sha256: str


class StagePlan(NamedTuple):
    sd_root: Path
    source_dir: Path
    volume_serial: str
    boot_hashes: dict[str, str]
    files: tuple[CopyRecord, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_volume_serial(root: Path) -> str:
    root = Path(root).resolve()
    anchor = root.anchor
    if not anchor:
        raise OSError(f"cannot determine volume root for {root}")
    serial = ctypes.c_uint32()
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(anchor),
        None,
        0,
        ctypes.byref(serial),
        None,
        None,
        None,
        0,
    )
    if not ok:
        raise ctypes.WinError()
    return f"{serial.value >> 16:04X}-{serial.value & 0xFFFF:04X}"


def parse_payload_manifest(path: Path) -> dict[str, str]:
    entries = {}
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError("invalid DPUV1.SHA line")
        digest, name = parts
        if name not in DPU_NAMES[:3] or name in entries:
            raise ValueError(f"unexpected DPUV1.SHA entry: {name}")
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError(f"invalid SHA-256 in DPUV1.SHA: {name}")
        entries[name] = digest.lower()
    if tuple(entries) != DPU_NAMES[:3]:
        raise ValueError("DPUV1.SHA entries or order do not match expected payloads")
    return entries


def preflight(
    sd_root: Path,
    source_dir: Path,
    *,
    expected_volume_serial: str = "54D7-D258",
    expected_boot_hashes: dict[str, str] = ORIGINAL_BOOT_HASHES,
) -> StagePlan:
    sd_root = Path(sd_root).resolve()
    source_dir = Path(source_dir).resolve()
    if not sd_root.is_dir():
        raise ValueError(f"SD root is not a directory: {sd_root}")
    if not source_dir.is_dir():
        raise ValueError(f"source directory is not a directory: {source_dir}")

    serial = read_volume_serial(sd_root).upper()
    if serial != expected_volume_serial.upper():
        raise ValueError(
            f"volume serial mismatch: expected {expected_volume_serial}, got {serial}"
        )

    actual_boot_hashes = {}
    for name in BOOT_NAMES:
        path = sd_root / name
        if not path.is_file():
            raise ValueError(f"missing original boot file: {name}")
        actual = file_sha256(path)
        expected = expected_boot_hashes[name].lower()
        if actual != expected:
            raise ValueError(
                f"original boot hash mismatch for {name}: expected {expected}, got {actual}"
            )
        actual_boot_hashes[name] = actual

    existing = [name for name in DPU_NAMES if (sd_root / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to stage over existing names: {', '.join(existing)}")

    for name in DPU_NAMES:
        if not (source_dir / name).is_file():
            raise ValueError(f"missing prepared artifact: {name}")
    payload_hashes = parse_payload_manifest(source_dir / "DPUV1.SHA")

    files = []
    for name in DPU_NAMES:
        source = source_dir / name
        digest = file_sha256(source)
        if name in payload_hashes and digest != payload_hashes[name]:
            raise ValueError(
                f"prepared artifact hash mismatch for {name}: "
                f"expected {payload_hashes[name]}, got {digest}"
            )
        files.append(
            CopyRecord(
                name=name,
                source=source,
                destination=sd_root / name,
                size=source.stat().st_size,
                sha256=digest,
            )
        )

    required = sum(item.size for item in files) + FREE_SPACE_MARGIN
    available = shutil.disk_usage(sd_root).free
    if available < required:
        raise OSError(
            f"insufficient free space: need {required} bytes, have {available} bytes"
        )
    return StagePlan(sd_root, source_dir, serial, actual_boot_hashes, tuple(files))


def stage(plan: StagePlan, *, dry_run: bool = False) -> dict:
    collisions = [item.name for item in plan.files if item.destination.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to stage over existing names: {', '.join(collisions)}"
        )

    report = {
        "dry_run": dry_run,
        "volume_serial": plan.volume_serial,
        "sd_root": str(plan.sd_root),
        "original_boot_hashes": dict(plan.boot_hashes),
        "files": {},
    }
    if dry_run:
        for item in plan.files:
            report["files"][item.name] = {
                "size": item.size,
                "source_sha256": item.sha256,
                "readback_sha256": None,
            }
        return report

    for item in plan.files:
        try:
            destination = item.destination.open("xb")
        except FileExistsError as error:
            raise FileExistsError(f"destination appeared during staging: {item.name}") from error
        with item.source.open("rb") as source, destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        readback_hash = file_sha256(item.destination)
        if item.destination.stat().st_size != item.size or readback_hash != item.sha256:
            raise OSError(f"read-back verification failed for {item.name}")
        report["files"][item.name] = {
            "size": item.size,
            "source_sha256": item.sha256,
            "readback_sha256": readback_hash,
        }

    for name, expected in plan.boot_hashes.items():
        actual = file_sha256(plan.sd_root / name)
        if actual != expected:
            raise OSError(f"original boot file changed during staging: {name}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Create-only staging for MZ7100 DPU files")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--sd-root", required=True, type=Path)
    parser.add_argument("--expected-volume-serial", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    plan = preflight(
        args.sd_root,
        args.source_dir,
        expected_volume_serial=args.expected_volume_serial,
    )
    result = stage(plan, dry_run=args.dry_run)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8", newline="\n") as output:
            output.write(rendered + "\n")
            output.flush()
            os.fsync(output.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
