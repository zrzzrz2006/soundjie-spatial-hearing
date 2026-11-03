#!/usr/bin/env python3
import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path, PurePath
from typing import NamedTuple


BOOT_NAMES = ("BOOT.bin", "uImage", "devicetree.dtb", "uEnv.txt")
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
    source_manifest: Path
    volume_serial: str
    filesystem_state: dict
    boot_hashes: dict[str, str]
    files: tuple[CopyRecord, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_volume_serial(root: Path) -> str:
    anchor = Path(root).resolve().anchor
    if not anchor:
        raise OSError(f"cannot determine volume root for {root}")
    serial = ctypes.c_uint32()
    ok = ctypes.windll.kernel32.GetVolumeInformationW(
        ctypes.c_wchar_p(anchor), None, 0, ctypes.byref(serial), None, None, None, 0
    )
    if not ok:
        raise ctypes.WinError()
    return f"{serial.value >> 16:04X}-{serial.value & 0xFFFF:04X}"


def read_filesystem_state(root: Path) -> dict:
    anchor = Path(root).resolve().anchor
    if anchor.startswith("\\\\?\\Volume{"):
        selector = f"-UniqueId '{anchor}'"
    else:
        drive = anchor[:1]
        if not drive:
            raise OSError(f"cannot determine volume root for {root}")
        selector = f"-DriveLetter '{drive}'"
    script = (
        f"$v=Get-Volume {selector}; "
        "$o=[pscustomobject]@{HealthStatus=[string]$v.HealthStatus;"
        "OperationalStatus=[string]($v.OperationalStatus -join ',');"
        "FileSystem=[string]$v.FileSystem};$o|ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=False, text=True, encoding="utf-8", errors="replace", capture_output=True,
    )
    if result.returncode != 0:
        raise OSError(f"cannot query filesystem state: {result.stderr.strip()}")
    data = json.loads(result.stdout.strip())
    health = data.get("HealthStatus", "")
    operational = data.get("OperationalStatus", "")
    clean = health.lower() == "healthy" and operational.lower() == "ok"
    return {
        "clean": clean,
        "health_status": health,
        "operational_status": operational,
        "filesystem": data.get("FileSystem", ""),
        "detail": f"HealthStatus={health}; OperationalStatus={operational}",
    }


def parse_manifest(path: Path, expected_names: tuple[str, ...]) -> dict[str, str]:
    entries = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise ValueError("invalid source manifest line")
        digest, name = parts
        if name not in expected_names or name in entries:
            raise ValueError(f"unexpected source manifest entry: {name}")
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError(f"invalid SHA-256 in source manifest: {name}")
        entries[name] = digest.lower()
    if tuple(entries) != expected_names:
        raise ValueError("source manifest entries or order do not match destinations")
    return entries


def validate_destinations(destinations: tuple[str, ...], manifest_name: str) -> None:
    if len(destinations) < 2 or len(set(destinations)) != len(destinations):
        raise ValueError("destination names must be unique")
    for name in destinations:
        if not name or PurePath(name).name != name or "/" in name or "\\" in name:
            raise ValueError(f"unsafe destination name: {name}")
    if manifest_name not in destinations:
        raise ValueError("source manifest must also be an explicit destination")


def preflight(
    sd_root: Path,
    source_dir: Path,
    source_manifest: Path,
    destinations: tuple[str, ...],
    *,
    expected_volume_serial: str,
    expected_boot_hashes: dict[str, str] = ORIGINAL_BOOT_HASHES,
) -> StagePlan:
    sd_root = Path(sd_root).resolve()
    source_dir = Path(source_dir).resolve()
    source_manifest = Path(source_manifest).resolve()
    destinations = tuple(destinations)
    if not sd_root.is_dir() or not source_dir.is_dir():
        raise ValueError("SD root and source directory must exist")
    if not source_manifest.is_file() or source_manifest.parent != source_dir:
        raise ValueError("source manifest must be a file directly inside source directory")
    validate_destinations(destinations, source_manifest.name)

    serial = read_volume_serial(sd_root).upper()
    if serial != expected_volume_serial.upper():
        raise ValueError(f"volume serial mismatch: expected {expected_volume_serial}, got {serial}")
    filesystem_state = read_filesystem_state(sd_root)
    if not filesystem_state.get("clean"):
        raise OSError(f"filesystem is not clean: {filesystem_state.get('detail', 'unknown')}")

    boot_hashes = {}
    for name in BOOT_NAMES:
        path = sd_root / name
        if not path.is_file():
            raise ValueError(f"missing original boot file: {name}")
        actual = file_sha256(path)
        expected = expected_boot_hashes[name].lower()
        if actual != expected:
            raise ValueError(f"original boot hash mismatch for {name}: expected {expected}, got {actual}")
        boot_hashes[name] = actual

    collisions = [name for name in destinations if (sd_root / name).exists()]
    if collisions:
        raise FileExistsError(f"refusing to stage over existing names: {', '.join(collisions)}")

    payload_names = tuple(name for name in destinations if name != source_manifest.name)
    manifest_hashes = parse_manifest(source_manifest, payload_names)
    files = []
    for name in destinations:
        source = source_dir / name
        if not source.is_file():
            raise ValueError(f"missing prepared artifact: {name}")
        digest = file_sha256(source)
        if name in manifest_hashes and digest != manifest_hashes[name]:
            raise ValueError(f"prepared artifact hash mismatch for {name}")
        files.append(CopyRecord(name, source, sd_root / name, source.stat().st_size, digest))

    required = sum(item.size for item in files) + FREE_SPACE_MARGIN
    available = shutil.disk_usage(sd_root).free
    if available < required:
        raise OSError(f"insufficient free space: need {required} bytes, have {available} bytes")
    return StagePlan(sd_root, source_dir, source_manifest, serial, filesystem_state, boot_hashes, tuple(files))


def stage(plan: StagePlan, *, dry_run: bool = False) -> dict:
    collisions = [item.name for item in plan.files if item.destination.exists()]
    if collisions:
        raise FileExistsError(f"refusing to stage over existing names: {', '.join(collisions)}")
    report = {
        "dry_run": dry_run,
        "volume_serial": plan.volume_serial,
        "filesystem_state": dict(plan.filesystem_state),
        "sd_root": str(plan.sd_root),
        "original_boot_hashes": dict(plan.boot_hashes),
        "files": {},
    }
    if dry_run:
        for item in plan.files:
            report["files"][item.name] = {
                "size": item.size, "source_sha256": item.sha256, "readback_sha256": None
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
        readback = file_sha256(item.destination)
        if item.destination.stat().st_size != item.size or readback != item.sha256:
            raise OSError(f"read-back verification failed for {item.name}")
        report["files"][item.name] = {
            "size": item.size, "source_sha256": item.sha256, "readback_sha256": readback
        }

    for name, expected in plan.boot_hashes.items():
        if file_sha256(plan.sd_root / name) != expected:
            raise OSError(f"original boot file changed during staging: {name}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Create-only additive DPU inference SD staging")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, action="append")
    parser.add_argument("--sd-root", required=True, type=Path)
    parser.add_argument("--expected-volume-serial", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    plan = preflight(
        args.sd_root, args.source_dir, args.source_manifest, tuple(args.destination),
        expected_volume_serial=args.expected_volume_serial,
    )
    result = stage(plan, dry_run=args.dry_run)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8", newline="\n") as report:
            report.write(rendered + "\n")
            report.flush()
            os.fsync(report.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
