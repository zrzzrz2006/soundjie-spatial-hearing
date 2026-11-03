#!/usr/bin/env python3
"""Verify listed SHA-256 hashes without writing files or operating hardware."""

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path, PureWindowsPath

ENTRY = re.compile(r"([0-9a-fA-F]{64}) [ *](.+)\Z")
RESERVED = re.compile(r"(?:con|prn|aux|nul|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3])(?:\.|$)", re.IGNORECASE)


def safe_parts(name):
    parts = name.split("/")
    if (PureWindowsPath(name).anchor or "\\" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or any(char in name for char in ':<>"|?*')
            or any(part in ("", ".", "..") or part.endswith((" ", "."))
                   or RESERVED.match(part) for part in parts)):
        raise ValueError("Unsafe manifest path: " + repr(name))
    return parts


def load_manifest(manifest):
    if not manifest.is_file():
        raise ValueError("Manifest must be a regular file")
    entries, seen = [], set()
    for lineno, line in enumerate(manifest.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        match = ENTRY.fullmatch(line)
        if not match:
            raise ValueError(f"Malformed SHA-256 entry at line {lineno}")
        digest, name = match.groups()
        safe_parts(name)
        if name.casefold() in seen:
            raise ValueError("Duplicate/case-colliding manifest path: " + name)
        seen.add(name.casefold())
        entries.append((name, digest.lower()))
    if not entries:
        raise ValueError("Empty manifest cannot verify a package")
    return sorted(entries)


def checked_file(root, name):
    candidate = root
    parts = safe_parts(name)
    for index, part in enumerate(parts):
        candidate = candidate / part
        info = candidate.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Links/reparse points are not allowed: " + name)
        expected_type = stat.S_ISREG if index == len(parts) - 1 else stat.S_ISDIR
        if not expected_type(info.st_mode):
            raise ValueError("Not a regular package file/directory: " + name)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("Path escapes package root: " + name)
    return resolved


def verify(root, manifest):
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Root must be a directory")
    entries = load_manifest(manifest)
    # Validate every path before reading any payload. Work on a quiescent directory.
    targets = [(name, expected, checked_file(root, name)) for name, expected in entries]
    files = []
    for name, expected, target in targets:
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        files.append({"path": name, "expected": expected, "actual": actual,
                      "status": "MATCH" if actual == expected else "MISMATCH"})
    return {"status": "VERIFIED" if all(item["status"] == "MATCH" for item in files) else "MISMATCH",
            "files": files, "checked": len(files),
            "scope": "Listed file bytes only; no authenticity, completeness, or rebuild claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Package root (read only)")
    parser.add_argument("manifest", type=Path, help="SHA-256 manifest path (read only)")
    args = parser.parse_args(argv)
    try:
        result = verify(args.root, args.manifest)
    except (OSError, ValueError, RuntimeError) as exc:
        result = {"status": "ERROR", "errors": [str(exc)]}
    print(json.dumps(result, indent=2, sort_keys=True))
    return {"VERIFIED": 0, "MISMATCH": 1, "ERROR": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
