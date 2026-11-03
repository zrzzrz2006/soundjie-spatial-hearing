#!/usr/bin/env python3
"""Fail-closed audit for the legacy DNNDK host archive."""

import argparse
import hashlib
import json
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_member(name: str, allowed_top_levels: set[str]) -> bool:
    if not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    return path.parts[0] in allowed_top_levels


def write_report(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
        newline="\n",
    )


def fail(reason: str, code: int = 1) -> int:
    print(reason, file=sys.stderr)
    return code


def audit(archive: Path, contract_path: Path, report: Path) -> int:
    if not archive.is_file():
        return fail(f"archive_missing:{archive}")
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return fail(f"contract_invalid:{error}")
    if contract.get("schema") != 1:
        return fail("contract_schema_unsupported")
    if archive.name != contract.get("archive_basename"):
        return fail("archive_basename_mismatch")

    allowed = set(contract.get("allowed_top_levels", []))
    required = tuple(contract.get("required_tools", []))
    host_subtree = str(contract.get("host_subtree", ""))
    if not allowed or not required or not host_subtree:
        return fail("contract_incomplete")

    try:
        with tarfile.open(archive, "r:*") as source:
            members = source.getmembers()
    except (tarfile.TarError, OSError) as error:
        return fail(f"archive_invalid:{error}")

    for member in members:
        if not safe_member(member.name, allowed):
            return fail(f"unsafe_archive_member:{member.name}")
        if member.issym() or member.islnk():
            link = PurePosixPath(member.linkname)
            if link.is_absolute() or ".." in link.parts:
                return fail(f"unsafe_archive_link:{member.name}")

    tool_paths: dict[str, str] = {}
    for tool in required:
        matches = [
            member
            for member in members
            if member.isfile()
            and PurePosixPath(member.name).name == tool
            and host_subtree in PurePosixPath(member.name).parts
            and member.mode & 0o111
        ]
        if not matches:
            return fail(f"missing_required_tool:{tool}")
        matches.sort(key=lambda item: (len(PurePosixPath(item.name).parts), item.name))
        tool_paths[tool] = matches[0].name

    actual_hash = file_sha256(archive)
    state = contract.get("state")
    values = {
        "ARCHIVE_BASENAME": archive.name,
        "ARCHIVE_BYTES": str(archive.stat().st_size),
        "ARCHIVE_SHA256": actual_hash,
        "SAFE_PATHS": "pass",
    }
    for tool in required:
        values[f"TOOL_{tool.upper()}"] = tool_paths[tool]

    if state == "candidate_only":
        values["RESULT"] = "candidate_only"
        write_report(report, values)
        return fail("candidate_only", 66)
    if state != "approved":
        return fail("contract_state_invalid")
    expected_hash = str(contract.get("approved_sha256") or "").lower()
    if not SHA256_RE.fullmatch(expected_hash):
        return fail("approved_sha256_invalid")
    if actual_hash != expected_hash:
        return fail("archive_sha256_mismatch")

    values["RESULT"] = "pass"
    write_report(report, values)
    print("RESULT=pass")
    print(f"ARCHIVE_SHA256={actual_hash}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    return audit(args.archive, args.contract, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
