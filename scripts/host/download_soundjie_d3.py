#!/usr/bin/env python3
"""Fail-closed downloader and create-only extractor for Soundjie D3."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tarfile
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from scripts.host.soundjie_d3_manifest import (
        SourceSpec,
        load_source_contract,
        sha256_file,
    )
except ModuleNotFoundError:
    from soundjie_d3_manifest import SourceSpec, load_source_contract, sha256_file


class DownloadError(RuntimeError):
    """A source could not be acquired without violating its contract."""


@dataclass(frozen=True)
class DownloadRecord:
    source_id: str
    path: str
    size_bytes: int
    sha256: str
    resumed: bool
    reused_existing: bool
    attempted_urls: tuple[str, ...]


def _record_download(root: Path, record: DownloadRecord) -> None:
    directory = root / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / "downloads.jsonl"
    with report.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _verify(path: Path, source: SourceSpec) -> tuple[int, str]:
    size = path.stat().st_size
    digest = sha256_file(path)
    if source.size_bytes is None or source.sha256 is None:
        raise DownloadError(f"source is not approved for download: {source.id}")
    if size != source.size_bytes:
        raise DownloadError(
            f"size mismatch for {source.id}: expected {source.size_bytes}, got {size}"
        )
    if digest != source.sha256:
        raise DownloadError(
            f"sha256 mismatch for {source.id}: expected {source.sha256}, got {digest}"
        )
    return size, digest


def _content_range_matches(value: str | None, offset: int, total: int) -> bool:
    if value is None:
        return False
    prefix = f"bytes {offset}-"
    return value.startswith(prefix) and value.endswith(f"/{total}")


def _fetch_url(url: str, partial: Path, expected_size: int) -> bool:
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "soundjie-d3-audited-downloader/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    restart_without_range = False
    with urlopen(request, timeout=60) as response:
        resumed = bool(
            offset
            and response.status == 206
            and _content_range_matches(response.headers.get("Content-Range"), offset, expected_size)
        )
        if offset and not resumed:
            restart_without_range = True
        else:
            mode = "ab" if resumed else "wb"
            with partial.open(mode) as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
                stream.flush()
                os.fsync(stream.fileno())
    if restart_without_range:
        partial.unlink()
        return _fetch_url(url, partial, expected_size)
    return resumed


def download_source(source: SourceSpec, root: Path) -> DownloadRecord:
    if source.status not in ("active", "fallback"):
        raise DownloadError(f"source is not active: {source.id}")
    if source.size_bytes is None or source.sha256 is None:
        raise DownloadError(f"source lacks approved size/hash: {source.id}")
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    final_path = downloads / source.archive_name
    partial = downloads / f"{source.archive_name}.partial"
    if final_path.exists():
        try:
            size, digest = _verify(final_path, source)
        except DownloadError as exc:
            raise DownloadError(f"existing target is not the contracted archive: {final_path}") from exc
        record = DownloadRecord(
            source.id,
            str(final_path),
            size,
            digest,
            False,
            True,
            tuple(),
        )
        _record_download(root, record)
        return record

    attempted = []
    resumed = False
    last_error: Exception | None = None
    for url in source.urls:
        attempted.append(url)
        try:
            resumed = _fetch_url(url, partial, source.size_bytes)
            if partial.exists() and partial.stat().st_size and not resumed:
                # A server that ignored Range returned a complete body into a
                # truncated file.  _fetch_url used wb, so no bytes were appended.
                pass
            size, digest = _verify(partial, source)
            os.replace(partial, final_path)
            record = DownloadRecord(
                source.id,
                str(final_path),
                size,
                digest,
                resumed,
                False,
                tuple(attempted),
            )
            _record_download(root, record)
            return record
        except (DownloadError, HTTPError, URLError, OSError) as exc:
            last_error = exc
            if partial.exists():
                partial.unlink()
    raise DownloadError(f"all URLs failed for {source.id}: {last_error}") from last_error


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise DownloadError(f"unsafe archive path: {name}")
    return path


def _write_members(members: Iterable[tuple[PurePosixPath, bytes]], root: Path) -> None:
    for relative, data in members:
        target = root.joinpath(*relative.parts)
        if target.exists():
            raise DownloadError(f"duplicate archive member: {relative.as_posix()}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _zip_members(path: Path) -> Iterable[tuple[PurePosixPath, bytes]]:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            relative = _safe_member_path(info.filename.rstrip("/"))
            mode = info.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if file_type == stat.S_IFLNK:
                raise DownloadError(f"archive link is forbidden: {info.filename}")
            if info.is_dir():
                continue
            if file_type not in (0, stat.S_IFREG):
                raise DownloadError(f"non-regular archive member: {info.filename}")
            yield relative, archive.read(info)


def _tar_members(path: Path) -> Iterable[tuple[PurePosixPath, bytes]]:
    with tarfile.open(path, "r:*") as archive:
        for info in archive.getmembers():
            relative = _safe_member_path(info.name.rstrip("/"))
            if info.isdir():
                continue
            if not info.isfile():
                raise DownloadError(f"archive link or special member is forbidden: {info.name}")
            extracted = archive.extractfile(info)
            if extracted is None:
                raise DownloadError(f"cannot read archive member: {info.name}")
            yield relative, extracted.read()


def extract_archive(archive: Path, destination: Path) -> Path:
    if destination.exists():
        raise DownloadError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        if zipfile.is_zipfile(archive):
            _write_members(_zip_members(archive), temporary)
        elif tarfile.is_tarfile(archive):
            _write_members(_tar_members(archive), temporary)
        else:
            raise DownloadError(f"unsupported archive format: {archive}")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists() and temporary.parent == destination.parent:
            shutil.rmtree(temporary)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    contract = load_source_contract(args.contract)
    root = args.root or contract.dataset_root
    source = next((item for item in contract.sources if item.id == args.source), None)
    if source is None:
        raise DownloadError(f"unknown source id: {args.source}")
    record = download_source(source, root)
    print(json.dumps(asdict(record), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
