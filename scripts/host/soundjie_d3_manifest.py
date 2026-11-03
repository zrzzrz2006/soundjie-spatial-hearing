#!/usr/bin/env python3
"""Strict contracts shared by the Soundjie D3 data pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Tuple


EXPECTED_DATASET_ROOT = r"F:\MLK\DPU_ZYNQ7000\datasets\soundjie_v1"
ROOT_KEYS = frozenset(("schema", "dataset_root", "sources"))
SOURCE_KEYS = frozenset(
    (
        "id",
        "status",
        "version",
        "official_page",
        "urls",
        "archive_name",
        "sha256",
        "size_bytes",
        "license",
        "license_url",
        "license_decision",
        "license_decision_rationale",
        "license_evidence",
        "competition_use",
        "allowed_classes",
        "fallback_source_id",
    )
)
SOURCE_STATUSES = frozenset(("active", "fallback", "candidate", "rejected"))
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """A versioned manifest violates its fail-closed schema."""


@dataclass(frozen=True)
class SourceSpec:
    id: str
    status: str
    version: str
    official_page: str
    urls: Tuple[str, ...]
    archive_name: str
    sha256: str | None
    size_bytes: int | None
    license: str
    license_url: str
    license_decision: str
    license_decision_rationale: str
    license_evidence: Tuple["LicenseEvidence", ...]
    competition_use: str
    allowed_classes: Tuple[str, ...]
    fallback_source_id: str | None


@dataclass(frozen=True)
class SourceContract:
    schema: int
    dataset_root: Path
    sources: Tuple[SourceSpec, ...]


@dataclass(frozen=True)
class LicenseDecision:
    source_id: str
    approved: bool
    license: str
    competition_use: str
    evidence_files: Tuple[str, ...]
    evidence_sha256: Tuple[Tuple[str, str], ...]
    reason: str


@dataclass(frozen=True)
class LicenseEvidence:
    path: str
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ContractError(f"{where} unknown keys: {','.join(unknown)}")
    if missing:
        raise ContractError(f"{where} missing keys: {','.join(missing)}")


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _parse_source(raw: Any, index: int) -> SourceSpec:
    if not isinstance(raw, dict):
        raise ContractError(f"sources[{index}] must be an object")
    _require_exact_keys(raw, SOURCE_KEYS, f"sources[{index}]")
    source_id = _nonempty_string(raw["id"], f"sources[{index}].id")
    status = _nonempty_string(raw["status"], f"{source_id}.status")
    if status not in SOURCE_STATUSES:
        raise ContractError(f"{source_id}.status unsupported: {status}")
    urls_raw = raw["urls"]
    if not isinstance(urls_raw, list) or any(
        not isinstance(item, str) or not item.startswith(("https://", "http://"))
        for item in urls_raw
    ):
        raise ContractError(f"{source_id}.urls must contain only HTTP(S) URLs")
    if status in ("active", "fallback") and not urls_raw:
        raise ContractError(f"{source_id}.urls must not be empty")
    allowed = raw["allowed_classes"]
    if not isinstance(allowed, list) or not allowed or any(
        not isinstance(item, str) or not item for item in allowed
    ):
        raise ContractError(f"{source_id}.allowed_classes must be non-empty strings")
    if len(set(allowed)) != len(allowed):
        raise ContractError(f"{source_id}.allowed_classes contains duplicates")

    license_name = raw["license"]
    if status != "rejected":
        _nonempty_string(license_name, f"{source_id}.license")
    elif not isinstance(license_name, str):
        raise ContractError(f"{source_id}.license must be a string")
    license_url = _nonempty_string(raw["license_url"], f"{source_id}.license_url")
    if not license_url.startswith("https://"):
        raise ContractError(f"{source_id}.license_url must use HTTPS")
    rationale = _nonempty_string(
        raw["license_decision_rationale"], f"{source_id}.license_decision_rationale"
    )
    evidence_raw = raw["license_evidence"]
    if not isinstance(evidence_raw, list) or not evidence_raw:
        raise ContractError(f"{source_id}.license_evidence must be non-empty")
    evidence = []
    for position, item in enumerate(evidence_raw):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ContractError(f"{source_id}.license_evidence[{position}] keys invalid")
        relative = _nonempty_string(item["path"], f"{source_id}.license_evidence.path")
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or "\\" in relative:
            raise ContractError(f"{source_id}.license_evidence path unsafe")
        digest = item["sha256"]
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise ContractError(f"{source_id}.license_evidence sha256 invalid")
        evidence.append(LicenseEvidence(relative, digest))

    sha256 = raw["sha256"]
    size_bytes = raw["size_bytes"]
    if status in ("active", "fallback"):
        if not isinstance(sha256, str) or not HEX64.fullmatch(sha256):
            raise ContractError(f"{source_id}.sha256 must be lowercase SHA-256")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0:
            raise ContractError(f"{source_id}.size_bytes must be positive")
        if raw["license_decision"] != "approved":
            raise ContractError(f"{source_id}.license_decision must be approved")
        if raw["competition_use"] != "approved":
            raise ContractError(f"{source_id}.competition_use must be approved")
    else:
        if sha256 is not None and (not isinstance(sha256, str) or not HEX64.fullmatch(sha256)):
            raise ContractError(f"{source_id}.sha256 must be null or lowercase SHA-256")
        if size_bytes is not None and (
            not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes <= 0
        ):
            raise ContractError(f"{source_id}.size_bytes must be null or positive")

    fallback = raw["fallback_source_id"]
    if fallback is not None and (not isinstance(fallback, str) or not fallback):
        raise ContractError(f"{source_id}.fallback_source_id must be null or non-empty")
    return SourceSpec(
        id=source_id,
        status=status,
        version=_nonempty_string(raw["version"], f"{source_id}.version"),
        official_page=_nonempty_string(raw["official_page"], f"{source_id}.official_page"),
        urls=tuple(urls_raw),
        archive_name=_nonempty_string(raw["archive_name"], f"{source_id}.archive_name"),
        sha256=sha256,
        size_bytes=size_bytes,
        license=license_name,
        license_url=license_url,
        license_decision=_nonempty_string(
            raw["license_decision"], f"{source_id}.license_decision"
        ),
        license_decision_rationale=rationale,
        license_evidence=tuple(evidence),
        competition_use=_nonempty_string(
            raw["competition_use"], f"{source_id}.competition_use"
        ),
        allowed_classes=tuple(allowed),
        fallback_source_id=fallback,
    )


def load_source_contract(path: Path) -> SourceContract:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read source contract: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractError("source contract must be an object")
    _require_exact_keys(raw, ROOT_KEYS, "contract")
    if raw["schema"] != 1:
        raise ContractError("schema must be 1")
    if raw["dataset_root"] != EXPECTED_DATASET_ROOT:
        raise ContractError(f"dataset_root must be exactly {EXPECTED_DATASET_ROOT}")
    if not isinstance(raw["sources"], list) or not raw["sources"]:
        raise ContractError("sources must be a non-empty array")
    sources = tuple(_parse_source(value, index) for index, value in enumerate(raw["sources"]))
    seen = set()
    for source in sources:
        if source.id in seen:
            raise ContractError(f"duplicate source id: {source.id}")
        seen.add(source.id)
    for source in sources:
        if source.fallback_source_id is not None and source.fallback_source_id not in seen:
            raise ContractError(
                f"{source.id}.fallback_source_id does not exist: {source.fallback_source_id}"
            )
    return SourceContract(1, Path(raw["dataset_root"]), sources)


def audit_license(source_dir: Path, source: SourceSpec) -> LicenseDecision:
    verified = []
    failure = None
    for item in source.license_evidence:
        path = source_dir.joinpath(*PurePosixPath(item.path).parts)
        if not path.is_file():
            failure = f"license evidence missing: {item.path}"
            break
        digest = sha256_file(path)
        if digest != item.sha256:
            failure = f"license evidence hash mismatch: {item.path}"
            break
        verified.append((item.path, digest))
    approved = (
        source.status in ("active", "fallback")
        and source.license_decision == "approved"
        and source.competition_use == "approved"
        and failure is None
        and len(verified) == len(source.license_evidence)
    )
    reason = source.license_decision_rationale if approved else (failure or "license evidence incomplete")
    return LicenseDecision(
        source.id,
        approved,
        source.license,
        source.competition_use,
        tuple(path for path, _ in verified),
        tuple(verified),
        reason,
    )
