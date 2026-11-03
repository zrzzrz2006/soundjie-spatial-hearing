#!/usr/bin/env python3
"""Map audited public recordings into the fixed Soundjie D3 eight-class catalog."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

try:
    from scripts.host.soundjie_d3_manifest import (
        EXPECTED_DATASET_ROOT,
        SourceSpec,
        audit_license,
        load_source_contract,
        sha256_file,
    )
except ModuleNotFoundError:
    from soundjie_d3_manifest import (
        EXPECTED_DATASET_ROOT,
        SourceSpec,
        audit_license,
        load_source_contract,
        sha256_file,
    )


CLASS_IDS = (
    "fire_smoke_alarm",
    "emergency_siren",
    "car_horn",
    "glass_breaking",
    "door_knock_bell",
    "distress_voice",
    "normal_speech",
    "background_unknown",
)
ROOT_KEYS = frozenset(("schema", "classes", "background_forbidden_terms", "background_exclusions"))
CLASS_KEYS = frozenset(
    ("index", "id", "display_name_zh", "semantic_status", "limitations", "source_labels")
)
EXCLUSION_KEYS = frozenset(("source_id", "label", "target_id"))
TRAINING_ROOT_KEYS = frozenset(
    ("schema", "seed", "split_ratios", "demo", "audio", "preparation", "augmentation", "training")
)
SPLIT_NAMES = ("train", "validation", "test")


class ClassMapError(ValueError):
    """The eight-class mapping is ambiguous or incomplete."""


@dataclass(frozen=True)
class ClassEntry:
    index: int
    id: str
    display_name_zh: str
    semantic_status: str
    limitations: str
    source_labels: Mapping[str, Tuple[str, ...]]


@dataclass(frozen=True)
class BackgroundExclusion:
    source_id: str
    label: str
    target_id: str


@dataclass(frozen=True)
class ClassMap:
    schema: int
    classes: Tuple[ClassEntry, ...]
    background_forbidden_terms: Tuple[str, ...]
    background_exclusions: Tuple[BackgroundExclusion, ...]

    def targets_for(self, source_id: str, labels: Tuple[str, ...]) -> Tuple[str, ...]:
        targets = []
        for item in self.classes:
            aliases = set(item.source_labels.get(source_id, ()))
            if aliases.intersection(labels):
                targets.append(item.id)
        return tuple(targets)


@dataclass(frozen=True)
class CatalogRecord:
    recording_id: str
    source_id: str
    source_group: str
    speaker_or_uploader: str
    license: str
    original_labels: Tuple[str, ...]
    target_id: str
    target_index: int
    path: str
    sha256: str


@dataclass(frozen=True)
class Rejection:
    source_id: str
    path: str
    original_labels: Tuple[str, ...]
    reason: str
    target_ids: Tuple[str, ...]
    details: str


@dataclass(frozen=True)
class SplitConfig:
    seed: int
    ratios: Mapping[str, float]
    demo_per_class_min: int
    demo_per_class_max: int
    license_priority: Tuple[str, ...]


@dataclass(frozen=True)
class SplitRecord:
    recording_id: str
    split: str
    source_id: str
    source_group: str
    speaker_or_uploader: str
    license: str
    original_labels: Tuple[str, ...]
    target_id: str
    target_index: int
    path: str
    sha256: str


@dataclass(frozen=True)
class SplitManifest:
    records: Tuple[SplitRecord, ...]
    seed: int
    ratios: Mapping[str, float]
    class_split_counts: Mapping[str, Mapping[str, int]]
    cross_split_group_count: int
    cross_split_speaker_count: int
    cross_split_hash_count: int


@dataclass(frozen=True)
class DemoRecord:
    recording_id: str
    source_id: str
    source_group: str
    speaker_or_uploader: str
    license: str
    original_labels: Tuple[str, ...]
    target_id: str
    target_index: int
    path: str
    sha256: str
    selection_reason: str


@dataclass(frozen=True)
class DemoManifest:
    records: Tuple[DemoRecord, ...]
    reserved_recording_ids: Tuple[str, ...]
    reserved_source_groups: Tuple[str, ...]
    reserved_speakers: Tuple[str, ...]
    reserved_sha256: Tuple[str, ...]


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ClassMapError(f"{where} keys mismatch: unknown={unknown}, missing={missing}")


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClassMapError(f"{where} must be a non-empty string")
    return value


def load_class_map(path: Path) -> ClassMap:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClassMapError(f"cannot read class map: {exc}") from exc
    if not isinstance(raw, dict):
        raise ClassMapError("class map must be an object")
    _exact_keys(raw, ROOT_KEYS, "class map")
    if raw["schema"] != 1:
        raise ClassMapError("schema must be 1")
    if not isinstance(raw["classes"], list) or len(raw["classes"]) != 8:
        raise ClassMapError("exactly eight classes are required")

    classes = []
    aliases: dict[tuple[str, str], str] = {}
    for position, value in enumerate(raw["classes"]):
        if not isinstance(value, dict):
            raise ClassMapError(f"classes[{position}] must be an object")
        _exact_keys(value, CLASS_KEYS, f"classes[{position}]")
        if value["index"] != position or value["id"] != CLASS_IDS[position]:
            raise ClassMapError("class indices and IDs must match the fixed eight-class order")
        if value["semantic_status"] not in ("direct", "proxy"):
            raise ClassMapError(f"{value['id']}.semantic_status must be direct or proxy")
        source_labels = value["source_labels"]
        if not isinstance(source_labels, dict) or not source_labels:
            raise ClassMapError(f"{value['id']}.source_labels must be non-empty")
        normalized: dict[str, Tuple[str, ...]] = {}
        for source_id, labels in source_labels.items():
            _text(source_id, f"{value['id']}.source_id")
            if not isinstance(labels, list) or not labels:
                raise ClassMapError(f"{value['id']}.{source_id} labels must be non-empty")
            checked = tuple(_text(label, f"{value['id']}.label") for label in labels)
            if len(set(checked)) != len(checked):
                raise ClassMapError(f"duplicate alias in {value['id']}.{source_id}")
            for label in checked:
                key = (source_id, label)
                if key in aliases:
                    raise ClassMapError(
                        f"alias collision for {source_id}:{label}: {aliases[key]} and {value['id']}"
                    )
                aliases[key] = value["id"]
            normalized[source_id] = checked
        classes.append(
            ClassEntry(
                position,
                value["id"],
                _text(value["display_name_zh"], f"{value['id']}.display_name_zh"),
                value["semantic_status"],
                _text(value["limitations"], f"{value['id']}.limitations"),
                normalized,
            )
        )

    forbidden_raw = raw["background_forbidden_terms"]
    if not isinstance(forbidden_raw, list) or not forbidden_raw:
        raise ClassMapError("background_forbidden_terms must be non-empty")
    forbidden = tuple(_text(term, "background_forbidden_terms") for term in forbidden_raw)
    if len(set(forbidden)) != len(forbidden):
        raise ClassMapError("background_forbidden_terms contains duplicates")

    exclusions = []
    if not isinstance(raw["background_exclusions"], list):
        raise ClassMapError("background_exclusions must be an array")
    for position, value in enumerate(raw["background_exclusions"]):
        if not isinstance(value, dict):
            raise ClassMapError(f"background_exclusions[{position}] must be an object")
        _exact_keys(value, EXCLUSION_KEYS, f"background_exclusions[{position}]")
        if value["target_id"] not in CLASS_IDS:
            raise ClassMapError(f"unknown target in background exclusion: {value['target_id']}")
        exclusion = BackgroundExclusion(
            _text(value["source_id"], "background exclusion source_id"),
            _text(value["label"], "background exclusion label"),
            value["target_id"],
        )
        if aliases.get((exclusion.source_id, exclusion.label)) != exclusion.target_id:
            raise ClassMapError(
                f"background exclusion does not match alias: {exclusion.source_id}:{exclusion.label}"
            )
        exclusions.append(exclusion)

    background = classes[7]
    for labels in background.source_labels.values():
        for label in labels:
            lowered = label.lower()
            if any(term.lower() in lowered for term in forbidden):
                raise ClassMapError(f"background alias contains forbidden target term: {label}")
    return ClassMap(1, tuple(classes), forbidden, tuple(exclusions))


def load_split_config(path: Path) -> SplitConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClassMapError(f"cannot read training config: {exc}") from exc
    if not isinstance(raw, dict):
        raise ClassMapError("training config must be an object")
    _exact_keys(raw, TRAINING_ROOT_KEYS, "training config")
    if raw["schema"] != 1:
        raise ClassMapError("training config schema must be 1")
    seed = raw["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed != 20261013:
        raise ClassMapError("training seed must be exactly 20261013")
    ratios = raw["split_ratios"]
    if not isinstance(ratios, dict) or tuple(ratios) != SPLIT_NAMES:
        raise ClassMapError("split_ratios must contain train, validation, test in order")
    checked_ratios: dict[str, float] = {}
    for name in SPLIT_NAMES:
        value = ratios[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ClassMapError(f"split ratio must be positive: {name}")
        checked_ratios[name] = float(value)
    if abs(sum(checked_ratios.values()) - 1.0) > 1e-12:
        raise ClassMapError("split ratios must sum to 1")
    demo = raw["demo"]
    if not isinstance(demo, dict) or set(demo) != {
        "per_class_min",
        "per_class_max",
        "license_priority",
    }:
        raise ClassMapError("demo config keys are invalid")
    minimum = demo["per_class_min"]
    maximum = demo["per_class_max"]
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or minimum < 2
        or maximum > 3
        or minimum > maximum
    ):
        raise ClassMapError("demo count must remain within two to three per class")
    augmentation = raw["augmentation"]
    if not isinstance(augmentation, dict) or set(augmentation) != {
        "partition", "gain_db", "time_shift_frames", "reverb_decay",
        "reverb_taps", "background_mix",
    }:
        raise ClassMapError("augmentation config keys are invalid")
    if augmentation["partition"] != "train_only":
        raise ClassMapError("augmentation must remain train-only")
    for key in ("gain_db", "reverb_decay", "background_mix"):
        bounds = augmentation[key]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in bounds)
            or float(bounds[0]) > float(bounds[1])
        ):
            raise ClassMapError(f"augmentation bounds are invalid: {key}")
    if (
        not isinstance(augmentation["time_shift_frames"], int)
        or isinstance(augmentation["time_shift_frames"], bool)
        or augmentation["time_shift_frames"] < 0
    ):
        raise ClassMapError("time_shift_frames must be a non-negative integer")
    if (
        not isinstance(augmentation["reverb_taps"], int)
        or isinstance(augmentation["reverb_taps"], bool)
        or augmentation["reverb_taps"] < 1
    ):
        raise ClassMapError("reverb_taps must be a positive integer")
    priority = demo["license_priority"]
    if priority != ["CC0", "CC-BY", "other_approved"]:
        raise ClassMapError("demo license priority must be CC0, CC-BY, other_approved")
    if not isinstance(raw["audio"], dict) or not isinstance(raw["training"], dict):
        raise ClassMapError("audio and training sections must be objects")
    return SplitConfig(seed, checked_ratios, minimum, maximum, tuple(priority))


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            self.parent[larger] = smaller


def _components(records: Sequence[CatalogRecord]) -> Tuple[Tuple[CatalogRecord, ...], ...]:
    ordered = tuple(sorted(records, key=lambda item: item.recording_id))
    if len({item.recording_id for item in ordered}) != len(ordered):
        raise ClassMapError("duplicate recording_id in catalog")
    groups = _DisjointSet(len(ordered))
    seen: dict[tuple[str, str], int] = {}
    for index, item in enumerate(ordered):
        keys = (
            ("source_group", f"{item.source_id}:{item.source_group}"),
            ("speaker", f"{item.source_id}:{item.speaker_or_uploader}"),
            ("sha256", item.sha256),
        )
        for key in keys:
            previous = seen.get(key)
            if previous is None:
                seen[key] = index
            else:
                groups.union(index, previous)
    buckets: dict[int, list[CatalogRecord]] = defaultdict(list)
    for index, item in enumerate(ordered):
        buckets[groups.find(index)].append(item)
    components = [tuple(sorted(items, key=lambda item: item.recording_id)) for items in buckets.values()]
    components.sort(key=lambda items: tuple(item.recording_id for item in items))
    return tuple(components)


def _license_rank(license_name: str) -> tuple[int, str]:
    normalized = license_name.upper().replace("_", "-")
    if normalized.startswith("CC0"):
        return 0, "CC0"
    if normalized.startswith("CC-BY-") and not normalized.startswith("CC-BY-NC"):
        return 1, "CC-BY"
    return 2, "other_approved"


def select_demo(
    records: Sequence[CatalogRecord], per_class_min: int = 2, per_class_max: int = 3
) -> DemoManifest:
    if per_class_min < 2 or per_class_max > 3 or per_class_min > per_class_max:
        raise ClassMapError("demo count must remain within two to three per class")
    components = _components(records)
    component_by_record = {
        item.recording_id: component_index
        for component_index, component in enumerate(components)
        for item in component
    }
    selected_components: set[int] = set()
    selected: list[DemoRecord] = []
    for class_index, class_id in enumerate(CLASS_IDS):
        candidates = []
        for item in sorted(records, key=lambda value: value.recording_id):
            if item.target_id != class_id:
                continue
            component_index = component_by_record[item.recording_id]
            rank, label = _license_rank(item.license)
            candidates.append((rank, item.recording_id, component_index, label, item))
        candidates.sort(key=lambda value: (value[0], value[1]))
        chosen_for_class = 0
        for _, _, component_index, license_label, item in candidates:
            if component_index in selected_components:
                continue
            selected_components.add(component_index)
            selected.append(
                DemoRecord(
                    item.recording_id,
                    item.source_id,
                    item.source_group,
                    item.speaker_or_uploader,
                    item.license,
                    item.original_labels,
                    item.target_id,
                    item.target_index,
                    item.path,
                    item.sha256,
                    f"{license_label}:{item.license};distinct_source_component",
                )
            )
            chosen_for_class += 1
            if chosen_for_class == per_class_max:
                break
        if chosen_for_class < per_class_min:
            raise ClassMapError(f"not enough independent demo groups for {class_id}")
    reserved_records = [item for index in selected_components for item in components[index]]
    return DemoManifest(
        tuple(sorted(selected, key=lambda item: (item.target_index, item.recording_id))),
        tuple(sorted({item.recording_id for item in reserved_records})),
        tuple(sorted({item.source_group for item in reserved_records})),
        tuple(sorted({item.speaker_or_uploader for item in reserved_records})),
        tuple(sorted({item.sha256 for item in reserved_records})),
    )


def _cross_split_count(records: Sequence[SplitRecord], field: str, include_source: bool) -> int:
    observed: dict[str, set[str]] = defaultdict(set)
    for item in records:
        value = getattr(item, field)
        key = f"{item.source_id}:{value}" if include_source else value
        observed[key].add(item.split)
    return sum(1 for splits in observed.values() if len(splits) > 1)


def assign_splits(records: Sequence[CatalogRecord], config: SplitConfig) -> SplitManifest:
    components = _components(records)
    if not components:
        raise ClassMapError("cannot split an empty catalog")
    totals = Counter(item.target_id for item in records)
    unknown = sorted(set(totals) - set(CLASS_IDS))
    if unknown:
        raise ClassMapError(f"split input has unknown classes: {','.join(unknown)}")
    present_classes = tuple(class_id for class_id in CLASS_IDS if totals[class_id] > 0)
    desired = {
        class_id: {name: totals[class_id] * config.ratios[name] for name in SPLIT_NAMES}
        for class_id in CLASS_IDS
    }
    counts = {class_id: {name: 0 for name in SPLIT_NAMES} for class_id in CLASS_IDS}
    assignments: dict[str, str] = {}

    def component_key(component: Tuple[CatalogRecord, ...]) -> str:
        identity = "\n".join(item.recording_id for item in component)
        return hashlib.sha256(f"{config.seed}:{identity}".encode("utf-8")).hexdigest()

    for component in sorted(components, key=component_key):
        additions = Counter(item.target_id for item in component)
        scored = []
        for split_index, split in enumerate(SPLIT_NAMES):
            deficit_score = sum(
                (desired[class_id][split] - counts[class_id][split]) * amount
                for class_id, amount in additions.items()
            )
            tie = hashlib.sha256(
                f"{config.seed}:{component_key(component)}:{split}".encode("utf-8")
            ).hexdigest()
            scored.append((-deficit_score, tie, split_index, split))
        split = min(scored)[3]
        for class_id, amount in additions.items():
            counts[class_id][split] += amount
        for item in component:
            assignments[item.recording_id] = split

    output = tuple(
        SplitRecord(
            item.recording_id,
            assignments[item.recording_id],
            item.source_id,
            item.source_group,
            item.speaker_or_uploader,
            item.license,
            item.original_labels,
            item.target_id,
            item.target_index,
            item.path,
            item.sha256,
        )
        for item in sorted(records, key=lambda value: value.recording_id)
    )
    for class_id in present_classes:
        absent = [name for name in SPLIT_NAMES if counts[class_id][name] == 0]
        if absent:
            raise ClassMapError(f"class {class_id} missing from splits: {','.join(absent)}")
    group_cross = _cross_split_count(output, "source_group", True)
    speaker_cross = _cross_split_count(output, "speaker_or_uploader", True)
    hash_cross = _cross_split_count(output, "sha256", False)
    if group_cross or speaker_cross or hash_cross:
        raise ClassMapError("group leakage detected after split assignment")
    return SplitManifest(
        output,
        config.seed,
        dict(config.ratios),
        {class_id: dict(counts[class_id]) for class_id in CLASS_IDS},
        group_cross,
        speaker_cross,
        hash_cross,
    )


def _default_class_map() -> ClassMap:
    return load_class_map(Path(__file__).resolve().parents[2] / "config" / "soundjie_d3_classes.json")


def catalog_recording(source_record: Mapping[str, Any]) -> CatalogRecord | Rejection:
    class_map = _default_class_map()
    source_id = str(source_record.get("source_id", ""))
    path_text = str(source_record.get("path", ""))
    labels_raw = source_record.get("original_labels", ())
    labels = tuple(str(label) for label in labels_raw) if isinstance(labels_raw, (list, tuple)) else ()
    targets = class_map.targets_for(source_id, labels)
    if len(targets) > 1:
        return Rejection(source_id, path_text, labels, "multiple_targets", targets, "recording maps to multiple classes")
    if not targets:
        return Rejection(source_id, path_text, labels, "unmapped_label", (), "no approved class alias")
    if source_record.get("license_decision") != "approved":
        return Rejection(source_id, path_text, labels, "license_not_approved", targets, "source license gate failed")
    target_id = targets[0]
    if target_id == "background_unknown":
        lowered = " ".join(labels).lower()
        if any(term.lower() in lowered for term in class_map.background_forbidden_terms):
            return Rejection(source_id, path_text, labels, "unsafe_background_label", targets, "target-related term in background label")
    path = Path(path_text)
    root = Path(EXPECTED_DATASET_ROOT).resolve()
    try:
        path.resolve().relative_to(root)
    except (OSError, ValueError):
        return Rejection(source_id, path_text, labels, "path_outside_dataset_root", targets, "path is not under fixed dataset root")
    if not path.is_file():
        return Rejection(source_id, path_text, labels, "missing_file", targets, "recording does not exist")
    digest = str(source_record.get("sha256") or sha256_file(path))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return Rejection(source_id, path_text, labels, "invalid_sha256", targets, "recording hash is malformed")
    target_index = CLASS_IDS.index(target_id)
    relative = path.resolve().relative_to(root).as_posix()
    return CatalogRecord(
        f"{source_id}:{relative}",
        source_id,
        _text(source_record.get("source_group"), "source_group"),
        _text(source_record.get("speaker_or_uploader"), "speaker_or_uploader"),
        _text(source_record.get("license"), "license"),
        labels,
        target_id,
        target_index,
        str(path.resolve()),
        digest,
    )


def _scan_esc50(root: Path, source: SourceSpec) -> list[Mapping[str, Any]]:
    candidates = sorted((root / "raw" / source.id).rglob("esc50.csv"))
    if len(candidates) != 1:
        raise ClassMapError(f"expected one ESC-50 metadata CSV, found {len(candidates)}")
    metadata = candidates[0]
    audio_root = metadata.parent.parent / "audio"
    records = []
    with metadata.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source_file = row.get("src_file", "") or row["filename"].split("-", 1)[0]
            records.append(
                {
                    "source_id": source.id,
                    "path": str(audio_root / row["filename"]),
                    "license": source.license,
                    "license_decision": source.license_decision,
                    "source_group": f"esc50:{source_file}",
                    "speaker_or_uploader": str(source_file),
                    "original_labels": [row["category"]],
                }
            )
    return records


def _scan_librispeech(root: Path, source: SourceSpec) -> list[Mapping[str, Any]]:
    files = sorted((root / "raw" / source.id).rglob("*.flac"))
    if not files:
        raise ClassMapError("no LibriSpeech FLAC files found")
    records = []
    for path in files:
        speaker = path.stem.split("-", 1)[0]
        records.append(
            {
                "source_id": source.id,
                "path": str(path),
                "license": source.license,
                "license_decision": source.license_decision,
                "source_group": f"librispeech:speaker:{speaker}",
                "speaker_or_uploader": speaker,
                "original_labels": ["speech"],
            }
        )
    return records


def _publish_jsonl(path: Path, rows: list[Any]) -> None:
    if path.exists():
        raise ClassMapError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ClassMapError(f"partial output already exists: {partial}")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(asdict(row), sort_keys=True, separators=(",", ":")))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ClassMapError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ClassMapError(f"partial output already exists: {partial}")
    try:
        with partial.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def _read_catalog(path: Path) -> Tuple[CatalogRecord, ...]:
    records = []
    expected = set(CatalogRecord.__dataclass_fields__)
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                raw = json.loads(line)
                if not isinstance(raw, dict) or set(raw) != expected:
                    raise ClassMapError(f"catalog line {line_number} has invalid keys")
                raw["original_labels"] = tuple(raw["original_labels"])
                record = CatalogRecord(**raw)
                if record.target_id not in CLASS_IDS or record.target_index != CLASS_IDS.index(record.target_id):
                    raise ClassMapError(f"catalog line {line_number} has invalid target")
                records.append(record)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise ClassMapError(f"cannot read catalog: {exc}") from exc
    if not records:
        raise ClassMapError("catalog is empty")
    return tuple(records)


def split_catalog(
    catalog_path: Path,
    config_path: Path,
    split_path: Path,
    demo_path: Path,
    audit_path: Path,
) -> Mapping[str, Any]:
    records = _read_catalog(catalog_path)
    config = load_split_config(config_path)
    if set(item.target_id for item in records) != set(CLASS_IDS):
        raise ClassMapError("actual catalog must contain all eight classes")
    demo = select_demo(records, config.demo_per_class_min, config.demo_per_class_max)
    reserved = set(demo.reserved_recording_ids)
    remaining = tuple(item for item in records if item.recording_id not in reserved)
    splits = assign_splits(remaining, config)
    split_ids = {item.recording_id for item in splits.records}
    split_groups = {item.source_group for item in splits.records}
    split_speakers = {item.speaker_or_uploader for item in splits.records}
    split_hashes = {item.sha256 for item in splits.records}
    demo_overlap_count = sum(
        (
            len(set(demo.reserved_recording_ids).intersection(split_ids)),
            len(set(demo.reserved_source_groups).intersection(split_groups)),
            len(set(demo.reserved_speakers).intersection(split_speakers)),
            len(set(demo.reserved_sha256).intersection(split_hashes)),
        )
    )
    if demo_overlap_count:
        raise ClassMapError("demo leakage detected")
    demo_counts = Counter(item.target_id for item in demo.records)
    if any(
        not config.demo_per_class_min <= demo_counts[class_id] <= config.demo_per_class_max
        for class_id in CLASS_IDS
    ):
        raise ClassMapError("demo class count gate failed")
    audit = {
        "schema": 1,
        "seed": config.seed,
        "catalog_record_count": len(records),
        "split_record_count": len(splits.records),
        "demo_selected_count": len(demo.records),
        "demo_reserved_record_count": len(demo.reserved_recording_ids),
        "demo_reserved_recording_ids": list(demo.reserved_recording_ids),
        "demo_class_counts": {class_id: demo_counts[class_id] for class_id in CLASS_IDS},
        "class_split_counts": splits.class_split_counts,
        "cross_split_group_count": splits.cross_split_group_count,
        "cross_split_speaker_count": splits.cross_split_speaker_count,
        "cross_split_hash_count": splits.cross_split_hash_count,
        "demo_overlap_count": demo_overlap_count,
    }
    _publish_jsonl(split_path, list(splits.records))
    _publish_jsonl(demo_path, list(demo.records))
    _publish_json(audit_path, audit)
    return audit


def curate(root: Path, source_contract_path: Path, catalog_path: Path, rejection_path: Path) -> tuple[int, int]:
    if root.resolve() != Path(EXPECTED_DATASET_ROOT).resolve():
        raise ClassMapError(f"root must be exactly {EXPECTED_DATASET_ROOT}")
    load_class_map(Path(__file__).resolve().parents[2] / "config" / "soundjie_d3_classes.json")
    contract = load_source_contract(source_contract_path)
    records: list[CatalogRecord] = []
    rejections: list[Rejection] = []
    for source in contract.sources:
        if source.status != "active":
            continue
        decision = audit_license(root / "raw" / source.id, source)
        if not decision.approved:
            raise ClassMapError(
                f"license audit failed for {source.id}: {decision.reason}"
            )
        if source.id == "esc50":
            raw = _scan_esc50(root, source)
        elif source.id == "librispeech_dev_clean":
            raw = _scan_librispeech(root, source)
        else:
            raise ClassMapError(f"no scanner for active source: {source.id}")
        for item in raw:
            result = catalog_recording(item)
            if isinstance(result, CatalogRecord):
                records.append(result)
            else:
                rejections.append(result)
    records.sort(key=lambda item: (item.target_index, item.source_id, item.recording_id))
    rejections.sort(key=lambda item: (item.source_id, item.path, item.reason))
    seen = {item.target_id for item in records}
    missing = [class_id for class_id in CLASS_IDS if class_id not in seen]
    if missing:
        raise ClassMapError(f"catalog has empty classes: {','.join(missing)}")
    _publish_jsonl(catalog_path, records)
    _publish_jsonl(rejection_path, rejections)
    return len(records), len(rejections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("curate", "split"), default="curate")
    parser.add_argument("--root", type=Path, default=Path(EXPECTED_DATASET_ROOT))
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--rejections", type=Path)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--demo-manifest", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.mode == "curate":
        if args.sources is None or args.catalog is None or args.rejections is None:
            parser.error("curate mode requires --sources, --catalog, and --rejections")
        accepted, rejected = curate(args.root, args.sources, args.catalog, args.rejections)
        report: Mapping[str, Any] = {"accepted": accepted, "rejected": rejected}
    else:
        if any(
            value is None
            for value in (
                args.catalog,
                args.training_config,
                args.splits,
                args.demo_manifest,
                args.audit,
            )
        ):
            parser.error(
                "split mode requires --catalog, --training-config, --splits, --demo-manifest, and --audit"
            )
        report = split_catalog(
            args.catalog,
            args.training_config,
            args.splits,
            args.demo_manifest,
            args.audit,
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
