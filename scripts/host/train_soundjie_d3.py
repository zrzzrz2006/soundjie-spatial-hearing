#!/usr/bin/env python3
"""Deterministic host training for the eight-class Soundjie D3 CNN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


class TrainingError(RuntimeError):
    """The fixed training or artifact contract was violated."""


@dataclass(frozen=True)
class ModelConfig:
    input_shape: tuple[int, int, int, int]
    num_classes: int
    channels: tuple[int, int, int, int]
    kernel_size: int
    padding: int
    pool_size: int
    flattened_features: int
    class_map_sha256: str
    frontend_sha256: str
    layer_names: tuple[str, ...]
    allowed_module_types: frozenset[str]


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    per_class: Mapping[str, Mapping[str, float | int]]
    confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class OverfitResult:
    accuracy: float
    logits: np.ndarray


@dataclass(frozen=True)
class TrainingResult:
    checkpoint: Path
    manifest: Path
    validation: EvaluationReport
    test: EvaluationReport


_MODEL_KEYS = {
    "schema", "input_shape", "num_classes", "channels", "kernel_size",
    "padding", "pool_size", "flattened_features", "layer_names",
    "allowed_module_types", "class_map_sha256", "frontend_sha256",
}


def load_model_config(path: Path) -> ModelConfig:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingError(f"cannot read model config: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _MODEL_KEYS:
        raise TrainingError("model config keys drift")
    if raw["schema"] != 1:
        raise TrainingError("model config schema drift")
    expected = {
        "input_shape": [1, 1, 64, 96],
        "num_classes": 8,
        "channels": [16, 32, 64, 96],
        "kernel_size": 3,
        "padding": 1,
        "pool_size": 2,
        "flattened_features": 2304,
    }
    for key, value in expected.items():
        if raw[key] != value:
            raise TrainingError(f"model config {key} drift")
    expected_names = tuple(
        name
        for index in range(1, 5)
        for name in (f"conv{index}", f"bn{index}", f"relu{index}", f"pool{index}")
    ) + ("flatten", "fc8")
    if tuple(raw["layer_names"]) != expected_names:
        raise TrainingError("model layer names drift")
    for key in ("class_map_sha256", "frontend_sha256"):
        value = raw[key]
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise TrainingError(f"model config {key} drift")
    return ModelConfig(
        tuple(raw["input_shape"]), raw["num_classes"], tuple(raw["channels"]),
        raw["kernel_size"], raw["padding"], raw["pool_size"],
        raw["flattened_features"], raw["class_map_sha256"], raw["frontend_sha256"],
        tuple(raw["layer_names"]),
        frozenset(raw["allowed_module_types"]),
    )


class SoundjieCNN(nn.Module):
    """DPU-friendly fixed four-block convolutional classifier."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        c1, c2, c3, c4 = config.channels
        kwargs = {"kernel_size": config.kernel_size, "padding": config.padding, "bias": False}
        self.conv1 = nn.Conv2d(1, c1, **kwargs)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(config.pool_size)
        self.conv2 = nn.Conv2d(c1, c2, **kwargs)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(config.pool_size)
        self.conv3 = nn.Conv2d(c2, c3, **kwargs)
        self.bn3 = nn.BatchNorm2d(c3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(config.pool_size)
        self.conv4 = nn.Conv2d(c3, c4, **kwargs)
        self.bn4 = nn.BatchNorm2d(c4)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(config.pool_size)
        self.flatten = nn.Flatten()
        self.fc8 = nn.Linear(config.flattened_features, config.num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.pool1(self.relu1(self.bn1(self.conv1(values))))
        values = self.pool2(self.relu2(self.bn2(self.conv2(values))))
        values = self.pool3(self.relu3(self.bn3(self.conv3(values))))
        values = self.pool4(self.relu4(self.bn4(self.conv4(values))))
        return self.fc8(self.flatten(values))


def validate_training_contract(
    manifest: Mapping[str, Any], class_map_sha256: str, frontend_sha256: str
) -> None:
    if manifest.get("class_map_sha256") != class_map_sha256:
        raise TrainingError("class map drift")
    if manifest.get("frontend_sha256") != frontend_sha256:
        raise TrainingError("frontend drift")


def audit_model_operators(model: SoundjieCNN) -> None:
    allowed = {"SoundjieCNN", "Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", "Flatten", "Linear"}
    found = {type(module).__name__ for module in model.modules()}
    unexpected = sorted(found - allowed)
    if unexpected:
        raise TrainingError(f"unapproved model operators: {','.join(unexpected)}")


def evaluate_logits(
    logits: np.ndarray, targets: np.ndarray, labels: Sequence[str]
) -> EvaluationReport:
    logits = np.asarray(logits)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 2 or logits.shape != (len(targets), len(labels)):
        raise TrainingError("logit shape drift")
    if len(labels) != 8 or len(set(labels)) != 8:
        raise TrainingError("label contract drift")
    if not np.isfinite(logits).all():
        raise TrainingError("non-finite logits")
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        if target < 0 or target >= len(labels):
            raise TrainingError("target outside label contract")
        confusion[target, prediction] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(labels):
        true_positive = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support
        }
    accuracy = float((predictions == targets).mean()) if len(targets) else 0.0
    return EvaluationReport(accuracy, per_class, tuple(tuple(map(int, row)) for row in confusion))


def apply_augmentation(
    feature: np.ndarray,
    partition: str,
    seed: int,
    recipe: Mapping[str, Any] | None = None,
    background: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(feature, dtype=np.float32)
    if partition != "train":
        return values.copy()
    if recipe is None:
        recipe = {
            "gain_db": [-1.0, 1.0], "time_shift_frames": 0,
            "reverb_decay": [0.0, 0.0], "reverb_taps": 1,
            "background_mix": [0.0, 0.0],
        }
    generator = np.random.default_rng(seed)
    gain_db = generator.uniform(*map(float, recipe["gain_db"]))
    augmented = values * np.float32(10.0 ** (gain_db / 20.0))
    maximum_shift = int(recipe["time_shift_frames"])
    shift = int(generator.integers(-maximum_shift, maximum_shift + 1)) if maximum_shift else 0
    if shift:
        shifted = np.full_like(augmented, -1.0)
        if shift > 0:
            shifted[..., shift:] = augmented[..., :-shift]
        else:
            shifted[..., :shift] = augmented[..., -shift:]
        augmented = shifted
    decay = float(generator.uniform(*map(float, recipe["reverb_decay"])))
    dry = augmented.copy()
    for tap in range(1, int(recipe["reverb_taps"])):
        augmented[..., tap:] += np.float32(decay**tap) * dry[..., :-tap]
    if background is not None:
        mix = np.float32(generator.uniform(*map(float, recipe["background_mix"])))
        augmented = (np.float32(1.0) - mix) * augmented + mix * np.asarray(background, dtype=np.float32)
    if not np.isfinite(augmented).all():
        raise TrainingError("augmentation produced non-finite feature")
    return augmented.astype(np.float32, copy=False)


def augment_training_epoch(
    features: np.ndarray,
    backgrounds: np.ndarray,
    recipe: Mapping[str, Any],
    seed: int,
    epoch: int,
) -> np.ndarray:
    if len(backgrounds) == 0:
        raise TrainingError("train split missing background examples for mixing")
    augmented = []
    for index, feature in enumerate(np.asarray(features, dtype=np.float32)):
        item_seed = seed + epoch * 1_000_003 + index * 101
        background_index = item_seed % len(backgrounds)
        augmented.append(
            apply_augmentation(feature, "train", item_seed, recipe, backgrounds[background_index])
        )
    return np.stack(augmented)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_prepared_arrays(
    manifest_path: Path, partitions: set[str], labels: Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if not partitions or not partitions.issubset({"train", "validation", "test", "demo"}):
        raise TrainingError("unsupported prepared partition")
    root = Path(manifest_path).parent
    selected: list[dict[str, Any]] = []
    features: list[np.ndarray] = []
    targets: list[int] = []
    try:
        lines = Path(manifest_path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise TrainingError(f"cannot read prepared manifest: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingError(f"prepared manifest JSON error at line {line_number}") from exc
        if record.get("split") not in partitions:
            continue
        relative = record.get("feature_path")
        expected_hash = record.get("feature_sha256")
        target = record.get("target_index")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise TrainingError(f"prepared record contract drift at line {line_number}")
        path = root / Path(relative)
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError as exc:
            raise TrainingError("prepared feature path escapes root") from exc
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise TrainingError(f"prepared feature hash drift: {relative}")
        try:
            feature = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise TrainingError(f"cannot load prepared feature: {relative}") from exc
        if feature.shape != (1, 64, 96) or feature.dtype != np.float32 or not np.isfinite(feature).all():
            raise TrainingError(f"prepared feature tensor drift: {relative}")
        if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < 8:
            raise TrainingError(f"prepared target drift: {relative}")
        if labels is not None and record.get("target_id") != labels[target]:
            raise TrainingError(f"prepared class map drift: {relative}")
        selected.append(record)
        features.append(feature)
        targets.append(target)
    if not selected:
        raise TrainingError("prepared partition is empty")
    return np.stack(features), np.asarray(targets, dtype=np.int64), selected


def compute_class_weights(targets: np.ndarray, class_count: int) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(targets, minlength=class_count)
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        raise TrainingError("train split missing classes: " + ",".join(map(str, missing)))
    weights = len(targets) / (class_count * counts.astype(np.float64))
    return weights.astype(np.float32)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_confusion_png(path: Path, confusion: np.ndarray) -> None:
    path = Path(path)
    if path.exists():
        raise TrainingError(f"confusion PNG exists: {path}")
    values = np.asarray(confusion, dtype=np.float64)
    if values.shape != (8, 8) or not np.isfinite(values).all() or (values < 0).any():
        raise TrainingError("confusion matrix shape drift")
    maximum = float(values.max())
    normalized = values / maximum if maximum else values
    cell = 24
    pixels = np.repeat(np.repeat(normalized, cell, axis=0), cell, axis=1)
    image = np.empty((pixels.shape[0], pixels.shape[1], 3), dtype=np.uint8)
    image[..., 0] = np.rint(255 * pixels).astype(np.uint8)
    image[..., 1] = np.rint(80 * (1.0 - pixels)).astype(np.uint8)
    image[..., 2] = np.rint(255 * (1.0 - pixels)).astype(np.uint8)
    raw = b"".join(b"\x00" + row.tobytes() for row in image)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", image.shape[1], image.shape[0], 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, 9))
    payload += _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise TrainingError(f"confusion PNG temporary exists: {temporary}")
    temporary.write_bytes(payload)
    _publish_create_only(temporary, path)


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _default_model_config() -> ModelConfig:
    return load_model_config(Path(__file__).resolve().parents[2] / "config" / "soundjie_d3_model.json")


def overfit_arrays(
    features: np.ndarray, labels: np.ndarray, seed: int, epochs: int
) -> OverfitResult:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.shape != (len(labels), 1, 64, 96) or not len(labels):
        raise TrainingError("overfit array shape drift")
    _set_deterministic(seed)
    model = SoundjieCNN(_default_model_config())
    model.train()
    inputs = torch.from_numpy(features.copy())
    targets = torch.from_numpy(labels.copy())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        if not torch.isfinite(loss):
            raise TrainingError("non-finite overfit loss")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(inputs).cpu().numpy()
    accuracy = float((logits.argmax(axis=1) == labels).mean())
    return OverfitResult(accuracy, logits)


def save_checkpoint_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise TrainingError(f"checkpoint exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise TrainingError(f"checkpoint temporary exists: {temporary}")
    try:
        torch.save(dict(payload), temporary)
        # Windows FlushFileBuffers requires a handle opened for writing.
        with temporary.open("r+b") as stream:
            os.fsync(stream.fileno())
        _publish_create_only(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _publish_create_only(temporary: Path, destination: Path) -> None:
    """Publish without any check-then-overwrite race."""
    try:
        try:
            if os.name == "nt":
                # MoveFile on Windows is atomic and fails when destination exists.
                os.rename(temporary, destination)
            else:
                os.link(temporary, destination)
        except FileExistsError as exc:
            raise TrainingError(f"artifact exists: {destination}") from exc
        except OSError as exc:
            # Fail closed instead of falling back to a visible, non-atomic copy.
            raise TrainingError(f"create-only publish unsupported: {exc}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_create_only(path: Path, value: Any) -> None:
    path = Path(path)
    if path.exists():
        raise TrainingError(f"artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise TrainingError(f"artifact temporary exists: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _publish_create_only(temporary, path)


def _save_npy_create_only(path: Path, value: np.ndarray) -> None:
    path = Path(path)
    if path.exists():
        raise TrainingError(f"artifact exists: {path}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise TrainingError(f"artifact temporary exists: {temporary}")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    _publish_create_only(temporary, path)


def _report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    return {
        "accuracy": report.accuracy,
        "per_class": report.per_class,
        "confusion_matrix": [list(row) for row in report.confusion_matrix],
    }


def _write_confusion_csv(path: Path, labels: Sequence[str], report: EvaluationReport) -> None:
    if path.exists():
        raise TrainingError(f"artifact exists: {path}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise TrainingError(f"artifact temporary exists: {temporary}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, report.confusion_matrix, strict=True):
            writer.writerow([label, *row])
        stream.flush()
        os.fsync(stream.fileno())
    _publish_create_only(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingError(f"JSON root must be an object: {path}")
    return value


def _predict(model: SoundjieCNN, features: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(features, dtype=np.float32))).cpu().numpy()


def verify_prepared_contract(
    manifest: Path,
    seed: int,
    labels: Sequence[str],
    model_config: ModelConfig,
    repo: Path,
) -> dict[str, Any]:
    manifest = Path(manifest)
    class_map_path = repo / "config" / "soundjie_d3_classes.json"
    frontend_path = repo / "config" / "logmel_frontend_v1.json"
    labels_path = repo / "models" / "soundjie_d3" / "labels.txt"
    class_map_hash = _sha256_file(class_map_path)
    frontend_hash = _sha256_file(frontend_path)
    validate_training_contract(
        {"class_map_sha256": class_map_hash, "frontend_sha256": frontend_hash},
        model_config.class_map_sha256,
        model_config.frontend_sha256,
    )
    if labels_path.read_text(encoding="utf-8").splitlines() != list(labels):
        raise TrainingError("labels file drift")
    class_map = _load_json(class_map_path)
    mapped_labels = [
        item.get("id")
        for item in sorted(class_map.get("classes", []), key=lambda item: item.get("index", -1))
    ]
    if mapped_labels != list(labels):
        raise TrainingError("class map and labels drift")
    audit = _load_json(manifest.parent / "preparation-audit.json")
    manifest_hash = _sha256_file(manifest)
    if audit.get("manifest_sha256") != manifest_hash:
        raise TrainingError("prepared manifest hash drift")
    if audit.get("frontend_sha256") != frontend_hash:
        raise TrainingError("frontend drift")
    dataset_root = manifest.parent.parent.parent
    split_manifest = dataset_root / "manifests" / "splits.jsonl"
    split_hash = _sha256_file(split_manifest)
    if audit.get("split_manifest_sha256") != split_hash:
        raise TrainingError("split manifest drift")
    preparation_payload = {
        "schema": 1,
        "frontend_sha256": frontend_hash,
        "seed": seed,
        "split_manifest_sha256": split_hash,
        "resampler": "scipy.resample_poly-kaiser5-float64-v1",
        "window_samples": 16000,
        "feature_shape": [1, 64, 96],
    }
    preparation_hash = _canonical_sha256(preparation_payload)
    if audit.get("preparation_contract_sha256") != preparation_hash:
        raise TrainingError("preparation contract drift")
    return {
        "prepared_manifest_sha256": manifest_hash,
        "split_manifest_sha256": split_hash,
        "preparation_contract_sha256": preparation_hash,
        "frontend_sha256": frontend_hash,
        "class_map_sha256": class_map_hash,
        "labels_sha256": _sha256_file(labels_path),
        "class_mapping_verified_by_target_id": True,
    }


def train_model(
    config: Mapping[str, Any], manifest: Path, output_dir: Path
) -> TrainingResult:
    """Train from a verified prepared manifest and create a new audited artifact set."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise TrainingError(f"output directory exists: {output_dir}")
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise TrainingError("training config missing training object")
    required = {"batch_size", "epochs", "overfit_epochs", "learning_rate", "weight_decay", "augmentation_partition"}
    if set(training) != required or training["augmentation_partition"] != "train_only":
        raise TrainingError("training config drift")
    augmentation = config.get("augmentation")
    augmentation_keys = {
        "partition", "gain_db", "time_shift_frames", "reverb_decay",
        "reverb_taps", "background_mix",
    }
    if not isinstance(augmentation, Mapping) or set(augmentation) != augmentation_keys:
        raise TrainingError("augmentation config drift")
    if augmentation["partition"] != "train_only":
        raise TrainingError("augmentation partition drift")
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TrainingError("training seed drift")
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    if batch_size <= 0 or epochs <= 0:
        raise TrainingError("training dimensions must be positive")

    repo = Path(__file__).resolve().parents[2]
    model_config_path = repo / "config" / "soundjie_d3_model.json"
    class_map_path = repo / "config" / "soundjie_d3_classes.json"
    frontend_path = repo / "config" / "logmel_frontend_v1.json"
    labels_path = repo / "models" / "soundjie_d3" / "labels.txt"
    labels = labels_path.read_text(encoding="utf-8").splitlines()
    if len(labels) != 8:
        raise TrainingError("labels file drift")
    model_config = load_model_config(model_config_path)
    prepared_contract = verify_prepared_contract(Path(manifest), seed, labels, model_config, repo)

    train_x, train_y, train_records = load_prepared_arrays(Path(manifest), {"train"}, labels)
    validation_x, validation_y, validation_records = load_prepared_arrays(Path(manifest), {"validation"}, labels)
    test_x, test_y, test_records = load_prepared_arrays(Path(manifest), {"test"}, labels)
    weights = compute_class_weights(train_y, 8)
    background_pool = train_x[train_y == 7]

    _set_deterministic(seed)
    model = SoundjieCNN(model_config)
    audit_model_operators(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights))
    generator = torch.Generator().manual_seed(seed)
    targets = torch.from_numpy(train_y)
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        epoch_features = augment_training_epoch(train_x, background_pool, augmentation, seed, epoch)
        inputs = torch.from_numpy(epoch_features)
        permutation = torch.randperm(len(targets), generator=generator)
        total_loss = 0.0
        for start in range(0, len(targets), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs[indices]), targets[indices])
            if not torch.isfinite(loss):
                raise TrainingError("non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        validation_logits = _predict(model, validation_x)
        validation_accuracy = float((validation_logits.argmax(axis=1) == validation_y).mean())
        history.append({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(targets),
            "validation_accuracy": validation_accuracy,
        })

    train_logits = _predict(model, train_x)
    validation_logits = _predict(model, validation_x)
    test_logits = _predict(model, test_x)
    if not all(np.isfinite(value).all() for value in (train_logits, validation_logits, test_logits)):
        raise TrainingError("non-finite final logits")
    combined_predictions = np.concatenate((validation_logits.argmax(1), test_logits.argmax(1)))
    if len(np.unique(combined_predictions)) < 2:
        raise TrainingError("class collapse: evaluation predicts only one class")
    validation_report = evaluate_logits(validation_logits, validation_y, labels)
    test_report = evaluate_logits(test_logits, test_y, labels)

    output_dir.mkdir(parents=True, exist_ok=False)
    upstream = {
        **prepared_contract,
        "model_config_sha256": _sha256_file(model_config_path),
        "training_config_sha256": _canonical_sha256(config),
    }
    checkpoint_path = output_dir / "soundjie_d3_float32.pt"
    save_checkpoint_create_only(checkpoint_path, {
        "schema": 1,
        "state_dict": model.state_dict(),
        "labels": labels,
        "seed": seed,
        "upstream": upstream,
    })
    _save_npy_create_only(output_dir / "train-logits.npy", train_logits)
    _save_npy_create_only(output_dir / "validation-logits.npy", validation_logits)
    _save_npy_create_only(output_dir / "test-logits.npy", test_logits)
    reports = {
        "schema": 1,
        "validation": _report_to_dict(validation_report),
        "test": _report_to_dict(test_report),
        "evaluation_predicted_class_count": int(len(np.unique(combined_predictions))),
    }
    _write_json_create_only(output_dir / "metrics.json", reports)
    _write_json_create_only(output_dir / "training-history.json", history)
    _write_confusion_csv(output_dir / "test-confusion.csv", labels, test_report)
    write_confusion_png(output_dir / "test-confusion.png", np.asarray(test_report.confusion_matrix))
    training_manifest = {
        "schema": 1,
        "seed": seed,
        "records": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "dependencies": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "upstream": upstream,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": _sha256_file(checkpoint_path),
        },
        "metrics": reports,
    }
    manifest_path = output_dir / "training-manifest.json"
    _write_json_create_only(manifest_path, training_manifest)
    return TrainingResult(checkpoint_path, manifest_path, validation_report, test_report)


def _run_overfit(args: argparse.Namespace) -> int:
    config = _load_json(args.config)
    prepared_manifest = Path(args.prepared_root) / "prepared-manifest.jsonl"
    seed = int(config["seed"])
    repo = Path(__file__).resolve().parents[2]
    labels = (repo / "models" / "soundjie_d3" / "labels.txt").read_text(encoding="utf-8").splitlines()
    model_config = load_model_config(repo / "config" / "soundjie_d3_model.json")
    prepared_contract = verify_prepared_contract(prepared_manifest, seed, labels, model_config, repo)
    features, targets, _ = load_prepared_arrays(prepared_manifest, {"train"}, labels)
    selected: list[int] = []
    per_class = args.samples // 8
    if args.samples % 8 or per_class < 1:
        raise TrainingError("overfit samples must be a positive multiple of 8")
    for target in range(8):
        indices = np.flatnonzero(targets == target)
        if len(indices) < per_class:
            raise TrainingError(f"insufficient overfit samples for class {target}")
        selected.extend(map(int, indices[:per_class]))
    epochs = int(config["training"]["overfit_epochs"])
    result = overfit_arrays(features[selected], targets[selected], seed, epochs)
    if result.accuracy < 0.95:
        raise TrainingError("overfit gate failed")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise TrainingError(f"output directory exists: {output_dir}")
    output_dir.mkdir(parents=True)
    _save_npy_create_only(output_dir / "overfit-logits.npy", result.logits)
    _write_json_create_only(output_dir / "overfit-report.json", {
        "schema": 1, "samples": args.samples, "epochs": epochs,
        "seed": seed, "accuracy": result.accuracy,
        "upstream": prepared_contract,
    })
    print(f"OVERFIT_ACCURACY={result.accuracy:.6f}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    config = _load_json(args.config)
    manifest = Path(args.prepared_root) / "prepared-manifest.jsonl"
    result = train_model(config, manifest, args.output_dir)
    print(f"VALIDATION_ACCURACY={result.validation.accuracy:.6f}")
    print(f"TEST_ACCURACY={result.test.accuracy:.6f}")
    print(f"TRAINING_MANIFEST={result.manifest}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    default_prepared = r"F:\MLK\DPU_ZYNQ7000\datasets\soundjie_v1\processed\smoke-a"
    overfit = subparsers.add_parser("overfit")
    overfit.add_argument("--config", type=Path, required=True)
    overfit.add_argument("--prepared-root", type=Path, default=Path(default_prepared))
    overfit.add_argument("--samples", type=int, default=32)
    overfit.add_argument("--output-dir", type=Path, required=True)
    overfit.set_defaults(handler=_run_overfit)
    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--prepared-root", type=Path, default=Path(default_prepared))
    train.add_argument("--output-dir", type=Path, required=True)
    train.set_defaults(handler=_run_train)
    args = parser.parse_args(argv)
    if args.version:
        print("soundjie-d3-training schema=1 torch=" + torch.__version__)
        return 0
    if not hasattr(args, "handler"):
        parser.error("a command is required")
    try:
        return int(args.handler(args))
    except TrainingError as exc:
        parser.exit(1, f"TRAINING_FAILED: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
