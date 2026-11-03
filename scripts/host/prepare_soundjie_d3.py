#!/usr/bin/env python3
"""Deterministic audio preparation using the board-validated D2 Log-Mel frontend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:
    from scripts.host.logmel_reference import (
        FrontendConfig,
        compute_logmel,
        encode_pcm_wav,
        load_frontend_config,
    )
    from scripts.host.soundjie_d3_manifest import sha256_file
except ModuleNotFoundError:
    from logmel_reference import FrontendConfig, compute_logmel, encode_pcm_wav, load_frontend_config
    from soundjie_d3_manifest import sha256_file


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


class AudioPreparationError(ValueError):
    """Audio cannot enter the reproducible D3 preparation pipeline."""


@dataclass(frozen=True)
class AudioBuffer:
    sample_rate: int
    samples: np.ndarray


@dataclass(frozen=True)
class WindowPolicy:
    mode: str
    seed: int
    window_samples: int = 16000


@dataclass(frozen=True)
class Window:
    start_sample: int
    samples: np.ndarray


@dataclass(frozen=True)
class PreparedRecord:
    recording_id: str
    split: str
    target_id: str
    target_index: int
    source_group: str
    source_path: str
    source_sha256: str
    start_sample_16k: int
    wav_path: str
    wav_sha256: str
    feature_path: str
    feature_sha256: str


@dataclass(frozen=True)
class PreparedManifest:
    records: Tuple[PreparedRecord, ...]
    frontend_sha256: str
    split_manifest_sha256: str
    preparation_contract_sha256: str
    manifest_sha256: str


def decode_audio(path: Path) -> AudioBuffer:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise AudioPreparationError(f"audio is missing or empty: {path}")
    try:
        samples, rate = sf.read(str(path), dtype="float64", always_2d=True)
    except (RuntimeError, ValueError, OSError) as exc:
        raise AudioPreparationError(f"cannot decode audio: {path}: {exc}") from exc
    if rate <= 0 or samples.ndim != 2 or samples.shape[0] == 0 or samples.shape[1] not in (1, 2):
        raise AudioPreparationError("unsupported audio shape or sample rate")
    if not np.isfinite(samples).all():
        raise AudioPreparationError("audio contains nonfinite samples")
    return AudioBuffer(int(rate), np.asarray(samples, dtype=np.float64))


def resample_mono_16k(audio: AudioBuffer) -> np.ndarray:
    samples = np.asarray(audio.samples, dtype=np.float64)
    if audio.sample_rate <= 0 or samples.ndim != 2 or samples.shape[1] not in (1, 2):
        raise AudioPreparationError("invalid AudioBuffer")
    if samples.shape[0] == 0 or not np.isfinite(samples).all():
        raise AudioPreparationError("audio samples must be finite and non-empty")
    mono = samples.mean(axis=1, dtype=np.float64)
    common = gcd(int(audio.sample_rate), 16000)
    up, down = 16000 // common, int(audio.sample_rate) // common
    output = resample_poly(mono, up, down, window=("kaiser", 5.0), padtype="constant")
    expected = int(math.ceil(len(mono) * 16000.0 / audio.sample_rate))
    if len(output) < expected:
        output = np.pad(output, (0, expected - len(output)))
    output = np.asarray(output[:expected], dtype=np.float64)
    if not np.isfinite(output).all():
        raise AudioPreparationError("resampling produced nonfinite samples")
    return output


def _assert_signal(samples: np.ndarray) -> None:
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise AudioPreparationError("audio window must be finite mono")
    if samples.size == 0 or float(np.max(samples) - np.min(samples)) <= 1e-7:
        raise AudioPreparationError("silent or constant audio is rejected")
    if float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) <= 1e-7:
        raise AudioPreparationError("silent audio is rejected")


def make_windows(samples: np.ndarray, policy: WindowPolicy) -> Iterator[Window]:
    samples = np.asarray(samples, dtype=np.float64)
    _assert_signal(samples)
    count = policy.window_samples
    if count != 16000 or policy.mode not in ("energy_center", "seeded"):
        raise AudioPreparationError("unsupported window policy")
    if samples.size <= count:
        left = (count - samples.size) // 2
        padded = np.pad(samples, (left, count - samples.size - left))
        yield Window(-left, np.asarray(padded, dtype=np.float64))
        return
    maximum_start = samples.size - count
    if policy.mode == "seeded":
        start = int(policy.seed % (maximum_start + 1))
    else:
        squared = np.square(samples, dtype=np.float64)
        cumulative = np.concatenate((np.zeros(1), np.cumsum(squared)))
        energy = cumulative[count:] - cumulative[:-count]
        start = int(np.argmax(energy))
    window = np.asarray(samples[start : start + count], dtype=np.float64)
    _assert_signal(window)
    yield Window(start, window)


def compute_feature(samples: np.ndarray, frontend: FrontendConfig) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.shape != (16000,):
        raise AudioPreparationError("audio window must contain exactly 16000 samples")
    _assert_signal(samples)
    try:
        feature = compute_logmel(samples, frontend).astype("<f4", copy=False)
    except ValueError as exc:
        raise AudioPreparationError(f"D2 frontend rejected audio: {exc}") from exc
    if feature.shape != (64, 96) or not np.isfinite(feature).all():
        raise AudioPreparationError("D2 feature must be finite [64,96]")
    if float(np.max(feature) - np.min(feature)) <= 1e-7:
        raise AudioPreparationError("constant feature is rejected")
    return feature.reshape(1, 64, 96)


def preparation_contract_hash(frontend: FrontendConfig, seed: int, split_hash: str) -> str:
    payload = {
        "schema": 1,
        "frontend_sha256": frontend.source_sha256,
        "seed": seed,
        "split_manifest_sha256": split_hash,
        "resampler": "scipy.resample_poly-kaiser5-float64-v1",
        "window_samples": 16000,
        "feature_shape": [1, 64, 96],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_demo_isolation(
    split_records: Sequence[Mapping[str, Any]], demo_records: Sequence[Mapping[str, Any]]
) -> None:
    fields = ("recording_id", "path", "source_group", "speaker_or_uploader", "sha256")
    demo_values = {field: {str(item[field]) for item in demo_records} for field in fields}
    for item in split_records:
        for field in fields:
            if str(item[field]) in demo_values[field]:
                raise AudioPreparationError(f"demo leakage through {field}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise AudioPreparationError("manifest row must be an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AudioPreparationError(f"cannot read manifest: {path}: {exc}") from exc
    if not rows:
        raise AudioPreparationError(f"manifest is empty: {path}")
    return rows


def _seed_for(seed: int, recording_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{recording_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _balanced_subset(rows: Sequence[dict[str, Any]], limits: Mapping[str, int]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["split"]), str(row["target_id"]))].append(row)
    selected = []
    for split in ("train", "validation", "test", "demo"):
        for class_id in CLASS_IDS:
            values = sorted(buckets[(split, class_id)], key=lambda item: item["recording_id"])
            limit = limits[split]
            if len(values) < limit:
                raise AudioPreparationError(f"not enough {split} recordings for {class_id}")
            selected.extend(values[:limit])
    return selected


def _float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    scaled = np.sign(samples) * np.floor(np.abs(samples) * 32768.0 + 0.5)
    return np.clip(scaled, -32768, 32767).astype("<i2")


def prepare_dataset(
    split_path: Path,
    demo_path: Path,
    frontend_path: Path,
    training_config_path: Path,
    output_root: Path,
) -> PreparedManifest:
    output_root = Path(output_root)
    if output_root.exists():
        raise AudioPreparationError(f"output already exists: {output_root}")
    split_rows = _read_jsonl(split_path)
    demo_rows = _read_jsonl(demo_path)
    validate_demo_isolation(split_rows, demo_rows)
    for row in demo_rows:
        row["split"] = "demo"
    config = json.loads(Path(training_config_path).read_text(encoding="utf-8"))
    limits = config["preparation"]["max_per_class"]
    if set(limits) != {"train", "validation", "test", "demo"}:
        raise AudioPreparationError("invalid preparation limits")
    selected = _balanced_subset(split_rows + demo_rows, limits)
    frontend = load_frontend_config(frontend_path)
    split_hash = sha256_file(split_path)
    contract_hash = preparation_contract_hash(frontend, int(config["seed"]), split_hash)
    temporary = output_root.with_name(output_root.name + ".partial")
    if temporary.exists():
        raise AudioPreparationError(f"partial output already exists: {temporary}")
    temporary.mkdir(parents=True)
    prepared = []
    try:
        for row in selected:
            split = str(row["split"])
            class_id = str(row["target_id"])
            recording_id = str(row["recording_id"])
            audio = resample_mono_16k(decode_audio(Path(row["path"])))
            mode = "energy_center" if int(row["target_index"]) <= 5 else "seeded"
            window = next(make_windows(audio, WindowPolicy(mode, _seed_for(config["seed"], recording_id))))
            feature = compute_feature(window.samples, frontend)
            stem = hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:20]
            wav_relative = Path("wav16k") / split / class_id / f"{stem}.wav"
            feature_relative = Path("features") / split / class_id / f"{stem}.npy"
            wav_target = temporary / wav_relative
            feature_target = temporary / feature_relative
            wav_target.parent.mkdir(parents=True, exist_ok=True)
            feature_target.parent.mkdir(parents=True, exist_ok=True)
            wav_target.write_bytes(encode_pcm_wav(_float_to_pcm16(window.samples), 1, 16000))
            with feature_target.open("xb") as stream:
                np.save(stream, feature, allow_pickle=False)
            prepared.append(
                PreparedRecord(
                    recording_id,
                    split,
                    class_id,
                    int(row["target_index"]),
                    str(row["source_group"]),
                    str(row["path"]),
                    str(row["sha256"]),
                    window.start_sample,
                    wav_relative.as_posix(),
                    sha256_file(wav_target),
                    feature_relative.as_posix(),
                    sha256_file(feature_target),
                )
            )
        manifest_rows = [asdict(item) for item in sorted(prepared, key=lambda item: (item.split, item.target_index, item.recording_id))]
        body = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in manifest_rows)
        manifest_path = temporary / "prepared-manifest.jsonl"
        manifest_path.write_text(body, encoding="utf-8", newline="\n")
        manifest_hash = sha256_file(manifest_path)
        audit = {
            "schema": 1,
            "record_count": len(prepared),
            "frontend_sha256": frontend.source_sha256,
            "split_manifest_sha256": split_hash,
            "preparation_contract_sha256": contract_hash,
            "manifest_sha256": manifest_hash,
        }
        (temporary / "preparation-audit.json").write_text(
            json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        os.replace(temporary, output_root)
    except Exception:
        import shutil

        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return PreparedManifest(
        tuple(sorted(prepared, key=lambda item: (item.split, item.target_index, item.recording_id))),
        frontend.source_sha256,
        split_hash,
        contract_hash,
        manifest_hash,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--frontend", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_dataset(
        args.splits, args.demo, args.frontend, args.training_config, args.output_root
    )
    print(json.dumps({"records": len(result.records), "manifest_sha256": result.manifest_sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
