#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import audit_logmel_d2_tensors as d2_audit
import logmel_reference as reference


CASES = (
    "silence", "impulse", "sine_440", "dual_tone", "noise_seeded",
    "clipping", "stereo_antiphase",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path, root):
    path = Path(path)
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_inputs(config_path: Path, real_wav: Path, output_dir: Path):
    config_path, real_wav, output_dir = map(Path, (config_path, real_wav, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    config = reference.load_frontend_config(config_path)
    _, real_metadata = reference.read_pcm_window(real_wav, 0, config.window_samples)
    last = real_metadata.total_samples - config.window_samples
    starts = sorted({0, last // 2, last})
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        synthetic_dir, real_dir = temporary / "synthetic", temporary / "real"
        synthetic_dir.mkdir()
        real_dir.mkdir()
        synthetic_records = []
        for name in CASES:
            samples, metadata = reference.synthesize_case(name, config)
            path = synthetic_dir / f"{name}.wav"
            path.write_bytes(reference.encode_pcm_wav(samples, metadata["channels"]))
            synthetic_records.append({**metadata, **record(path, temporary)})

        real_records = []
        raw = Path(real_wav).read_bytes()
        pcm = np.frombuffer(
            raw[real_metadata.data_offset:real_metadata.data_offset + real_metadata.data_bytes],
            dtype="<i2").reshape(real_metadata.total_samples, real_metadata.channels)
        for start in starts:
            samples = pcm[start:start + config.window_samples].reshape(-1)
            path = real_dir / f"start-{start:010d}.wav"
            path.write_bytes(reference.encode_pcm_wav(
                samples, real_metadata.channels, real_metadata.sample_rate_hz))
            real_records.append({"start_sample": start, **record(path, temporary)})

        manifest = {
            "schema": 1,
            "config": {"sha256": config.source_sha256},
            "synthetic_cases": synthetic_records,
            "real_wav": {
                "source_name": real_wav.name,
                "size": real_wav.stat().st_size,
                "sha256": sha256(real_wav),
                "channels": real_metadata.channels,
                "sample_rate_hz": real_metadata.sample_rate_hz,
                "bits_per_sample": real_metadata.bits_per_sample,
                "total_samples": real_metadata.total_samples,
                "window_start_samples": starts,
                "windows": real_records,
            },
        }
        (temporary / "source-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n")
        os.replace(temporary, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def float_stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    valid = values[finite]
    return {
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "mean": float(valid.mean()) if valid.size else None,
        "rms": float(np.sqrt(np.mean(valid * valid))) if valid.size else None,
        "nonfinite": int(values.size - valid.size),
    }


def int8_stats(values):
    values = np.asarray(values, dtype=np.int8)
    count = values.size
    saturated_low = int(np.count_nonzero(values == -128))
    saturated_high = int(np.count_nonzero(values == 127))
    return {
        "min": int(values.min()), "max": int(values.max()),
        "unique_values": int(np.unique(values).size),
        "zero_count": int(np.count_nonzero(values == 0)),
        "negative_count": int(np.count_nonzero(values < 0)),
        "positive_count": int(np.count_nonzero(values > 0)),
        "saturated_low": saturated_low,
        "saturated_high": saturated_high,
        "saturated_low_rate": saturated_low / count,
        "saturated_high_rate": saturated_high / count,
    }


def _write_case(root, name, source, samples, config, input_scale):
    features = reference.compute_logmel(samples, config)
    tensor = reference.quantize_int8(features, input_scale)
    cases = root / "cases"
    float_path, int8_path = cases / f"{name}.f32le", cases / f"{name}.i8"
    json_path = cases / f"{name}.json"
    float_path.write_bytes(features.astype("<f4", copy=False).tobytes(order="C"))
    int8_path.write_bytes(tensor.tobytes(order="C"))
    item = {
        "name": name,
        "source": source,
        "float": {**record(float_path, root), **float_stats(features)},
        "int8": {**record(int8_path, root), **int8_stats(tensor)},
    }
    json_path.write_text(
        json.dumps(item, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    item["report"] = record(json_path, root)
    return item, tensor


def build_golden(config_path: Path, real_wav: Path, input_scale: float,
                 scale_usage: str, output_dir: Path):
    config_path, real_wav, output_dir = map(Path, (config_path, real_wav, output_dir))
    input_scale = float(input_scale)
    if scale_usage not in ("parity_test_only", "runtime_dpu"):
        raise ValueError("invalid scale usage")
    if scale_usage == "runtime_dpu" and input_scale != 128.0:
        raise ValueError("runtime_dpu requires input scale 128")
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing output: {output_dir}")
    config = reference.load_frontend_config(config_path)
    reference.quantize_int8(np.zeros((64, 96), dtype=np.float32), input_scale)
    _, real_metadata = reference.read_pcm_window(real_wav, 0, config.window_samples)
    last = real_metadata.total_samples - config.window_samples
    starts = sorted({0, last // 2, last})
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        (temporary / "cases").mkdir()
        items = []
        real_tensors = []
        for name in CASES:
            pcm, metadata = reference.synthesize_case(name, config)
            wav_path = temporary / f".{name}.wav"
            wav_path.write_bytes(reference.encode_pcm_wav(pcm, metadata["channels"]))
            samples, _ = reference.read_pcm_window(wav_path, 0, config.window_samples)
            item, _ = _write_case(
                temporary, name, {"kind": "synthetic", "case": name},
                samples, config, input_scale)
            items.append(item)
            wav_path.unlink()
        for start in starts:
            samples, _ = reference.read_pcm_window(
                real_wav, start, config.window_samples)
            name = f"xvf3800-start-{start:010d}"
            item, tensor = _write_case(
                temporary, name,
                {"kind": "real_wav", "source_name": real_wav.name,
                 "source_sha256": sha256(real_wav), "start_sample": start},
                samples, config, input_scale)
            items.append(item)
            real_tensors.append({"name": name, "tensor": tensor})
        quality = (d2_audit.audit_real_windows(real_tensors)
                   if scale_usage == "runtime_dpu" else None)
        manifest = {
            "schema": 2,
            "config_sha256": config.source_sha256,
            "profile_id": config.profile_id,
            "profile_domain": config.profile_domain,
            "profile_lower_bound": config.profile_lower_bound,
            "profile_upper_bound": config.profile_upper_bound,
            "profile_mean": list(config.profile_mean),
            "profile_std": list(config.profile_std),
            "input_scale": input_scale,
            "scale_usage": scale_usage,
            "tensor_shape": list(config.tensor_shape),
            "cases": items,
            "real_window_quality": quality,
        }
        d2_audit.audit_manifest_contract(manifest)
        (temporary / "golden-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n")
        os.replace(temporary, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inputs = subparsers.add_parser("inputs")
    inputs.add_argument("--config", required=True, type=Path)
    inputs.add_argument("--real-wav", required=True, type=Path)
    inputs.add_argument("--output-dir", required=True, type=Path)
    build = subparsers.add_parser("build")
    build.add_argument("--config", required=True, type=Path)
    build.add_argument("--real-wav", required=True, type=Path)
    build.add_argument("--input-scale", required=True, type=float)
    build.add_argument("--scale-usage", required=True,
                       choices=("parity_test_only", "runtime_dpu"))
    build.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "inputs":
        result = build_inputs(args.config, args.real_wav, args.output_dir)
        print(json.dumps({"schema": result["schema"], "output": str(args.output_dir)}))
    elif args.command == "build":
        result = build_golden(args.config, args.real_wav, args.input_scale,
                              args.scale_usage, args.output_dir)
        print(json.dumps({"schema": result["schema"], "case_count": len(result["cases"])}))


if __name__ == "__main__":
    main()
