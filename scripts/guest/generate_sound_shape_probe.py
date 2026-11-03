#!/usr/bin/env python3
"""Generate deterministic calibration inputs and bootstrap Caffe weights.

Calibration-only mode deliberately uses only the Python standard library so the
corpus can be reproduced before Caffe/DNNDK is installed.
"""

import argparse
import hashlib
import json
import math
import random
import struct
from pathlib import Path


SHAPE = [1, 1, 64, 96]
SAMPLE_COUNT = 16


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_values(index, seed):
    height, width = SHAPE[2], SHAPE[3]
    if index == 0:
        return "zero", [0.0] * (height * width)
    if index == 1:
        values = []
        for row in range(height):
            for col in range(width):
                tone = 0.55 * math.sin(2.0 * math.pi * col / 12.0)
                envelope = 0.35 + 0.65 * row / (height - 1)
                values.append(tone * envelope)
        return "pattern", values

    generator = random.Random(seed + index * 104729)
    values = []
    for row in range(height):
        for col in range(width):
            harmonic = 0.45 * math.sin(
                2.0 * math.pi * (index + 1) * col / width + row * 0.07
            )
            impulse = 0.35 if (col + index * 3) % (11 + index) == 0 else 0.0
            noise = generator.uniform(-0.12, 0.12)
            values.append(max(-1.0, min(1.0, harmonic + impulse + noise)))
    return "synthetic", values


def write_pgm(path, values):
    pixels = bytes(
        max(0, min(255, int(round((value + 1.0) * 127.5))))
        for value in values
    )
    path.write_bytes(b"P5\n96 64\n255\n" + pixels)


def generate_calibration(output_dir, seed):
    calibration = output_dir / "calibration"
    raw_dir = calibration / "raw"
    image_dir = calibration / "images"
    raw_dir.mkdir(parents=True, exist_ok=False)
    image_dir.mkdir(parents=True, exist_ok=False)

    samples = []
    all_min = float("inf")
    all_max = float("-inf")
    list_lines = []
    for index in range(SAMPLE_COUNT):
        kind, values = sample_values(index, seed)
        raw_name = "sample_{:03d}.bin".format(index)
        image_name = "sample_{:03d}.pgm".format(index)
        raw_path = raw_dir / raw_name
        image_path = image_dir / image_name
        raw_path.write_bytes(struct.pack("<{}f".format(len(values)), *values))
        write_pgm(image_path, values)
        sample_min, sample_max = min(values), max(values)
        all_min = min(all_min, sample_min)
        all_max = max(all_max, sample_max)
        samples.append({
            "index": index,
            "kind": kind,
            "file": "calibration/raw/{}".format(raw_name),
            "image": "calibration/images/{}".format(image_name),
            "min": round(sample_min, 9),
            "max": round(sample_max, 9),
            "sha256": sha256(raw_path),
            "image_sha256": sha256(image_path),
        })
        list_lines.append("{} 0\n".format(image_name))

    (calibration / "list.txt").write_text("".join(list_lines), encoding="utf-8")
    manifest = {
        "schema": 1,
        "seed": seed,
        "shape": SHAPE,
        "dtype": "float32-le",
        "sample_count": SAMPLE_COUNT,
        "min": round(all_min, 9),
        "max": round(all_max, 9),
        "samples": samples,
    }
    manifest_path = output_dir / "calibration-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def generate_caffe_weights(deploy, output_path, seed):
    try:
        import caffe
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "Caffe and NumPy are required unless --calibration-only is used: {}".format(exc)
        )

    caffe.set_mode_cpu()
    net = caffe.Net(str(deploy), caffe.TEST)
    rng = np.random.RandomState(seed)
    for layer_name in sorted(net.params.keys()):
        blobs = net.params[layer_name]
        if layer_name.startswith("bn"):
            blobs[0].data[...] = 0.0
            blobs[1].data[...] = 1.0
            blobs[2].data[...] = 1.0
        elif layer_name.startswith("scale"):
            blobs[0].data[...] = 1.0
            if len(blobs) > 1:
                blobs[1].data[...] = 0.0
        elif layer_name == "logits":
            blobs[0].data[...] = rng.uniform(-0.05, 0.05, blobs[0].data.shape)
            # Keep the zero-input probe observable after INT8 quantization.  A
            # network with zero bias throughout can legitimately emit eight
            # zero bytes for zero input, which is indistinguishable from a
            # runner that never copied DPU output.
            blobs[1].data[...] = np.linspace(-0.35, 0.35, blobs[1].data.size)
        elif layer_name.startswith("conv"):
            blobs[0].data[...] = rng.uniform(-0.05, 0.05, blobs[0].data.shape)
            blobs[1].data[...] = rng.uniform(-0.01, 0.01, blobs[1].data.shape)
        else:
            blobs[0].data[...] = rng.uniform(-0.05, 0.05, blobs[0].data.shape)
            for blob in blobs[1:]:
                blob.data[...] = 0.0
    net.save(str(output_path))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--deploy", type=Path)
    parser.add_argument("--calibration-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit("output directory already exists: {}".format(args.output_dir))
    args.output_dir.mkdir(parents=True)
    manifest = generate_calibration(args.output_dir, args.seed)
    if not args.calibration_only:
        default_deploy = Path(__file__).resolve().parents[2] / "models" / "sound_shape_probe" / "deploy.prototxt"
        deploy = (args.deploy or default_deploy).resolve()
        if not deploy.is_file():
            raise SystemExit("missing deploy prototxt: {}".format(deploy))
        generate_caffe_weights(deploy, args.output_dir / "float.caffemodel", args.seed)
    print("CALIBRATION_MANIFEST={}".format(manifest))


if __name__ == "__main__":
    main()
