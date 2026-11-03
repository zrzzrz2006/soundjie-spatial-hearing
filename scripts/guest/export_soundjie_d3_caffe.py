#!/usr/bin/env python3
"""Fuse Soundjie D3 BatchNorm, write Caffe weights, and verify logits parity."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


EXPORTED_LAYERS = ("conv1", "conv2", "conv3", "conv4", "fc8")
EXPECTED_ARRAY_SHAPES = {
    "conv1.weight": (16, 1, 3, 3), "conv1.bias": (16,),
    "conv2.weight": (32, 16, 3, 3), "conv2.bias": (32,),
    "conv3.weight": (64, 32, 3, 3), "conv3.bias": (64,),
    "conv4.weight": (96, 64, 3, 3), "conv4.bias": (96,),
    "fc8.weight": (8, 2304), "fc8.bias": (8,),
}
EXPECTED_STATE_SHAPES = {
    "conv1.weight": (16, 1, 3, 3),
    "conv2.weight": (32, 16, 3, 3),
    "conv3.weight": (64, 32, 3, 3),
    "conv4.weight": (96, 64, 3, 3),
    "fc8.weight": (8, 2304),
    "fc8.bias": (8,),
}
for _index, _channels in enumerate((16, 32, 64, 96), 1):
    for _suffix in ("weight", "bias", "running_mean", "running_var"):
        EXPECTED_STATE_SHAPES["bn{}.{}".format(_index, _suffix)] = (_channels,)
    EXPECTED_STATE_SHAPES["bn{}.num_batches_tracked".format(_index)] = ()


class ExportError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_fused_arrays(arrays):
    if set(arrays) != set(EXPECTED_ARRAY_SHAPES):
        raise ExportError("weight contract keys differ")
    for name, expected in EXPECTED_ARRAY_SHAPES.items():
        value = np.asarray(arrays[name])
        if value.shape != expected or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ExportError("weight contract shape/dtype differs: " + name)


def validate_checkpoint_contract(checkpoint, manifest, labels):
    if checkpoint.get("schema") != 1 or manifest.get("schema") != 1:
        raise ExportError("checkpoint contract drift: schema")
    if checkpoint.get("labels") != list(labels):
        raise ExportError("checkpoint contract drift: labels")
    if checkpoint.get("upstream") != manifest.get("upstream"):
        raise ExportError("checkpoint contract drift: upstream")


def _tensor_array(value, name):
    try:
        return value.detach().cpu().numpy()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise ExportError("state dict contract tensor differs: " + name) from exc


def validate_state_dict(state_dict):
    if set(state_dict) != set(EXPECTED_STATE_SHAPES):
        raise ExportError("state dict contract keys differ")
    for name, expected_shape in EXPECTED_STATE_SHAPES.items():
        value = _tensor_array(state_dict[name], name)
        expected_dtype = np.int64 if name.endswith("num_batches_tracked") else np.float32
        if value.shape != expected_shape or value.dtype != expected_dtype:
            raise ExportError("state dict contract shape/dtype differs: " + name)
        if not np.isfinite(value).all():
            raise ExportError("state dict contract contains non-finite value: " + name)


def validate_source_files(upstream, model_config_path, prepared_manifest, labels_path, model_config):
    bindings = (
        ("model config", model_config_path, "model_config_sha256"),
        ("prepared manifest", prepared_manifest, "prepared_manifest_sha256"),
        ("labels", labels_path, "labels_sha256"),
    )
    for display, path, key in bindings:
        if sha256_file(path) != upstream.get(key):
            raise ExportError(display + " SHA drift")
    if model_config.class_map_sha256 != upstream.get("class_map_sha256"):
        raise ExportError("model config class map SHA drift")
    if model_config.frontend_sha256 != upstream.get("frontend_sha256"):
        raise ExportError("model config frontend SHA drift")


def fuse_state_dict(state_dict, epsilon=1e-5):
    validate_state_dict(state_dict)
    arrays = {}
    for index in range(1, 5):
        conv = "conv{}".format(index)
        bn = "bn{}".format(index)
        weight = state_dict[conv + ".weight"].detach().cpu().numpy().astype(np.float32)
        gamma = state_dict[bn + ".weight"].detach().cpu().numpy().astype(np.float32)
        beta = state_dict[bn + ".bias"].detach().cpu().numpy().astype(np.float32)
        mean = state_dict[bn + ".running_mean"].detach().cpu().numpy().astype(np.float32)
        variance = state_dict[bn + ".running_var"].detach().cpu().numpy().astype(np.float32)
        scale = gamma / np.sqrt(variance + np.float32(epsilon))
        arrays[conv + ".weight"] = (weight * scale[:, None, None, None]).astype(np.float32)
        arrays[conv + ".bias"] = (beta - mean * scale).astype(np.float32)
    arrays["fc8.weight"] = state_dict["fc8.weight"].detach().cpu().numpy().astype(np.float32)
    arrays["fc8.bias"] = state_dict["fc8.bias"].detach().cpu().numpy().astype(np.float32)
    validate_fused_arrays(arrays)
    return arrays


def assess_parity(reference, candidate, threshold=1e-4):
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if reference.shape != (16, 8) or candidate.shape != reference.shape:
        raise ExportError("parity shape drift")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ExportError("parity contains non-finite logits")
    difference = np.abs(reference - candidate)
    maximum = float(difference.max())
    reference_argmax = reference.argmax(axis=1)
    candidate_argmax = candidate.argmax(axis=1)
    argmax_matches = int(np.sum(reference_argmax == candidate_argmax))
    effective_threshold = float(np.float32(threshold))
    if not maximum < effective_threshold or argmax_matches != 16:
        raise ExportError("parity failed: max_abs_error={} argmax={}/16".format(maximum, argmax_matches))
    return {
        "max_abs_error": maximum,
        "per_sample_max_abs_error": [float(value) for value in difference.max(axis=1)],
        "reference_argmax": [int(value) for value in reference_argmax],
        "candidate_argmax": [int(value) for value in candidate_argmax],
        "argmax_matches": argmax_matches,
        "sample_count": 16,
    }


def export_weights(arrays, net):
    validate_fused_arrays(arrays)
    if set(net.params.keys()) != set(EXPORTED_LAYERS):
        raise ExportError("Caffe parameter layer contract drift")
    for layer in EXPORTED_LAYERS:
        blobs = net.params[layer]
        if len(blobs) != 2:
            raise ExportError("Caffe layer must expose weight and bias: " + layer)
        for blob, suffix in zip(blobs, ("weight", "bias")):
            value = arrays[layer + "." + suffix]
            if tuple(blob.data.shape) != value.shape:
                raise ExportError("Caffe blob shape drift: {}.{}".format(layer, suffix))
            blob.data[...] = value


def _publish_create_only(temporary, destination):
    temporary = Path(temporary)
    destination = Path(destination)
    try:
        if os.name == "nt":
            os.rename(str(temporary), str(destination))
        else:
            os.link(str(temporary), str(destination))
    except FileExistsError as exc:
        raise ExportError("artifact exists: " + str(destination)) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path, value):
    path = Path(path)
    if path.exists():
        raise ExportError("artifact exists: " + str(path))
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise ExportError("temporary exists: " + str(temporary))
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _publish_create_only(temporary, path)


def _save_npy(path, value):
    path = Path(path)
    if path.exists():
        raise ExportError("artifact exists: " + str(path))
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        np.save(stream, np.asarray(value, dtype=np.float32), allow_pickle=False)
    _publish_create_only(temporary, path)


def command_fuse(args):
    import torch
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from scripts.host import train_soundjie_d3 as training

    output = Path(args.output_dir)
    if output.exists():
        raise ExportError("output directory exists: " + str(output))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    manifest = json.loads(Path(args.training_manifest).read_text(encoding="utf-8"))
    labels = Path(args.labels).read_text(encoding="utf-8").splitlines()
    validate_checkpoint_contract(checkpoint, manifest, labels)
    if sha256_file(args.checkpoint) != manifest["checkpoint"]["sha256"]:
        raise ExportError("checkpoint SHA drift")
    model_config = training.load_model_config(args.model_config)
    validate_source_files(
        checkpoint["upstream"], args.model_config, args.prepared_manifest,
        args.labels, model_config,
    )
    arrays = fuse_state_dict(checkpoint["state_dict"])
    features, targets, _ = training.load_prepared_arrays(args.prepared_manifest, {"test"}, labels)
    selected = []
    for target in range(8):
        indices = np.flatnonzero(targets == target)
        if len(indices) < 2:
            raise ExportError("parity set needs two test samples per class")
        selected.extend(map(int, indices[:2]))
    parity_inputs = features[selected]
    model = training.SoundjieCNN(model_config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    reference_logits = training._predict(model, parity_inputs)
    if parity_inputs.shape != (16, 1, 64, 96) or reference_logits.shape != (16, 8):
        raise ExportError("parity reference shape drift")
    output.mkdir(parents=True)
    npz_path = output / "fused-weights.npz"
    temporary = output / "fused-weights.npz.partial"
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
    _publish_create_only(temporary, npz_path)
    inputs_path = output / "parity-inputs.npy"
    logits_path = output / "pytorch-logits.npy"
    _save_npy(inputs_path, parity_inputs)
    _save_npy(logits_path, reference_logits)
    _write_json(output / "fused-manifest.json", {
        "schema": 1,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "training_manifest_sha256": sha256_file(args.training_manifest),
        "deploy_sha256": sha256_file(args.deploy),
        "weights_sha256": sha256_file(npz_path),
        "parity_inputs_sha256": sha256_file(inputs_path),
        "pytorch_logits_sha256": sha256_file(logits_path),
        "parity_target_indices": [int(targets[index]) for index in selected],
        "upstream": checkpoint["upstream"],
    })
    print("FUSED_WEIGHTS=" + str(npz_path))


def command_caffe(args):
    try:
        import caffe
    except ImportError as exc:
        raise ExportError("Caffe Python is required: " + str(exc))
    output = Path(args.output_dir)
    if output.exists():
        raise ExportError("output directory exists: " + str(output))
    with np.load(args.weights, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    caffe.set_mode_cpu()
    net = caffe.Net(str(args.deploy), caffe.TEST)
    export_weights(arrays, net)
    inputs = np.load(args.inputs, allow_pickle=False)
    reference = np.load(args.reference_logits, allow_pickle=False)
    if inputs.shape != (16, 1, 64, 96):
        raise ExportError("parity input shape drift")
    candidate = []
    for feature in inputs:
        net.blobs["data"].data[...] = feature[None, ...]
        net.forward()
        candidate.append(np.asarray(net.blobs["fc8"].data[0], dtype=np.float32).copy())
    candidate = np.stack(candidate)
    report = assess_parity(reference, candidate)
    output.mkdir(parents=True)
    model = output / "soundjie_d3_float32.caffemodel"
    partial = output / "soundjie_d3_float32.caffemodel.partial"
    net.save(str(partial))
    _publish_create_only(partial, model)
    caffe_logits = output / "caffe-logits.npy"
    _save_npy(caffe_logits, candidate)
    report.update({
        "schema": 1,
        "caffemodel_sha256": sha256_file(model),
        "weights_sha256": sha256_file(args.weights),
        "deploy_sha256": sha256_file(args.deploy),
        "inputs_sha256": sha256_file(args.inputs),
        "reference_logits_sha256": sha256_file(args.reference_logits),
        "caffe_logits_sha256": sha256_file(caffe_logits),
    })
    _write_json(output / "caffe-export-manifest.json", report)
    print("MAX_ABS_ERROR={:.9g}".format(report["max_abs_error"]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    fuse = commands.add_parser("fuse")
    fuse.add_argument("--checkpoint", required=True, type=Path)
    fuse.add_argument("--training-manifest", required=True, type=Path)
    fuse.add_argument("--labels", required=True, type=Path)
    fuse.add_argument("--deploy", required=True, type=Path)
    fuse.add_argument("--model-config", required=True, type=Path)
    fuse.add_argument("--prepared-manifest", required=True, type=Path)
    fuse.add_argument("--output-dir", required=True, type=Path)
    fuse.set_defaults(handler=command_fuse)
    caffe_parser = commands.add_parser("caffe")
    caffe_parser.add_argument("--weights", required=True, type=Path)
    caffe_parser.add_argument("--deploy", required=True, type=Path)
    caffe_parser.add_argument("--inputs", required=True, type=Path)
    caffe_parser.add_argument("--reference-logits", required=True, type=Path)
    caffe_parser.add_argument("--output-dir", required=True, type=Path)
    caffe_parser.set_defaults(handler=command_caffe)
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.error("a command is required")
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        args.handler(args)
        return 0
    except ExportError as exc:
        print("EXPORT_FAILED: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
