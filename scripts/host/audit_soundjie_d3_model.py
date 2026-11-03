#!/usr/bin/env python3
"""Create-only calibration staging and fail-closed D3 compiler auditing."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


class ModelAuditError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_compiler_contract(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    compiler = value.get("compiler")
    if not isinstance(compiler, dict):
        raise ModelAuditError("missing compiler contract")
    return compiler


def expected_compile_evidence(contract):
    return {
        "dnnc_version": contract["accepted_dnnc_versions"][-1],
        "dpu_target": contract["dpu_target"],
        "dpu": contract["dpu"],
        "cpu_arch": contract["cpu_arch"],
        "abi": contract["abi"],
        "kernels": [contract["kernel"]],
        "kernel_type": contract["kernel_type"],
        "input_node": contract["input_node"],
        "output_node": contract["output_node"],
        "output_count": contract["output_count"],
        "unsupported_layers": [],
    }


def validate_compile_evidence(contract, evidence):
    if evidence.get("dnnc_version") not in contract["accepted_dnnc_versions"]:
        raise ModelAuditError("wrong compiler version")
    for key in ("dpu_target", "dpu", "cpu_arch", "abi", "kernel_type",
                "input_node", "output_node"):
        if evidence.get(key) != contract[key]:
            raise ModelAuditError("wrong compile evidence: " + key)
    if evidence.get("kernels") != [contract["kernel"]]:
        raise ModelAuditError("kernel count or name differs")
    if evidence.get("kernel_type") == "CPUKernel":
        raise ModelAuditError("CPUKernel is forbidden")
    if evidence.get("unsupported_layers"):
        raise ModelAuditError("unsupported layers present")
    if evidence.get("output_count") != contract["output_count"]:
        raise ModelAuditError("output count differs")


def validate_upstream_hashes(expected, actual):
    if set(expected) != set(actual):
        raise ModelAuditError("upstream hash drift: keys")
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ModelAuditError("upstream hash drift: " + key)


def validate_model_hash_chain(fused, caffe_export, calibration, metrics, actual):
    try:
        expected = {
            "fused_weights_sha256": fused["weights_sha256"],
            "float_caffemodel_sha256": caffe_export["caffemodel_sha256"],
            "actual_float_deploy_sha256": fused["deploy_sha256"],
            "export_float_deploy_sha256": fused["deploy_sha256"],
            "export_parity_inputs_sha256": fused["parity_inputs_sha256"],
            "metrics_parity_inputs_sha256": fused["parity_inputs_sha256"],
            "export_reference_logits_sha256": fused["pytorch_logits_sha256"],
            "metrics_reference_logits_sha256": fused["pytorch_logits_sha256"],
            "prepared_manifest_sha256": fused["upstream"]["prepared_manifest_sha256"],
            "quantized_caffemodel_sha256": metrics["quantized_caffemodel_sha256"],
        }
        observed = {
            "fused_weights_sha256": caffe_export["weights_sha256"],
            "float_caffemodel_sha256": actual["float_caffemodel_sha256"],
            "actual_float_deploy_sha256": actual["float_deploy_sha256"],
            "export_float_deploy_sha256": caffe_export["deploy_sha256"],
            "export_parity_inputs_sha256": caffe_export["inputs_sha256"],
            "metrics_parity_inputs_sha256": metrics["inputs_sha256"],
            "export_reference_logits_sha256": caffe_export["reference_logits_sha256"],
            "metrics_reference_logits_sha256": metrics["reference_logits_sha256"],
            "prepared_manifest_sha256": calibration["prepared_manifest_sha256"],
            "quantized_caffemodel_sha256": actual["quantized_caffemodel_sha256"],
        }
    except (KeyError, TypeError) as exc:
        raise ModelAuditError("upstream hash drift: incomplete chain") from exc
    validate_upstream_hashes(expected, observed)


def _macro_f1(predictions, targets, class_count):
    values = []
    for target in range(class_count):
        true_positive = int(((predictions == target) & (targets == target)).sum())
        false_positive = int(((predictions == target) & (targets != target)).sum())
        false_negative = int(((predictions != target) & (targets == target)).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        values.append(0.0 if denominator == 0 else 2.0 * true_positive / denominator)
    return sum(values) / class_count


def classification_metrics(reference_logits, quantized_logits, targets):
    import numpy as np
    reference_logits = np.asarray(reference_logits)
    quantized_logits = np.asarray(quantized_logits)
    targets = np.asarray(targets)
    if reference_logits.shape != quantized_logits.shape or reference_logits.ndim != 2:
        raise ModelAuditError("classification logits shape differs")
    if reference_logits.shape[1] != 8 or targets.shape != (reference_logits.shape[0],):
        raise ModelAuditError("classification output count differs")
    if not np.isfinite(reference_logits).all() or not np.isfinite(quantized_logits).all():
        raise ModelAuditError("classification logits contain non-finite values")
    if set(targets.tolist()) != set(range(8)):
        raise ModelAuditError("classification targets do not cover eight classes")
    reference_predictions = reference_logits.argmax(axis=1)
    quantized_predictions = quantized_logits.argmax(axis=1)
    unique_reference = len(set(reference_predictions.tolist()))
    unique_quantized = len(set(quantized_predictions.tolist()))
    if unique_quantized < unique_reference:
        raise ModelAuditError("quantized class collapse")
    reference_f1 = _macro_f1(reference_predictions, targets, 8)
    quantized_f1 = _macro_f1(quantized_predictions, targets, 8)
    f1_delta = quantized_f1 - reference_f1
    match_count = int((reference_predictions == quantized_predictions).sum())
    match_ratio = match_count / float(targets.shape[0])
    if f1_delta < -0.05:
        raise ModelAuditError("quantized macro F1 regression")
    if match_ratio < 0.90:
        raise ModelAuditError("quantized argmax parity regression")
    return {
        "sample_count": int(targets.shape[0]),
        "class_count": 8,
        "reference_macro_f1": reference_f1,
        "quantized_macro_f1": quantized_f1,
        "macro_f1_delta": f1_delta,
        "argmax_match_count": match_count,
        "argmax_match_ratio": match_ratio,
        "reference_unique_predictions": unique_reference,
        "quantized_unique_predictions": unique_quantized,
    }


def _legacy_node_metadata(kernel_info_text, label):
    inline = re.search(
        r"(?im)^\s*" + re.escape(label) +
        r"\s*:\s*([A-Za-z0-9_]+)\(0\)(?:\s*:\s*([0-9*]+))?\s*$",
        kernel_info_text,
    )
    if inline:
        return inline
    return re.search(
        re.escape(label) +
        r".*?\n\s*([A-Za-z0-9_]+)\(0\)\s*(?::\s*([0-9*]+))?",
        kernel_info_text,
        re.IGNORECASE | re.DOTALL,
    )


def _network_output_count(network_text, output_node):
    if network_text is None:
        raise ModelAuditError("missing compiled network output metadata")
    counts = []
    for start, end in _layer_spans(network_text):
        layer = network_text[start:end]
        name = re.search(r'\bname\s*:\s*"([^"]+)"', layer)
        top = re.search(r'\btop\s*:\s*"([^"]+)"', layer)
        if not name or name.group(1) != output_node:
            continue
        if top and top.group(1) != output_node:
            continue
        value = re.search(r"\bnum_output\s*:\s*([0-9]+)", layer)
        if value:
            counts.append(int(value.group(1)))
    if len(counts) != 1:
        raise ModelAuditError("missing or ambiguous compiled output count")
    return counts[0]


def extract_compile_evidence(
    contract, kernel_info_text, elf_bytes, compile_log_text, network_text=None
):
    if re.search(r"unsupported\s+(?:layer|operator|op)", compile_log_text, re.IGNORECASE):
        raise ModelAuditError("unsupported layer or operator present")
    if re.search(r"CPUKernel", compile_log_text, re.IGNORECASE):
        raise ModelAuditError("CPUKernel is forbidden")
    version = re.search(rb"dnnc version (v[0-9]+\.[0-9]+)", elf_bytes)
    target = re.search(rb"DPU Target\s*:\s*(v[0-9]+\.[0-9]+\.[0-9]+)", elf_bytes)
    dpu = re.search(rb"DPU_([0-9]+FA)", elf_bytes)
    if not (version and target and dpu):
        raise ModelAuditError("missing embedded compiler metadata")
    kernel_names = re.findall(
        r"(?im)^\s*Kernel Name\s*:\s*(\S+)\s*$", kernel_info_text
    )
    kernel_types = re.findall(
        r"(?im)^\s*(?:Kernel )?Type\s*:\s*(\S+)\s*$", kernel_info_text
    )
    input_match = _legacy_node_metadata(kernel_info_text, "Input Node(s)")
    output_match = _legacy_node_metadata(kernel_info_text, "Output Node(s)")
    if len(kernel_names) != 1 or len(kernel_types) != 1 or not input_match or not output_match:
        raise ModelAuditError("missing or ambiguous kernel metadata")
    if output_match.group(2):
        dimensions = [int(value) for value in output_match.group(2).split("*")]
        output_count = 1
        for value in dimensions:
            output_count *= value
    else:
        output_count = _network_output_count(network_text, output_match.group(1))
    return {
        "dnnc_version": version.group(1).decode("ascii"),
        "dpu_target": target.group(1).decode("ascii"),
        "dpu": dpu.group(1).decode("ascii"),
        "cpu_arch": contract["cpu_arch"],
        "abi": contract["abi"],
        "kernels": kernel_names,
        "kernel_type": kernel_types[0],
        "input_node": input_match.group(1),
        "output_node": output_match.group(1),
        "output_count": output_count,
        "unsupported_layers": [],
    }


def validate_calibration_records(calibration, demo):
    if not calibration:
        raise ModelAuditError("empty calibration")
    demo_ids = {item["recording_id"] for item in demo}
    demo_groups = {item["source_group"] for item in demo}
    for item in calibration:
        if item.get("split") != "train":
            raise ModelAuditError("calibration must contain train records only")
        if item["recording_id"] in demo_ids or item["source_group"] in demo_groups:
            raise ModelAuditError("demo overlap in calibration")


def select_calibration_records(records, per_class=8):
    selected = []
    for target in range(8):
        candidates = sorted(
            (item for item in records if item.get("split") == "train" and item.get("target_index") == target),
            key=lambda item: item["recording_id"],
        )
        if len(candidates) < per_class:
            raise ModelAuditError("insufficient calibration class: {}".format(target))
        selected.extend(candidates[:per_class])
    return selected


def publish_manifest(path, value):
    path = Path(path)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise ModelAuditError("artifact already exists: " + str(path)) from exc


def _layer_spans(text):
    spans = []
    for match in re.finditer(r"(?m)^layer\s*\{", text):
        depth = 0
        for index in range(match.start(), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), index + 1))
                    break
        else:
            raise ModelAuditError("unterminated Caffe layer")
    return spans


def make_quantizer_prototxt(deploy_text, hdf5_list_path):
    body = re.sub(r'(?m)^input:\s*"data"\s*\n', "", deploy_text, count=1)
    body = re.sub(r"(?m)^input_shape\s*\{[^\n]*\}\s*\n", "", body, count=1)
    layer = (
        'layer {\n  name: "calibration_input"\n  type: "HDF5Data"\n'
        '  top: "data"\n  top: "label"\n'
        '  hdf5_data_param { source: "' + str(hdf5_list_path) + '" batch_size: 1 shuffle: false }\n}\n'
    )
    return body.split("\n", 1)[0] + "\n" + layer + body.split("\n", 1)[1].lstrip("\n")


def make_runtime_prototxt(quantizer_text):
    spans = [
        span for span in _layer_spans(quantizer_text)
        if 'type: "HDF5Data"' in quantizer_text[span[0]:span[1]]
    ]
    if len(spans) != 1:
        raise ModelAuditError("expected exactly one HDF5Data layer")
    start, end = spans[0]
    body = (quantizer_text[:start] + quantizer_text[end:]).lstrip("\n")
    for layer_start, layer_end in _layer_spans(body):
        layer = body[layer_start:layer_end]
        if 'type: "Input"' in layer and 'top: "data"' in layer:
            return body
    return 'input: "data"\ninput_shape { dim: 1 dim: 1 dim: 64 dim: 96 }\n' + body


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    expected = commands.add_parser("expected")
    expected.add_argument("--contract", required=True, type=Path)
    verify = commands.add_parser("verify-calibration")
    verify.add_argument("--calibration-manifest", required=True, type=Path)
    verify.add_argument("--demo-manifest", required=True, type=Path)
    assets = commands.add_parser("make-assets")
    assets.add_argument("--input-root", required=True, type=Path)
    assets.add_argument("--work-dir", required=True, type=Path)
    runtime = commands.add_parser("make-runtime")
    runtime.add_argument("--input", required=True, type=Path)
    runtime.add_argument("--output", required=True, type=Path)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--prototxt", required=True, type=Path)
    evaluate.add_argument("--caffemodel", required=True, type=Path)
    evaluate.add_argument("--inputs", required=True, type=Path)
    evaluate.add_argument("--reference-logits", required=True, type=Path)
    evaluate.add_argument("--targets-manifest", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    audit = commands.add_parser("audit")
    audit.add_argument("--contract", required=True, type=Path)
    audit.add_argument("--compile-log", required=True, type=Path)
    audit.add_argument("--elf", required=True, type=Path)
    audit.add_argument("--kernel-info", required=True, type=Path)
    audit.add_argument("--prototxt", required=True, type=Path)
    audit.add_argument("--fused-manifest", required=True, type=Path)
    audit.add_argument("--caffe-export-manifest", required=True, type=Path)
    audit.add_argument("--float-deploy", required=True, type=Path)
    audit.add_argument("--float-caffemodel", required=True, type=Path)
    audit.add_argument("--quantized-caffemodel", required=True, type=Path)
    audit.add_argument("--compiler-binary", required=True, type=Path)
    audit.add_argument("--compiler-version", required=True, type=Path)
    audit.add_argument("--platform-dcf", required=True, type=Path)
    audit.add_argument("--quantize-command", required=True, type=Path)
    audit.add_argument("--compile-command", required=True, type=Path)
    audit.add_argument("--metrics", required=True, type=Path)
    audit.add_argument("--calibration-manifest", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command is None:
        parser.error("a command is required")
    if args.command == "expected":
        print(json.dumps(expected_compile_evidence(load_compiler_contract(args.contract)), sort_keys=True))
    elif args.command == "verify-calibration":
        calibration = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))["records"]
        demo = json.loads(args.demo_manifest.read_text(encoding="utf-8"))["records"]
        validate_calibration_records(calibration, demo)
    elif args.command == "make-assets":
        import h5py
        import numpy as np
        data = np.load(args.input_root / "calibration.npy", allow_pickle=False)
        labels = np.load(args.input_root / "calibration-targets.npy", allow_pickle=False)
        if data.shape != (64, 1, 64, 96) or labels.shape != (64,):
            raise ModelAuditError("calibration tensor shape differs")
        args.work_dir.mkdir(parents=True, exist_ok=True)
        h5_path = args.work_dir / "calibration.h5"
        with h5py.File(str(h5_path), "w") as handle:
            handle.create_dataset("data", data=data.astype("float32"))
            handle.create_dataset("label", data=labels.astype("float32").reshape(-1, 1))
        list_path = args.work_dir / "calibration-h5-list.txt"
        list_path.write_text(str(h5_path) + "\n", encoding="utf-8")
        deploy = (args.input_root / "deploy.prototxt").read_text(encoding="utf-8")
        (args.work_dir / "quantize.prototxt").write_text(
            make_quantizer_prototxt(deploy, list_path), encoding="utf-8"
        )
    elif args.command == "make-runtime":
        args.output.write_text(make_runtime_prototxt(args.input.read_text(encoding="utf-8")), encoding="utf-8")
    elif args.command == "evaluate":
        import caffe
        import numpy as np
        inputs = np.load(args.inputs, allow_pickle=False).astype("float32")
        reference = np.load(args.reference_logits, allow_pickle=False).astype("float32")
        targets_value = json.loads(args.targets_manifest.read_text(encoding="utf-8"))
        targets = np.asarray(targets_value["parity_target_indices"], dtype=np.int64)
        if inputs.shape != (16, 1, 64, 96):
            raise ModelAuditError("parity input tensor shape differs")
        caffe.set_mode_cpu()
        net = caffe.Net(str(args.prototxt), str(args.caffemodel), caffe.TEST)
        logits = []
        for value in inputs:
            net.blobs["data"].data[...] = value.reshape(1, 1, 64, 96)
            result = net.forward()
            output = result.get("fc8", net.blobs["fc8"].data)
            logits.append(np.asarray(output).reshape(-1).copy())
        quantized = np.stack(logits).astype("float32")
        metrics = classification_metrics(reference, quantized, targets)
        metrics.update({
            "inputs_sha256": sha256_file(args.inputs),
            "reference_logits_sha256": sha256_file(args.reference_logits),
            "quantized_caffemodel_sha256": sha256_file(args.caffemodel),
        })
        publish_manifest(args.output, metrics)
    elif args.command == "audit":
        contract = load_compiler_contract(args.contract)
        evidence = extract_compile_evidence(
            contract,
            args.kernel_info.read_text(encoding="utf-8", errors="replace"),
            args.elf.read_bytes(),
            args.compile_log.read_text(encoding="utf-8", errors="replace"),
            args.prototxt.read_text(encoding="utf-8", errors="replace"),
        )
        validate_compile_evidence(contract, evidence)
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        calibration = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
        fused = json.loads(args.fused_manifest.read_text(encoding="utf-8"))
        caffe_export = json.loads(
            args.caffe_export_manifest.read_text(encoding="utf-8")
        )
        actual_hashes = {
            "float_caffemodel_sha256": sha256_file(args.float_caffemodel),
            "float_deploy_sha256": sha256_file(args.float_deploy),
            "quantized_caffemodel_sha256": sha256_file(args.quantized_caffemodel),
        }
        validate_model_hash_chain(
            fused, caffe_export, calibration, metrics, actual_hashes
        )
        manifest = {
            "schema": 1,
            "result": "pass",
            "compile_evidence": evidence,
            "quantization_metrics": metrics,
            "artifacts": {
                "elf_sha256": sha256_file(args.elf),
                "kernel_info_sha256": sha256_file(args.kernel_info),
                "compile_log_sha256": sha256_file(args.compile_log),
                "runtime_prototxt_sha256": sha256_file(args.prototxt),
                "quantized_caffemodel_sha256": actual_hashes["quantized_caffemodel_sha256"],
                "compiler_binary_sha256": sha256_file(args.compiler_binary),
                "compiler_version_sha256": sha256_file(args.compiler_version),
                "platform_dcf_sha256": sha256_file(args.platform_dcf),
                "quantize_command_sha256": sha256_file(args.quantize_command),
                "compile_command_sha256": sha256_file(args.compile_command),
            },
            "upstream": {
                "calibration_manifest_sha256": sha256_file(args.calibration_manifest),
                "prepared_manifest_sha256": calibration["prepared_manifest_sha256"],
                "fused_manifest_sha256": sha256_file(args.fused_manifest),
                "caffe_export_manifest_sha256": sha256_file(args.caffe_export_manifest),
                "checkpoint_sha256": fused["checkpoint_sha256"],
                "training_manifest_sha256": fused["training_manifest_sha256"],
                "float_deploy_sha256": actual_hashes["float_deploy_sha256"],
                "float_caffemodel_sha256": actual_hashes["float_caffemodel_sha256"],
                "class_map_sha256": fused["upstream"]["class_map_sha256"],
                "frontend_sha256": fused["upstream"]["frontend_sha256"],
                "labels_sha256": fused["upstream"]["labels_sha256"],
                "model_config_sha256": fused["upstream"]["model_config_sha256"],
                "split_manifest_sha256": fused["upstream"]["split_manifest_sha256"],
                "training_config_sha256": fused["upstream"]["training_config_sha256"],
            },
        }
        publish_manifest(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
