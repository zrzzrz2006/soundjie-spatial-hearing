#!/usr/bin/env python3
"""Build the deterministic, create-only Soundjie D3 board-validation bundle."""

import gzip
import hashlib
import io
import json
import math
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np


OUTPUTS = ("SJD3.TGZ", "SJD3.SHA256")
CLASS_IDS = (
    "fire_smoke_alarm", "emergency_siren", "car_horn", "glass_breaking",
    "door_knock_bell", "distress_voice", "normal_speech", "background_unknown",
)
BASE_MEMBERS = (
    "runtime/dpu-4.14.0-xilinx.ko",
    "runtime/sound_shape_runner",
    "model/dpu_soundjie_d3.elf",
    "dnndk/pkgs/bin/dexplorer",
    "dnndk/pkgs/lib/libn2cube.so",
    "dnndk/pkgs/lib/libhineon.so",
    "config/logmel_frontend_v1.json",
    "config/soundjie_d3_model.json",
    "model/labels.txt",
    "manifest/soundjie-d3-model-manifest.json",
    "manifest/prepared-manifest.jsonl",
    "manifest/demo.jsonl",
    "manifest/calibration-manifest.json",
    "licenses/ESC-50-LICENSE",
    "licenses/LibriSpeech-LICENSE.TXT",
    "probe/run_soundjie_d3.sh",
)
CASE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Record(NamedTuple):
    path: Path
    size: int
    sha256: str


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path)
    return Record(path, path.stat().st_size, file_sha256(path))


def _safe(name):
    value = PurePosixPath(name)
    return bool(name) and not value.is_absolute() and ".." not in value.parts and "\\" not in name


def expected_members(cases):
    names = list(BASE_MEMBERS)
    for case in cases:
        names.append(f"inputs/{case['id']}.i8")
        if case["split"] == "demo":
            names.append(f"demo/{case['id']}.wav")
    names.extend(("manifest/ATTRIBUTION.md", "manifest/cases.tsv",
                  "manifest/contract.tsv", "manifest/SHA256SUMS"))
    return tuple(names)


def _tar_info(name, size, mode):
    item = tarfile.TarInfo(name)
    item.size, item.mode, item.mtime = size, mode, 0
    item.uid = item.gid = 0
    item.uname = item.gname = ""
    return item


def _add_bytes(archive, name, data, mode):
    archive.addfile(_tar_info(name, len(data), mode), io.BytesIO(data))


def _publish_create_only(source, destination):
    source, destination = Path(source), Path(destination)
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        created = False
        try:
            with source.open("rb") as incoming, destination.open("xb") as outgoing:
                created = True
                for block in iter(lambda: incoming.read(1024 * 1024), b""):
                    outgoing.write(block)
                outgoing.flush()
                os.fsync(outgoing.fileno())
        except Exception:
            if created:
                destination.unlink()
            raise
    source.unlink()


def _strict_json(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def _jsonl(payload):
    rows = []
    for number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL line {number}") from error
    return rows


def _snapshot_member(item):
    name = item.get("name")
    if not _safe(name):
        raise ValueError("archive member allowlist mismatch")
    path = Path(item.get("source", ""))
    if not path.is_file():
        raise ValueError(f"missing source: {name}")
    payload = path.read_bytes()
    if len(payload) != item.get("size"):
        raise ValueError(f"wrong-size source: {name}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != str(item.get("sha256", "")).lower():
        raise ValueError(f"SHA-256 mismatch: {name}")
    mode = int(str(item.get("mode", "")), 8)
    if mode not in (0o644, 0o755):
        raise ValueError("unsupported mode")
    return name, payload, mode, digest


def _validate_model(model):
    expected = {
        "architecture": "DPU_1152FA", "cpu_arch": "arm32", "abi": 1,
        "kernel": "soundjie_d3", "input_node": "conv1", "output_node": "fc8",
        "input_shape": [1, 1, 64, 96], "input_size": 6144,
        "output_size": 8,
    }
    if any(model.get(key) != value for key, value in expected.items()):
        raise ValueError("model contract mismatch")
    for key in ("input_scale", "output_scale"):
        value = float(model.get(key, math.nan))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("model contract mismatch")


def _validate_model_manifest(payload, elf_sha, member_hashes, expected_git_commit):
    value = json.loads(payload.decode("utf-8"))
    evidence = value.get("compile_evidence", {})
    expected = {
        "kernels": ["soundjie_d3"], "input_node": "conv1", "output_node": "fc8",
        "output_count": 8, "abi": 1, "cpu_arch": "arm32", "dpu": "1152FA",
        "dpu_target": "v1.4.0", "kernel_type": "DPUKernel",
    }
    if value.get("schema") != 1 or value.get("result") != "pass" or any(
            evidence.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("model manifest contract mismatch")
    if evidence.get("unsupported_layers") != []:
        raise ValueError("model manifest contract mismatch")
    bound = value.get("artifacts", {}).get("elf_sha256")
    if bound != elf_sha:
        raise ValueError("model manifest ELF SHA-256 mismatch")
    upstream = value.get("upstream", {})
    bindings = {
        "frontend_sha256": "config/logmel_frontend_v1.json",
        "model_config_sha256": "config/soundjie_d3_model.json",
        "labels_sha256": "model/labels.txt",
        "prepared_manifest_sha256": "manifest/prepared-manifest.jsonl",
        "calibration_manifest_sha256": "manifest/calibration-manifest.json",
        "demo_manifest_sha256": "manifest/demo.jsonl",
    }
    if any(upstream.get(key) != member_hashes[name] for key, name in bindings.items()):
        raise ValueError("model manifest upstream hash chain mismatch")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", str(expected_git_commit or ""))
        or upstream.get("git_commit") != expected_git_commit
    ):
        raise ValueError("model manifest upstream hash chain mismatch")


def _validate_cases(cases, classes, prepared_rows, demo_rows, calibration):
    if not isinstance(cases, list) or len(cases) != 24:
        raise ValueError("case coverage mismatch")
    prepared = {(row.get("recording_id"), row.get("split")): row for row in prepared_rows}
    demo_index = {row.get("recording_id"): row for row in demo_rows}
    calibration_ids = {row.get("recording_id") for row in calibration.get("records", [])}
    seen_ids, test_groups, test_hashes = set(), set(), set()
    counts = {(index, split): 0 for index in range(8) for split in ("test", "demo")}
    snapshots = []
    for case in cases:
        case_id = case.get("id", "")
        split = case.get("split")
        index = case.get("target_index")
        if not CASE_ID.fullmatch(case_id) or case_id in seen_ids:
            raise ValueError("unsafe or duplicate case id")
        seen_ids.add(case_id)
        if split not in ("test", "demo") or not isinstance(index, int) or not 0 <= index < 8:
            raise ValueError("case contract mismatch")
        if case.get("target_id") != classes[index]:
            raise ValueError("class contract mismatch")
        counts[(index, split)] += 1
        required = ("recording_id", "source_group", "source_sha256", "tensor_sha256")
        if any(not case.get(key) for key in required) or not all(
                SHA256.fullmatch(str(case[key])) for key in ("source_sha256", "tensor_sha256")):
            raise ValueError("case provenance mismatch")
        if split == "demo" and (
                case["source_group"] in test_groups or case["source_sha256"] in test_hashes):
            raise ValueError("demo overlap with test")
        prepared_row = prepared.get((case["recording_id"], split))
        if prepared_row is None or prepared_row.get("source_group") != case["source_group"]:
            raise ValueError("case is not bound to prepared manifest")
        tensor = Path(case.get("tensor_source", ""))
        if not tensor.is_file() or tensor.stat().st_size != 6144 or file_sha256(tensor) != case["tensor_sha256"]:
            raise ValueError("tensor size or SHA-256 mismatch")
        values = np.frombuffer(tensor.read_bytes(), dtype=np.int8)
        if not np.any(values):
            raise ValueError("all-zero input tensor")
        snapshots.append((f"inputs/{case_id}.i8", tensor.read_bytes(), 0o644,
                          case["tensor_sha256"]))
        if split == "test":
            test_groups.add(case["source_group"])
            test_hashes.add(case["source_sha256"])
        else:
            if case["recording_id"] in calibration_ids:
                raise ValueError("demo overlap with calibration")
            if case["recording_id"] not in demo_index:
                raise ValueError("demo missing from demo manifest")
            attribution = case.get("attribution", {})
            required_attribution = (
                "source_id", "license", "creator", "source_url", "license_url",
                "modification_notice", "source_citation",
            )
            if any(not attribution.get(key) for key in required_attribution):
                raise ValueError("missing demo attribution")
            if (
                attribution["creator"] == case["source_group"].split(":")[-1]
                or attribution["source_url"].rstrip("/")
                in ("https://github.com/karolpiczak/ESC-50", "https://www.openslr.org/12",
                    "https://example.invalid/dataset")
            ):
                raise ValueError("recording-specific attribution required")
            wav = Path(case.get("wav_source", ""))
            if not wav.is_file() or file_sha256(wav) != case.get("wav_sha256"):
                raise ValueError("demo WAV SHA-256 mismatch")
            snapshots.append((f"demo/{case_id}.wav", wav.read_bytes(), 0o644,
                              case["wav_sha256"]))
    if any(counts[(index, "test")] != 1 or counts[(index, "demo")] != 2
           for index in range(8)):
        raise ValueError("case coverage mismatch")
    for case in cases:
        if case["split"] == "demo" and (
                case["source_group"] in test_groups or case["source_sha256"] in test_hashes):
            raise ValueError("demo overlap with test")
    return snapshots


def build_bundle(config_path, output_dir):
    data = _strict_json(config_path)
    if data.get("schema") != 1 or tuple(data.get("classes", ())) != CLASS_IDS:
        raise ValueError("bundle contract mismatch")
    _validate_model(data.get("model", {}))
    items = data.get("archive_members", [])
    if tuple(item.get("name") for item in items) != BASE_MEMBERS:
        raise ValueError("archive member allowlist mismatch")
    sources = [_snapshot_member(item) for item in items]
    payloads = {name: payload for name, payload, _, _ in sources}
    hashes = {name: digest for name, _, _, digest in sources}
    labels = tuple(payloads["model/labels.txt"].decode("ascii").splitlines())
    if labels != CLASS_IDS:
        raise ValueError("labels/class contract mismatch")
    _validate_model_manifest(payloads["manifest/soundjie-d3-model-manifest.json"],
                             hashes["model/dpu_soundjie_d3.elf"], hashes,
                             data.get("git_commit"))
    prepared = _jsonl(payloads["manifest/prepared-manifest.jsonl"])
    demos = _jsonl(payloads["manifest/demo.jsonl"])
    calibration = json.loads(payloads["manifest/calibration-manifest.json"].decode("utf-8"))
    cases = data.get("cases", [])
    case_sources = _validate_cases(cases, labels, prepared, demos, calibration)
    sources.extend(case_sources)

    attribution_lines = ["# Soundjie D3 demo attribution", ""]
    for case in cases:
        if case["split"] == "demo":
            item = case["attribution"]
            attribution_lines.append(
                f"- `{case['id']}` — {item['creator']}; {item['license']} "
                f"({item['license_url']}); {item['source_id']}; {item['source_url']}; "
                f"{item['source_citation']}; modification: {item['modification_notice']}")
    attribution = ("\n".join(attribution_lines) + "\n").encode("utf-8")
    case_lines = []
    for case in cases:
        wav = f"demo/{case['id']}.wav" if case["split"] == "demo" else "-"
        case_lines.append("\t".join((
            case["id"], case["split"], f"inputs/{case['id']}.i8", wav,
            str(case["target_index"]), case["target_id"], case["recording_id"],
            case["source_group"], case["tensor_sha256"],
        )) + "\n")
    cases_tsv = "".join(case_lines).encode("utf-8")
    model = data["model"]
    contract_tsv = (
        "SCHEMA\t1\nCLASS_COUNT\t8\nCASE_COUNT\t24\n"
        f"MODEL_KERNEL\t{model['kernel']}\nINPUT_NODE\t{model['input_node']}\n"
        f"OUTPUT_NODE\t{model['output_node']}\nINPUT_SHAPE\t1,1,64,96\n"
        f"INPUT_SIZE\t{model['input_size']}\nINPUT_SCALE\t{model['input_scale']:.9g}\n"
        f"OUTPUT_SIZE\t{model['output_size']}\nOUTPUT_SCALE\t{model['output_scale']:.9g}\n"
        f"MODEL_SHA256\t{hashes['model/dpu_soundjie_d3.elf']}\n"
        f"RUNNER_SHA256\t{hashes['runtime/sound_shape_runner']}\n"
        f"LABELS_SHA256\t{hashes['model/labels.txt']}\n"
    ).encode("ascii")
    generated = [
        ("manifest/ATTRIBUTION.md", attribution, 0o644),
        ("manifest/cases.tsv", cases_tsv, 0o644),
        ("manifest/contract.tsv", contract_tsv, 0o644),
    ]
    all_without_sums = sources + [
        (name, payload, mode, hashlib.sha256(payload).hexdigest())
        for name, payload, mode in generated
    ]
    sums = "".join(f"{digest}  {name}\n" for name, _, _, digest in all_without_sums)
    sums_bytes = sums.encode("ascii")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError("refusing to replace existing output")
    with tempfile.TemporaryDirectory(prefix=".sjd3-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / OUTPUTS[0]
        with archive_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for name, payload, mode, _ in sources:
                        _add_bytes(archive, name, payload, mode)
                    for name, payload, mode in generated:
                        _add_bytes(archive, name, payload, mode)
                    _add_bytes(archive, "manifest/SHA256SUMS", sums_bytes, 0o644)
        archive_record = record(archive_path)
        (temporary / OUTPUTS[1]).write_text(
            f"{archive_record.sha256}  {OUTPUTS[0]}\n", encoding="ascii", newline="\n")
        for name in OUTPUTS:
            _publish_create_only(temporary / name, output_dir / name)
    return {name: record(output_dir / name) for name in OUTPUTS}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({key: value._asdict() for key, value in
                      build_bundle(args.config, args.output_dir).items()}, indent=2, default=str))
