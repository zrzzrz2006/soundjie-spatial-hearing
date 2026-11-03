#!/usr/bin/env python3
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_logmel_d2_tensors_for_sound_bundle",
    Path(__file__).with_name("audit_logmel_d2_tensors.py"))
d2_audit = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(d2_audit)

OUTPUTS = ("SJD2.TGZ", "SJD2.SHA256")
CASES = (
    "xvf3800-start-0000000000",
    "xvf3800-start-0000072000",
    "xvf3800-start-0000144000",
)
MODEL = {
    "architecture": "DPU_1152FA", "cpu_arch": "arm32",
    "kernel": "sound_shape_probe", "input_node": "conv1",
    "output_node": "logits", "input_shape": [1, 1, 64, 96],
    "class_count": 8,
}
BASE_MEMBERS = (
    "runtime/dpu-4.14.0-xilinx.ko",
    "runtime/sound_shape_runner",
    "model/dpu_sound_shape_probe.elf",
    "dnndk/pkgs/bin/dexplorer",
    "dnndk/pkgs/lib/libn2cube.so",
    "dnndk/pkgs/lib/libhineon.so",
    "frontend/logmel_cli",
    "config/logmel_frontend_v1.json",
    "manifest/dpu-input-contract.txt",
)


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


def expected_members():
    names = list(BASE_MEMBERS)
    for case in CASES:
        names.extend((f"inputs/{case}.wav", f"golden/{case}.f32le",
                      f"golden/{case}.i8"))
    names.append("probe/run_sound_wav_dpu.sh")
    return tuple(names)


def safe(name):
    value = PurePosixPath(name)
    return bool(name) and not value.is_absolute() and ".." not in value.parts and "\\" not in name


def tar_info(name, size, mode):
    item = tarfile.TarInfo(name)
    item.size, item.mode, item.mtime = size, mode, 0
    item.uid = item.gid = 0
    item.uname = item.gname = ""
    return item


def add_bytes(archive, name, data, mode):
    archive.addfile(tar_info(name, len(data), mode), io.BytesIO(data))


def publish_create_only(source, destination):
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


def build_bundle(config_path, output_dir):
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if data.get("schema") != 2 or tuple(data.get("cases", ())) != CASES:
        raise ValueError("bundle contract mismatch")
    if data.get("normalization_profile") != d2_audit.EXPECTED_PROFILE_ID:
        raise ValueError("normalization profile mismatch")
    scale = float(data.get("input_scale"))
    if not math.isfinite(scale) or scale != 128.0:
        raise ValueError("input scale must equal measured value 128")
    if any(data.get("model", {}).get(key) != value for key, value in MODEL.items()):
        raise ValueError("model contract mismatch")
    items = data.get("archive_members", [])
    names = tuple(item.get("name") for item in items)
    if names != expected_members() or any(not safe(name) for name in names):
        raise ValueError("archive member allowlist mismatch")

    sources = []
    for item in items:
        path = Path(item["source"])
        if not path.is_file():
            raise ValueError(f"missing or wrong-size source: {item['name']}")
        payload = path.read_bytes()
        if len(payload) != int(item["size"]):
            raise ValueError(f"missing or wrong-size source: {item['name']}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != str(item["sha256"]).lower():
            raise ValueError(f"SHA-256 mismatch: {item['name']}")
        mode = int(str(item["mode"]), 8)
        if mode not in (0o644, 0o755):
            raise ValueError("unsupported mode")
        sources.append((item["name"], payload, mode, digest))
    by_name = {name: digest for name, _, _, digest in sources}
    payload_by_name = {name: payload for name, payload, _, _ in sources}
    config_digest = by_name["config/logmel_frontend_v1.json"]
    if config_digest != data.get("config_sha256"):
        raise ValueError("config SHA-256 mismatch")
    frontend = json.loads(payload_by_name["config/logmel_frontend_v1.json"].decode("utf-8"))
    if (frontend.get("schema") != 2 or
            frontend.get("normalization", {}).get("id") != d2_audit.EXPECTED_PROFILE_ID):
        raise ValueError("frontend profile mismatch")
    real_tensors = [
        {"name": case,
         "tensor": np.frombuffer(payload_by_name[f"golden/{case}.i8"], dtype=np.int8)}
        for case in CASES
    ]
    quality = d2_audit.audit_real_windows(real_tensors)

    case_lines = []
    for case in CASES:
        start = int(case.rsplit("-", 1)[1])
        case_lines.append(f"{case}\tinputs/{case}.wav\tgolden/{case}.f32le\t"
                          f"golden/{case}.i8\t{start}\t128\n")
    cases_tsv = "".join(case_lines).encode("ascii")
    contract_tsv = (
        "SCHEMA\t2\n"
        f"NORMALIZATION_PROFILE\t{d2_audit.EXPECTED_PROFILE_ID}\n"
        "INPUT_SCALE\t128\n"
        "INPUT_SIZE\t6144\n"
        "MODEL_KERNEL\tsound_shape_probe\n"
        "INPUT_NODE\tconv1\n"
        "OUTPUT_NODE\tlogits\n"
        f"MODEL_SHA256\t{by_name['model/dpu_sound_shape_probe.elf']}\n"
        f"RUNNER_SHA256\t{by_name['runtime/sound_shape_runner']}\n"
        f"FRONTEND_SHA256\t{by_name['frontend/logmel_cli']}\n"
        f"CONFIG_SHA256\t{by_name['config/logmel_frontend_v1.json']}\n"
        f"INPUT_CONTRACT_SHA256\t{by_name['manifest/dpu-input-contract.txt']}\n"
    ).encode("ascii")
    sums = "".join(f"{digest}  {name}\n" for name, _, _, digest in sources)
    sums += f"{hashlib.sha256(cases_tsv).hexdigest()}  manifest/cases.tsv\n"
    sums += f"{hashlib.sha256(contract_tsv).hexdigest()}  manifest/contract.tsv\n"
    sums_bytes = sums.encode("ascii")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError("refusing to replace existing output")
    with tempfile.TemporaryDirectory(prefix=".sjd2-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / OUTPUTS[0]
        with archive_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for name, payload, mode, _ in sources:
                        add_bytes(archive, name, payload, mode)
                    add_bytes(archive, "manifest/cases.tsv", cases_tsv, 0o644)
                    add_bytes(archive, "manifest/contract.tsv", contract_tsv, 0o644)
                    add_bytes(archive, "manifest/SHA256SUMS", sums_bytes, 0o644)
        archive_record = record(archive_path)
        (temporary / OUTPUTS[1]).write_text(
            f"{archive_record.sha256}  {OUTPUTS[0]}\n", encoding="ascii", newline="\n")
        for name in OUTPUTS:
            publish_create_only(temporary / name, output_dir / name)
    return {name: record(output_dir / name) for name in OUTPUTS}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({key: value._asdict() for key, value in
                      build_bundle(args.config, args.output_dir).items()}, indent=2, default=str))
