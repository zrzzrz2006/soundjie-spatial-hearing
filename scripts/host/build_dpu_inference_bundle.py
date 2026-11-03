#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple


OUTPUT_NAMES = ("DPUY1.TGZ", "DPUY1.SHA")
ARCHIVE_NAMES = (
    "runtime/dpu-4.14.0-xilinx.ko",
    "runtime/dpu_yolo_probe",
    "dnndk/pkgs/bin/dexplorer",
    "dnndk/pkgs/lib/libn2cube.so",
    "dnndk/pkgs/lib/libhineon.so",
    "probe/run_real_dpu_inference.sh",
)


class ArtifactRecord(NamedTuple):
    path: Path
    size: int
    sha256: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(item: dict, label: str) -> Path:
    source = Path(item["source"])
    if not source.is_file():
        raise ValueError(f"missing source for {label}: {source}")
    if source.stat().st_size != int(item["size"]):
        raise ValueError(f"size mismatch for {label}")
    actual = file_sha256(source)
    if actual != str(item["sha256"]).lower():
        raise ValueError(f"SHA-256 mismatch for {label}: expected {item['sha256']}, got {actual}")
    return source


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def tar_info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def add_bytes(archive: tarfile.TarFile, name: str, payload: bytes, mode: int) -> None:
    archive.addfile(tar_info(name, len(payload), mode), io.BytesIO(payload))


def artifact(path: Path) -> ArtifactRecord:
    return ArtifactRecord(path, path.stat().st_size, file_sha256(path))


def build_bundle(config_path: Path, output_dir: Path) -> dict[str, ArtifactRecord]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema") != 1:
        raise ValueError("unsupported config schema")
    model = config.get("model", {})
    required_model = {
        "architecture": "DPU_1152FA",
        "target": "v1.4.0",
        "kernel": "yolo",
        "input_node": "layer0_conv",
        "output_nodes": ["layer81_conv", "layer93_conv", "layer105_conv", "layer117_conv"],
    }
    for key, expected in required_model.items():
        if model.get(key) != expected:
            raise ValueError(f"model contract mismatch: {key}")

    records = config.get("bundle", {}).get("archive_members", [])
    names = tuple(item.get("name") for item in records)
    if names != ARCHIVE_NAMES or any(not safe_member(name) for name in names):
        raise ValueError("archive member allowlist, order, or path mismatch")

    members = []
    for item in records:
        name = item["name"]
        source = verify_source(item, name)
        mode = int(str(item["mode"]), 8)
        if mode not in (0o644, 0o755):
            raise ValueError(f"unsupported mode for {name}")
        members.append((name, source, mode, str(item["sha256"]).lower()))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to replace existing outputs: {', '.join(existing)}")

    contract = {key: model[key] for key in required_model}
    contract_bytes = (json.dumps(contract, sort_keys=True, indent=2) + "\n").encode("utf-8")
    member_manifest = "".join(
        f"{digest}  {name}\n" for name, _, _, digest in members
    )
    contract_hash = hashlib.sha256(contract_bytes).hexdigest()
    manifest_bytes = (
        member_manifest + f"{contract_hash}  manifest/inference-contract.json\n"
    ).encode("ascii")

    with tempfile.TemporaryDirectory(prefix=".dpuy1-build-", dir=output_dir) as temp_name:
        temp = Path(temp_name)
        archive_path = temp / "DPUY1.TGZ"
        with archive_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for name, source, mode, _ in members:
                        add_bytes(archive, name, source.read_bytes(), mode)
                    add_bytes(archive, "manifest/inference-contract.json", contract_bytes, 0o644)
                    add_bytes(archive, "manifest/SHA256SUMS", manifest_bytes, 0o644)
        archive_record = artifact(archive_path)
        (temp / "DPUY1.SHA").write_text(
            f"{archive_record.sha256}  DPUY1.TGZ\n", encoding="ascii", newline="\n"
        )
        staged = {name: artifact(temp / name) for name in OUTPUT_NAMES}
        for name in OUTPUT_NAMES:
            os.replace(temp / name, output_dir / name)

    final = {name: artifact(output_dir / name) for name in OUTPUT_NAMES}
    for name in OUTPUT_NAMES:
        if final[name].size != staged[name].size or final[name].sha256 != staged[name].sha256:
            raise RuntimeError(f"output changed after move: {name}")
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic real-DPU inference bundle")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = build_bundle(args.config, args.output_dir)
    print(json.dumps({name: item._asdict() for name, item in result.items()}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
