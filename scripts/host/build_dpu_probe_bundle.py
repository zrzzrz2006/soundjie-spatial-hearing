#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import NamedTuple


OUTPUT_NAMES = ("DPUV1.BIT", "DPUV1.DTB", "DPUV1.TGZ", "DPUV1.SHA")
ARCHIVE_NAMES = (
    "runtime/dpu-4.14.0-xilinx.ko",
    "runtime/dpu_smoke",
    "dnndk/pkgs/bin/dexplorer",
    "dnndk/pkgs/lib/libn2cube.so",
    "dnndk/pkgs/lib/libhineon.so",
    "probe/run_ephemeral_dpu_probe.sh",
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


def verify_source(record: dict, label: str) -> Path:
    source = Path(record["source"])
    if not source.is_file():
        raise ValueError(f"missing source for {label}: {source}")
    actual_size = source.stat().st_size
    expected_size = int(record["size"])
    if actual_size != expected_size:
        raise ValueError(
            f"size mismatch for {label}: expected {expected_size}, got {actual_size}"
        )
    actual_hash = file_sha256(source)
    expected_hash = str(record["sha256"]).lower()
    if actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {label}: expected {expected_hash}, got {actual_hash}"
        )
    return source


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


def build_archive(path: Path, members: list[tuple[str, Path, int, str]]) -> None:
    manifest = "".join(f"{digest}  {name}\n" for name, _, _, digest in members).encode(
        "ascii"
    )
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
            ) as archive:
                for name, source, mode, _ in members:
                    payload = source.read_bytes()
                    archive.addfile(tar_info(name, len(payload), mode), io.BytesIO(payload))
                archive.addfile(
                    tar_info("manifest/SHA256SUMS", len(manifest), 0o644),
                    io.BytesIO(manifest),
                )


def record(path: Path) -> ArtifactRecord:
    return ArtifactRecord(path=path, size=path.stat().st_size, sha256=file_sha256(path))


def build_bundle(config_path: Path, output_dir: Path) -> dict[str, ArtifactRecord]:
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != 1:
        raise ValueError("unsupported config schema")
    if tuple(config.get("outputs", {}).keys()) != ("DPUV1.BIT", "DPUV1.DTB"):
        raise ValueError("outputs must be DPUV1.BIT and DPUV1.DTB in order")

    member_records = config.get("archive_members", [])
    member_names = tuple(item.get("name") for item in member_records)
    if member_names != ARCHIVE_NAMES:
        raise ValueError("archive member allowlist or order mismatch")

    verified_outputs = {
        name: verify_source(source_record, name)
        for name, source_record in config["outputs"].items()
    }
    verified_members = []
    for item in member_records:
        name = item["name"]
        source = verify_source(item, name)
        mode = int(str(item["mode"]), 8)
        if mode not in (0o644, 0o755):
            raise ValueError(f"unsupported mode for {name}: {item['mode']}")
        verified_members.append((name, source, mode, str(item["sha256"]).lower()))

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(f"refusing to replace existing outputs: {', '.join(existing)}")

    with tempfile.TemporaryDirectory(prefix=".dpu-build-", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        shutil.copyfile(verified_outputs["DPUV1.BIT"], temp_dir / "DPUV1.BIT")
        shutil.copyfile(verified_outputs["DPUV1.DTB"], temp_dir / "DPUV1.DTB")
        build_archive(temp_dir / "DPUV1.TGZ", verified_members)

        preliminary = {
            name: record(temp_dir / name)
            for name in ("DPUV1.BIT", "DPUV1.DTB", "DPUV1.TGZ")
        }
        sha_text = "".join(
            f"{preliminary[name].sha256}  {name}\n"
            for name in ("DPUV1.BIT", "DPUV1.DTB", "DPUV1.TGZ")
        )
        (temp_dir / "DPUV1.SHA").write_text(sha_text, encoding="ascii", newline="\n")

        temp_records = {name: record(temp_dir / name) for name in OUTPUT_NAMES}
        for name in OUTPUT_NAMES:
            os.replace(temp_dir / name, output_dir / name)

    final_records = {name: record(output_dir / name) for name in OUTPUT_NAMES}
    for name in OUTPUT_NAMES:
        if final_records[name].size != temp_records[name].size:
            raise RuntimeError(f"output size changed after move: {name}")
        if final_records[name].sha256 != temp_records[name].sha256:
            raise RuntimeError(f"output SHA-256 changed after move: {name}")
    return final_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic MZ7100 DPU probe media")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    artifacts = build_bundle(args.config, args.output_dir)
    print(
        json.dumps(
            {
                name: {
                    "path": str(item.path),
                    "size": item.size,
                    "sha256": item.sha256,
                }
                for name, item in artifacts.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
