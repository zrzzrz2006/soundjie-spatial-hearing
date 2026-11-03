#!/usr/bin/env python3
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple


OUTPUTS = ("SJD0.TGZ", "SJD0.SHA256")
MEMBERS = (
    "runtime/dpu-4.14.0-xilinx.ko",
    "runtime/sound_shape_probe",
    "model/dpu_sound_shape_probe.elf",
    "dnndk/pkgs/bin/dexplorer",
    "dnndk/pkgs/lib/libn2cube.so",
    "dnndk/pkgs/lib/libhineon.so",
    "probe/run_sound_shape_probe.sh",
)
MODEL = {
    "architecture": "DPU_1152FA", "cpu_arch": "arm32",
    "kernel": "sound_shape_probe", "input_node": "conv1",
    "output_node": "logits", "input_shape": [1, 1, 64, 96], "class_count": 8,
}


class Record(NamedTuple):
    path: Path
    size: int
    sha256: str


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def record(path: Path) -> Record:
    return Record(path, path.stat().st_size, sha256(path))


def safe(name: str) -> bool:
    value = PurePosixPath(name)
    return bool(name) and not value.is_absolute() and ".." not in value.parts and "\\" not in name


def info(name: str, size: int, mode: int) -> tarfile.TarInfo:
    item = tarfile.TarInfo(name)
    item.size, item.mode, item.mtime = size, mode, 0
    item.uid = item.gid = 0
    item.uname = item.gname = ""
    return item


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes, mode: int) -> None:
    archive.addfile(info(name, len(data), mode), io.BytesIO(data))


def build_bundle(config_path: Path, output_dir: Path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema") != 1:
        raise ValueError("unsupported schema")
    if any(config.get("model", {}).get(key) != value for key, value in MODEL.items()):
        raise ValueError("model contract mismatch")
    items = config.get("bundle", {}).get("archive_members", [])
    names = tuple(item.get("name") for item in items)
    if names != MEMBERS or any(not safe(name) for name in names):
        raise ValueError("archive member allowlist mismatch")

    sources = []
    for item in items:
        path = Path(item["source"])
        if not path.is_file() or path.stat().st_size != int(item["size"]):
            raise ValueError(f"missing or wrong-size source: {item['name']}")
        actual = sha256(path)
        if actual != str(item["sha256"]).lower():
            raise ValueError(f"SHA-256 mismatch: {item['name']}")
        mode = int(str(item["mode"]), 8)
        if mode not in (0o644, 0o755):
            raise ValueError(f"unsupported mode: {item['name']}")
        sources.append((item["name"], path, mode, actual))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError("refusing to replace existing output")
    contract = (json.dumps(MODEL, sort_keys=True, indent=2) + "\n").encode()
    sums = "".join(f"{digest}  {name}\n" for name, _, _, digest in sources)
    sums += f"{hashlib.sha256(contract).hexdigest()}  manifest/model-contract.json\n"

    with tempfile.TemporaryDirectory(prefix=".sjd0-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / OUTPUTS[0]
        with archive_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for name, path, mode, _ in sources:
                        add_bytes(archive, name, path.read_bytes(), mode)
                    add_bytes(archive, "manifest/model-contract.json", contract, 0o644)
                    add_bytes(archive, "manifest/SHA256SUMS", sums.encode("ascii"), 0o644)
        archive_record = record(archive_path)
        (temporary / OUTPUTS[1]).write_text(
            f"{archive_record.sha256}  {OUTPUTS[0]}\n", encoding="ascii", newline="\n")
        for name in OUTPUTS:
            os.replace(temporary / name, output_dir / name)
    return {name: record(output_dir / name) for name in OUTPUTS}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({k: v._asdict() for k, v in build_bundle(args.config, args.output_dir).items()},
                     indent=2, default=str))
