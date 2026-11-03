#!/usr/bin/env python3
"""Build the deterministic, create-only Soundjie D3 live inference bundle."""

import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple


OUTPUTS = ("SJD3L1.TGZ", "SJD3L1.SHA")
CLASSES = (
    "fire_smoke_alarm", "emergency_siren", "car_horn", "glass_breaking",
    "door_knock_bell", "distress_voice", "normal_speech", "background_unknown",
)
REQUIRED = {
    "runtime/dpu-4.14.0-xilinx.ko", "runtime/sound_shape_runner", "runtime/wav_probe",
    "runtime/logmel_cli", "runtime/live_window_selector", "runtime/live_result_fuser",
    "model/dpu_soundjie_d3.elf", "model/labels.txt", "model/display_labels_zh.txt",
    "dnndk/pkgs/bin/dexplorer", "dnndk/pkgs/lib/libn2cube.so",
    "dnndk/pkgs/lib/libhineon.so", "config/logmel_frontend_v1.json",
    "config/soundjie_d3_model.json", "config/soundjie_live_v1.tsv",
    "manifest/soundjie-d3-model-manifest.json", "manifest/prepared-manifest.jsonl",
    "manifest/demo.jsonl", "manifest/calibration-manifest.json",
    "licenses/ESC-50-LICENSE", "licenses/LibriSpeech-LICENSE.TXT",
    "probe/run_soundjie_d3_live_once.sh", "probe/run_soundjie_d3_live_wav.sh",
    "audio/load-usb-audio.sh", "audio/board-tools/xvf3800_probe.sh",
    "audio/board-tools/xvf3800_record.sh", "audio/bin/arecord",
    "audio/lib/libasound.so.2", "audio/share/alsa/alsa.conf",
    "audio/modules/snd-hwdep.ko", "audio/modules/snd-rawmidi.ko",
    "audio/modules/snd-usbmidi-lib.ko", "audio/modules/snd-usb-audio.ko", "README.md",
}
GENERATED = {"manifest/contract.tsv", "manifest/demo-playback.tsv", "manifest/SHA256SUMS"}
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Record(NamedTuple):
    path: Path
    size: int
    sha256: str


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path)
    return Record(path, path.stat().st_size, file_sha256(path))


def safe_member(name):
    value = PurePosixPath(name)
    return bool(name) and not value.is_absolute() and ".." not in value.parts and "\\" not in name


def tar_info(name, size, mode):
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = size, mode, 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def add_bytes(archive, name, payload, mode):
    archive.addfile(tar_info(name, len(payload), mode), io.BytesIO(payload))


def publish_create_only(source, destination):
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        with Path(source).open("rb") as incoming, Path(destination).open("xb") as outgoing:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    Path(source).unlink()


def _clean_field(value, field):
    value = str(value or "")
    if not value or "\t" in value or "\r" in value or "\n" in value:
        raise ValueError(f"missing or unsafe demo attribution: {field}")
    return value


def build_bundle(config_path, output_dir):
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if data.get("schema") != 1 or tuple(data.get("classes", ())) != CLASSES:
        raise ValueError("bundle contract mismatch")
    provenance = data.get("provenance")
    if provenance is not None:
        base = Path(provenance.get("base_bundle", ""))
        expected = str(provenance.get("base_bundle_sha256", "")).lower()
        if not base.is_file() or not SHA256.fullmatch(expected) or file_sha256(base) != expected:
            raise ValueError("SJD3C provenance mismatch")

    snapshots = []
    names = []
    for item in data.get("archive_members", []):
        name = item.get("name", "")
        if not safe_member(name) or name in GENERATED or name in names:
            raise ValueError("archive member allowlist mismatch")
        path = Path(item.get("source", ""))
        if not path.is_file():
            raise ValueError(f"missing source: {name}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != item.get("size") or digest != str(item.get("sha256", "")).lower():
            raise ValueError(f"source size or SHA-256 mismatch: {name}")
        mode = int(str(item.get("mode", "")), 8)
        if mode not in (0o644, 0o755):
            raise ValueError("unsupported mode")
        names.append(name)
        snapshots.append((name, payload, mode, digest))
    by_name = {name: (payload, digest) for name, payload, _, digest in snapshots}
    if not REQUIRED.issubset(by_name):
        raise ValueError("archive member allowlist missing required member")

    demos = data.get("demos", [])
    if not isinstance(demos, list) or len(demos) != 16:
        raise ValueError("demo coverage mismatch")
    counts = {name: 0 for name in CLASSES}
    rows = ["id\twav_member\ttensor_member\tdirect_logits_member\tEXPECTED_TENSOR_SHA256\tDIRECT_OUTPUT_SHA256\ttarget_index\ttarget_id\tsource_id\tlicense\tcreator\tsource_url\tlicense_url\tmodification_notice\n"]
    seen = set()
    for demo in demos:
        case_id = str(demo.get("id", ""))
        index = demo.get("target_index")
        target = demo.get("target_id")
        if not SAFE_ID.fullmatch(case_id) or case_id in seen or not isinstance(index, int) or not 0 <= index < 8 or target != CLASSES[index]:
            raise ValueError("demo contract mismatch")
        seen.add(case_id)
        counts[target] += 1
        members = [demo.get(key, "") for key in ("wav_member", "tensor_member", "direct_logits_member")]
        if any(member not in by_name for member in members):
            raise ValueError("demo member is not in archive allowlist")
        if not members[0].startswith("demo/") or not members[1].startswith("inputs/") or not members[2].startswith("direct/"):
            raise ValueError("demo member contract mismatch")
        if len(by_name[members[1]][0]) != 6144 or len(by_name[members[2]][0]) != 8:
            raise ValueError("demo tensor or direct output size mismatch")
        fields = [_clean_field(demo.get(key), key) for key in (
            "source_id", "license", "creator", "source_url", "license_url", "modification_notice")]
        rows.append("\t".join([case_id, *members, by_name[members[1]][1], by_name[members[2]][1],
                               str(index), target, *fields]) + "\n")
    if set(counts.values()) != {2}:
        raise ValueError("demo coverage mismatch")

    contract = (
        "SCHEMA\t1\nCLASS_COUNT\t8\nMODEL_KERNEL\tsoundjie_d3\nINPUT_NODE\tconv1\n"
        "OUTPUT_NODE\tfc8\nINPUT_SHAPE\t1,1,64,96\nINPUT_SIZE\t6144\nINPUT_SCALE\t128\n"
        "OUTPUT_SIZE\t8\nOUTPUT_SCALE\t0.0625\nAUDIO_USB_ID\t2886:001a\n"
        "AUDIO_FORMAT\tS16_LE\nAUDIO_CHANNELS\t2\nAUDIO_RATE\t16000\nCAPTURE_SECONDS\t6\n"
        f"MODEL_SHA256\t{by_name['model/dpu_soundjie_d3.elf'][1]}\n"
        f"RUNNER_SHA256\t{by_name['runtime/sound_shape_runner'][1]}\n"
        f"FRONTEND_SHA256\t{by_name['runtime/logmel_cli'][1]}\n"
    ).encode("ascii")
    playback = "".join(rows).encode("utf-8")
    generated = (("manifest/contract.tsv", contract), ("manifest/demo-playback.tsv", playback))
    sums = "".join(f"{sha}  {name}\n" for name, _, _, sha in snapshots)
    sums += "".join(f"{hashlib.sha256(payload).hexdigest()}  {name}\n" for name, payload in generated)
    sums = sums.encode("ascii")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / name).exists() for name in OUTPUTS):
        raise FileExistsError("refusing to replace existing output")
    with tempfile.TemporaryDirectory(prefix=".sjd3l1-", dir=output_dir) as temporary:
        temporary = Path(temporary)
        archive_path = temporary / OUTPUTS[0]
        with archive_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for name, payload, mode, _ in snapshots:
                        add_bytes(archive, name, payload, mode)
                    add_bytes(archive, generated[0][0], generated[0][1], 0o644)
                    add_bytes(archive, generated[1][0], generated[1][1], 0o644)
                    add_bytes(archive, "manifest/SHA256SUMS", sums, 0o644)
        if archive_path.stat().st_size >= 54 * 1024 * 1024:
            raise ValueError("bundle exceeds 54 MiB FAT budget")
        archive_sha = file_sha256(archive_path)
        (temporary / OUTPUTS[1]).write_text(f"{archive_sha}  {OUTPUTS[0]}\n", encoding="ascii", newline="\n")
        for name in OUTPUTS:
            publish_create_only(temporary / name, output_dir / name)
    return {name: record(output_dir / name) for name in OUTPUTS}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({k: v._asdict() for k, v in build_bundle(args.config, args.output_dir).items()}, indent=2, default=str))
