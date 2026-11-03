#!/usr/bin/env python3
"""Create-only full two-clip / physical-six application release, never a board test.

Usage (from the repository root; paths may be Windows or native Linux):
  python -B -m scripts.host.build_soundjie_two_clip_release \
    --runtime-dir PATH --ui-dir PATH --classifier-dir PATH \
    --model-elf PATH/dpu_soundjie_alert5bg.elf \
    --model-sha256 ecd4c9f4d65faf5c5083520b6a19623d1df01cd94dbe0d7e9a8324656ba917c7 \
    --thresholds 0.525,0.9,0.575,0.6,0.675 --training-receipt thresholds-fixed.json \
    --bank-assets-dir PATH/bank-assets --output-dir NEW_DIRECTORY

Each ARM directory supplies the named binary and build-report.txt. Reports must
bind SHA256, ARM/v7/hard-float/interpreter/GLIBC/dependencies; classifier reports
also bind MODEL_SHA256, FRONTEND_SHA256, PHYSICAL_OUTPUTS=6, STORAGE_OUTPUTS=6,
KERNEL_NAME=soundjie_alert5bg, MODEL_BASENAME=dpu_soundjie_alert5bg.elf.
Supplied binaries are never stripped or modified. Live readelf inspection uses
WSL Ubuntu on Windows, native readelf elsewhere. No source artifact is executed.

The required training receipt binds contract, model, kernel, six outputs and
five foreground thresholds; its metrics are provenance, not onboard acceptance.
The bank-manifest.json files map contains relative names -> SHA256 strings.
Only the two launch scripts receive CRLF-to-LF normalization. All preserved base
files retain exact bytes AND tar modes. Old model/probe/TV/manifest metadata is
removed from the in-memory derived payload, never from source directories.

Output: SJ6V1.TGZ (sj-two-clip-release/), release-manifest.json, SHA256SUMS, and
SJ6V1.SHA written LAST after readback. A partial directory without SJ6V1.SHA is
incomplete and is retained for diagnosis. All validation precedes output mkdir.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import subprocess
import tarfile
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = Path("F:/MLK/DPU_ZYNQ7000/artifacts/hdmi-road-presentation-20261101a/release/SJ4U2.TGZ"
                    if os.name == "nt" else "/mnt/f/MLK/DPU_ZYNQ7000/artifacts/hdmi-road-presentation-20261101a/release/SJ4U2.TGZ")
BASE_SHA = "6ad340a970fef70bb718f2ea08523e7910956ef60164f0ec794a42cfe2301206"
CONTRACT = "soundjie-alert5bg-v3"
KERNEL = "soundjie_alert5bg"
MODEL = "dpu_soundjie_alert5bg.elf"
LABELS = ["siren", "car_horn", "doorbell", "dog_bark", "speech", "background_unknown"]
MAX_FILE = 64*1024*1024
MAX_TOTAL = 128*1024*1024
BANK_FILES = {"bank.tsv", "ambulance/reference.features", "ambulance/cues.tsv",
              "dialogue/reference.features", "dialogue/cues.tsv", "traffic.bin",
              "dialogue-horn-start.txt", "dialogue-horn-end.txt"}
DPU_NAMES = {".deephi."+name+"."+KERNEL for name in ("strtab", "metadata", "tensor", "node", "parameter", "code")}


class Entry(NamedTuple):
    data: bytes
    mode: int = 0o644


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode("utf-8")


def read_json(data):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate JSON key: "+key)
            out[key] = value
        return out
    def invalid(value):
        raise ValueError("nonfinite JSON value: "+value)
    result = json.loads(data.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid)
    require(isinstance(result, dict), "JSON object required")
    return result


def safe_name(name):
    require(isinstance(name, str) and 0 < len(name) <= 240 and
            re.fullmatch(r"[A-Za-z0-9_./-]+", name) and
            all(part not in ("", ".", "..") for part in name.split("/")) and
            PurePosixPath(name).as_posix() == name, "unsafe relative path: "+str(name))
    return name


def no_links(path):
    """Reject symbolic links and Windows junction/reparse components, including parents."""
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if not os.path.lexists(part):
            continue
        info = part.lstat()
        require(not stat.S_ISLNK(info.st_mode) and not getattr(info, "st_file_attributes", 0) & 0x400,
                "source/output path contains a link or reparse point: "+str(part))
    return path


def checked_file(path, maximum=MAX_FILE):
    path = no_links(path)
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and not info.st_mode & 0o7000 and info.st_size <= maximum,
            "nonregular, privileged-mode or oversized source: "+str(path))
    with path.open("rb") as stream:
        data = stream.read(maximum+1)
    require(len(data) == info.st_size and len(data) <= maximum, "source size changed: "+str(path))
    return data


def verify_checksums(entries):
    require("SHA256SUMS" in entries, "archive lacks SHA256SUMS")
    found = {}
    for line in entries["SHA256SUMS"].data.decode("ascii").splitlines():
        require(len(line) >= 67 and line[64:66] == "  ", "malformed checksum line")
        digest, name = line[:64], safe_name(line[66:])
        require(valid_hash(digest) and name not in found and name != "SHA256SUMS", "invalid/duplicate checksum")
        found[name] = digest
    require(set(found) == set(entries)-{"SHA256SUMS"}, "checksum inventory differs")
    require(all(sha(entries[name].data) == digest for name, digest in found.items()), "payload checksum drift")


def read_archive(path, expected_sha, prefix):
    raw = checked_file(path)
    require(valid_hash(expected_sha) and sha(raw) == expected_sha, "archive SHA256 mismatch")
    safe_name(prefix)
    # Bound decompression before tarfile processes any extension header.
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
        unpacked = stream.read(MAX_TOTAL+2*1024*1024+1)
    require(len(unpacked) <= MAX_TOTAL+2*1024*1024, "decompressed archive too large")
    entries, seen, total = {}, set(), 0
    with tarfile.open(fileobj=io.BytesIO(unpacked), mode="r:") as archive:
        for index, item in enumerate(archive):
            require(index < 2048, "too many archive members")
            name = item.name.rstrip("/") if item.isdir() else item.name
            safe_name(name)
            require(name == prefix or name.startswith(prefix+"/"), "wrong archive prefix")
            require(name.casefold() not in seen, "duplicate/case-colliding archive path")
            seen.add(name.casefold())
            require(not item.linkname and not item.pax_headers and
                    ((item.isfile() and item.mode in (0o644, 0o755)) or (item.isdir() and item.mode == 0o755)),
                    "unsafe archive type, mode, link or extension")
            if item.isdir():
                continue
            relative = safe_name(name[len(prefix)+1:])
            require(0 <= item.size <= MAX_FILE, "oversized archive member")
            total += item.size
            require(total <= MAX_TOTAL, "archive payload too large")
            data = archive.extractfile(item).read(item.size+1)
            require(len(data) == item.size, "truncated archive member")
            entries[relative] = Entry(data, item.mode)
    names = {name.casefold() for name in entries}
    for name in names:
        require(not any(str(parent) in names for parent in PurePosixPath(name).parents), "file is an ancestor of another member")
    verify_checksums(entries)
    return entries


def write_archive(path, entries):
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name, entry in sorted(entries.items()):
                    safe_name(name)
                    require(entry.mode in (0o644, 0o755), "invalid output mode")
                    item = tarfile.TarInfo("sj-two-clip-release/"+name)
                    item.size, item.mode = len(entry.data), entry.mode
                    item.mtime = item.uid = item.gid = 0
                    archive.addfile(item, io.BytesIO(entry.data))


def slice_bytes(data, offset, size):
    require(0 <= offset <= len(data) and 0 <= size <= len(data)-offset, "ELF range outside file/section")
    return data[offset:offset+size]


def elf_sections(raw, *, relocatable):
    header = struct.unpack("<16sHHIIIIIHHHHHH", slice_bytes(raw, 0, 52))
    ident, kind, machine, version, _, phoff, shoff, flags, ehsize, phsize, phnum, shsize, shnum, names_index = header
    require(ident[:7] == b"\x7fELF\x01\x01\x01" and machine == 40 and version == 1 and
            (kind == 1 if relocatable else kind in (2,3)) and flags >> 24 == 5,
            "ELF32 little-endian ARM EABI5 required")
    require(ehsize == 52 and shsize == 40 and 0 < names_index < shnum <= 4096, "invalid ELF section table")
    require(not phnum or phsize == 32, "invalid ELF program table")
    if phnum:
        slice_bytes(raw, phoff, phnum*phsize)
    require(not relocatable or (not phnum and not phoff), "model must be relocatable")
    require(relocatable or flags & 0x400, "ARM executable must use hard-float ABI")
    headers = list(struct.iter_unpack("<10I", slice_bytes(raw, shoff, shnum*40)))
    require(headers[0] == (0,)*10 and headers[names_index][1] == 3, "invalid ELF null/string section")
    strings = slice_bytes(raw, headers[names_index][4], headers[names_index][5])
    result, ranges = {}, [(0,52),(shoff,shnum*40)]
    for row in headers[1:]:
        require(row[0] < len(strings), "ELF name offset outside strings")
        end = strings.find(b"\0", row[0])
        require(end >= 0, "unterminated ELF section name")
        name = strings[row[0]:end].decode("ascii")
        require(name and name not in result, "empty/duplicate ELF section name")
        require(not relocatable or row[1] != 8, "raw model has NOBITS section")
        data = b"" if row[1] == 8 else slice_bytes(raw, row[4], row[5])
        result[name] = data
        if data:
            ranges.append((row[4],len(data)))
    ordered = sorted((offset,offset+size) for offset,size in ranges if size)
    require(all(a[1] <= b[0] for a,b in zip(ordered,ordered[1:])), "overlapping ELF sections")
    return result


def model_layout(raw):
    sections = elf_sections(raw, relocatable=True)
    require({name for name in sections if name.startswith(".deephi.")} == DPU_NAMES, "wrong model kernel/DPU sections")
    part = lambda name: sections[".deephi."+name+"."+KERNEL]
    require(all(part(name) for name in ("strtab","metadata","tensor","node","parameter","code")), "empty DPU section")
    require(len(part("metadata")) == 64 and struct.unpack("<16I",part("metadata"))[2:5] == (0,5,52), "unsupported DPU metadata")
    require(len(part("tensor")) == 6*52 and len(part("node")) == 5*160, "unsupported DPU tensor/node layout")
    tensors = list(struct.iter_unpack("<13I",part("tensor")))
    first, last = tensors[0], tensors[-1]
    require(first[0:4] == (2,64,96,1) and first[5:7] == (6144,8) and first[8] == 1, "wrong model input shape")
    require(last[0:4] == (4,1,1,6) and last[5:7] == (6,8) and last[8] == 6 and 0 <= last[7] <= 31,
            "model must expose six physical/storage output channels")
    nodes = list(struct.iter_unpack("<40I",part("node")))
    strings = part("strtab")
    for name, row in zip(("conv1","conv2","conv3","conv4","fc8"),nodes):
        require(row[0] == 1 and row[1] < len(strings) and strings[row[1]:].split(b"\0",1)[0] == name.encode(), "wrong DPU node binding")
    fc = nodes[-1]
    require(fc[19:21] == (6,8) and fc[23:28] == (2,6,1,1,1) and
            fc[29:31] == (13824,8) and fc[33:40] == (2,6,96,4,6,0,0), "FC6 parameter shape differs")
    weights = slice_bytes(part("parameter"),fc[28],fc[29])
    slice_bytes(part("parameter"),fc[18],fc[19])
    require(all(any(weights[i*2304:(i+1)*2304]) for i in range(6)), "model contains a padded/empty output row")
    describe = lambda row: dict(shape_hwc=list(row[1:4]),size=row[5],bits=row[6],fix_pos=row[7],channel_stride=row[8])
    return dict(schema=1,contract=CONTRACT,kernel=KERNEL,elf_sha256=sha(raw),physical_output_count=6,
                logical_output_count=6,storage_output_count=6,background_index=5,padding=None,
                input=describe(first),output=describe(last),dpu_sections_sha256={name:sha(sections[name]) for name in sorted(DPU_NAMES)})


def readelf(path):
    path = no_links(path)
    command = ["readelf", "-h", "-A", "-l", "-d", "--version-info"]
    if os.name == "nt":
        require(bool(path.drive) and len(path.drive) == 2, "WSL requires a local drive path")
        command = ["wsl.exe", "-d", "Ubuntu", "--", "env", "LC_ALL=C", *command,
                   "/mnt/"+path.drive[0].lower()+path.as_posix().split(":",1)[1]]
    else:
        command.append(str(path))
    result = subprocess.run(command, check=True, capture_output=True, timeout=60, env=dict(os.environ,LC_ALL="C"))
    return result.stdout.decode("utf-8")


def audit_abi(text):
    versions = [tuple(map(int,value)) for value in re.findall(r"GLIBC_(\d+)\.(\d+)",text)]
    needed = re.findall(r"Shared library: \[([^]]+)\]",text)
    require(re.search(r"Machine:\s+ARM\b",text) and re.search(r"Class:\s+ELF32\b",text) and
            "little endian" in text and "Version5 EABI" in text and re.search(r"Tag_CPU_arch:\s+v7\b",text) and
            "Tag_ABI_VFP_args: VFP registers" in text and "/lib/ld-linux-armhf.so.3" in text and
            versions and max(versions) <= (2,17), "ARMv7 hard-float / GLIBC <= 2.17 ABI audit failed")
    allowed = {"libc.so.6","libm.so.6","libpthread.so.0","libdl.so.2","librt.so.1","ld-linux-armhf.so.3",
               "libn2cube.so","libhineon.so","libasound.so.2"}
    require(needed and set(needed) <= allowed, "unsupported dynamic dependency (including dynamic libstdc++/libgcc)")
    return dict(max_glibc=".".join(map(str,max(versions))),needed=sorted(set(needed)))


def load_program(folder, name, model, model_sha, frontend_sha):
    binary = Path(folder)/name
    data = checked_file(binary)
    sections = elf_sections(data, relocatable=False)
    report_bytes = checked_file(Path(folder)/"build-report.txt",65536)
    report = {}
    for line in report_bytes.decode("utf-8").splitlines():
        if not line:
            continue
        key, separator, value = line.partition("=")
        require(separator and key not in report and re.fullmatch(r"[A-Z0-9_]+",key), "malformed/duplicate build report field")
        report[key] = value
    expected = dict(MACHINE="ARM",CPU_ARCH="v7",HARD_FLOAT="yes",ELF_INTERPRETER="/lib/ld-linux-armhf.so.3",SHA256=sha(data))
    if name == "soundjie_d4_classifier":
        expected.update(MODEL_SHA256=model_sha,FRONTEND_SHA256=frontend_sha,PHYSICAL_OUTPUTS="6",STORAGE_OUTPUTS="6",
                        KERNEL_NAME=KERNEL,MODEL_BASENAME=MODEL)
        require({key for key in sections if key.startswith(".deephi.")} == DPU_NAMES and ".symtab" in sections,
                "classifier lacks model sections/symbol table")
        raw_sections = elf_sections(model,relocatable=True)
        require(all(sections[key] == raw_sections[key] for key in DPU_NAMES), "linked classifier DPU sections differ from model")
        require(model_sha.encode() in data and MODEL.encode() in data, "classifier lacks embedded model hash/path")
    else:
        require(b"--log-max-bytes" in data and CONTRACT.encode() in data, "new contract and bounded logs required")
        if name == "soundjie_d4_runtime":
            require(b"--tv-bank" in data, "runtime lacks bank option")
    require(all(report.get(key) == value for key,value in expected.items()), "build report/binary/model mismatch: "+name)
    if "STRIP_MODE" in report or "UNSTRIPPED_SHA256" in report:
        require(report.get("STRIP_MODE") in ("debug-only","--strip-debug","strip-debug") and
                valid_hash(report.get("UNSTRIPPED_SHA256")), "invalid debug-only stripping receipt")
    inspection = readelf(binary)
    abi = audit_abi(inspection)
    require(report.get("MAX_GLIBC") == abi["max_glibc"] and
            sorted(report.get("DT_NEEDED","").split(",")) == abi["needed"], "report disagrees with actual ELF ABI")
    require(checked_file(binary) == data, "binary changed during readelf inspection")
    return data, report_bytes, inspection.encode("utf-8"), abi


def finite_floats(raw, bound):
    require(len(raw)%4 == 0, "unaligned float payload")
    values = [row[0] for row in struct.iter_unpack("<f",raw)]
    require(all(math.isfinite(value) and abs(value) <= bound for value in values), "nonfinite/out-of-range feature/model value")
    return values


def validate_cues(raw, duration):
    require(len(raw) <= 65536, "cue file too large")
    ids, end = set(), 0.
    for line in raw.decode("utf-8").splitlines():
        if not line:
            continue
        fields = line.split("\t")
        require(len(fields) == 5 and len(line.encode()) <= 512 and len(ids) < 100, "invalid cue row/count")
        cue, start, finish, speaker, text = fields
        require(0 < len(cue.encode()) <= 32 and cue not in ids and 0 < len(text.encode()) <= 256 and
                all(ord(c) >= 32 and ord(c) != 127 for c in cue+text) and speaker in ("A","B","C","?"), "invalid cue identity/speaker/text")
        start, finish = float(start), float(finish)
        require(math.isfinite(start) and math.isfinite(finish) and end <= start < finish <= duration, "invalid/overlapping cue time")
        end = finish
        ids.add(cue)


def load_bank(folder):
    folder = no_links(folder)
    require(folder.is_dir(), "bank directory missing")
    actual = set()
    for path in folder.rglob("*"):
        no_links(path)
        if not path.is_dir():
            actual.add(path.relative_to(folder).as_posix())
    require(actual == BANK_FILES | {"bank-manifest.json"}, "bank file inventory differs")
    manifest_bytes = checked_file(folder/"bank-manifest.json",1024*1024)
    manifest = read_json(manifest_bytes)
    pins = manifest.get("files")
    require(isinstance(pins,dict) and set(pins) == BANK_FILES and manifest.get("old_archive_sha256") == BASE_SHA,
            "bank pins/base binding missing")
    payload = {}
    for name, expected in pins.items():
        safe_name(name)
        data = checked_file(folder/name,3072016 if name.endswith(".features") else 65536)
        require(valid_hash(expected) and sha(data) == expected, "bank asset hash mismatch: "+name)
        payload[name] = data
    require(len(payload["bank.tsv"]) <= 4096, "bank TSV too large")
    rows = [line.split("\t") for line in payload["bank.tsv"].decode("utf-8").splitlines() if line]
    require(len(rows) == 2 and all(len(row) == 4 for row in rows), "release requires exactly two bank rows")
    ids, profiles, durations = set(), set(), {}
    for clip, profile, feature, cues in rows:
        require(0 < len(clip.encode()) <= 64 and all(ord(c) >= 32 and ord(c) != 127 for c in clip) and clip not in ids,
                "invalid/duplicate clip ID")
        directory = {"ambulance_traffic":"ambulance","dialogue":"dialogue"}.get(profile)
        require(directory and profile not in profiles and feature == directory+"/reference.features" and cues == directory+"/cues.tsv",
                "bank profile/relative assets differ")
        raw = payload[feature]
        require(len(raw) >= 16 and raw[:8] == b"SJTVF2\0\0", "invalid reference header")
        count, dims = struct.unpack_from("<II",raw,8)
        require(297 <= count <= 24000 and dims == 32 and len(raw) == 16+count*128, "invalid reference size")
        finite_floats(raw[16:],100)
        duration = ((count-1)*80+512)/16000.
        validate_cues(payload[cues],duration)
        ids.add(clip);profiles.add(profile);durations[profile] = duration
    start, end = (float(payload["dialogue-horn-"+key+".txt"].decode("ascii").strip()) for key in ("start","end"))
    require(math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= durations["dialogue"], "invalid dialogue horn interval")
    traffic = payload["traffic.bin"]
    require(len(traffic) == 20000 and traffic[:8] == b"SJTF1\0\0\0" and struct.unpack_from("<III",traffic,8) == (96,48,3), "invalid traffic model")
    values = finite_floats(traffic[20:],10000)
    require(all(value >= .00001 for value in values[96:192]), "invalid traffic normalization scale")
    payload["bank-manifest.json"] = manifest_bytes
    return payload, dict(clips=sorted(ids),profiles=sorted(profiles),dialogue_horn_start=start,dialogue_horn_end=end,
                         manifest_sha256=sha(manifest_bytes))


def thresholds_list(values):
    require(isinstance(values,(list,tuple)) and len(values) == 5, "exactly five foreground thresholds required")
    require(all(type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1 for value in values), "thresholds must be finite in [0,1]")
    return [float(value) for value in values]


def build_release(*, runtime_dir, ui_dir, classifier_dir, model_elf, model_sha256, thresholds,
                  bank_assets_dir, output_dir, training_receipt, base=DEFAULT_BASE, repo_root=ROOT,
                  hop_samples=6400):
    output = no_links(output_dir)
    if os.path.lexists(output):
        raise FileExistsError(output)
    thresholds = thresholds_list(thresholds)
    require(type(hop_samples) is int and hop_samples in (4000,6400,8000,12000), "unsupported inference hop")
    for folder in (runtime_dir,ui_dir,classifier_dir,bank_assets_dir):
        folder = no_links(folder)
        require(folder.is_dir() and folder not in output.parents, "output must not be inside an input directory")
    original = read_archive(base,BASE_SHA,"sj-alert4-release")
    required = {"runtime/soundjie_d4_runtime","runtime/soundjie_d4_classifier","runtime/soundjie_hdmi_ui",
                "runtime/xvf3800_doa","runtime/dpu-4.14.0-xilinx.ko","audio/bin/arecord","audio/lib/libasound.so.2",
                "dnndk/pkgs/lib/libn2cube.so","dnndk/pkgs/lib/libhineon.so","config/soundjie_d4_runtime.tsv",
                "config/logmel_frontend_v1.json","model/dpu_soundjie_d3.elf","tv/reference.features","tv/cues.tsv","tv/traffic.bin",
                "board/run_soundjie_d4_session.sh","board/run_soundjie_tv_session.sh","manifest/release.json"}
    require(required <= set(original) and all(original[name].data for name in required), "incomplete base application")
    model = checked_file(model_elf)
    require(valid_hash(model_sha256) and sha(model) == model_sha256, "model SHA256 mismatch")
    layout = model_layout(model)
    receipt_bytes = checked_file(training_receipt,4*1024*1024)
    receipt = read_json(receipt_bytes)
    require(receipt.get("contract") == CONTRACT and receipt.get("elf_sha256") == model_sha256 and receipt.get("kernel") == KERNEL and
            receipt.get("physical_output_count") == 6 and receipt.get("logical_output_count") == 6 and receipt.get("labels") == LABELS and
            receipt.get("background_index") == 5 and receipt.get("board_tested") is False and receipt.get("independent_acceptance") is False and
            valid_hash(receipt.get("checkpoint_sha256")) and thresholds_list(receipt.get("alert_thresholds")) == thresholds,
            "training receipt model/contract/threshold binding differs")
    if "foreground_thresholds" in receipt:
        require(thresholds_list(receipt["foreground_thresholds"]) == thresholds, "foreground threshold receipt differs")
    bank, bank_info = load_bank(bank_assets_dir)
    # These groups contain metadata/results for the previous model or TV clip.
    stale = {name for name in original if name.startswith(("model/","manifest/","probe/","tv/")) or name in
             {"SHA256SUMS","board/probe_soundjie_alert4.sh","evidence/regression.log",
              "evidence/soundjie_d4_runtime-readelf.txt","evidence/soundjie_hdmi_ui-readelf.txt","evidence/soundjie_d4_classifier-readelf.txt"}}
    payload = {name:entry for name,entry in original.items() if name not in stale}
    replaced, source_pins, abi_reports = set(), {}, {}
    def put(name, data, mode=0o644):
        payload[safe_name(name)] = Entry(data,mode)
        replaced.add(name)
    for folder, name in ((runtime_dir,"soundjie_d4_runtime"),(ui_dir,"soundjie_hdmi_ui"),(classifier_dir,"soundjie_d4_classifier")):
        data, report, inspection, abi = load_program(folder,name,model,model_sha256,sha(original["config/logmel_frontend_v1.json"].data))
        put("runtime/"+name,data,0o755)
        put("evidence/"+name+"-build-report.txt",report)
        put("evidence/"+name+"-readelf.txt",inspection)
        source_pins[str(Path(folder)/name)] = sha(data)
        source_pins[str(Path(folder)/"build-report.txt")] = sha(report)
        abi_reports[name] = abi
    repo = no_links(repo_root)
    classes_bytes = checked_file(repo/"config/soundjie_alert5bg_classes.json",65536)
    classes = read_json(classes_bytes)
    require(classes.get("contract") == CONTRACT and classes.get("labels") == LABELS and classes.get("physical_output_count") == 6 and
            classes.get("logical_output_count") == 6 and classes.get("background_index") == 5 and classes.get("speech_index") == 4,
            "class configuration differs from physical-six contract")
    put("config/classes.json",classes_bytes)
    source_pins[str(repo/"config/soundjie_alert5bg_classes.json")] = sha(classes_bytes)
    atlas_path = repo/"assets/hdmi/soundjie_two_clip_glyph_atlas.bin"
    atlas = checked_file(atlas_path,4*1024*1024)
    require(len(atlas) >= 16 and atlas[:5] == b"SJGA1" and len(atlas) == 16+struct.unpack_from("<I",atlas,12)[0]*40,
            "invalid glyph atlas")
    put("assets/hdmi/soundjie_glyph_atlas.bin",atlas)
    source_pins[str(atlas_path)] = sha(atlas)
    for name in ("run_soundjie_d4_session.sh","run_soundjie_tv_session.sh"):
        path = repo/"scripts/board"/name
        source = checked_file(path,65536)
        script = source.replace(b"\r\n",b"\n")
        require(script.startswith(b"#!/bin/sh\n") and b"\r" not in script and b"\0" not in script, "invalid board shell script")
        if name == "run_soundjie_d4_session.sh":
            require(all(token in script for token in (b"--tv-bank",CONTRACT.encode(),b"--log-max-bytes")), "launcher lacks required bank/contract/log support")
        put("board/"+name,script,0o755)
        source_pins[str(path)] = sha(source)
    put("model/"+MODEL,model)
    put("model/storage-layout.json",json_bytes(layout))
    put("model/thresholds-fixed.json",receipt_bytes)
    put("config/class-contract.txt",(CONTRACT+"\n").encode())
    put("config/alert-thresholds.txt",(",".join(format(value,".15g") for value in thresholds)+"\n").encode())
    put("config/hop-samples.txt",(str(hop_samples)+"\n").encode("ascii"))
    put("config/log-policy.txt",b"bounded-jsonl-v1\n")
    for name, data in bank.items():
        put("tv/"+name,data)
        source_pins[str(Path(bank_assets_dir)/name)] = sha(data)
    source_pins[str(Path(model_elf))] = model_sha256
    source_pins[str(Path(training_receipt))] = sha(receipt_bytes)
    unchanged = {name:sha(entry.data) for name,entry in original.items() if name not in stale|replaced}
    require(all(payload[name] == original[name] for name in unchanged), "unapproved base payload/mode mutation")
    report = dict(schema=1,contract=CONTRACT,kernel=KERNEL,model_basename=MODEL,model_sha256=model_sha256,
                  physical_output_count=6,logical_output_count=6,storage_output_count=6,background_index=5,speech_index=4,
                  thresholds=thresholds,foreground_threshold_count=5,hop_samples=hop_samples,window_samples=16000,
                  log_policy="bounded-jsonl-v1",log_max_bytes_default=1048576,application_only=True,
                  board_tested=False,sd_written=False,cold_boot_verified=False,independent_acceptance=False,boot_files_modified=False,
                  model_changed=True,runtime_changed=True,ui_changed=True,base_archive_sha256=BASE_SHA,
                  base_archive_path=str(base),archive_prefix="sj-two-clip-release",bank=bank_info,
                  training_receipt_sha256=sha(receipt_bytes),training_receipt_path="model/thresholds-fixed.json",
                  abi=abi_reports,source_sha256=source_pins,removed_base_paths=sorted(set(original)-set(payload)),
                  changed_base_paths=sorted(name for name in payload if name in original and payload[name] != original[name]),
                  unchanged_payload_sha256=unchanged,payload_sha256={name:sha(entry.data) for name,entry in sorted(payload.items())},
                  payload_modes={name:format(entry.mode,"04o") for name,entry in sorted(payload.items())},
                  verification_scope="host integrity/ABI only; no board, live replay, timing or physical acceptance claimed")
    put("manifest/release.json",json_bytes(report))
    put("SHA256SUMS","".join(sha(entry.data)+"  "+name+"\n" for name,entry in sorted(payload.items())).encode("ascii"))
    require(sum(len(entry.data) for entry in payload.values()) <= MAX_TOTAL, "output payload too large")
    verify_checksums(payload)
    # Sources are checked again before any output is created. Never overwrite,
    # clean up, strip, extractall, or change Git/board/source files.
    for path, expected in source_pins.items():
        require(sha(checked_file(path)) == expected, "source changed during build: "+path)
    require(sha(checked_file(base)) == BASE_SHA, "base changed during build")
    no_links(output)
    output.mkdir(parents=True,exist_ok=False)
    archive = output/"SJ6V1.TGZ"
    write_archive(archive,payload)
    archive_sha = sha(checked_file(archive))
    require(read_archive(archive,archive_sha,"sj-two-clip-release") == payload, "archive readback differs")
    result = dict(report,archive_sha256=archive_sha,archive_bytes=archive.stat().st_size,
                  unpacked_bytes=sum(len(entry.data) for entry in payload.values()),
                  payload_sha256={name:sha(entry.data) for name,entry in sorted(payload.items())},
                  payload_modes={name:format(entry.mode,"04o") for name,entry in sorted(payload.items())})
    with (output/"release-manifest.json").open("xb") as stream:
        stream.write(json_bytes(result))
    sums = (archive_sha+"  SJ6V1.TGZ\n").encode("ascii")
    with (output/"SHA256SUMS").open("xb") as stream:
        stream.write(sums)
    with (output/"SJ6V1.SHA").open("xb") as stream:
        stream.write(sums)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base",type=Path,default=DEFAULT_BASE,help="must match the fixed SJ4U2 SHA256")
    for name in ("runtime-dir","ui-dir","classifier-dir","model-elf","training-receipt","bank-assets-dir","output-dir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--model-sha256",required=True)
    parser.add_argument("--thresholds",required=True,help="five comma-separated foreground thresholds")
    parser.add_argument("--hop-samples",type=int,choices=(4000,6400,8000,12000),default=6400)
    args = vars(parser.parse_args(argv))
    try:
        args["thresholds"] = [float(value) for value in args["thresholds"].split(",")]
        result = build_release(**args)
    except (ValueError,OSError,tarfile.TarError,subprocess.SubprocessError) as error:
        parser.exit(2,"release refused: "+str(error)+"\n")
    print(json.dumps(dict(archive=str(Path(args["output_dir"])/"SJ6V1.TGZ"),sha256=result["archive_sha256"],
                          bytes=result["archive_bytes"],board_tested=False),indent=2))


if __name__ == "__main__":
    main()
