#!/usr/bin/env python3
"""Fail-closed audit for a DNNC sound-shape probe build."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import NamedTuple


class CompileAuditError(RuntimeError):
    pass


class CompileAudit(NamedTuple):
    ok: bool
    kernel: str
    elf_sha256: str
    elf_size: int
    target: str
    cpu_arch: str
    dnnc_version: str
    dpu_target: str


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(reason, detail=""):
    suffix = ": {}".format(detail) if detail else ""
    raise CompileAuditError(reason + suffix)


def parse_markers(text):
    markers = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([A-Z_]+)=(.*)", line.strip())
        if match:
            key, value = match.groups()
            if key in markers and markers[key] != value.strip():
                fail("conflicting_marker", key)
            markers[key] = value.strip()
    return markers


def audit_kernel_info(contract, kernel_info_path):
    kernel_info_path = Path(kernel_info_path)
    if not kernel_info_path.is_file():
        fail("missing_kernel_info", str(kernel_info_path))
    text = kernel_info_path.read_text(encoding="utf-8", errors="replace")
    fields = {}
    for key, pattern in {
        "kernel": r"Kernel Name\s*:\s*(\S+)",
        "type": r"Kernel Type\s*:\s*(\S+)",
        "input": r"Input Node\(s\).*?\n\s*([A-Za-z0-9_]+)\(0\)",
        "output": r"Output Node\(s\).*?\n\s*([A-Za-z0-9_]+)\(0\)",
    }.items():
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            fail("missing_kernel_info_field", key)
        fields[key] = match.group(1)
    if fields["kernel"] != contract["kernel"]:
        fail("wrong_kernel", fields["kernel"])
    if fields["type"] != "DPUKernel":
        fail("wrong_kernel_type", fields["type"])
    if fields["input"] != contract["input"]["name"]:
        fail("wrong_runtime_input", fields["input"])
    if fields["output"] != contract["output"]["name"]:
        fail("wrong_runtime_output", fields["output"])


def audit_embedded_compiler(contract, elf_path):
    data = elf_path.read_bytes()
    version_match = re.search(rb"dnnc version (v[0-9]+\.[0-9]+)", data)
    target_match = re.search(rb"DPU Target\s*:\s*(v[0-9]+\.[0-9]+\.[0-9]+)", data)
    dpu_match = re.search(rb"DPU_([0-9]+FA)", data)
    if not (version_match and target_match and dpu_match):
        fail("missing_compiler_metadata")

    version = version_match.group(1).decode("ascii")
    target = target_match.group(1).decode("ascii")
    dpu = dpu_match.group(1).decode("ascii")
    compiler = contract["compiler"]
    if version not in compiler["accepted_dnnc_versions"]:
        fail("wrong_dnnc_version", version)
    if target != compiler["dpu_target"]:
        fail("wrong_dpu_target", target)
    if dpu != compiler["dpu"]:
        fail("wrong_embedded_dpu", dpu)
    return version, target


def audit_compile(contract, log_path, elf_path, kernel_info_path=None):
    log_path = Path(log_path)
    elf_path = Path(elf_path)
    if not log_path.is_file():
        fail("missing_log", str(log_path))
    if not elf_path.is_file():
        fail("missing_elf", str(elf_path))
    if elf_path.stat().st_size == 0:
        fail("empty_elf", str(elf_path))
    dnnc_version, dpu_target = audit_embedded_compiler(contract, elf_path)
    if kernel_info_path is not None:
        audit_kernel_info(contract, kernel_info_path)

    text = log_path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"unsupported\s+(?:layer|operator|op)", text, re.IGNORECASE):
        fail("unsupported_layer")
    if re.search(
        r"CPUKernel[^\n]*(?:Convolution|Pooling|InnerProduct|\bConv\b|\bPool\b|\bFC\b)",
        text,
        re.IGNORECASE,
    ):
        fail("core_cpu_fallback")

    markers = parse_markers(text)
    required = ("TARGET", "CPU", "KERNELS", "INPUT", "OUTPUT", "SUBGRAPHS", "FALLBACK")
    missing = [key for key in required if not markers.get(key)]
    if missing:
        fail("missing_markers", ",".join(missing))

    expected_target = contract["compiler"]["dpu"]
    expected_cpu = contract["compiler"]["cpu_arch"]
    if markers["TARGET"] != expected_target:
        fail("wrong_target", markers["TARGET"])
    if markers["CPU"] != expected_cpu:
        fail("wrong_cpu_arch", markers["CPU"])
    if markers["INPUT"] != contract["input"]["name"]:
        fail("wrong_input", markers["INPUT"])
    if markers["OUTPUT"] != contract["output"]["name"]:
        fail("wrong_output", markers["OUTPUT"])

    kernels = [item.strip() for item in markers["KERNELS"].split(",") if item.strip()]
    if len(kernels) != 1:
        fail("kernel_count", str(len(kernels)))
    if kernels[0] != contract["kernel"]:
        fail("wrong_kernel", kernels[0])
    if markers["SUBGRAPHS"] != "1":
        fail("subgraph_count", markers["SUBGRAPHS"])

    fallback = markers["FALLBACK"]
    if fallback.lower() != "none":
        if re.search(r"Convolution|Pooling|InnerProduct|\bConv\b|\bPool\b|\bFC\b", fallback, re.IGNORECASE):
            fail("core_cpu_fallback", fallback)
        fail("unexplained_cpu_fallback", fallback)

    return CompileAudit(
        ok=True,
        kernel=kernels[0],
        elf_sha256=file_sha256(elf_path),
        elf_size=elf_path.stat().st_size,
        target=markers["TARGET"],
        cpu_arch=markers["CPU"],
        dnnc_version=dnnc_version,
        dpu_target=dpu_target,
    )


def write_report(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "RESULT=pass",
        "KERNEL={}".format(result.kernel),
        "TARGET={}".format(result.target),
        "CPU_ARCH={}".format(result.cpu_arch),
        "DNNC_VERSION={}".format(result.dnnc_version),
        "DPU_TARGET={}".format(result.dpu_target),
        "ELF_SIZE={}".format(result.elf_size),
        "ELF_SHA256={}".format(result.elf_sha256),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--elf", required=True, type=Path)
    parser.add_argument("--kernel-info", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = audit_compile(contract, args.log, args.elf, args.kernel_info)
    write_report(args.report, result)
    print("RESULT=pass")


if __name__ == "__main__":
    main()
