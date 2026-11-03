#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
from typing import NamedTuple


class ModelAuditError(ValueError):
    pass


class ModelContract(NamedTuple):
    expected_size: int
    expected_sha256: str
    target: str
    architecture: str
    kernel: str
    input_node: str
    output_nodes: tuple[str, ...]


class ModelAudit(NamedTuple):
    ok: bool
    path: str
    size: int
    sha256: str
    elf_class: str
    machine: str
    eabi: str
    target: str
    architecture: str
    kernel: str
    input_node: str
    output_nodes: tuple[str, ...]


def _contains_c_string(data: bytes, value: str) -> bool:
    needle = value.encode("ascii") + b"\x00"
    start = 0
    while True:
        index = data.find(needle, start)
        if index < 0:
            return False
        if index == 0 or data[index - 1] == 0:
            return True
        start = index + 1


def audit_model(path: Path, contract: ModelContract) -> ModelAudit:
    path = Path(path)
    data = path.read_bytes()
    size = len(data)
    if size != contract.expected_size:
        raise ModelAuditError(f"size_mismatch:expected={contract.expected_size}:actual={size}")

    digest = hashlib.sha256(data).hexdigest()
    if digest.lower() != contract.expected_sha256.lower():
        raise ModelAuditError(
            f"sha256_mismatch:expected={contract.expected_sha256.lower()}:actual={digest}"
        )

    if len(data) < 52 or data[:4] != b"\x7fELF":
        raise ModelAuditError("not_elf32")
    if data[4] != 1 or data[5] != 1:
        raise ModelAuditError("not_little_endian_elf32")
    machine = int.from_bytes(data[18:20], "little")
    if machine != 40:
        raise ModelAuditError(f"wrong_elf_machine:{machine}")
    flags = int.from_bytes(data[36:40], "little")
    eabi = (flags >> 24) & 0xFF
    if eabi != 5:
        raise ModelAuditError(f"wrong_eabi:{eabi}")

    if f"DPU Target : {contract.target}".encode("ascii") not in data:
        raise ModelAuditError("missing_target_marker")
    if contract.architecture.encode("ascii") not in data:
        raise ModelAuditError("missing_architecture_marker")
    if f"_dpu_{contract.kernel}_".encode("ascii") not in data:
        raise ModelAuditError("missing_kernel_symbol")
    if not _contains_c_string(data, contract.input_node):
        raise ModelAuditError("missing_input_node")
    for node in contract.output_nodes:
        if not _contains_c_string(data, node):
            raise ModelAuditError(f"missing_output_node:{node}")

    return ModelAudit(
        ok=True,
        path=str(path.resolve()),
        size=size,
        sha256=digest,
        elf_class="ELF32",
        machine="ARM",
        eabi="EABI5",
        target=contract.target,
        architecture=contract.architecture,
        kernel=contract.kernel,
        input_node=contract.input_node,
        output_nodes=contract.output_nodes,
    )


def load_contract(config_path: Path) -> tuple[Path, ModelContract]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    model = config["model"]
    return Path(model["path"]), ModelContract(
        expected_size=int(model["size"]),
        expected_sha256=str(model["sha256"]),
        target=str(model["target"]),
        architecture=str(model["architecture"]),
        kernel=str(model["kernel"]),
        input_node=str(model["input_node"]),
        output_nodes=tuple(model["output_nodes"]),
    )


def format_text(result: ModelAudit) -> str:
    lines = [
        "MODEL_AUDIT_STATUS=pass",
        f"PATH={result.path}",
        f"SIZE={result.size}",
        f"SHA256={result.sha256}",
        f"ELF_CLASS={result.elf_class}",
        f"MACHINE={result.machine}",
        f"EABI={result.eabi}",
        f"DPU_TARGET={result.target}",
        f"DPU_ARCHITECTURE={result.architecture}",
        f"KERNEL={result.kernel}",
        f"INPUT_NODE={result.input_node}",
    ]
    lines.extend(f"OUTPUT_NODE_{index}={node}" for index, node in enumerate(result.output_nodes))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a pinned DNNDK DPU model ELF")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    model_path, contract = load_contract(args.config)
    result = audit_model(model_path, contract)
    if args.format == "json":
        print(json.dumps(result._asdict(), indent=2, sort_keys=True))
    else:
        print(format_text(result), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
