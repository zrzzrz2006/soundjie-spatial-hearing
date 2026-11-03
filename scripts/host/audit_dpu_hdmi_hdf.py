#!/usr/bin/env python3
import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import NamedTuple


class HdfAuditError(ValueError):
    pass


class HdfAudit(NamedTuple):
    path: str
    dpu_present: bool
    vdma_present: bool
    framebuffer_present: bool
    dpu_hp_ports: tuple[str, ...]
    display_hp_ports: tuple[str, ...]
    hp_conflicts: tuple[str, ...]
    clocks: tuple[str, ...]
    interrupts: tuple[str, ...]
    action: str


TEXT_MEMBERS = ("system_bd.tcl", "system.hwh", "sysdef.xml")


def _read_design_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = {name.replace("\\", "/"): name for name in archive.namelist()}
            chunks = []
            for wanted in TEXT_MEMBERS:
                actual = names.get(wanted)
                if actual:
                    chunks.append(archive.read(actual).decode("utf-8", errors="replace"))
    except (zipfile.BadZipFile, OSError) as exc:
        raise HdfAuditError(f"not_zip_hdf:{path}") from exc
    if not chunks:
        raise HdfAuditError("missing_design_metadata")
    return "\n".join(chunks)


def _ports_for_lines(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    ports = set()
    for line in text.splitlines():
        lower = line.lower()
        if any(marker in lower for marker in markers):
            ports.update(x.upper() for x in re.findall(r"S_AXI_(HP[0-3])", line, re.I))
    return tuple(sorted(ports))


def _names_matching(text: str, pattern: str) -> tuple[str, ...]:
    return tuple(sorted(set(re.findall(pattern, text, re.I))))


def audit_hdf(path: Path) -> HdfAudit:
    path = Path(path)
    text = _read_design_text(path)
    lower = text.lower()
    dpu_present = any(x in lower for x in ("xilinx.com:ip:dpu", "dpu_eu", "dpuczdx8g"))
    vdma_present = any(x in lower for x in ("axi_vdma", "xlnx,axi-vdma"))
    framebuffer_present = any(
        x in lower for x in ("vdma-fb", "topic,vdma-fb", "axi_dynclk", "rgb2dvi", "v_tc")
    )
    dpu_ports = _ports_for_lines(text, ("dpu",))
    display_ports = _ports_for_lines(text, ("vdma", "display", "framebuffer", "mm2s"))
    if not dpu_present:
        dpu_ports = ()
    if not vdma_present:
        display_ports = ()
    conflicts = tuple(sorted(set(dpu_ports) & set(display_ports))) if dpu_present and vdma_present else ()
    clocks = _names_matching(text, r"\b(?:clk_dpu(?:_2x)?|pixel_clk|vid_io_out_clk)\b")
    interrupts = _names_matching(text, r"\b(?:dpu_interrupt|mm2s_introut|vdma[^\s/]*irq)\b")

    if dpu_present and not vdma_present:
        action = "hardware_merge_required"
    elif vdma_present and not dpu_present:
        action = "dpu_hardware_missing"
    elif dpu_present and vdma_present and conflicts:
        action = "resolve_hardware_conflicts"
    elif dpu_present and vdma_present:
        action = "vivado_feasibility_report_required"
    else:
        action = "unsupported_hardware_description"

    return HdfAudit(
        path=str(path.resolve()),
        dpu_present=dpu_present,
        vdma_present=vdma_present,
        framebuffer_present=framebuffer_present,
        dpu_hp_ports=dpu_ports,
        display_hp_ports=display_ports,
        hp_conflicts=conflicts,
        clocks=clocks,
        interrupts=interrupts,
        action=action,
    )


def format_report(result: HdfAudit) -> str:
    def joined(values):
        return ",".join(values) if values else "none"

    return "\n".join(
        [
            "SOUNDJIE_D0_DPU_HDMI_HDF_REPORT_SCHEMA=1",
            f"HDF_PATH={result.path}",
            f"DPU_PRESENT={'yes' if result.dpu_present else 'no'}",
            f"VDMA_PRESENT={'yes' if result.vdma_present else 'no'}",
            f"FRAMEBUFFER_PRESENT={'yes' if result.framebuffer_present else 'no'}",
            f"DPU_HP_PORTS={joined(result.dpu_hp_ports)}",
            f"DISPLAY_HP_PORTS={joined(result.display_hp_ports)}",
            f"HP_CONFLICTS={joined(result.hp_conflicts)}",
            f"CLOCK_MARKERS={joined(result.clocks)}",
            f"INTERRUPT_MARKERS={joined(result.interrupts)}",
            f"ACTION={result.action}",
            "RESULT=pass",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only DPU/HDMI HDF feasibility audit")
    parser.add_argument("--hdf", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit_hdf(args.hdf)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = json.dumps(result._asdict(), indent=2, sort_keys=True) + "\n" if args.json else format_report(result)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
