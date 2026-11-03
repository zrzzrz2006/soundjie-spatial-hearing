#!/usr/bin/env python3
"""Create-only staging of the audited sound-shaped DPU bundle."""

import argparse
import importlib.util
import json
import os
from pathlib import Path


DESTINATIONS = ("SJD0.TGZ", "SJD0.SHA256")
MANIFEST_NAME = "SJD0.SHA256"


def _load_generic_stager():
    path = Path(__file__).with_name("stage_additive_inference_sd.py")
    spec = importlib.util.spec_from_file_location("sound_shape_generic_stager", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGER = _load_generic_stager()


def preflight(sd_root: Path, source_dir: Path, expected_volume_serial: str):
    source_dir = Path(source_dir)
    return STAGER.preflight(
        Path(sd_root), source_dir, source_dir / MANIFEST_NAME, DESTINATIONS,
        expected_volume_serial=expected_volume_serial,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage the create-only SJD0 sound DPU bundle")
    parser.add_argument("--sd-root", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--expected-volume-serial", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    plan = preflight(args.sd_root, args.source_dir, args.expected_volume_serial)
    result = STAGER.stage(plan, dry_run=args.dry_run)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8", newline="\n") as report:
        report.write(rendered + "\n")
        report.flush()
        os.fsync(report.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
