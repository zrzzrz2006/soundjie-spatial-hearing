#!/usr/bin/env python3
"""Read a timing report conservatively; never certify hardware signoff."""

import argparse
import json
import math
import re
from decimal import Decimal
from pathlib import Path

METRICS = ("WNS", "TNS", "WHS", "THS")
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
CHECKS = {
    "no_clock": r"register/latch pins? with no clock",
    "unconstrained_internal_endpoints": r"pins? that are not constrained for (?:maximum|minimum) delay(?! due to)",
    "no_input_delay": r"input ports? with no input delay specified",
    "no_output_delay": r"(?:output )?ports? with no output delay specified",
}


def number(token):
    if not NUMBER.fullmatch(token):
        return None
    value = float(token)
    return value if math.isfinite(value) and (value != 0 or Decimal(token) == 0) else None


def parse_metrics(text):
    """Use separator column spans so blank cells cannot shift hold/pulse data."""
    lines = text.expandtabs().splitlines()
    headers = [i for i, line in enumerate(lines)
               if re.match(r"^\s*WNS\(ns\)\s+TNS\(ns\)", line)]
    values, errors = {}, []
    if headers:
        if len(headers) != 1:
            return {}, ["Multiple summary tables; select one report/run."]
        i = headers[0]
        if i + 2 >= len(lines) or not re.fullmatch(r"[\s-]+", lines[i + 1]):
            return {}, ["Missing summary separator/data row."]
        spans = list(re.finditer(r"-+", lines[i + 1]))
        for span in spans:
            start, end = span.span()
            label = lines[i][start:end].strip()
            token = lines[i + 2][start:end].strip()
            metric = re.fullmatch(r"(WNS|TNS|WHS|THS|WPWS|TPWS)\(ns\)", label)
            if metric:
                key = metric[1]
                if key in values:
                    errors.append("Duplicate metric: " + key)
                values[key] = number(token)
            elif "Failing Endpoints" in label or "Total Endpoints" in label:
                values[label] = number(token)
    else:
        # A deliberately small interchange format: one NAME[ (ns) ] = VALUE per line.
        for match in re.finditer(
            r"^\s*(WNS|TNS|WHS|THS|WPWS|TPWS)(?:\(ns\))?\s*[:=]\s*(\S+)\s*$",
            text, re.MULTILINE
        ):
            key, token = match.groups()
            if key in values:
                errors.append("Duplicate metric: " + key)
            values[key] = number(token)
    if not values:
        errors.append("No supported global timing summary found.")
    return values, errors


def audit_text(text):
    values, errors = parse_metrics(text)
    metrics = {key: values.get(key) for key in METRICS}
    missing = [key for key, value in metrics.items() if value is None]
    counts = {}
    for key, phrase in CHECKS.items():
        matches = re.findall(r"There (?:are|is) (\d+) " + phrase, text, re.IGNORECASE)
        counts[key] = max(map(int, matches)) if matches else None
    findings = [key + "=" + str(value) for key, value in counts.items() if value]
    # Other nonzero check_timing findings also require review, including loops.
    sections = re.split(r"(?m)^\s*\d+\. checking (\w+)\s*$", text)
    for i in range(1, len(sections), 2):
        section, body = sections[i:i + 2]
        body = body.split("| Design Timing Summary", 1)[0]
        if section not in CHECKS and re.search(r"There (?:are|is) [1-9]\d*\b", body):
            findings.append("check_timing: " + section)
    # Also recognize a common exported coverage count; never infer zero from absence.
    for match in re.finditer(r"(?im)^\s*Unconstrained (?:paths|endpoints)\s*[:=][ \t]*(.*)$", text):
        token = match[1].strip()
        if not re.fullmatch(r"[0-9]+", token):
            errors.append("Unknown unconstrained path/endpoint count: " + token)
        elif int(token):
            findings.append("unconstrained=" + match[1])
    coverage = "unconstrained" if findings else (
        "unknown" if any(value is None for value in counts.values()) else "clear")
    violations = [key for key, value in values.items()
                  if value is not None and (value < 0 or ("Failing Endpoints" in key and value > 0))]
    for key, value in values.items():
        if value is None and key not in METRICS:
            errors.append("Missing/invalid value: " + key)
        if key in ("TNS", "THS", "TPWS") and value is not None and value > 0:
            errors.append("Negative-slack total must not be positive: " + key)
        if "Total Endpoints" in key and value == 0:
            errors.append("No analyzed endpoints: " + key)
    if re.search(r"timing constraints are not met|Slack\s*\(VIOLATED\)", text, re.IGNORECASE):
        violations.append("report states a timing violation")
    status = ("VIOLATIONS" if violations else "UNCONSTRAINED" if findings else
              "INCOMPLETE" if missing or errors or coverage == "unknown" else "REPORT_CHECK_PASS")
    return {
        "status": status, "metrics_ns": metrics, "missing_metrics": missing,
        "coverage": coverage, "coverage_counts": counts, "findings": findings,
        "violations": violations, "errors": errors, "signoff": False,
        "scope": "Reported timing and recognized coverage checks only; no hardware signoff.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Timing report path (read only)")
    args = parser.parse_args(argv)
    try:
        if not args.report.is_file():
            raise FileNotFoundError("Report is missing or is not a regular file")
        result = audit_text(args.report.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError) as exc:
        result = {"status": "UNREADABLE_REPORT", "metrics_ns": dict.fromkeys(METRICS),
                  "signoff": False, "errors": [str(exc)]}
    result["report"] = str(args.report)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "REPORT_CHECK_PASS" else (
        1 if result["status"] in ("VIOLATIONS", "UNCONSTRAINED") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
