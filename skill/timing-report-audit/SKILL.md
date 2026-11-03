---
name: timing-report-audit
description: Audit FPGA timing reports for setup and hold metrics, missing evidence, and unconstrained paths before reviewing implementation claims. Use for offline report review, not synthesis or hardware operation.
---

# Timing report audit

## Applicability

Use for a supplied timing report from any project. The Python 3.9+ standard-library
CLI accepts any file path; it supports Vivado-style whitespace summary tables in
nanoseconds and one-per-line `WNS=...`, `TNS=...`, `WHS=...`, `THS=...` exports.
Other formats require adaptation against actual samples; unrecognized data is
incomplete evidence. Utilization and structural feasibility are supporting
evidence, not substitutes for timing or board tests.

## Use

Run from this skill directory, substituting the actual report path:

```text
python -B scripts/audit_timing.py /path/to/timing_summary.rpt
```

Read the JSON `metrics_ns`, `missing_metrics`, `coverage_counts`, `findings`, and
`errors` together. WNS/TNS describe setup; WHS/THS describe hold. Missing, blank,
NA, non-finite, or unrecognized cells stay `null`; WPWS/TPWS never fill hold cells.
The fixed-width table must have its dash separator immediately before its data
row. Select one global report per invocation; do not merge per-clock rows.

| Status | Exit | Meaning |
| --- | --- | --- |
| REPORT_CHECK_PASS | 0 | Four metrics present, worst slacks nonnegative, totals zero, recognized coverage checks clear |
| VIOLATIONS | 1 | Negative slack, failing endpoints, or an explicit timing violation |
| UNCONSTRAINED | 1 | Nonzero constraint/coverage finding; missing metrics can coexist |
| INCOMPLETE | 2 | Missing metrics, invalid/ambiguous data, or unknown coverage |
| UNREADABLE_REPORT | 2 | Missing, non-file, unreadable, or non-UTF-8 input |

Coverage requires explicit counts for no clock, unconstrained internal endpoints,
missing input delay, and missing output delay in the supported `There are/is N ...`
report sentences. A table of contents or absence of a warning does not establish
zero. Nonzero additional `check_timing` sections and exported
`Unconstrained paths: N` / `Unconstrained endpoints: N` also block a pass.
Exceptions are not automatically waived. Status precedence is violation,
unconstrained, incomplete, then report check pass; inspect all fields.

## Failure handling

Return the evidence and limitations. Request the missing hold/coverage report or
review the constraints and exceptions when needed. Do not rewrite a report,
silently substitute zeros, invoke implementation tools, or change hardware.
Even exit 0 has `signoff: false`: report freshness, report completeness, clock
intent, exceptions, CDC, pulse-width scope, and routed-design provenance still
need review. A report success sentence alone never certifies the design.

## Observations

The supplied source timing report has WNS 0.254 ns and TNS 0.000 ns, no WHS/THS,
and coverage counts 1/5/5/31 (no clock/internal endpoints/input/output delays).
It therefore returns `UNCONSTRAINED`, despite its success sentence. Its command
uses `-delay_type max`. The companion feasibility report's `RESULT=PASS` does not
add hold evidence. See [shared evidence and checks](../README.md) for source
identity and the tested effects; no historical collaboration is inferred.
