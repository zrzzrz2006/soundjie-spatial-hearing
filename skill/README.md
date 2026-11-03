# Contest skill package

Two reusable offline skills distilled from FPGA report review and artifact
handoff. This is a contest skill contribution; it adds no application runtime
features. Each folder is independently reusable with Python 3.9+ and its standard
library, and has an ASCII lowercase name plus YAML `name` and `description`.

| Skill | Shared reuse | Tested effect |
| --- | --- | --- |
| [timing-report-audit](timing-report-audit/SKILL.md) | Review another project's supplied timing report without board-specific paths | Extract four setup/hold metrics, expose absent hold/coverage, reject negative slack and unconstrained findings |
| [package-manifest-verify](package-manifest-verify/SKILL.md) | Check a stable package handoff using a chosen root and SHA-256 manifest | Repeatable hash checks, tamper/missing-file detection, rejection of traversal and unsafe Windows paths |

Use them together to bind an audit to exact report bytes. These are file-level
checks, not claims of hardware signoff, a reproducible synthesis build, or
measured productivity improvement. No PYNQ support or execution is claimed.
No runtime code, original artifacts, or hardware is modified.

## Evidence and observations

The original files under `hardware/reports/` in the repository are read-only inputs.
[tests/source-reports.sha256](tests/source-reports.sha256) records their exact
SHA-256 identities for repeat verification.

| Input | Observed evidence | Limit |
| --- | --- | --- |
| timing_summary.rpt | WNS 0.254 ns, TNS 0.000 ns; WHS/THS absent; no-clock 1, unconstrained internal endpoints 5, missing input delay 5, missing output delay 31 | `-delay_type max`; the success sentence does not cover missing hold/constraint evidence |
| utilization.rpt | Routed state; LUTs 35682/277400 (12.86%), registers 77318/554800 (13.94%), BRAM tiles 127.5/755 (16.89%), DSPs 211/2020 (10.45%) | Resource occupancy does not prove functional operation or complete timing closure |
| pl-feasibility-report.txt | DPU/VDMA/framebuffer implementation flags are 1; WNS=0.254; RESULT=PASS | Structural report statements, not an independent on-board test or hold report |

Timing and utilization headers carry `2026-10-24` report timestamps. Those are
artifact metadata, not evidence of a collaboration on that date. The feasibility
text has no date field. The [collaborative debugging discussions](../report/ai_collaboration/README.md)
describe timing-constraint and HDMI voltage reviews. The generic skill scripts
were extracted for this submission and demonstrate reusable file-level checks.

## Run checks

Commands from the repository root, using Python 3.9 or later:

```sh
python -B -m unittest discover -s skill/tests -p "test_*.py" -v
python -B skill/timing-report-audit/scripts/audit_timing.py hardware/reports/timing_summary.rpt
python -B skill/package-manifest-verify/scripts/verify_manifest.py . skill/tests/source-reports.sha256
python -B skill/package-manifest-verify/scripts/verify_manifest.py skill skill/sha256sums.txt
```

The real timing audit intentionally exits 1 with `UNCONSTRAINED`, WHS/THS `null`,
and `signoff: false`. Both manifest checks should exit 0. Unit-test fixtures are
created and removed only inside `skill/tests/`; `-B` suppresses bytecode files.
The scripts print JSON and perform no writes. The package manifest excludes
itself and temporary test files; newly added/unlisted files are not verified.

Validation performed with Python 3.12.14: 13 unit tests ran, 12 passed and the
optional real symlink test was skipped because Windows denied symlink creation
(privilege 1314). Traversal, drive/UNC paths, reserved names, alternate streams,
hash tampering, missing files, and repeated read-only verification were tested.
This skip is not evidence that link handling passed on this host.

Both skills passed the skill-creator `quick_validate.py` authoring check using
an existing PyYAML dependency. No dependency was installed or modified.
Both shipped CLIs and all tests use only the Python standard library.
