---
name: package-manifest-verify
description: Verify a reproducible package handoff against a SHA-256 file manifest with portable path validation. Use for offline artifact integrity checks, not deployment, device flashing, or build reproducibility certification.
---

# Package manifest verification

## Applicability

Use for any project with a package directory and a trusted SHA-256 manifest.
The Python 3.9+ standard-library CLI reads files only. It does not extract an
archive, copy a package, rewrite its manifest, run payloads, or operate hardware.

## Use

From this skill directory, pass the package root and manifest as separate paths:

```text
python -B scripts/verify_manifest.py /path/to/package /path/to/sha256sums.txt
```

The manifest can be outside the package. Paths inside it are relative to the
package root, never relative to the manifest. Use UTF-8, one SHA-256 entry per
line, in `sha256sum` text or binary-marker format:

```text
<64 hexadecimal characters><space><space>relative/path.bin
<64 hexadecimal characters><space>*relative/path.bin
```

The angle-bracket descriptions above denote fields, not literal input. Blank
lines are ignored; comments and escaped-filename variants are unsupported.
Use forward slashes; filenames may contain spaces. Entries are sorted for
deterministic JSON output; upper/lowercase digest hex is equivalent.

| Result | Exit | Action |
| --- | --- | --- |
| VERIFIED | 0 | All listed regular files match |
| MISMATCH | 1 | Inspect expected/actual hashes; preserve the mismatching artifact |
| ERROR | 2 | Correct missing files, invalid manifest, unsafe paths, or unreadable inputs |

## Failure handling

Reject absolute/drive/UNC paths, traversal, empty/dot components, backslashes,
control characters, Windows reserved names, alternate data streams, trailing
dots/spaces, duplicate/case-colliding entries, symlinks, and Windows reparse
points. Reject directories and special files as payloads. Validate every listed
path before hashing any payload. Never repair a mismatch by replacing its
expected digest without independent provenance.

Use a stable, quiescent package directory: portable path checks are not an atomic
sandbox against concurrent malicious filesystem changes. The root itself is
resolved first. This verifier checks listed bytes only; extra files are not
scanned. An empty manifest is an error. Trust and completeness of the manifest,
artifact provenance, and reproducibility of the build remain separate reviews.

## Observations

The supplied reports can be checked without copying or changing them using
`../tests/source-reports.sha256` and the source project as root. Synthetic tests
demonstrate repeatable results, tamper detection, missing-file errors, and path
rejection. The bundled `../sha256sums.txt` covers this skill package except the
manifest itself. See [shared evidence and checks](../README.md) for validation
results and limits; matching hashes do not establish board operation.
