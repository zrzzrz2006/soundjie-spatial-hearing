#!/usr/bin/env python3
"""Create a provenance-hardened Soundjie D3 bundle configuration.

This utility is deliberately create-only: it derives a new model manifest and
bundle config without modifying the previously hardware-validated package.
"""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path


SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
ESC_LINE = re.compile(
    r"^- \[(?P<clip>[^]]+)\.ogg\]: clip derived from (?P<title>.*?) "
    r"\((?P<url>https?://[^)]+)\) by (?P<creator>.*?) "
    r"\[(?P<license>[^]]+)\]\s*$"
)
ESC_LICENSES = {
    "CC0": ("CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"),
    "CC-BY": ("CC-BY-3.0", "https://creativecommons.org/licenses/by/3.0/"),
    "CC-BY-NC": (
        "CC-BY-NC-3.0",
        "https://creativecommons.org/licenses/by-nc/3.0/",
    ),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_esc50_attribution(text, transformed_filename):
    """Return the original Freesound attribution for one ESC-50 recording."""
    match = re.fullmatch(r"(?P<clip>.+)-\d+\.wav", Path(transformed_filename).name)
    if not match:
        raise ValueError(f"ESC-50 attribution filename is invalid: {transformed_filename}")
    wanted = match.group("clip")
    for line in text.splitlines():
        parsed = ESC_LINE.match(line)
        if parsed and parsed.group("clip") == wanted:
            license_name = parsed.group("license")
            if license_name not in ESC_LICENSES:
                raise ValueError(f"unsupported ESC-50 attribution license: {license_name}")
            normalized_license, license_url = ESC_LICENSES[license_name]
            source_url = parsed.group("url").replace(
                "http://www.freesound.org/", "https://freesound.org/", 1
            ).replace("https://www.freesound.org/", "https://freesound.org/", 1)
            return {
                "source_id": f"ESC-50:{wanted}",
                "license": normalized_license,
                "creator": parsed.group("creator").strip(),
                "source_url": source_url,
                "license_url": license_url,
                "source_citation": (
                    f"ESC-50 clip {wanted}, derived from "
                    f"{parsed.group('title').strip()}"
                ),
                "modification_notice": (
                    "ESC-50 source converted to mono 16 kHz and cropped to a "
                    "one-second WAV; Log-Mel tensor derived for inference"
                ),
            }
    raise ValueError(f"ESC-50 attribution not found for {transformed_filename}")


def parse_librispeech_readers(text):
    """Parse LibriSpeech SPEAKERS.TXT into speaker-id/name pairs."""
    readers = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        parts = [part.strip() for part in line.split("|", 4)]
        if len(parts) == 5 and parts[0].isdigit() and parts[4]:
            readers[parts[0]] = parts[4]
    return readers


def _record(path, name, mode="0644"):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "mode": mode,
        "name": name,
        "sha256": file_sha256(path),
        "size": path.stat().st_size,
        "source": str(path),
    }


def _member(config, name):
    matches = [item for item in config.get("archive_members", []) if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"expected one archive member: {name}")
    item = matches[0]
    source = Path(item.get("source", ""))
    if not source.is_file() or file_sha256(source) != item.get("sha256"):
        raise ValueError(f"archive member is missing or changed: {name}")
    return item


def _librispeech_attribution(recording_id, readers):
    relative = recording_id.split(":", 1)[-1]
    stem = Path(relative).stem
    speaker_id = stem.split("-", 1)[0]
    creator = readers.get(speaker_id)
    if not creator:
        raise ValueError(f"LibriSpeech reader not found: {speaker_id}")
    return {
        "source_id": f"LibriSpeech:dev-clean:{stem}",
        "license": "CC-BY-4.0",
        "creator": creator,
        "source_url": f"https://www.openslr.org/resources/12/dev-clean.tar.gz#{stem}",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_citation": (
            f"LibriSpeech dev-clean recording {stem}, reader {speaker_id} ({creator})"
        ),
        "modification_notice": (
            "LibriSpeech source converted to mono 16 kHz and cropped to a "
            "one-second WAV; Log-Mel tensor derived for inference"
        ),
    }


def upgrade_config(input_config, esc50_license, librispeech_license,
                   librispeech_speakers, git_commit, output_dir):
    """Create the upgraded config and derived manifest in a new directory."""
    if not GIT_COMMIT.fullmatch(git_commit):
        raise ValueError("git commit must be 40 lowercase hexadecimal characters")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    config = json.loads(Path(input_config).read_text(encoding="utf-8"))
    model_member = _member(config, "manifest/soundjie-d3-model-manifest.json")
    demo_member = _member(config, "manifest/demo.jsonl")
    model_manifest = json.loads(Path(model_member["source"]).read_text(encoding="utf-8"))
    model_manifest.setdefault("upstream", {})["demo_manifest_sha256"] = demo_member["sha256"]
    model_manifest["upstream"]["git_commit"] = git_commit

    derived_manifest = output_dir / "soundjie-d3-model-manifest.json"
    derived_manifest.write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    updated = copy.deepcopy(config)
    updated["git_commit"] = git_commit
    members = []
    for item in updated["archive_members"]:
        if item["name"] == "manifest/soundjie-d3-model-manifest.json":
            members.append(_record(derived_manifest, item["name"]))
        elif item["name"] == "probe/run_soundjie_d3.sh":
            members.extend((
                _record(esc50_license, "licenses/ESC-50-LICENSE"),
                _record(librispeech_license, "licenses/LibriSpeech-LICENSE.TXT"),
            ))
            members.append(item)
        else:
            members.append(item)
    updated["archive_members"] = members

    esc_text = Path(esc50_license).read_text(encoding="utf-8")
    readers = parse_librispeech_readers(
        Path(librispeech_speakers).read_text(encoding="utf-8")
    )
    for case in updated.get("cases", []):
        if case.get("split") != "demo":
            continue
        recording_id = case.get("recording_id", "")
        filename = Path(recording_id.split(":", 1)[-1]).name
        if recording_id.startswith("esc50:"):
            case["attribution"] = parse_esc50_attribution(esc_text, filename)
        elif recording_id.startswith("librispeech_dev_clean:"):
            case["attribution"] = _librispeech_attribution(recording_id, readers)
        else:
            raise ValueError(f"unsupported demo source: {recording_id}")

    output_config = output_dir / "bundle-input-config.json"
    output_config.write_text(
        json.dumps(updated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-config", required=True, type=Path)
    parser.add_argument("--esc50-license", required=True, type=Path)
    parser.add_argument("--librispeech-license", required=True, type=Path)
    parser.add_argument("--librispeech-speakers", required=True, type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = upgrade_config(**vars(args))
    print(result)


if __name__ == "__main__":
    main()
