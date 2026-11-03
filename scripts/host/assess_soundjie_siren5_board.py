"""Compare physical DPU probe outputs with the pinned fixed-point references.

This reads archives without extracting them and reports conversion agreement.
It does not tune thresholds or claim independent acoustic acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_archive(path):
    result = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name:
                raise ValueError("unsafe archive path")
            if member.isdir():
                continue
            if not member.isfile() or member.name in result:
                raise ValueError("nonregular or duplicate archive member")
            result[member.name] = archive.extractfile(member).read()
    return result


def parse_output(data):
    values = data.decode("ascii").split()
    if len(values) != 9:
        raise ValueError("expected class, confidence, milliseconds and six probabilities")
    class_id = int(values[0])
    numbers = np.asarray([float(v) for v in values[1:]], dtype=np.float64)
    confidence, elapsed = numbers[:2]
    scores = numbers[2:]
    if not np.isfinite(numbers).all() or not 0 <= class_id < 6 or elapsed <= 0:
        raise ValueError("invalid DPU numeric output")
    if np.any(scores < 0) or np.any(scores > 1) or abs(scores.sum() - 1) > 1e-6:
        raise ValueError("unnormalized DPU probabilities")
    if class_id != int(scores.argmax()) or abs(confidence - scores[class_id]) > 1e-8:
        raise ValueError("DPU top1/confidence disagrees with probabilities")
    return class_id, float(elapsed), scores


def merge(scores):
    return np.asarray([scores[0] + scores[1], *scores[2:]], dtype=np.float64)


def decide(scores):
    top = int(scores.argmax())
    thresholds = [.4, .6500000000000002, .4, .4]
    return 4 if top < 4 and scores[top] < thresholds[top] else top


def assess(package, manifest, results):
    pinned = json.loads(manifest.read_text(encoding="utf-8"))
    if sha(package) != pinned["archive_sha256"]:
        raise ValueError("package hash mismatch")
    payload = read_archive(package)
    for name, expected in pinned["payload_sha256"].items():
        if hashlib.sha256(payload["sj-siren5/" + name]).hexdigest() != expected:
            raise ValueError("package payload hash mismatch: " + name)
    reference = json.loads(payload["sj-siren5/probe/reference.json"])
    if len(reference["records"]) != 28:
        raise ValueError("expected 28 references")
    files = read_archive(results)
    basenames = {}
    for name, value in files.items():
        key = PurePosixPath(name).name
        if key in basenames:
            raise ValueError("duplicate result basename")
        basenames[key] = value
    expected_names = {PurePosixPath(r["file"]).name + suffix
                      for r in reference["records"] for suffix in (".txt", ".stderr")}
    if set(basenames) != expected_names:
        raise ValueError("probe result set differs from 28 paired outputs")
    records = []
    for ref in reference["records"]:
        name = PurePosixPath(ref["file"]).name
        top, ms, actual = parse_output(basenames[name + ".txt"])
        logits = np.asarray(ref["caffe_fixed_logits"], dtype=np.float64)
        if logits.shape != (6,) or not np.isfinite(logits).all():
            raise ValueError("invalid fixed-point reference")
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        physical5, fixed5 = merge(actual), merge(probs)
        records.append(dict(file=name, recording_id=ref["recording_id"],
            original_target=ref["original_target"], physical6=actual.tolist(),
            fixed6=probs.tolist(), physical_top1=top, fixed_top1=int(probs.argmax()),
            physical5=physical5.tolist(), fixed5=fixed5.tolist(),
            physical_decision=decide(physical5), fixed_decision=decide(fixed5),
            max_abs_probability_error=float(np.abs(actual - probs).max()), inference_ms=ms,
            stderr=basenames[name + ".stderr"].decode("utf-8", errors="replace")))
    times = np.asarray([r["inference_ms"] for r in records])
    return dict(schema=1, package_sha256=sha(package), results_sha256=sha(results),
        model_sha256=pinned["model_sha256"], protocol_valid=True, count=len(records),
        physical6_top1_agreement=sum(r["physical_top1"] == r["fixed_top1"] for r in records),
        merged5_decision_agreement=sum(r["physical_decision"] == r["fixed_decision"] for r in records),
        max_abs_probability_error=max(r["max_abs_probability_error"] for r in records),
        inference_ms=dict(min=float(times.min()), median=float(np.median(times)),
                          p95=float(np.percentile(times, 95)), max=float(times.max())),
        scope="physical DPU on development replay PCM; conversion check, not independent acoustic acceptance",
        independent_acceptance=False, records=records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package", "manifest", "results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.package, args.manifest, args.results)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
