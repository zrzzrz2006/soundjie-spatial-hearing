#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import struct
from pathlib import Path


SOURCE_SHA256 = "c31d210962408a00de8e2ebe2f2fc26824d7a4939d4eb15d347761fb2a0b39a6"
UI_TEXT = "声界当前声音置信度方向不可用强度系统正常故障音频延迟毫秒静音削波前右后左火灾或烟雾报警器应急车辆警笛汽车鸣笛玻璃破碎敲门或门铃紧急人声普通说话背景未知类别候选"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-text", type=Path)
    args = parser.parse_args()
    if digest(args.source) != SOURCE_SHA256:
        raise SystemExit("Unifont source SHA-256 mismatch")
    if args.output.exists():
        raise SystemExit("output already exists")
    wanted = set(range(0x20, 0x7F)) | {ord(ch) for ch in UI_TEXT}
    if args.extra_text:
        wanted |= {ord(ch) for ch in args.extra_text.read_text(encoding='utf-8') if ord(ch)>=0x20}
    glyphs = {}
    with gzip.open(args.source, "rt", encoding="ascii") as stream:
        for line in stream:
            code_text, bitmap = line.rstrip().split(":", 1)
            codepoint = int(code_text, 16)
            if codepoint not in wanted:
                continue
            raw = bytes.fromhex(bitmap)
            if len(raw) == 16:
                width = 8
                rows = [value << 8 for value in raw]
            elif len(raw) == 32:
                width = 16
                rows = [int.from_bytes(raw[i:i + 2], "big") for i in range(0, 32, 2)]
            else:
                continue
            glyphs[codepoint] = (width, rows)
    missing = sorted(wanted - set(glyphs))
    if missing:
        raise SystemExit("missing glyphs: " + ",".join(f"U+{value:04X}" for value in missing))
    payload = bytearray(b"SJGA1\0\0\0")
    payload += struct.pack("<II", 1, len(glyphs))
    for codepoint in sorted(glyphs):
        width, rows = glyphs[codepoint]
        payload += struct.pack("<IB3x16H", codepoint, width, *rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"GLYPHS={len(glyphs)}")
    print(f"SHA256={hashlib.sha256(payload).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
