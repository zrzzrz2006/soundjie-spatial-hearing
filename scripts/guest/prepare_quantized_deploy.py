#!/usr/bin/env python3
"""Replace a quantizer-only ImageData layer with the fixed runtime input."""

import argparse
import re
from pathlib import Path


INPUT_DECLARATION = 'input: "data"\ninput_shape { dim: 1 dim: 1 dim: 64 dim: 96 }\n'


def layer_spans(text):
    spans = []
    for match in re.finditer(r"(?m)^layer\s*\{", text):
        depth = 0
        for index in range(match.start(), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((match.start(), index + 1))
                    break
        else:
            raise ValueError("unterminated layer block")
    return spans


def prepare(text):
    image_spans = [
        span for span in layer_spans(text)
        if re.search(r'(?m)^\s*type:\s*"ImageData"\s*$', text[span[0]:span[1]])
    ]
    if len(image_spans) != 1:
        raise ValueError("expected exactly one ImageData layer, found {}".format(len(image_spans)))
    start, end = image_spans[0]
    remainder = (text[:start] + text[end:]).lstrip("\n")
    return INPUT_DECLARATION + remainder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("output already exists: {}".format(args.output))
    try:
        result = prepare(args.input.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(str(exc))
    args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
