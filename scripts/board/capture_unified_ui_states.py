#!/usr/bin/env python3
"""Read actual HDMI frames when the live UI reports each requested state.

Diagnostic observer only: does not change snapshots, scores, models or display.
Requires capture_hdmi_readonly.py beside this script. Python 3.5 compatible.
"""
import argparse
import json
import os
import time

from capture_hdmi_readonly import capture


def latest(path):
    with open(path, 'rb') as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 65536))
        raw = stream.read()
    lines = raw.split(b'\n')
    # A producer may be halfway through its newest JSON line.
    for line in reversed(lines[:-1]):
        try:
            return json.loads(line.decode('utf-8'))
        except (ValueError, UnicodeError):
            pass
    return None


def state_name(state):
    if not state or state.get('layout') != 'unified_live_caption_v2':
        return None
    if not all(state.get(k) for k in
               ('classification_fresh', 'usb_ok', 'dpu_ok', 'framebuffer_ok')):
        return None
    if state.get('session_state') != 'running':
        return None
    age = time.monotonic() * 1000 - state.get('monotonic_ms', 0)
    if not 0 <= age <= 1000:
        return None
    if state.get('tv_matched'):
        if state.get('caption_visible') and state.get('fine_events_visible'):
            return 'scene-caption'
    elif state.get('alert_active'):
        return 'ordinary-alert'
    elif state.get('legacy_class_id') == 4:
        return 'background'
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jsonl', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    source = os.path.realpath(args.jsonl)
    output = os.path.abspath(args.output)
    if not source.startswith('/run/') or os.path.dirname(output) != '/run':
        raise ValueError('input and new output must be under /run')
    if not 1 <= args.timeout <= 600:
        raise ValueError('timeout outside 1..600 seconds')
    os.mkdir(output)  # create-only, never overwrite an earlier evidence directory
    captured = set()
    end = time.monotonic() + args.timeout
    with open(os.path.join(output, 'captures.jsonl'), 'x') as log:
        while time.monotonic() < end and len(captured) < 3:
            try:
                before = latest(source)
                name = state_name(before)
                if name and name not in captured:
                    started = time.monotonic()
                    capture(os.path.join(output, name + '.png'))
                    after = latest(source)
                    log.write(json.dumps(dict(
                        state=name, before=before, after=after,
                        started_monotonic_ms=int(started * 1000),
                        finished_monotonic_ms=int(time.monotonic() * 1000),
                        scope='actual framebuffer read; capture can span a frame update'
                    ), ensure_ascii=True) + '\n')
                    log.flush()
                    captured.add(name)
            except (OSError, ValueError) as error:
                # Report the failure. Never silently retry a partly-created PNG.
                log.write(json.dumps(dict(error=str(error))) + '\n')
                log.flush()
                return 2
            time.sleep(0.1)
    print(json.dumps(dict(captured=sorted(captured), complete=len(captured) == 3)))
    return 0 if len(captured) == 3 else 3


if __name__ == '__main__':
    raise SystemExit(main())
