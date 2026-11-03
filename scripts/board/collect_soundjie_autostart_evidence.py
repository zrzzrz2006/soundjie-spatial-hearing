#!/usr/bin/env python3
"""Read-only evidence from an already automatically started Soundjie session.

No driver, process, mount or persistent file is changed. Output is a compressed,
SHA-pinned JSON envelope on serial; compatible with board Python 3.5.
"""
import argparse
import base64
import gzip
import hashlib
import json
import time
from pathlib import Path


def read_jsonl(path):
    raw = path.read_bytes()
    complete = raw[:raw.rfind(b'\n') + 1]
    rows = [json.loads(line.decode('utf-8')) for line in complete.splitlines() if line]
    return rows, {'bytes': len(complete), 'sha256': hashlib.sha256(complete).hexdigest(), 'records': len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True, choices=['coldboot1', 'coldboot2', 'handoff'])
    args = parser.parse_args()
    session = Path('/run/sj-auto-session')
    runtime, runtime_source = read_jsonl(session / 'inference/timeline.jsonl')
    ui, ui_source = read_jsonl(session / 'logs/ui.jsonl')
    doa, doa_source = read_jsonl(session / 'state/doa.jsonl')
    runtime = [r for r in runtime if r.get('sequence', 0) > 0]
    if not runtime or not ui:
        raise SystemExit('missing running evidence')
    # Ignore only startup UI frames before the first published inference.
    steady_ui = [u for u in ui if u['monotonic_ms'] >= runtime[0]['timestamp_ms'] + 2000]
    pid = int((session / 'supervisor.pid').read_text().strip())
    stderr = {name: (session / 'logs' / name).read_text() for name in
              ('capture.stderr', 'runtime.stderr', 'doa.stderr', 'ui.stderr')}
    report = {
        'label': args.label, 'board_monotonic_seconds': time.monotonic(),
        'session': str(session), 'supervisor_pid': pid,
        'supervisor_cmd': Path('/proc/{}/cmdline'.format(pid)).read_bytes().replace(b'\0', b' ').decode(),
        'pid1_cmd': Path('/proc/1/cmdline').read_bytes().replace(b'\0', b' ').decode(),
        'startup_log': Path('/run/sj-startup.log').read_text(),
        'startup_script_sha256': hashlib.sha256(Path('/run/sj-init').read_bytes()).hexdigest(),
        'dpu_info': Path('/run/sj-auto-dpu.txt').read_text(),
        'asound_cards': Path('/proc/asound/cards').read_text(),
        'mounts': Path('/proc/mounts').read_text(),
        'block_stats': {name: [int(v) for v in Path('/sys/block/{}/stat'.format(name)).read_text().split()]
                        for name in ('mmcblk0', 'mmcblk1')},
        'status': json.loads((session / 'status.json').read_text()) if (session / 'status.json').exists() else None,
        'runtime_source': runtime_source, 'ui_source': ui_source, 'doa_source': doa_source,
        'runtime_windows': len(runtime),
        'runtime_span_seconds': (runtime[-1]['timestamp_ms'] - runtime[0]['timestamp_ms']) / 1000.0,
        'runtime_bad_windows': sum(not (r.get('usb_ok') and r.get('dpu_ok')) for r in runtime),
        'runtime_points': [[r.get(k) for k in ('sequence', 'timestamp_ms', 'usb_ok', 'dpu_ok', 'inference_ms', 'end_to_end_ms')] for r in runtime],
        'last_runtime': runtime[-1],
        'steady_ui_frames': len(steady_ui),
        'steady_ui_bad_frames': sum(not all(u.get(k) for k in ('usb_ok', 'dpu_ok', 'framebuffer_ok', 'classification_fresh')) or u.get('display_class_count') != 6 for u in steady_ui),
        'ui_tail': ui[-10:], 'doa_tail': doa[-10:],
        'stderr': stderr, 'capture_overruns': stderr['capture.stderr'].count('overrun!!!'),
    }
    blob = gzip.compress(json.dumps(report, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    print('SJ_EVIDENCE_BEGIN {} {} {}'.format(args.label, len(blob), hashlib.sha256(blob).hexdigest()))
    data = base64.b64encode(blob).decode('ascii')
    for pos in range(0, len(data), 76):
        print(data[pos:pos + 76])
    print('SJ_EVIDENCE_END {}'.format(args.label))


if __name__ == '__main__':
    main()
