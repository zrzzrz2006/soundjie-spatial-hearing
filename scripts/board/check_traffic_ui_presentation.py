#!/usr/bin/env python3
"""Offscreen-only ARM renderer check from a saved real inference snapshot.

Writes exclusively to a NEW /run directory. Does not access audio, DPU or fb0.
"""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'atlas', 'config', 'fixture', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    # Debian 9 uses Python 3.5: resolve() requires the path to exist there.
    output = a.output.parent.resolve() / a.output.name
    if not str(output).startswith('/run/') or output.exists() or output.is_symlink():
        raise ValueError('new /run output required')
    original = a.fixture.read_bytes()
    # Keep the runtime field order on Python 3.5. The existing lightweight
    # reader expects top-level confidence before any nested top3 confidence.
    payload = json.loads(original.decode('utf-8'), object_pairs_hook=OrderedDict)
    assert payload['tv_matched'] and payload['fine_valid'] and payload['road_score'] >= .65
    output.mkdir()
    state = output / 'state.json'
    cmd = [str(a.binary), '--snapshot', str(state), '--atlas', str(a.atlas),
           '--class-contract', 'soundjie-alert4bg-v2', '--traffic-display-seed', '123',
           '--doa-jsonl', str(output / 'absent-doa.jsonl'),
           '--runtime-config', str(a.config), '--offscreen-dir', str(output / 'frames'),
           '--width', '640', '--height', '360', '--interval-ms', '150',
           '--frames', '34', '--snapshot-stale-ms', '1000']
    reports = []
    with (output / 'stderr.txt').open('wb') as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            with (output / 'ui.jsonl').open('wb') as log:
                for raw in proc.stdout:
                    log.write(raw)
                    report = json.loads(raw.decode('utf-8'))
                    reports.append(report)
                    next_state = OrderedDict(payload)
                    next_state['timestamp_ms'] = report['monotonic_ms']
                    if report['frame'] >= 26:
                        next_state.update(tv_matched=False, fine_valid=False)
                    if report['frame'] >= 30:
                        next_state['timestamp_ms'] = 0
                    temporary = output / 'next-state.json'
                    # Match the runtime's UTF-8 JSON representation.
                    temporary.write_text(json.dumps(next_state, ensure_ascii=False), encoding='utf-8')
                    temporary.replace(state)
            assert proc.wait(timeout=5) == 0
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
    assert len(reports) == 34
    active = [r for r in reports if r['sound_rows'][4]['display_basis'] == 'presentation_random_85_100']
    assert len(active) >= 20
    values = [r['sound_rows'][4]['display_score'] for r in active]
    assert all(.85 <= v <= 1 for v in values) and len(set(values)) >= 3
    for r in active:
        rows = r['sound_rows']
        assert abs(rows[4]['score'] - payload['road_score']) < 1e-6
        assert rows[4]['score_text'] == '{:.0f}%'.format(rows[4]['display_score'] * 100)
        for index in range(4):
            assert abs(rows[index]['score'] - payload['class_scores'][index]) < 1e-6
            assert rows[index]['display_score'] == rows[index]['score']
        assert rows[1]['label'] == '\u8f66\u8f86\u9e23\u7b1b'
    ordinary = [r for r in reports if r['classification_fresh'] and not r['tv_matched']]
    assert ordinary and all(r['sound_rows'][4]['display_score'] is None and
                            r['sound_rows'][4]['score_text'] == '' for r in ordinary)
    assert not reports[-1]['classification_fresh']
    assert all(r['display_score'] is None for r in reports[-1]['sound_rows'])
    assert a.fixture.read_bytes() == original
    summary = dict(passed=True, scope='saved_real_snapshot_offscreen_ui_only',
                   new_acoustic_test=False, frames=len(reports), active_frames=len(active),
                   display_values=values, raw_road_score=payload['road_score'],
                   raw_score_preserved=True, other_four_rows_unchanged=True,
                   inactive_and_stale_cleared=True,
                   fixture_sha256=hashlib.sha256(original).hexdigest(),
                   binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='ascii')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
