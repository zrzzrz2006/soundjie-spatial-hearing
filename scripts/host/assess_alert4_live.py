#!/usr/bin/env python3
"""Assess a captured curated replay without extraction, playback or board access.

CLI: python -m scripts.host.assess_alert4_live --session session.tar.gz
       --playlist replay-a/playlist --output new-report.json

Requires NumPy and SciPy. Output is create-only (or stdout if --output omitted).
Reception gates are fixed diagnostic criteria, not tuned accuracy thresholds.
The schedule is selected development material, never independent ground truth.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import wave

import numpy as np
from scipy.signal import correlate


RATE = 16000
RETAIN_LIMIT = 240
CONTRACT = 'soundjie-alert4bg-v2'
MAX_SAMPLES = 180 * RATE
GATES = dict(global_coverage=.90, global_raw_correlation=.20,
             global_envelope_correlation=.50, event_coverage=.98,
             event_correlation=.20, event_rms_dbfs=-60., event_clipping_rate=.01,
             local_search_ms=250)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def _json(raw):
    def constant(value):
        raise ValueError('non-finite JSON number: ' + value)
    return json.loads(raw, object_pairs_hook=_object, parse_constant=constant)


def _jsonl(raw, name):
    result = []
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = _json(line)
            require(isinstance(value, dict), 'expected JSON object')
        except ValueError as error:
            raise ValueError('%s line %d: %s' % (name, number, error)) from error
        result.append(value)
    return result


def _member_name(name):
    while name.startswith('./'):
        name = name[2:]
    name = name.rstrip('/')
    require(name and not name.startswith('/') and '\\' not in name and ':' not in name
            and '\0' not in name and all(x not in ('', '.', '..') for x in name.split('/')),
            'unsafe archive member name: ' + repr(name))
    return name


def load_session(source):
    """Read selected regular members in memory; never call tar.extract*()."""
    kwargs = {'fileobj': source} if hasattr(source, 'read') else {'name': str(source)}
    with tarfile.open(mode='r:*', **kwargs) as stream:
        members, total, ignored = {}, 0, 0
        for member in stream:
            if member.name in ('.', './') and member.isdir():
                continue
            name = _member_name(member.name)
            require(name not in members, 'duplicate archive member: ' + name)
            require(not member.issym() and not member.islnk(), 'archive links are unsupported: ' + name)
            require(len(members) < 10000, 'too many archive members')
            total += member.size
            require(0 <= member.size <= 64 * 1024 * 1024 and total <= 512 * 1024 * 1024,
                    'archive exceeds assessment size limits')
            members[name] = member
            if not member.isfile() and not member.isdir():
                ignored += 1  # The supervisor's pcm.fifo is metadata, never opened.
        suffix = 'inference/timeline.jsonl'
        timelines = [name for name in members if name == suffix or name.endswith('/' + suffix)]
        require(len(timelines) == 1, 'expected exactly one inference/timeline.jsonl')
        timeline_name = timelines[0]
        prefix = timeline_name[:-len(suffix)]

        def read(name, optional=False):
            member = members.get(name)
            if member is None and optional:
                return b''
            require(member is not None and member.isfile(), 'missing regular payload: ' + name)
            with stream.extractfile(member) as handle:
                data = handle.read(member.size + 1)
            require(len(data) == member.size, 'truncated archive payload: ' + name)
            return data

        timeline = _jsonl(read(timeline_name), timeline_name)
        require(timeline, 'empty inference timeline')
        payloads = {}
        for name in members:
            if not name.startswith(prefix + 'inference/'):
                continue
            basename = name[len(prefix + 'inference/'):]
            match = re.fullmatch(r'capture-([1-9][0-9]*)\.s16', basename)
            if match and int(match.group(1)) <= RETAIN_LIMIT:
                payloads[basename] = read(name)
        result = dict(timeline=timeline, payloads=payloads, session_prefix=prefix,
                      ignored_special_members=ignored)
        missing = []
        for key, relative in [('ui', 'logs/ui.jsonl'), ('doa', 'state/doa.jsonl')]:
            raw = read(prefix + relative, optional=True)
            result[key] = _jsonl(raw, relative)
            if not raw:
                missing.append(relative)
        result['missing_reports'] = missing
        return result


def _integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid ' + label)
    return value


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def reconstruct(timeline, payloads):
    """Merge [window_end_sample-16000, window_end_sample); holes stay unobserved."""
    windows, previous, last_end, unretained = [], 0, 0, 0
    for item in timeline:
        sequence = _integer(item.get('sequence'), 'sequence', 1)
        require(sequence > previous, 'duplicate or out-of-order timeline sequence')
        previous = sequence
        end = _integer(item.get('window_end_sample', 0), 'window_end_sample')
        if end == 0:
            require(not item.get('capture_file'), 'payload without a sample coordinate')
            continue
        require(end >= RATE and end > last_end, 'invalid or out-of-order window end')
        last_end = end
        require(item.get('class_contract') == CONTRACT, 'timeline class contract differs')
        require(type(item.get('class_id')) is int and -1 <= item['class_id'] <= 4,
                'invalid decision class')
        require(_finite(item.get('timestamp_ms')) and item['timestamp_ms'] >= 0,
                'invalid reception timestamp')
        require(type(item.get('usb_ok')) is bool and type(item.get('dpu_ok')) is bool,
                'missing inference health evidence')
        if sequence > RETAIN_LIMIT:
            unretained += 1
            continue
        name = 'capture-%d.s16' % sequence
        require(item.get('capture_file') == name, 'missing or mismatched retained capture_file')
        require(name in payloads, 'missing retained payload: ' + name)
        require(isinstance(payloads[name], bytes) and len(payloads[name]) == 32000,
                'capture must contain exactly 32000 bytes: ' + name)
        windows.append(item)
    require(windows, 'no retained PCM windows to assess')
    size = windows[-1]['window_end_sample']
    require(size <= MAX_SAMPLES, 'retained sample span exceeds 180 seconds')
    samples, observed = np.zeros(size, dtype='<i2'), np.zeros(size, dtype=bool)
    for item in windows:
        end = item['window_end_sample']; start = end - RATE
        pcm = np.frombuffer(payloads[item['capture_file']], dtype='<i2')
        overlap = observed[start:end]
        require(np.array_equal(samples[start:end][overlap], pcm[overlap]),
                'conflicting PCM overlap at sequence %d' % item['sequence'])
        samples[start:end] = pcm
        observed[start:end] = True
    edges = np.flatnonzero(np.diff(np.r_[False, ~observed, False]))
    gaps = [dict(start_sample=int(a), end_sample=int(b)) for a, b in zip(edges[::2], edges[1::2])]
    report = dict(sample_rate=RATE, retained_sequence_limit=RETAIN_LIMIT,
                  retained_windows=len(windows), unretained_windows=unretained,
                  first_sequence=windows[0]['sequence'], last_sequence=windows[-1]['sequence'],
                  span_samples=size, span_seconds=size / RATE,
                  observed_samples=int(observed.sum()), missing_samples=int((~observed).sum()),
                  gaps=gaps, capture_gap_states=sum(r.get('doa_reason') == 'capture_gap' for r in timeline),
                  timeline_rows=len(timeline),
                  coordinate_rule='one-second windows end at window_end_sample; holes are not silence')
    return dict(samples=samples, observed=observed, windows=windows, report=report)


def _ncc(samples, observed, reference, minimum_coverage, lag_min=None, lag_max=None):
    """Masked zero-mean normalized cross-correlation, including partial overlap."""
    x = np.asarray(samples, dtype=np.float64)
    mask = np.asarray(observed, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    require(x.ndim == y.ndim == mask.ndim == 1 and len(x) == len(mask)
            and len(x) > 0 and len(y) > 0 and np.isfinite(x).all() and np.isfinite(y).all(),
            'invalid correlation input')
    x = x * mask
    lags = np.arange(1 - len(y), len(x))
    left = np.maximum(lags, 0)
    right = np.minimum(lags + len(y), len(x))

    def sums(values):
        prefix = np.r_[0., np.cumsum(values)]
        return prefix[right] - prefix[left]

    count = sums(mask)
    n = np.maximum(count, 1.)
    sx, sxx = sums(x), sums(x * x)
    sy = correlate(mask, y, mode='full', method='fft')
    syy = correlate(mask, y * y, mode='full', method='fft')
    xy = correlate(x, y, mode='full', method='fft')
    vx, vy = np.maximum(sxx - sx * sx / n, 0), np.maximum(syy - sy * sy / n, 0)
    denominator = np.sqrt(vx * vy)
    valid = (count >= minimum_coverage * len(y)) & (denominator > 1e-8)
    if lag_min is not None:
        valid &= lags >= lag_min
    if lag_max is not None:
        valid &= lags <= lag_max
    if not np.any(valid):
        return dict(offset_samples=None, correlation=None, absolute_correlation=None, coverage=0.)
    scores = np.zeros_like(xy)
    np.divide(xy - sx * sy / n, denominator, out=scores, where=valid)
    scores = np.clip(scores, -1., 1.)
    rank = np.where(valid, np.abs(scores), -1.)
    best = int(np.argmax(rank))
    return dict(offset_samples=int(lags[best]), correlation=float(scores[best]),
                absolute_correlation=float(abs(scores[best])), coverage=float(count[best] / len(y)))


def align_reference(samples, observed, reference):
    raw = _ncc(samples, observed, reference, GATES['global_coverage'])
    # A 10-ms RMS envelope tolerates microphone filtering/clock drift. It may
    # choose the coarse offset only; every event still needs local waveform NCC.
    block = 160
    a, b = len(samples) // block, len(reference) // block
    envelope = dict(offset_samples=None, correlation=None, absolute_correlation=None, coverage=0.)
    if a and b:
        x = np.asarray(samples[:a * block], dtype=float).reshape(a, block)
        y = np.asarray(reference[:b * block], dtype=float).reshape(b, block)
        mask = np.asarray(observed[:a * block]).reshape(a, block)
        envelope = _ncc(np.sqrt(np.mean(x * x, axis=1)), mask.mean(1) >= .98,
                        np.sqrt(np.mean(y * y, axis=1)), GATES['global_coverage'])
        if envelope['offset_samples'] is not None:
            envelope['offset_samples'] *= block
    method = 'raw' if (raw['absolute_correlation'] or 0) >= GATES['global_raw_correlation'] else 'envelope'
    chosen = raw if method == 'raw' else envelope
    threshold = GATES['global_raw_correlation'] if method == 'raw' else GATES['global_envelope_correlation']
    passed = (chosen['absolute_correlation'] or 0) >= threshold
    if method == 'envelope':
        passed = passed and chosen['correlation'] > 0
    return dict(raw=raw, envelope=envelope, method=method,
                offset_samples=chosen['offset_samples'] if passed else None,
                offset_seconds=chosen['offset_samples'] / RATE if passed else None,
                gate_pass=passed, reasons=[] if passed else ['weak_or_absent_global_reception'],
                offset_convention='capture_sample = playback_sample + offset_samples',
                polarity_inversion_allowed=True)


def assess_events(reconstruction, reference, schedule, alignment):
    samples, observed = reconstruction['samples'], reconstruction['observed']
    result = []
    jitter = round(GATES['local_search_ms'] * RATE / 1000)
    for event in schedule['events']:
        start, end = event['start_sample'], event['end_sample']
        ref = reference[start:end]
        reasons = list(alignment.get('reasons', []))
        local = dict(offset_samples=None, correlation=None, absolute_correlation=None, coverage=0.)
        offset = alignment.get('offset_samples')
        predicted = start + offset if offset is not None else None
        if predicted is not None:
            a, b = max(0, predicted - jitter), min(len(samples), predicted + len(ref) + jitter)
            if a < b:
                local = _ncc(samples[a:b], observed[a:b], ref, GATES['event_coverage'],
                             predicted - jitter - a, predicted + jitter - a)
                if local['offset_samples'] is not None:
                    local['offset_samples'] += a
        aligned_start = local['offset_samples'] if local['offset_samples'] is not None else predicted
        coverage, rms_dbfs, clipping = 0., None, None
        if aligned_start is not None:
            a, b = max(0, aligned_start), min(len(samples), aligned_start + len(ref))
            received = samples[a:b][observed[a:b]] if a < b else np.array([], dtype='<i2')
            coverage = len(received) / len(ref)
            if len(received):
                rms = float(np.sqrt(np.mean((received.astype(float) / 32768.) ** 2)))
                rms_dbfs = 20 * math.log10(rms) if rms > 0 else -120.
                clipping = float(np.mean((received == -32768) | (received == 32767)))
        if not alignment['gate_pass']:
            reasons.append('global_alignment_unconfirmed')
        if coverage < GATES['event_coverage']:
            reasons.append('insufficient_captured_event_coverage')
        if (local['absolute_correlation'] or 0) < GATES['event_correlation']:
            reasons.append('weak_or_absent_event_waveform_match')
        if rms_dbfs is None or rms_dbfs < GATES['event_rms_dbfs']:
            reasons.append('received_level_too_low')
        if clipping is not None and clipping > GATES['event_clipping_rate']:
            reasons.append('excessive_received_clipping')
        if reconstruction['report']['capture_gap_states']:
            reasons.append('runtime_capture_clock_discontinuity')
        quality = not reasons
        full, boundary = [], []
        if aligned_start is not None:
            aligned_end = aligned_start + len(ref)
            for item in reconstruction['windows']:
                right = item['window_end_sample']; left = right - RATE
                if left >= aligned_start and right <= aligned_end:
                    full.append(item)
                elif right > aligned_start and left < aligned_end:
                    boundary.append(item)
        eligible = [r for r in full if r['usb_ok'] and r['dpu_ok'] and 0 <= r['class_id'] <= 4]
        matches = [r for r in eligible if r['class_id'] == event['target_index']]
        assessable = quality and bool(eligible)
        status = 'not_assessable'
        if assessable:
            status = 'observed_match' if len(matches) == len(eligible) else 'mixed' if matches else 'observed_mismatch'
        first = matches[0] if matches and assessable else None
        processing = first.get('end_to_end_ms') if first else None
        first_delay = ((first['window_end_sample'] - aligned_start) * 1000 / RATE + processing
                       if first and _finite(processing) and processing >= 0 else None)
        overlap_matches = sorted((r for r in full + boundary
                                  if r['usb_ok'] and r['dpu_ok'] and r['class_id'] == event['target_index']),
                                 key=lambda r: r['window_end_sample'])
        first_overlap = overlap_matches[0] if overlap_matches and quality else None
        overlap_processing = first_overlap.get('end_to_end_ms') if first_overlap else None
        overlap_delay = ((first_overlap['window_end_sample'] - aligned_start) * 1000 / RATE + overlap_processing
                         if first_overlap and _finite(overlap_processing) and overlap_processing >= 0 else None)
        result.append(dict(
            event_index=event['event_index'], target_index=event['target_index'],
            expected_display=event.get('expected_display'), recording_id=event.get('recording_id'),
            scheduled_start_s=start / RATE, scheduled_end_s=end / RATE,
            aligned_capture_start_sample=aligned_start,
            local_offset_samples=aligned_start - start if aligned_start is not None else None,
            reception=dict(gate_pass=quality, reasons=sorted(set(reasons)), coverage=coverage,
                           waveform_correlation=local['correlation'],
                           absolute_waveform_correlation=local['absolute_correlation'],
                           rms_dbfs=rms_dbfs, clipping_rate=clipping),
            full_window_count=len(full), boundary_window_count=len(boundary),
            eligible_window_count=len(eligible), unhealthy_window_count=len(full) - len(eligible),
            decision_counts=dict(sorted(Counter(str(r['class_id']) for r in full).items())),
            boundary_decision_counts=dict(sorted(Counter(str(r['class_id']) for r in boundary).items())),
            decision_window_scope='decision_counts/status use fully-contained one-second windows; boundary decisions are separate diagnostics',
            rejected_window_count=sum(r.get('classification_rejected') is True for r in full),
            decision_status=status, expected_decision_windows=len(matches) if assessable else None,
            first_expected_sequence=first['sequence'] if first else None,
            first_expected_window_end_reference_s=(start + first['window_end_sample'] - aligned_start) / RATE if first else None,
            first_expected_result_after_aligned_onset_ms=first_delay,
            first_expected_overlapping_sequence=first_overlap['sequence'] if first_overlap else None,
            first_expected_overlapping_result_after_aligned_onset_ms=overlap_delay,
            first_expected_overlapping_window_start_from_onset_ms=(first_overlap['window_end_sample'] - RATE - aligned_start) * 1000 / RATE if first_overlap else None,
            timing_scope='first_expected_result uses fully-contained windows; overlapping timing may include preceding-event context; both include receive-to-result processing and neither is physical end-to-end latency',
            exclusions=[] if eligible else ['no healthy fully-contained retained decision windows'],
            independent_accuracy=False))
    return result


def _stats(values):
    values = list(values)
    valid = [float(x) for x in values if _finite(x) and x >= 0]
    result = dict(count=len(valid), unavailable_count=len(values) - len(valid))
    for key, percentile in [('p50', 50), ('p90', 90), ('p95', 95), ('p99', 99)]:
        result[key] = float(np.percentile(valid, percentile)) if valid else None
    result['max'] = max(valid) if valid else None
    return result


def summarize_reports(timeline, ui, doa):
    from bisect import bisect_right
    inference = [r for r in timeline if r.get('window_end_sample', 0) > 0]
    latency = {name: _stats(r.get(field) for r in inference) for name, field in [
        ('receive_to_result', 'end_to_end_ms'), ('clip_match', 'clip_match_ms'),
        ('fine_model', 'fine_model_ms'), ('classifier_process', 'classifier_ms'), ('dpu', 'inference_ms')]}
    background_score, background_alert, suppression, missing = [], [], [], []
    for index, r in enumerate(ui):
        fields = ['class_contract', 'legacy_class_id', 'classification_fresh', 'session_state',
                  'classification_rejected', 'main_score_visible', 'main_display_confidence',
                  'legacy_class_scores', 'alert_active', 'usb_ok', 'dpu_ok']
        if any(key not in r for key in fields):
            missing.append(index)
        background = r.get('legacy_class_id') == 4 or r.get('display_label') == '背景'
        inactive = (background or r.get('classification_rejected') is True
                    or r.get('classification_fresh') is not True or r.get('session_state') != 'running'
                    or r.get('usb_ok') is not True or r.get('dpu_ok') is not True)
        scores = (r.get('main_score_visible') is True or r.get('main_display_confidence') is not None
                  or bool(r.get('legacy_class_scores')))
        alert = r.get('alert_active') is True
        if background and scores:
            background_score.append(index)
        if background and alert:
            background_alert.append(index)
        if inactive and (scores or alert):
            suppression.append(index)
    completed = sorted((r['timestamp_ms'] + r['end_to_end_ms'], r['timestamp_ms']) for r in inference
                       if _finite(r.get('timestamp_ms')) and _finite(r.get('end_to_end_ms'))
                       and r['end_to_end_ms'] >= 0)
    completion_times = [r[0] for r in completed]
    estimates = []
    for r in ui:
        now, paint, fb = r.get('monotonic_ms'), r.get('paint_ms'), r.get('framebuffer_ms')
        if r.get('classification_fresh') is not True or not all(_finite(v) and v >= 0 for v in (now, paint, fb)):
            continue
        index = bisect_right(completion_times, now) - 1
        if index >= 0:
            estimates.append(now + paint + fb - completed[index][1])
    exact_age = _stats([])
    exact_age['reason'] = 'UI reports omit the source snapshot timestamp/sequence; exact frame age cannot be measured'
    estimated_age = _stats(estimates)
    estimated_age['scope'] = 'estimate using latest completed timeline state at UI monotonic_ms plus paint/fb; no source-snapshot binding'
    return dict(
        latency_scope='end_to_end_ms is reception-to-result age, excluding window accumulation and physical/USB/acoustic delay',
        latency_ms=latency,
        health=dict(inference_windows=len(inference),
                    usb_ok_windows=sum(r.get('usb_ok') is True for r in inference),
                    dpu_ok_windows=sum(r.get('dpu_ok') is True for r in inference),
                    all_inference_windows_usb_dpu_ok=bool(inference) and all(r.get('usb_ok') is True and r.get('dpu_ok') is True for r in inference),
                    non_window_unavailable_states=sum(r.get('window_end_sample', 0) == 0 for r in timeline),
                    unavailable_state_scope='includes shutdown/fault/gap states; do not infer cause from USB=false alone',
                    doa_rows=len(doa), doa_valid_rows=sum(r.get('valid') is True for r in doa),
                    doa_reasons=dict(Counter(str(r.get('reason', 'missing')) for r in doa))),
        ui=dict(frames=len(ui), paint_ms=_stats(r.get('paint_ms') for r in ui),
                framebuffer_ms=_stats(r.get('framebuffer_ms') for r in ui),
                exact_frame_age_ms=exact_age, estimated_snapshot_age_at_fb_complete_ms=estimated_age,
                framebuffer_ok_frames=sum(r.get('framebuffer_ok') is True for r in ui),
                fresh_frames=sum(r.get('classification_fresh') is True for r in ui),
                background_score_violation_count=len(background_score),
                background_alert_violation_count=len(background_alert),
                inactive_score_or_alert_violation_count=len(suppression),
                violation_frame_indices=sorted(set(background_score + background_alert + suppression)),
                missing_visibility_evidence_frames=missing,
                contract_mismatch_frames=[i for i, r in enumerate(ui) if r.get('class_contract') != CONTRACT],
                evidence_scope='reported main UI visibility; inference class_scores are diagnostic and are not UI violations'))


def _validate_schedule(schedule, reference):
    require(schedule.get('contract') == CONTRACT and schedule.get('rate') == RATE,
            'playlist contract/rate differs')
    require(schedule.get('independent_accuracy') is False, 'playlist must disclose development selection')
    require(schedule.get('total_samples') == len(reference) == 54 * RATE,
            'expected full 54-second playback reference')
    events = schedule.get('events')
    require(isinstance(events, list) and len(events) == 10, 'expected ten replay events')
    for i, event in enumerate(events):
        require(event.get('event_index') == i and event.get('target_index') == i % 5
                and event.get('start_sample') == (4 + 5 * i) * RATE
                and event.get('end_sample') == (7 + 5 * i) * RATE,
                'playlist event schedule differs')


def assess_loaded(session, schedule, reference):
    reconstruction = reconstruct(session['timeline'], session['payloads'])
    _validate_schedule(schedule, reference)
    alignment = align_reference(reconstruction['samples'], reconstruction['observed'], reference)
    events = assess_events(reconstruction, reference, schedule, alignment)
    assessable = [e for e in events if e['decision_status'] != 'not_assessable']
    return dict(schema=1, contract=CONTRACT, scope='curated development replay; schedule labels are not independent accuracy',
                independent_accuracy=False, tuning_performed=False, gates=GATES,
                gate_scope='fixed diagnostic heuristics, not calibrated recognition thresholds',
                reconstruction=reconstruction['report'], alignment=alignment, events=events,
                summary=dict(event_count=len(events), assessable_events=len(assessable),
                             excluded_events=len(events) - len(assessable),
                             events_with_expected_decision=sum(e['expected_decision_windows'] > 0 for e in assessable),
                             events_without_expected_decision=sum(e['expected_decision_windows'] == 0 for e in assessable)),
                reports=summarize_reports(session['timeline'], session.get('ui', []), session.get('doa', [])),
                missing_reports=session.get('missing_reports', []),
                session_prefix=session.get('session_prefix'),
                ignored_special_members=session.get('ignored_special_members', 0))


def assess(session_path, playlist):
    playlist = Path(playlist)
    schedule_raw = (playlist / 'schedule.json').read_bytes()
    schedule = _json(schedule_raw)
    wav_path = playlist / 'playback.wav'
    wav_hash = hashlib.sha256(wav_path.read_bytes()).hexdigest()
    require(schedule.get('playback_sha256') == wav_hash, 'playback WAV hash differs from schedule')
    with wave.open(str(wav_path), 'rb') as reader:
        require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype())
                == (1, 2, RATE, 'NONE'), 'reference must be mono 16kHz signed PCM16 WAV')
        raw = reader.readframes(reader.getnframes())
        require(len(raw) == reader.getnframes() * 2, 'truncated reference WAV')
    reference = np.frombuffer(raw, dtype='<i2')
    result = assess_loaded(load_session(session_path), schedule, reference)
    digest = hashlib.sha256()
    with Path(session_path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    result['inputs'] = dict(session=str(session_path), session_sha256=digest.hexdigest(),
                            playlist=str(playlist), playback_sha256=wav_hash,
                            schedule_sha256=hashlib.sha256(schedule_raw).hexdigest())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path, help='captured session TAR.GZ')
    parser.add_argument('--playlist', required=True, type=Path, help='directory containing schedule.json and playback.wav')
    parser.add_argument('--output', type=Path, help='new JSON report; never overwritten (default stdout)')
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(args.output)
        result = assess(args.session, args.playlist)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output is None:
            sys.stdout.write(text)
        else:
            with args.output.open('x', encoding='utf-8', newline='\n') as handle:
                handle.write(text)
    except (OSError, ValueError, tarfile.TarError, wave.Error) as error:
        print('assessment rejected: ' + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
