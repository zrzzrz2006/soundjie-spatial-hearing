"""Create-only, deliberately selected development replay; never controls a board.

Build and verify are silent. Only the explicit ``play`` subcommand uses winsound.
The stored FixedNeuron scores describe original one-second validation windows,
not the gain-adjusted three-second excerpts or acoustic microphone reception.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
import wave

import numpy as np

from scripts.host.deploy_alert4_candidate import (
    CONTRACT, LABELS, check_compiled, check_files, probabilities,
    require_hash, staged_inputs,
)
from scripts.host.logmel_reference import load_frontend_config
from scripts.host.prepare_soundjie_d3 import decode_audio, resample_mono_16k, compute_feature

RATE = 16000
CLIP_SAMPLES = 48000
LABELS_ZH = ['警笛', '车辆鸣笛', '门铃', '狗叫', '背景']
DEFAULT_COMPILE = Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_alert4bg/20261031a/compile-20261031a-sidecar-a')
BIAS = ('Confidence-ranked, correct development-validation cases selected with the '
        'same model and thresholds used for selection/calibration. Deliberately biased '
        'demonstration, not independent accuracy or a physical-DPU/acoustic pass.')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_text(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(value)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def centered_excerpt(samples, validation_start, clip_samples=CLIP_SAMPLES, window_samples=RATE):
    """Retain the exact validation center, padding instead of clamping/rewrapping."""
    samples = np.asarray(samples, dtype=np.float64)
    if (samples.ndim != 1 or not samples.size or not np.isfinite(samples).all()
            or type(validation_start) is not int or type(clip_samples) is not int
            or type(window_samples) is not int or not 0 < window_samples <= clip_samples
            or (clip_samples - window_samples) % 2):
        raise ValueError('invalid centered excerpt request')
    if max(0, validation_start) >= min(len(samples), validation_start + window_samples):
        raise ValueError('validation window does not overlap source')
    offset = (clip_samples - window_samples) // 2
    start = validation_start - offset
    a, b = max(0, start), min(len(samples), start + clip_samples)
    output = np.zeros(clip_samples, dtype=np.float64)
    output[a-start:b-start] = samples[a:b]
    return output, dict(excerpt_start_sample_16k=start, excerpt_samples=clip_samples,
                       validation_start_sample_16k=validation_start,
                       validation_offset_samples=offset, validation_samples=window_samples,
                       left_padding_samples=max(0, -start),
                       right_padding_samples=max(0, start + clip_samples - len(samples)))


def uniform_gain_pcm(clips, peak_limit=.5):
    """One attenuation for the entire playlist; no AGC, boosts or clipping limiter."""
    if not clips or not math.isfinite(peak_limit) or not 0 < peak_limit <= .9:
        raise ValueError('invalid gain request')
    arrays = [np.asarray(x, dtype=np.float64) for x in clips]
    if any(x.ndim != 1 or not x.size or not np.isfinite(x).all() for x in arrays):
        raise ValueError('invalid source samples')
    peaks = [float(np.max(np.abs(x))) for x in arrays]
    if min(peaks) <= 1e-6:
        raise ValueError('silent excerpt')
    gain = min(1., peak_limit / max(peaks))
    pcm = [np.rint(x * gain * 32768.).astype('<i2') for x in arrays]
    clipped = sum(int(np.count_nonzero((x == -32768) | (x == 32767))) for x in pcm)
    if clipped or any(np.max(np.abs(x.astype(np.int32))) > math.ceil(peak_limit*32768) for x in pcm):
        raise ValueError('PCM gain overflow')
    return pcm, dict(policy='same attenuation for all excerpts; no boost or limiter',
                     digital_gain=gain, source_max_peak=max(peaks), peak_limit=peak_limit,
                     output_peak_dbfs=20*math.log10(max(peaks)*gain), clipped_samples=clipped)


def identities(row):
    result = set()
    for key in ('recording_id', 'sampling_identity', 'pcm_sha256_16k'):
        if row.get(key):
            result.add((key, row[key]))
    for key in ('source_group', 'uploader_group'):
        if row.get(key):
            result.add(('group', row[key]))
    for key in ('sha256', 'parent_sha256'):
        if row.get(key):
            result.add(('source_sha256', row[key]))
    return result


def select_cases(rows, sources, probs, thresholds):
    """Two distinct groups per alert and one speech/glass control, no training reuse."""
    probs = np.asarray(probs, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if (probs.shape != (len(rows), 5) or not np.isfinite(probs).all()
            or np.any(probs < 0) or np.any(probs > 1)
            or not np.allclose(probs.sum(1), 1., atol=1e-6)
            or thresholds.shape != (4,) or not np.isfinite(thresholds).all()
            or np.any(thresholds < 0) or np.any(thresholds > 1)):
        raise ValueError('invalid probabilities or thresholds')
    by_id = {r['recording_id']: r for r in sources}
    if len(by_id) != len(sources) or len({r['recording_id'] for r in rows}) != len(rows):
        raise ValueError('duplicate recording ID')
    train_ids = set().union(*(identities(r) for r in sources if r['purpose'] == 'train'))
    candidates = []
    for i, row in enumerate(rows):
        src = by_id.get(row['recording_id'])
        if (src is None or row['purpose'] != 'validation' or row.get('derivation') is not None
                or src['purpose'] != 'validation' or src.get('derivation') is not None
                or src['class_contract'] != CONTRACT or src['sha256'] != row['parent_sha256']
                or src['target_index'] != row['target_index']
                or src['source_group'] != row['source_group']
                or src.get('sampling_identity') != row.get('sampling_identity')
                or src.get('uploader_group') != row.get('uploader_group')):
            raise ValueError('validation/source binding differs')
        if (identities(row) | identities(src)) & train_ids:
            raise ValueError('validation source overlaps training')
        target = row['target_index']
        if type(target) is not int or not 0 <= target <= 4:
            raise ValueError('invalid target')
        # Prefer explicit siren recordings, rather than using two alarm-clock
        # samples merely because alarms share the new siren output channel.
        if target == 0 and src['negative_subtype'] != 'siren':
            continue
        floor = max(.9, thresholds[target]) if target < 4 else .8
        if int(probs[i].argmax()) != target or probs[i, target] < floor:
            continue
        candidates.append(dict(recording_id=row['recording_id'], target_index=target,
            validation_index=i, source=src, validation_source=row,
            fixed_reference=dict(scope='original unscaled one-second development-validation window only',
                probabilities=probs[i].tolist(), confidence=float(probs[i, target]),
                argmax_index=target, decision_index=target,
                threshold=float(thresholds[target]) if target < 4 else None)))
    chosen, used = [], set()
    slots = [(0, None), (1, None), (2, None), (3, None), (4, 'ordinary_speech'),
             (0, None), (1, None), (2, None), (3, None), (4, 'glass')]
    for target, subtype in slots:
        pool = [r for r in candidates if r['target_index'] == target
                and (subtype is None or r['source']['negative_subtype'] == subtype)]
        pool.sort(key=lambda r: (-r['fixed_reference']['confidence'], r['recording_id']))
        item = next((r for r in pool if not (identities(r['source']) |
                                            identities(r['validation_source'])) & used), None)
        if item is None:
            raise ValueError('not enough correct, high-confidence disjoint sources: %s %s' % (target, subtype))
        used.update(identities(item['source']) | identities(item['validation_source']))
        chosen.append(item)
    return chosen


def write_wav(path, pcm):
    pcm = np.asarray(pcm)
    if pcm.ndim != 1 or pcm.dtype != np.dtype('<i2') or not pcm.size:
        raise ValueError('mono little-endian int16 PCM required')
    with Path(path).open('xb') as f:
        with wave.open(f, 'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
            w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(str(path), 'rb') as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) != (1, 2, RATE, 'NONE'):
            raise ValueError('unexpected playback WAV format')
        result = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2')
        if len(result) != w.getnframes():
            raise ValueError('truncated WAV')
        return result


def write_bundle(output, clips, selected, provenance):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (len(clips) != 10 or len(selected) != 10
            or Counter(r['target_index'] for r in selected) != {i: 2 for i in range(5)}
            or any(x.shape != (CLIP_SAMPLES,) or x.dtype != np.dtype('<i2') for x in clips)):
        raise ValueError('ten three-second int16 clips required, two per output')
    # Four seconds of lead silence, 10*(3 seconds clip + 2 seconds gap),
    # including the final two-second observation tail: exactly 54 seconds.
    waveform = np.zeros(54*RATE, dtype='<i2')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'references').mkdir()
    events = []
    for i, (pcm, item) in enumerate(zip(clips, selected)):
        target = item['target_index']; start = (4+5*i)*RATE; end = start+CLIP_SAMPLES
        name = 'references/%02d_%s.wav' % (i+1, LABELS[target])
        write_wav(output / name, pcm)
        waveform[start:end] = pcm
        event = dict(item, event_index=i, start_sample=start, end_sample=end,
            start_s=start/RATE, end_s=end/RATE, reference_wav=name,
            reference_sha256=sha((output/name).read_bytes()), reference_pcm_sha256=sha(pcm.tobytes()),
            expected_display=LABELS_ZH[target], expected_alert=target < 4,
            expected_main_score_visible=target < 4,
            independent_accuracy=False, physical_reception_confirmed=False,
            fixed_reference_applies_to_rendered_audio=False)
        events.append(event)
    write_wav(output / 'playback.wav', waveform)
    plan = dict(schema=1, contract=CONTRACT, rate=RATE, total_samples=len(waveform),
        duration_s=len(waveform)/RATE, lead_seconds=4, gap_seconds=2, tail_seconds=2,
        clip_seconds=3, capture_seconds=60, events=events, selection_bias=BIAS,
        provenance=provenance, playback_sha256=sha((output/'playback.wav').read_bytes()),
        independent_accuracy=False, board_tested=False, played=False,
        labels_are_schedule_not_observed_predictions=True,
        confidence_scores_are_internal_provenance_not_background_UI=True)
    write_json(output / 'schedule.json', plan)
    write_text(output / 'schedule.tsv', 'index\tstart_s\tend_s\tlabel\tsubtype\trecording_id\n' + ''.join(
        '%d\t%.1f\t%.1f\t%s\t%s\t%s\n' % (r['event_index']+1, r['start_s'], r['end_s'],
            r['expected_display'], r['source']['negative_subtype'], r['recording_id']) for r in events))
    write_text(output / 'README.md', bundle_readme(output, plan))
    pins = {p.relative_to(output).as_posix(): sha(p.read_bytes())
            for p in sorted(output.rglob('*')) if p.is_file()}
    write_text(output / 'SHA256SUMS', ''.join(h+'  '+name+'\n' for name, h in pins.items()))
    verify_bundle(output)
    return plan


def verify_bundle(directory):
    directory = Path(directory)
    pins = {}
    for line in (directory/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if name in pins:
            raise ValueError('duplicate checksum entry')
        pins[name] = digest
    check_files(directory, pins, {'schedule.json', 'schedule.tsv', 'README.md', 'playback.wav'})
    plan = read_json(directory/'schedule.json')
    if (plan['contract'] != CONTRACT or plan['rate'] != RATE or plan['duration_s'] != 54
            or len(plan['events']) != 10 or plan['board_tested'] or plan['independent_accuracy']):
        raise ValueError('unexpected replay contract')
    require_hash(directory/'playback.wav', plan['playback_sha256'])
    wave_data = read_wav(directory/'playback.wav')
    expected = np.zeros(54*RATE, dtype='<i2')
    seen = set()
    for i, item in enumerate(plan['events']):
        start, end = (4+5*i)*RATE, (7+5*i)*RATE
        name = item['reference_wav']; target = item['target_index']
        if (name not in pins or item['start_sample'] != start or item['end_sample'] != end
                or item['expected_alert'] != (target < 4)
                or item['expected_main_score_visible'] != (target < 4)
                or item['expected_display'] != LABELS_ZH[target]):
            raise ValueError('schedule offsets or display policy differ')
        identity = identities(item['source']) | identities(item['validation_source'])
        if seen & identity:
            raise ValueError('replay repeats source identity')
        seen.update(identity)
        require_hash(directory/name, item['reference_sha256'])
        pcm = read_wav(directory/name)
        if pcm.shape != (CLIP_SAMPLES,) or sha(pcm.tobytes()) != item['reference_pcm_sha256']:
            raise ValueError('reference PCM differs')
        expected[start:end] = pcm
    if not np.array_equal(wave_data, expected) or plan['total_samples'] != len(expected):
        raise ValueError('playback/clip/silence assembly differs')
    return plan


def bundle_readme(directory, plan):
    lines = ['# 四类提醒受控演示回放', '',
        '**用途：筛选过的开发验证集演示，不是独立准确率验收。尚未播放，也未上板实测。**', '',
        '54 秒，16 kHz/单声道/PCM16；前置静音 4 秒，每段 3 秒，间隔及结尾静音 2 秒。',
        '警笛、车辆鸣笛、门铃、狗叫各两段；普通说话和玻璃破碎各一段，预期仅显示“背景”、不提醒、不显示主模型分数。',
        '所有片段以原始一秒验证窗口为中心；越过原文件边界补零，不重新寻找高能量位置。统一降低音量、无增强及限幅。',
        'JSON 中的置信度仅对应原始、未调整音量的一秒窗口；不得解释为三秒播放片段或麦克风回录的准确率。',
        '54 秒播放时间由 Windows 音频 API 启动时计时，声卡缓冲/声学传播另有延迟；验收以板端实际回录对齐为准。', '',
        '## 仅校验（不播放）', '', '```powershell',
        "$env:PYTHONPATH='E:/Soundjie_Work/d4'",
        "& 'F:/MLK/DPU_ZYNQ7000/envs/soundjie_d4_host/Scripts/python.exe' -B -m scripts.host.soundjie_alert4_replay verify --directory '" + directory.as_posix() + "'", '```', '',
        '## 明确开始播放时才运行', '', '```powershell',
        "& 'F:/MLK/DPU_ZYNQ7000/envs/soundjie_d4_host/Scripts/python.exe' -B -m scripts.host.soundjie_alert4_replay play --directory '" + directory.as_posix() + "'", '```', '',
        '播放使用 Windows 当前默认外放设备，不能证明麦克风已收到；本工具不控制串口、板卡、SD 卡或 Docker。', '',
        '|开始–结束（秒）|预期显示|来源 / 许可|', '|---|---|---|']
    for r in plan['events']:
        s = r['source']; title = s.get('title') or s['recording_id']
        title = title.replace('|', '/').replace('\n', ' ')
        author = s.get('author') or s.get('speaker_or_uploader') or '见原始数据集'
        lines.append('|%.0f–%.0f|%s|[%s](%s)，%s，%s|' % (
            r['start_s'], r['end_s'], r['expected_display'], title,
            s.get('original_url', ''), author, s.get('license', '见来源')))
    lines += ['', '完整 SHA256、作者/许可原始字段、训练/验证身份、窗口起点、padding、统一增益、固定点模型/阈值溯源保存在 `schedule.json`。',
              '未执行人工试听；标题和标签来自已验证的源元数据，声音本身仍须通过实际播放与回录确认。', '']
    return '\n'.join(lines)


def load_inputs(compile_dir):
    inputs = compile_dir/'input'; fixed = compile_dir/'container-output/fixed'
    compiled = compile_dir/'container-output/build'
    staged, meta = staged_inputs(inputs)
    check_compiled(inputs, compiled)
    fm = read_json(fixed/'fixed-evaluation.json')
    check_files(fixed, fm['artifacts'], {'validation-fixed.npy'})
    policy = read_json(compile_dir/'thresholds-fixed.json')
    for key, path in [('assessment_sha256', compile_dir/'assessment.json'),
                      ('compiled_manifest_sha256', compiled/'compiled-manifest.json'),
                      ('fixed_manifest_sha256', fixed/'fixed-evaluation.json'),
                      ('elf_sha256', compiled/'compiled/dpu_soundjie_d3.elf')]:
        require_hash(path, policy[key])
    if (policy['contract'] != CONTRACT or policy['labels'] != LABELS or policy['background_index'] != 4
            or policy['checkpoint_sha256'] != meta['checkpoint_sha256']
            or fm['compiled_manifest_sha256'] != policy['compiled_manifest_sha256']
            or fm['input_manifest_sha256'] != sha((inputs/'input-manifest.json').read_bytes())
            or fm['contract'] != CONTRACT or fm['status'] != 'fixed_evaluation_complete'):
        raise ValueError('model/fixed-validation/threshold provenance differs')
    rows = read_json(inputs/'validation-sources.json')
    sources = [json.loads(x) for x in (inputs/'source-manifest.jsonl').read_text(encoding='utf-8').splitlines() if x]
    y = np.load(inputs/'validation-targets.npy', allow_pickle=False)
    p = probabilities(np.load(fixed/'validation-fixed.npy', allow_pickle=False))
    if not np.array_equal(y, [r['target_index'] for r in rows]) or len(p) != len(rows):
        raise ValueError('fixed logits, target order and source rows differ')
    paths = [inputs/'input-manifest.json', inputs/'source-manifest.jsonl', inputs/'validation-sources.json',
             inputs/'validation-targets.npy', inputs/'frontend.json', fixed/'validation-fixed.npy',
             compile_dir/'thresholds-fixed.json', fixed/'fixed-evaluation.json', compiled/'compiled-manifest.json']
    pins = {str(path.resolve()): sha(path.read_bytes()) for path in paths}
    return rows, sources, p, policy, dict(contract=CONTRACT, input_pins=pins,
        threshold_policy=policy, checkpoint_sha256=staged['checkpoint_sha256'],
        compiled_elf_sha256=policy['elf_sha256'], script_sha256=sha(Path(__file__).read_bytes()))


def build(compile_dir, output):
    compile_dir, output = Path(compile_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows, sources, p, policy, provenance = load_inputs(compile_dir)
    selected = select_cases(rows, sources, p, policy['alert_thresholds'])
    frontend = load_frontend_config(compile_dir/'input/frontend.json')
    clips = []
    for item in selected:
        source, row = item['source'], item['validation_source']
        path = Path(source['path']); require_hash(path, source['sha256'])
        samples = resample_mono_16k(decode_audio(path))
        full_pcm = np.clip(np.rint(samples*32768), -32768, 32767).astype('<i2')
        if sha(full_pcm.tobytes()) != row['pcm_sha256_16k']:
            raise ValueError('decoded source PCM differs')
        clip, window = centered_excerpt(samples, row['start_sample_16k'])
        start = window['validation_offset_samples']
        center = np.clip(np.rint(clip[start:start+RATE]*32768), -32768, 32767).astype('<i2')
        feature = compute_feature(center.astype(np.float64)/32768., frontend)
        if sha(feature.tobytes()) != row['feature_sha256']:
            raise ValueError('validation center no longer produces the pinned feature')
        item['excerpt'] = dict(window, original_center_pcm_sha256=sha(center.tobytes()),
                               original_center_feature_verified=True)
        # Recheck after decoding; neither originals nor training artifacts are changed.
        require_hash(path, source['sha256'])
        clips.append(clip)
    pcm, gain = uniform_gain_pcm(clips)
    provenance['uniform_gain'] = gain
    provenance['selected_source_count'] = len(selected)
    provenance['selected_source_groups'] = len({r['source']['source_group'] for r in selected})
    provenance['train_source_overlap'] = 0
    provenance['validation_pool_count'] = len(rows)
    for item, clip in zip(selected, pcm):
        item['digital_gain'] = gain['digital_gain']
        item['excerpt_rms_dbfs'] = float(20*np.log10(max(1e-12, np.sqrt(np.mean((clip.astype(np.float64)/32768.)**2)))))
        item['excerpt_peak'] = float(np.max(np.abs(clip.astype(np.int32)))/32768.)
    return write_bundle(output, pcm, selected, provenance)


def play(directory):
    """Explicit invocation only; synchronous default-output playback, no board access."""
    directory = Path(directory)
    plan = verify_bundle(directory)
    import winsound
    stamp = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    log = directory / ('playback-run-' + stamp + '.json')
    start, clock = time.time(), time.perf_counter()
    report = dict(started_unix=start, playback_sha256=plan['playback_sha256'],
        requested_duration_s=plan['duration_s'], api='winsound synchronous Windows default output',
        physical_reception_confirmed=False)
    print(json.dumps(dict(stage='playback_started', unix_time=start, duration_s=plan['duration_s'])), flush=True)
    try:
        winsound.PlaySound(str(directory/'playback.wav'), winsound.SND_FILENAME | winsound.SND_NODEFAULT)
        elapsed = time.perf_counter()-clock
        if elapsed < plan['duration_s']-.5:
            raise RuntimeError('playback returned early')
        report.update(status='API_completed_not_acoustic_verification', elapsed_seconds=elapsed)
    except Exception as exc:
        report.update(status='failed', elapsed_seconds=time.perf_counter()-clock, error=str(exc))
        raise
    finally:
        write_json(log, report)
    print(json.dumps(dict(stage='playback_finished', elapsed_seconds=elapsed, log=str(log))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    make = sub.add_parser('build', help='create a new bundle without playback')
    make.add_argument('--compile-dir', type=Path, default=DEFAULT_COMPILE)
    make.add_argument('--output', type=Path, required=True)
    for action in ['verify', 'play']:
        command = sub.add_parser(action)
        command.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'play':
        play(args.directory)
        return
    plan = build(args.compile_dir, args.output) if args.action == 'build' else verify_bundle(args.directory)
    print(json.dumps(dict(stage=args.action, contract=plan['contract'], events=len(plan['events']),
        duration_s=plan['duration_s'], playback_sha256=plan['playback_sha256'],
        independent_accuracy=False, board_tested=False, played=False), ensure_ascii=False))


if __name__ == '__main__':
    main()
