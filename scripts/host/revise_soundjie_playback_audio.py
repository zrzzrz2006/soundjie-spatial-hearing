"""Revise the authorized filming audio; retain originals and verify PCM readback."""
import hashlib
import io
import json
import os
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, 'F:/MLK/DPU_ZYNQ7000/tools/tv-audio-20261030/packages')
import av

ROOT = Path('C:/Users/32450/Desktop/presentation')
REPO = Path('E:/Soundjie_Work/d4')
RECEIPT = REPO/'evidence/soundjie-d4/filming-audio-revision-20261102.json'
BACKUP = ROOT/'05_来源与核对/音源修订前备份_20261102'
RATE = 16000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data, new=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb' if new else 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    assert path.read_bytes() == data


def preserve(relative):
    original, backup = ROOT/relative, BACKUP/relative
    if not backup.exists():
        write(backup, original.read_bytes(), new=True)
    return backup


def encode(x):
    pcm = np.rint(x).astype('<i2')
    assert np.max(np.abs(x)) < 32767
    out = io.BytesIO()
    with wave.open(out, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return out.getvalue()


def decode(path):
    chunks = []
    resampler = av.AudioResampler(format='s16', layout='mono', rate=RATE)
    with av.open(str(path)) as c:
        for frame in c.decode(audio=0):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().flatten().astype(np.float64))
        for f in resampler.resample(None):
            chunks.append(f.to_ndarray().flatten().astype(np.float64))
    return np.concatenate(chunks)


def loop(x, seconds):
    count, fade = round(seconds*RATE), 160
    ramp = np.linspace(0, 1, fade, endpoint=False)
    y = x.copy()
    while len(y) < count:
        join = y[-fade:]*(1-ramp)+x[:fade]*ramp
        y = np.concatenate((y[:-fade], join, x[fade:]))
    y = y[:count].copy()
    y[:80] *= np.linspace(0, 1, 80)
    y[-80:] *= np.linspace(1, 0, 80)
    return y


def info(name, raw):
    with wave.open(io.BytesIO(raw), 'rb') as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
        frames = w.getnframes()
        pcm = np.frombuffer(w.readframes(frames), '<i2').astype(np.int32)
    assert np.count_nonzero(np.abs(pcm) >= 32767) == 0
    return dict(file=name, bytes=len(raw), sha256=sha(raw), rate=RATE, channels=1,
                width=2, frames=frames, duration_s=frames/RATE,
                peak=int(np.max(np.abs(pcm))), full_scale_samples=0)


def main():
    if RECEIPT.exists():
        report = json.loads(RECEIPT.read_text(encoding='utf-8'))
        for row in report['files']:
            assert sha((ROOT/row['file']).read_bytes()) == row['sha256']
        print('ALREADY_PUBLISHED_AND_VERIFIED')
        return
    dog_name = '02_单独音频/A04_狗叫_3秒.wav'
    dog_original = preserve(dog_name)
    assert sha(dog_original.read_bytes()) == 'd13d2e1e9d4b99fff4446a960e8633d8531151330066f09694eeffddfb3e2086'
    # User identified the unwanted voice before the repeated barks. Discard the
    # entire first 1.02s; retain the three later bark transients and natural gaps.
    # End before the original file's partially cut-off final bark.
    dog = decode(dog_original)[round(1.02*RATE):round(2.62*RATE)]
    manifest_path = Path('F:/MLK/DPU_ZYNQ7000/datasets/soundjie_alert5bg_20261101/manifest.jsonl')
    rows = [json.loads(line) for line in manifest_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    source_rows = {key: next(r for r in rows if r['recording_id'] == 'fsd50k:'+fid)
                   for key, fid in [('doorbell', '361564'), ('siren', '223824')]}
    audio, sources = {}, []
    for key, row in source_rows.items():
        path = Path(row['path'])
        assert sha(path.read_bytes()) == row['sha256']
        assert row['license'] == 'CC0-1.0'
        x = decode(path)
        gain = 20000/np.max(np.abs(x))
        audio[key] = x*gain
        sources.append({k: row[k] for k in ('recording_id', 'title', 'author', 'path', 'sha256', 'license', 'original_url')}
                       | {'applied_gain': float(gain), 'source_kind': 'existing licensed sound-effect recording'})
    door = np.pad(audio['doorbell'], (0, 3*RATE-len(audio['doorbell'])))
    siren = audio['siren'][RATE:4*RATE]
    dog20 = loop(dog, 20)
    gains = np.interp(np.arange(20*RATE)/RATE, [0, 5, 10, 15, 20], [.25, .5, 1, .5, .25])
    double = lambda x: np.concatenate((np.zeros(3*RATE), x, np.zeros(3*RATE), x, np.zeros(6*RATE)))
    arrays = {
        '02_单独音频/A01_警笛_3秒.wav': siren,
        '02_单独音频/A03_门铃_3秒.wav': door,
        dog_name: loop(dog, 3),
        '03_带间隔拍摄音轨/P01_警笛_两次播放_18秒.wav': double(siren),
        '03_带间隔拍摄音轨/P03_门铃_两次播放_18秒.wav': double(door),
        '03_带间隔拍摄音轨/P06_狗叫_连续20秒.wav': dog20,
        '03_带间隔拍摄音轨/P07_狗叫_音量连续变化20秒.wav': dog20*gains,
        '07_修正版音频/P01_警笛_新音源20秒.wav': loop(audio['siren'][800:-800], 20),
        '07_修正版音频/P03_门铃_叮咚20秒.wav': loop(audio['doorbell'], 20),
        '07_修正版音频/P06_狗叫_连续20秒.wav': dog20,
        '07_修正版音频/P07_狗叫_音量连续变化20秒.wav': dog20*gains,
    }
    payloads = {name: encode(x) for name, x in arrays.items()}
    records = [info(name, raw) for name, raw in payloads.items()]
    assert all(r['frames'] == 320000 for r in records if r['file'].startswith('07_'))
    backups = []
    for name, raw in payloads.items():
        if (ROOT/name).exists():
            old = preserve(name)
            backups.append(dict(file=name, backup=str(old), sha256=sha(old.read_bytes())))
        write(ROOT/name, raw)
        assert info(name, (ROOT/name).read_bytes()) == next(r for r in records if r['file'] == name)
    report = dict(revision='20261102-audio-fix', recorded_at=datetime.now().astimezone().isoformat(),
                  dog_source=str(dog_original), dog_retained_interval_s=[1.02, 2.62],
                  dog_discarded_intervals_s=[[0, 1.02], [2.62, 3]], loop_crossfade_ms=10,
                  dog_inserted_silence_s=0,
                  dog_volume_envelope=dict(seconds=[0, 5, 10, 15, 20], gain=[.25, .5, 1, .5, .25]),
                  scope='Playback audio editing only; no classifier change or new board/acoustic acceptance',
                  sources=sources, backups=backups, files=records, board_accessed=False,
                  human_audition_performed=False, played_aloud=False)
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2)+'\n').encode('utf-8')
    write(ROOT/'05_来源与核对/音源修订记录_20261102.json', report_bytes, new=True)
    note = '''# 2026-11-02 音源修订

当前试听、拍摄优先使用 `07_修正版音频` 中的四条20秒文件。

- P06/P07：从旧狗叫3秒片段仅保留1.02–2.62秒的后段狗叫，再循环拼接。
  固定音量版和低→高→低版均20秒，未添加静音间隔。
- P03：改为 MatthewWong 的 [Ding Dong](https://freesound.org/people/MatthewWong/sounds/361564/)，CC0-1.0。
- P01：改为 WBJB1 的 [Wail and Yelp](https://freesound.org/people/WBJB1/sounds/223824/)，CC0-1.0。
- 狗叫来源仍为 moffet 的 [whimpering and barking.mp3](https://freesound.org/people/moffet/sounds/72579/)，CC0-1.0；裁剪后循环，P07加音量包络。

原目录中对应的A01/A03/A04、P01/P03和两条20秒狗叫也已同步更新。原P04及旧人声P06/P07保留，但不用于此次拍摄。
替换前副本在 `05_来源与核对/音源修订前备份_20261102`。
本轮校验时长、采样格式、哈希、PCM完整性及无满幅削波；未重新进行板端外放识别验收。
旧来源说明及旧媒体检查报告是上次版本记录，本次改动以本说明与音源修订记录JSON为准。
'''
    write(ROOT/'07_修正版音频/使用说明.md', note.encode('utf-8'), new=True)
    manifest_name = '05_来源与核对/素材清单与SHA256.json'
    manifest_backup = preserve(manifest_name)
    manifest = json.loads(manifest_backup.read_text(encoding='utf-8'))
    replacement = {r['file']: r | {'operation': '20261102 revised playback; see 音源修订记录_20261102.json',
                                  'revision': report['revision'], 'evidence_scope': report['scope']} for r in records}
    manifest['files'] = [replacement.pop(r['file'], r) for r in manifest['files']] + list(replacement.values())
    manifest['audio_revision_record'] = '05_来源与核对/音源修订记录_20261102.json'
    write(ROOT/manifest_name, (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    index_name = '05_来源与核对/交付索引.json'
    index = json.loads(preserve(index_name).read_text(encoding='utf-8'))
    index['audio_revision'] = report['revision']
    index['audio_revision_record'] = manifest['audio_revision_record']
    index['audio_revision_scope'] = report['scope']
    indexed = {r['file']: r for r in index['files']}
    changed = [*payloads, manifest_name, manifest['audio_revision_record'], '07_修正版音频/使用说明.md']
    for name in changed:
        data = (ROOT/name).read_bytes()
        indexed[name] = dict(file=name, bytes=len(data), sha256=sha(data))
    index['files'] = sorted(indexed.values(), key=lambda r: r['file'])
    index['files_before_index'] = len(index['files'])
    index['bytes_before_index'] = sum(r['bytes'] for r in index['files'])
    index['mib_before_index'] = round(index['bytes_before_index']/1024**2, 2)
    index['updated_at'] = report['recorded_at']
    index['backup_directory_not_in_index'] = str(BACKUP)
    write(ROOT/index_name, (json.dumps(index, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
    write(RECEIPT, report_bytes, new=True)
    print(json.dumps(dict(result='REVISED_AUDIO_READY', files=len(records), new_folder=str(ROOT/'07_修正版音频'),
                          dog_duration_s=20, no_full_scale_samples=True, board_accessed=False), ensure_ascii=True))


if __name__ == '__main__':
    main()
