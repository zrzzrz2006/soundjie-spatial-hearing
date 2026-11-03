"""Create two 20-second continuous dog-bark WAVs without added silent gaps."""
import hashlib
import json
import os
import wave
from pathlib import Path
import numpy as np

root = Path('C:/Users/32450/Desktop/presentation')
source = root/'02_单独音频/A04_狗叫_3秒.wav'
outputs = [root/'03_带间隔拍摄音轨/P06_狗叫_连续20秒.wav',
           root/'03_带间隔拍摄音轨/P07_狗叫_音量连续变化20秒.wav']
if any(p.exists() for p in outputs):
    raise SystemExit('Output exists; inspect before replacement')
with wave.open(str(source), 'rb') as f:
    assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000)
    clip = np.frombuffer(f.readframes(f.getnframes()), dtype='<i2').astype(np.float64)
rate, count, fade = 16000, 320000, 160
assert len(clip) > 2*fade
# A 10ms overlap smooths only the loop boundary. No silence is inserted;
# the natural brief spacing between individual barks remains in the recording.
continuous = clip.copy()
ramp = np.linspace(0, 1, fade, endpoint=False)
while len(continuous) < count:
    join = continuous[-fade:]*(1-ramp) + clip[:fade]*ramp
    continuous = np.concatenate((continuous[:-fade], join, clip[fade:]))
continuous = continuous[:count]
gains = np.interp(np.arange(count)/rate, [0, 5, 10, 15, 20], [.25, .5, 1, .5, .25])
report = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'loop_crossfade_ms': 10, 'inserted_silence_seconds': 0,
          'volume_envelope': {'seconds':[0,5,10,15,20], 'gains':[.25,.5,1,.5,.25]},
          'board_accessed': False, 'played_aloud': False, 'files': []}
for path, audio in zip(outputs, (continuous, continuous*gains)):
    pcm = np.rint(audio).clip(-32768,32767).astype('<i2')
    assert len(pcm) == count and np.max(np.abs(pcm.astype(np.int32))) < 32767
    with path.open('xb') as handle:
        with wave.open(handle, 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate); f.writeframes(pcm.tobytes())
        handle.flush(); os.fsync(handle.fileno())
    with wave.open(str(path), 'rb') as f:
        assert f.getnframes() == count and f.readframes(count) == pcm.tobytes()
    report['files'].append({'path': str(path), 'duration_seconds': 20,
                            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'peak_pcm': int(np.max(np.abs(pcm.astype(np.int32))))})
receipt = Path('E:/Soundjie_Work/d4/evidence/soundjie-d4/continuous-dog-playback-20261102.json')
with receipt.open('x',encoding='utf-8',newline='\n') as f:
    json.dump(report,f,ensure_ascii=False,indent=2); f.write('\n')
print(json.dumps(report,ensure_ascii=True))
