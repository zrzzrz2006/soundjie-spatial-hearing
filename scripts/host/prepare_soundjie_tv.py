"""Create a bounded, provenance-checked known-clip reference; never overwrite media."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import wave


def validate_config(cfg):
    if cfg.get('schema') != 1 or cfg.get('sample_rate') != 16000:
        raise ValueError('unsupported schema or sample rate')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', cfg.get('clip_id', '')):
        raise ValueError('invalid clip id')
    if not re.fullmatch(r'[0-9a-f]{64}', cfg.get('source_sha256', '')):
        raise ValueError('invalid source hash')
    start, end = cfg['start_seconds'], cfg['end_seconds']
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, end)):
        raise ValueError('nonfinite interval')
    if start < 0 or not 0 < end-start <= 120:
        raise ValueError('invalid interval')
    seen, previous = set(), 0
    cues=cfg.get('cues', [])
    if len(cues)>100: raise ValueError('too many cues')
    for cue in cues:
        a,b=cue['start'],cue['end']
        if not all(isinstance(x,(float,int)) and math.isfinite(x) for x in (a,b)):
            raise ValueError('nonfinite cue')
        if not (previous <= a < b <= end-start): raise ValueError('cue overlap or range')
        if cue['id'] in seen or not re.fullmatch(r'[a-zA-Z0-9_-]{1,32}',cue['id']):
            raise ValueError('invalid cue id')
        if cue['speaker'] not in ('A','B','C','?'): raise ValueError('speaker id')
        text=cue['text']
        if not isinstance(text,str) or not text.strip() or len(text.encode('utf-8'))>256 or any(ord(c)<32 for c in text):
            raise ValueError('invalid caption text')
        if cue['status'] not in ('candidate','confirmed_anchor','two_model_agreement','asr_context_consensus','verified_text'):
            raise ValueError('caption verification status')
        seen.add(cue['id']);previous=b
    return round((end-start)*16000)


def enabled_cues(cfg):
    validate_config(cfg)
    return [c for c in cfg['cues'] if c['status'] != 'candidate']


def file_hash(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(cfg, source, output, ffmpeg):
    count=validate_config(cfg)
    source,output=Path(source),Path(output)
    if output.exists(): raise FileExistsError(output)
    before=file_hash(source)
    if before!=cfg['source_sha256']: raise ValueError('source hash differs')
    output.mkdir(parents=True,exist_ok=False)
    wav=output/'reference.wav'
    result=subprocess.run([str(ffmpeg),'-hide_banner','-nostdin','-v','error','-n',
        '-ss',str(cfg['start_seconds']),'-i',str(source),'-t',str(cfg['end_seconds']-cfg['start_seconds']),
        '-map','0:a:0','-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(wav)],capture_output=True)
    (output/'decode.log').write_bytes(result.stderr)
    result.check_returncode()
    with wave.open(str(wav)) as f:
        if (f.getnchannels(),f.getsampwidth(),f.getframerate(),f.getnframes())!=(1,2,16000,count):
            raise ValueError('decoded PCM format/count differs')
        raw=f.readframes(count)
    if file_hash(source)!=before: raise ValueError('source changed during intake')
    (output/'reference.s16le').write_bytes(raw)
    active=enabled_cues(cfg)
    (output/'cues.tsv').write_text(''.join(f"{c['id']}\t{c['start']:.3f}\t{c['end']:.3f}\t{c['speaker']}\t{c['text']}\n" for c in active),encoding='utf-8')
    manifest=dict(cfg,source_path=str(source.resolve()),audio_sha256=file_hash(wav),pcm_sha256=hashlib.sha256(raw).hexdigest(),samples=count,enabled_cue_count=len(active),scope='known_clip_not_general_asr')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['config','source','output','ffmpeg']: p.add_argument('--'+key,required=True)
    a=p.parse_args()
    result=prepare(json.loads(Path(a.config).read_text(encoding='utf-8')),a.source,a.output,a.ffmpeg)
    print(json.dumps(result,ensure_ascii=False))
