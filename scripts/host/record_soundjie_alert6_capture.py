"""Record this frozen capture's receipt and a guarded, train-only adaptation subset."""
import base64
import hashlib
import json
from pathlib import Path
from scripts.host.acquire_soundjie_fsd50k import save_verified
from scripts.host.build_soundjie_d4_manifest import digest_file
from scripts.host.soundjie_alert6 import validate_rows
from scripts.host.train_soundjie_alert6 import write_json


def main():
    root=Path('F:/MLK/DPU_ZYNQ7000/artifacts/soundjie_alert6_replay/20261031a')
    capture=root/'capture-20261031b'
    alignment=json.loads((capture/'alignment-report.json').read_text(encoding='utf-8'))
    paired=json.loads((capture/'paired-host-diagnostics.json').read_text(encoding='utf-8'))
    raw=Path('evidence/soundjie-tv/tv1-boot-20261031.serial.log').read_bytes()
    chunk=raw.replace(b'\r',b'').split(b'\nSJ_ALERT6_CAPTURE_BEGIN\n',1)[1].split(b'\nSJ_ALERT6_CAPTURE_END\n',1)[0]
    archive=base64.b64decode(b''.join(chunk.split()),validate=True)
    archive_sha=hashlib.sha256(archive).hexdigest()
    if archive_sha!='c6019d2d8674e5b235ee2661d1d1a70708ad171c4951d268622b412ee7331d05':
        raise ValueError('archive mismatch')
    if digest_file(capture/'capture-stereo.wav')!=alignment['recording_sha256']:
        raise ValueError('capture drift')
    rows=[]
    for event in alignment['events']:
        if not event['adaptation_allowed']:continue
        if event['purpose']!='train' or event['quarantined'] or not event['reception']['passed']:
            raise ValueError('ineligible adaptation source')
        if digest_file(event['captured_wav'])!=event['captured_sha256']:raise ValueError('excerpt drift')
        row={k:event[k] for k in ['source_group','target_id','target_index','negative_subtype','license']}
        # User-provided recordings do not have FSD labels, authors or source URLs.
        row.update({k:event[k] for k in ['original_url','original_labels','license_url','author','competition_use'] if k in event})
        row.update(recording_id='replay:20261031b:'+event['recording_id'],source_id='acoustic_replay',
            source_recording_id=event.get('source_recording_id') or event['recording_id'],
            parent_recording_id=event['recording_id'],parent_sha256=event['sha256'],
            path=event['captured_wav'],sha256=event['captured_sha256'],purpose='train',split='train',
            domain='acoustic_replay',session='20261031b',weak_clip_label=True,independent_D5=False,
            capture_parent_sha256=alignment['recording_sha256'],reception=event['reception'],
            adaptation_allowed=True,derivation={'method':'actual_speaker_XVF3800_capture_equal_mix',
                'alignment_sha256':digest_file(capture/'alignment-report.json')})
        rows.append(row)
    validate_rows(rows,require_all=False)
    if len(rows)!=13 or set(r['target_index'] for r in rows)!=set(range(6)):raise ValueError('unexpected train subset')
    manifest=capture/'adaptation-train-subset.jsonl'
    save_verified(manifest,b''.join((json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n').encode('utf-8') for row in rows))
    ui=[]
    for line in raw.splitlines():
        if line.startswith(b'{"panels"'):
            try:ui.append(json.loads(line))
            except ValueError:pass
    if not ui or not all(ui[-1][k] for k in ['usb_ok','dpu_ok','framebuffer_ok','classification_fresh']):
        raise ValueError('no restored live UI evidence')
    evidence=dict(schema=1,date='2026-10-31',status='capture_and_host_diagnosis_complete_new_model_not_deployed',
        raw_wav_sha256=alignment['recording_sha256'],gzip_sha256=archive_sha,
        source_manifest_sha256=json.loads((root/'schedule.json').read_text())['source_manifest_sha256'],
        files={str(p):digest_file(p) for p in [root/'schedule.json',root/'playback.wav',root/'playback-run.json',
            capture/'capture-stereo.wav',capture/'alignment-report.json',capture/'paired-host-diagnostics.json',manifest]},
        sample_rate=16000,channels=2,format='WAV PCM16 little endian',duration_seconds=124,
        start_sync=alignment['start_sync'],end_sync=alignment['end_sync'],clock_ratio=alignment['clock_ratio'],
        clipped_fraction=alignment['clipped_fraction'],reception_verified=20,quarantined=2,
        adaptation_train_sources=13,replay_validation_sources=7,independent_acceptance=False,new_training_run=False,new_DPU_inference=False,
        summary=paired['summary'],results_interpretation='Known-source weak-label replay diagnostics, separated train/validation; no population accuracy claim.',
        label_boundary_issue=dict(recording_id='fsd50k:372251',title='Standard Emergency Warning Signal (SEWS)',
            original_labels=['Siren','Alarm'],action='flag semantic boundary; preserve frozen labels and scores, do not relabel to improve score'),
        events=[{k:e[k] for k in ['event_index','recording_id','purpose','target_index','captured_rms_dbfs','reception','quarantined','adaptation_allowed']}
                for e in alignment['events']],
        board=dict(session='/run/sj-tv1-dual-demo-20261031b',supervisor_pid=26659,restored_ui=ui[-1],
            sd_written=False,old_model_retained=True,power='on'),
        invalid_attempt=dict(directory='/run/sj-alert6-replay-20261031a',
            reason='host playback never started due Python3.13 SND_SYNC; capture interrupted and excluded'),
        tests=dict(replay_suite=8,failures=0,review_findings_fixed=['silent events quarantine','original native PCM format verification']))
    write_json(Path('evidence/soundjie-d4/alert6-acoustic-results-20261031.json'),evidence)
    print(json.dumps(dict(evidence_written=True,native_capture_sha_verified=True,gzip_sha_verified=True,
        adaptation_train_sources=len(rows),board_health=ui[-1]['health_text'])))


if __name__=='__main__':main()
