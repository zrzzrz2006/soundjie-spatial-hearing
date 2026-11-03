import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "assets" / "hdmi" / "soundjie_glyph_atlas.bin"
SOURCES = [ROOT / "src" / name for name in (
    "framebuffer_surface.cpp", "soundjie_hdmi_ui.cpp", "soundjie_d4_doa_stream.cpp",
    "soundjie_d4_state.cpp", "xvf3800_control_protocol.cpp", "sha256.cpp")]


def wsl_path(path: Path) -> str:
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


def snapshot(sector="front", valid=True, dbfs=-18.0):
    return {
        "sequence": 9, "timestamp_ms": 1234, "class_id": 1,
        "label_zh": "应急车辆警笛", "confidence": 0.87, "top3": [],
        "doa_raw_deg": ("front", "front_right", "right", "rear_right", "rear", "rear_left", "left", "front_left").index(sector) * 45 if valid else None,
        "doa_deg": ("front", "front_right", "right", "rear_right", "rear", "rear_left", "left", "front_left").index(sector) * 45 if valid else None,
        "doa_sector": sector if valid else None, "doa_valid": valid,
        "doa_reason": "ok" if valid else "no_speech", "rms": 0.2, "dbfs": dbfs,
        "peak": 0.5, "clipping_rate": 0.0, "audio_state": "normal",
        "usb_ok": True, "dpu_ok": True, "framebuffer_ok": True,
        "inference_ms": 32.5, "end_to_end_ms": 81.0,
    }


class SoundjieHdmiUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.binary = Path(cls.temporary.name) / "soundjie_hdmi_ui"
        result = subprocess.run(
            ["wsl.exe", "g++", "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
             f"-I{wsl_path(ROOT / 'src')}", *(wsl_path(p) for p in SOURCES),
             "-o", wsl_path(cls.binary)], text=True, encoding="utf-8", errors="replace",
            capture_output=True)
        if result.returncode:
            raise AssertionError(f"HDMI UI build failed\n{result.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_bounded_live_jsonl_does_not_duplicate_stdout_and_keeps_latest_frame(self):
        with tempfile.TemporaryDirectory(dir=self.temporary.name) as d:
            root=Path(d);state=root/'state.json';doa=root/'doa.jsonl';log=root/'ui.jsonl'
            state.write_text(json.dumps(snapshot(),ensure_ascii=False),encoding='utf-8')
            doa.write_text('')
            r=subprocess.run(['wsl','-d','Ubuntu','--',wsl_path(self.binary),
                '--snapshot',wsl_path(state),'--atlas',wsl_path(ATLAS),'--doa-jsonl',wsl_path(doa),
                '--runtime-config',wsl_path(ROOT/'config/soundjie_d4_runtime.tsv'),
                '--offscreen-dir',wsl_path(root/'frames'),'--width','640','--height','360',
                '--frames','12','--interval-ms','20','--jsonl',wsl_path(log),
                '--log-max-bytes','4096'],capture_output=True,timeout=30)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(r.stdout,b'')
            self.assertLessEqual(log.stat().st_size,4096)
            rows=[json.loads(s) for s in log.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(rows[-1]['frame'],11)
            self.assertIsNone(rows[-1]['arrow_tip'])

    def render(self, payload, width=640, height=360):
        root = Path(tempfile.mkdtemp(dir=self.temporary.name))
        state, ppm, report = root / "state.json", root / "ui.ppm", root / "report.json"
        state.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        before = hashlib.sha256(state.read_bytes()).hexdigest()
        result = subprocess.run(
            ["wsl.exe", wsl_path(self.binary), "--snapshot", wsl_path(state),
             "--atlas", wsl_path(ROOT/'assets/hdmi/soundjie_tv_glyph_atlas_v4.bin' if payload.get('tv_enabled') else ATLAS), "--offscreen", wsl_path(ppm),
             "--report", wsl_path(report), "--width", str(width), "--height", str(height)],
            text=True, encoding="utf-8", errors="replace", capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(hashlib.sha256(state.read_bytes()).hexdigest(), before)
        return json.loads(report.read_text(encoding="utf-8")), ppm.read_bytes()

    def test_layout_contains_class_confidence_compass_strength_health_latency(self):
        report, ppm = self.render(snapshot())
        self.assertEqual(report["panels"], ["class", "confidence", "compass", "strength", "health", "latency"])
        self.assertTrue(ppm.startswith(b"P6\n640 360\n255\n"))

    def test_compass_arrow_rotates_for_eight_sectors(self):
        tips = {}
        for sector in ("front", "front_right", "right", "rear_right", "rear", "rear_left", "left", "front_left"):
            tips[sector] = self.render(snapshot(sector))[0]["arrow_tip"]
        self.assertEqual(len({tuple(value) for value in tips.values()}), 8)
        self.assertLess(tips["front"][1], tips["rear"][1])
        self.assertGreater(tips["right"][0], tips["left"][0])

    def test_invalid_doa_draws_unavailable_without_old_arrow(self):
        report, _ = self.render(snapshot(valid=False))
        self.assertIsNone(report["arrow_tip"])
        self.assertEqual(report["doa_text"], "方向不可用")
        self.assertEqual(report["doa_reason"], "no_speech")

    def test_arrow_uses_precise_angle_not_only_sector(self):
        payload = snapshot("front_right")
        payload["doa_deg"] = 30
        thirty, _ = self.render(payload)
        payload["doa_deg"] = 60
        sixty, _ = self.render(payload)
        self.assertNotEqual(thirty["arrow_tip"], sixty["arrow_tip"])
        self.assertEqual(thirty["doa_angle_deg"], 30)
        self.assertIn("30", thirty["doa_text"])
        self.assertEqual(thirty["doa_reason"], "ok")

    def test_live_refresh_expires_doa_and_classification_without_new_inference(self):
        from test_soundjie_d4_doa_stream import row
        with tempfile.TemporaryDirectory(dir=ROOT / ".superpowers") as folder:
            root = Path(folder)
            state, doa, frames = root / "state.json", root / "doa.jsonl", root / "frames"
            config = root / "runtime.tsv"
            config.write_text("key\tvalue\ndoa_stale_ms\t200\ndoa_ema_alpha\t0.5\n", encoding="utf-8")
            proc = subprocess.Popen([
                "wsl.exe", "-d", "Ubuntu", "--", wsl_path(self.binary),
                "--snapshot", wsl_path(state), "--atlas", wsl_path(ATLAS),
                "--doa-jsonl", wsl_path(doa), "--runtime-config", wsl_path(config),
                "--frames", "12", "--interval-ms", "100", "--snapshot-stale-ms", "400",
                "--offscreen-dir", wsl_path(frames), "--width", "640", "--height", "360"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                first = proc.stdout.readline()
                if not first:
                    self.fail(proc.stderr.read().decode(errors="replace"))
                initial = json.loads(first)
                payload = snapshot()
                payload.update(tv_enabled=True,tv_matched=True,fine_valid=True,tv_position_seconds=14,tv_score=.9,
                    caption_text='打开百度地图',caption_speaker='A',caption_cue='open_map',
                    ambulance_score=.92,road_score=.9,speech_score=.8)
                payload["timestamp_ms"] = initial["monotonic_ms"]
                state.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                doa.write_bytes((row(initial["monotonic_ms"], 90) + "\n").encode())
                output, errors = proc.communicate(timeout=20)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()
            self.assertEqual(proc.returncode, 0, errors.decode(errors="replace"))
            reports = [initial] + [json.loads(line) for line in output.splitlines()]
            self.assertEqual(len(reports), 12)
            self.assertFalse(initial["classification_fresh"])
            self.assertTrue(any(r["classification_fresh"] and r["arrow_tip"] is not None for r in reports))
            self.assertTrue(any(r['caption_visible'] for r in reports))
            final = reports[-1]
            self.assertFalse(final["classification_fresh"])
            self.assertFalse(final["legacy_visible"])
            self.assertEqual(final["legacy_class_scores"], [])
            self.assertFalse(final['caption_visible'])
            self.assertEqual(final['caption_text'],'')
            self.assertFalse(final["usb_ok"])
            self.assertFalse(final["dpu_ok"])
            self.assertFalse(final["framebuffer_ok"])
            self.assertEqual(final["strength_ratio"], 0)
            self.assertIsNone(final["arrow_tip"])
            self.assertEqual(final["doa_reason"], "stale")
            self.assertEqual(len(list(frames.glob("*.ppm"))), 12)
            self.assertGreater(final["monotonic_ms"] - initial["monotonic_ms"], 900)

    def test_strength_bar_clamps_without_nan(self):
        low, _ = self.render(snapshot(dbfs=-999.0))
        high, _ = self.render(snapshot(dbfs=999.0))
        self.assertEqual(low["strength_ratio"], 0.0)
        self.assertEqual(high["strength_ratio"], 1.0)

    def test_requested_stop_is_neutral_and_clears_stale_measurements(self):
        payload = snapshot()
        payload['session_state'] = 'stopped'
        report, _ = self.render(payload)
        self.assertEqual(report['session_state'], 'stopped')
        self.assertEqual(report['health_text'], 'USB STOP  DPU STOP  HDMI --')
        self.assertFalse(report['classification_fresh'])
        self.assertIsNone(report['arrow_tip'])
        self.assertEqual(report['strength_ratio'], 0)
        self.assertFalse(report['usb_ok'])
        self.assertFalse(report['dpu_ok'])

    def test_fault_is_not_rendered_as_normal_stop(self):
        payload = snapshot(valid=False)
        payload.update(session_state='fault', usb_ok=False, dpu_ok=False, framebuffer_ok=False)
        report, _ = self.render(payload)
        self.assertEqual(report['session_state'], 'fault')
        self.assertIn('ERR', report['health_text'])

    def test_utf8_labels_use_audited_atlas(self):
        report, _ = self.render(snapshot())
        self.assertGreater(report["label_glyphs"], 0)
        self.assertEqual(report["atlas_sha256"], "ced4413fb0b2dab694ac64cbae679b5aca0f8080648a3516a9e4b824b2aae527")

    def test_known_clip_caption_and_independent_events_are_displayed(self):
        p=snapshot();p.update(tv_enabled=True,tv_matched=True,tv_position_seconds=14.1,tv_score=.9,
            caption_text='打开"百度地图"',caption_speaker='A',caption_cue='open_map',fine_valid=True,
            ambulance_score=.92,road_score=.88,speech_score=.81)
        r,_=self.render(p)
        self.assertEqual(r['caption_text'],'打开"百度地图"')
        self.assertEqual(r['display_label'],p['label_zh'])
        self.assertEqual(r['tv_display_label'],'救护车警笛 + 车流')
        self.assertTrue(r['caption_visible']);self.assertEqual(r['caption_speaker'],'A')
        p['tv_matched']=False
        r,_=self.render(p);self.assertFalse(r['caption_visible']);self.assertNotIn('救护车',r['tv_display_label'])
        self.assertEqual(r['display_label'],p['label_zh'])
        p['session_state']='stopped'
        r,_=self.render(p);self.assertFalse(r['caption_visible'])

    def test_fault_clears_even_fresh_caption_and_fine_event(self):
        p=snapshot();p.update(session_state='fault',tv_enabled=True,tv_matched=True,fine_valid=True,
            tv_position_seconds=14,tv_score=.9,caption_text='旧字幕',caption_speaker='A',caption_cue='old',
            ambulance_score=.9,road_score=.9,speech_score=.9)
        r,_=self.render(p);self.assertFalse(r['caption_visible']);self.assertNotIn('救护车',r['display_label'])

    def test_long_caption_is_clipped_inside_left_panel(self):
        p=snapshot();p.update(tv_enabled=True,tv_matched=True,tv_position_seconds=14,tv_score=.9,
            caption_text='打开百度地图'*10,caption_speaker='A',caption_cue='long',fine_valid=True,
            ambulance_score=.9,road_score=.9,speech_score=.9)
        r,_=self.render(p)
        self.assertTrue(r['caption_truncated'])
        self.assertLessEqual(r['caption_max_right'],r['caption_panel_right'])

    def test_bad_utf8_live_snapshot_clears_instead_of_exiting(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.superpowers') as folder:
            root=Path(folder);state=root/'state.json';doa=root/'doa.jsonl';config=root/'runtime.tsv'
            config.write_text('key\tvalue\ndoa_stale_ms\t200\ndoa_ema_alpha\t0.5\n',encoding='utf-8')
            proc=subprocess.Popen(['wsl','-d','Ubuntu','--',wsl_path(self.binary),'--snapshot',wsl_path(state),
                '--atlas',wsl_path(ATLAS),'--doa-jsonl',wsl_path(doa),'--runtime-config',wsl_path(config),
                '--frames','8','--interval-ms','100','--snapshot-stale-ms','1500',
                '--offscreen-dir',wsl_path(root/'frames'),'--width','640','--height','360'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                first=json.loads(proc.stdout.readline());p=snapshot()
                p.update(timestamp_ms=first['monotonic_ms'],tv_enabled=True,tv_matched=True,fine_valid=True,
                    tv_position_seconds=14,tv_score=.9,caption_text='BAD_UTF8',caption_speaker='A',caption_cue='bad',
                    ambulance_score=.9,road_score=.9,speech_score=.9)
                state.write_bytes(json.dumps(p,ensure_ascii=False).encode().replace(b'BAD_UTF8',b'\xc3'))
                output,err=proc.communicate(timeout=12)
                self.assertEqual(proc.returncode,0,err)
                reports=[json.loads(s) for s in output.splitlines()]
                self.assertTrue(all(not r['caption_visible'] for r in reports))
            finally:
                if proc.poll() is None:proc.kill();proc.communicate()

    def test_tv_display_exposes_weak_scene_label_status(self):
        p=snapshot();p.update(tv_enabled=True,tv_matched=True,fine_valid=True,tv_position_seconds=14,tv_score=.9,
            caption_text='',caption_speaker='',caption_cue='',ambulance_score=.9,road_score=.9,speech_score=.1)
        r,_=self.render(p)
        self.assertEqual(r['label_basis'],'legacy_dpu')
        self.assertEqual(r['tv_label_basis'],'user_scene_weak_not_source_verified')

    def test_eight_dpu_scores_remain_visible_without_clip_match(self):
        p=snapshot();p.update(tv_enabled=True,tv_matched=False,fine_valid=False,
            tv_position_seconds=14,tv_score=.2,caption_text='',caption_speaker='',
            ambulance_score=.9,road_score=.9,speech_score=.1,
            class_scores=[.01,.87,.02,.02,.02,.02,.02,.02])
        r,_=self.render(p)
        self.assertTrue(r['legacy_visible'])
        self.assertEqual(r['display_label'],p['label_zh'])
        self.assertEqual(r['legacy_class_scores'],p['class_scores'])
        self.assertEqual(r['tv_display_label'],'等待片段匹配')
        self.assertFalse(r['fine_events_visible'])

    def test_conflicting_models_are_shown_separately_not_fused(self):
        p=snapshot();p.update(class_id=3,label_zh='玻璃破碎',confidence=.74,
            class_scores=[.01,.04,.06,.74,.02,.03,.05,.05],
            tv_enabled=True,tv_matched=True,fine_valid=True,tv_position_seconds=14,tv_score=.85,
            caption_text='打开百度地图',caption_speaker='A',ambulance_score=.95,road_score=.8,speech_score=.9)
        for width,height in ((1280,720),(640,360)):
            r,_=self.render(p,width,height)
            self.assertEqual(r['display_label'],'玻璃破碎')
            self.assertEqual(r['legacy_confidence'],.74)
            self.assertEqual(r['tv_display_label'],'救护车警笛 + 车流')
            self.assertTrue(r['caption_visible'])
            self.assertEqual(len(r['legacy_class_scores']),8)
            self.assertEqual(r['layout'],'dual_live_clip_v1')
            self.assertLessEqual(r['caption_max_right'],r['caption_panel_right'])

    def test_dpu_unavailable_never_substitutes_clip_scores(self):
        p=snapshot();p.update(dpu_ok=False,tv_enabled=True,tv_matched=True,fine_valid=True,
            tv_position_seconds=14,tv_score=.9,caption_text='打开百度地图',caption_speaker='A',
            ambulance_score=.9,road_score=.9,speech_score=.8,class_scores=[.01,.87,.02,.02,.02,.02,.02,.02])
        r,_=self.render(p)
        self.assertFalse(r['legacy_visible'])
        self.assertEqual(r['display_label'],'不可用')
        self.assertEqual(r['legacy_class_scores'],[])
        self.assertTrue(r['caption_visible'])

    def test_absent_class_vector_is_not_fabricated_from_top1(self):
        r,_=self.render(snapshot())
        self.assertEqual(r['legacy_class_scores'],[])

    def test_compact_score_digits_remain_distinguishable(self):
        crops=[]
        for value in (.13,.18,.19):
            p=snapshot();p.update(tv_enabled=True,tv_matched=False,fine_valid=False,
                tv_position_seconds=0,tv_score=0,caption_text='',caption_speaker='',
                ambulance_score=0,road_score=0,speech_score=0,
                class_scores=[value,.4,.05,.05,.05,.05,.1,.3-value])
            _,ppm=self.render(p)
            pixels=ppm.split(b'\n',3)[3]
            # First score's percentage text only; exclude its changing bar.
            crops.append(b''.join(pixels[(y*640+188)*3:(y*640+208)*3] for y in range(107,117)))
        self.assertEqual(len(set(crops)),3,'13%, 18%, 19% must not alias to one bitmap')

    def test_rgb565_and_xrgb8888_pixel_conversion(self):
        for fmt, expected in (("RGB565", "0xfc00"), ("XRGB8888", "0xffff8000")):
            result = subprocess.run(
                ["wsl.exe", wsl_path(self.binary), "--pixel", "255,128,0", "--format", fmt],
                text=True, encoding="utf-8", errors="replace", capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), expected)

    def test_missing_or_unsupported_framebuffer_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake = Path(temporary) / "fb0"
            fake.write_bytes(b"not-a-framebuffer")
            for target in (Path(temporary) / "missing", fake):
                result = subprocess.run(
                    ["wsl.exe", wsl_path(self.binary), "--framebuffer", wsl_path(target)],
                    text=True, encoding="utf-8", errors="replace", capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("framebuffer", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
