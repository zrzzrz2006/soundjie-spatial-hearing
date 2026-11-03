"""Native C++11 bank integration tests; all generated files stay in F: artifacts.

Run with Python + numpy on Windows (WSL Ubuntu g++) or Linux. Override
SOUNDJIE_BANK_TEST_ROOT to move the retained test artifacts when needed.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get(
    "SOUNDJIE_BANK_TEST_ROOT",
    "F:/MLK/DPU_ZYNQ7000/artifacts/two-clip-matcher-20261101a"
    if os.name == "nt" else "/mnt/f/MLK/DPU_ZYNQ7000/artifacts/two-clip-matcher-20261101a"))


def native_path(path):
    path = str(path)
    return "/mnt/" + path[0].lower() + path[2:].replace("\\", "/") if os.name == "nt" else path


def native_run(args):
    prefix = ["wsl.exe", "-d", "Ubuntu", "--"] if os.name == "nt" else []
    return subprocess.run(prefix + [str(a) for a in args], capture_output=True)


API_TEST = r'''
#include "known_clip_matcher.h"
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace soundjie;
static void require(bool value) { if (!value) throw std::runtime_error("API assertion failed"); }
static void invalid(const std::vector<ClipReference>& refs, double t=.65, double m=.05, double cross=.05) {
  bool threw=false;
  try { KnownClipBank bank(refs,t,m,cross); } catch(const std::exception&) { threw=true; }
  require(threw);
}
int main(int argc,char** argv) { try {
  require(argc==3);
  const auto refs=read_clip_bank(argv[1]);
  invalid({}); auto bad=refs; bad.push_back(refs[0]); invalid(bad);
  bad=refs; bad[0].id=""; invalid(bad);
  bad=refs; bad[0].profile="other"; invalid(bad);
  bad=refs; bad[0].features.resize(296); invalid(bad);
  bad=refs; bad[0].features.resize(24001); invalid(bad);
  bad=refs; bad[0].features[0][0]=std::numeric_limits<float>::infinity(); invalid(bad);
  bad=refs; bad[0].features[0][0]=101; invalid(bad);
  bad=refs; bad[0].cues[0].start=std::numeric_limits<double>::quiet_NaN(); invalid(bad);
  bad=refs; bad[0].cues[0].end=999; invalid(bad);
  bad=refs; bad[0].cues[0].speaker="D"; invalid(bad);
  bad=refs; bad[0].cues[0].text=""; invalid(bad);
  bad=refs; bad[0].cues[0].text="\xc3"; invalid(bad);
  bad=refs; bad[0].cues[0].id="bad\t"; invalid(bad);
  bad=refs; bad[0].cues[1].id=bad[0].cues[0].id; invalid(bad);
  bad=refs; bad[0].cues[1].start=bad[0].cues[0].start; invalid(bad);
  bad=refs; bad[0].cues.resize(101,bad[0].cues[0]); invalid(bad);
  invalid(refs,-.1); invalid(refs,.65,-.1); invalid(refs,.65,.05,1.1);
  invalid(refs,.65,.05,std::numeric_limits<double>::quiet_NaN());
  KnownClipBank bank(refs);
  require(!bank.match().matched && bank.match().clip_id.empty());
  std::ifstream pcm(argv[2],std::ios::binary); char b[2]; unsigned n=0;
  while(n<64000 && pcm.read(b,2)) { bank.push(static_cast<int16_t>(static_cast<unsigned char>(b[0])|(static_cast<unsigned char>(b[1])<<8))); ++n; }
  const auto first=bank.match(); require(first.matched && first.clip_id==refs[0].id);
  for(int i=0;i<8;++i) { const auto same=bank.match(); require(same.matched && same.position_seconds==first.position_seconds && same.text==first.text); }
  bank.reset(); const auto cleared=bank.match();
  require(!cleared.matched && cleared.clip_id.empty() && cleared.scene_profile.empty() && cleared.text.empty() && cleared.cue_id.empty() && cleared.speaker.empty());
  for(int i=0;i<24000;++i) bank.push(0);
  require(!bank.match().matched);
  ClipMatch original; require(original.clip_id.empty() && original.scene_profile.empty());
  std::cout<<"API ok\n"; return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 2; } }
'''


class KnownClipBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix="bank-tests-", dir=ARTIFACTS))
        print("Retained bank test artifacts:", cls.work, flush=True)
        if not (ROOT / "src/known_clip_bank_cli.cpp").exists():
            raise AssertionError("KnownClipBank offline CLI is not implemented")
        cls.binary = cls.work / "bank"
        cls.old_binary = cls.work / "single"
        harness = cls.work / "api.cpp"
        harness.write_text(API_TEST, encoding="utf-8")
        cls.api_binary = cls.work / "api"
        for source, binary in [(ROOT / "src/known_clip_bank_cli.cpp", cls.binary),
                               (ROOT / "src/known_clip_cli.cpp", cls.old_binary),
                               (harness, cls.api_binary)]:
            proc = native_run(["g++", "-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror",
                               "-I" + native_path(ROOT / "src"),
                               native_path(ROOT / "src/fft_radix2.cpp"),
                               native_path(ROOT / "src/known_clip_matcher.cpp"),
                               native_path(source), "-o", native_path(binary)])
            if proc.returncode:
                raise AssertionError(proc.stderr.decode("utf-8", "replace"))
        cls.signals = []
        for index, seed in enumerate((14, 99)):
            rng = np.random.default_rng(seed)
            signal = np.zeros(16000 * 9)
            for k in range(45):
                t = np.arange(3200) / 16000
                freq = rng.uniform(180, 2600)
                signal[k*3200:(k+1)*3200] = .13*np.sin(2*np.pi*freq*t) + .08*np.sin(2*np.pi*freq*1.83*t)
            cls.signals.append(signal)
            raw = cls.work / (str(index) + ".raw")
            cls.write_pcm(raw, signal)
            cls.invoke(["--make-reference", native_path(raw), "--output", native_path(cls.work / (str(index) + ".features"))])
            (cls.work / (str(index) + ".tsv")).write_text(
                "q1\t1.6\t3.2\tA\t%s：第一句\nq2\t3.6\t5.2\tB\t%s：第二句\nq3\t5.6\t7.8\tC\t%s：第三句\n" % (("急救" if index == 0 else "对话",)*3), encoding="utf-8")
        cls.manifest = cls.work / "bank.tsv"
        cls.bank_rows = "ambulance\tambulance_traffic\t0.features\t0.tsv\ndialogue\tdialogue\t1.features\t1.tsv\n"
        cls.manifest.write_text(cls.bank_rows, encoding="utf-8")

    @staticmethod
    def write_pcm(path, signal):
        path.write_bytes(np.clip(np.rint(signal*32768), -32768, 32767).astype("<i2").tobytes())

    @classmethod
    def invoke(cls, args, binary=None, ok=True):
        proc = native_run([native_path(binary or cls.binary), *args])
        if not ok:
            return proc
        if proc.returncode:
            raise AssertionError(proc.stderr.decode("utf-8", "replace"))
        return [json.loads(line) for line in proc.stdout.decode("utf-8").splitlines() if line.startswith("{")]

    def match(self, signal, manifest=None, extra=()):
        raw = self.work / (self._testMethodName + ".raw")
        self.write_pcm(raw, signal)
        return self.invoke(["--bank", native_path(manifest or self.manifest), "--pcm", native_path(raw), *extra])

    def assert_cleared(self, rows):
        self.assertTrue(rows)
        self.assertTrue(all(not r["matched"] and not any(r[k] for k in
                        ("clip_id", "scene_profile", "cue_id", "speaker", "text", "caption")) for r in rows))

    def test_distinct_clips_local_positions_and_all_speakers(self):
        for index, (clip, profile, text) in enumerate((("ambulance", "ambulance_traffic", "急救"),
                                                      ("dialogue", "dialogue", "对话"))):
            rows = self.match(self.signals[index][16000:16000*8] * .4)
            matched = [r for r in rows if r["matched"]]
            self.assertGreater(len(matched), 20)
            self.assertTrue(all(r["clip_id"] == clip and r["scene_profile"] == profile for r in matched))
            self.assertLess(max(abs(r["position_seconds"]-r["input_seconds"]-1) for r in matched), .10)
            self.assertEqual({r["speaker"] for r in matched if r["text"]}, {"A", "B", "C"})
            self.assertTrue(all(r["text"].startswith(text) for r in matched if r["text"]))
            self.assertTrue(all(r["caption"] == r["text"] and r["match_ms"] >= 0 for r in rows))

    def test_switches_both_directions_without_stale_captions(self):
        rows = self.match(np.r_[self.signals[0][:64000], self.signals[1][32000:96000], self.signals[0][64000:128000]])
        for lo, hi, clip, prefix, offset in [(1.8, 4, "ambulance", "急救", 0),
                                            (5.8, 8, "dialogue", "对话", -2),
                                            (9.8, 12, "ambulance", "急救", -4)]:
            tail = [r for r in rows if lo <= r["input_seconds"] <= hi]
            self.assertTrue(tail)
            self.assertTrue(all(r["matched"] and r["clip_id"] == clip for r in tail))
            self.assertTrue(all(r["text"].startswith(prefix) for r in tail if r["text"]))
            self.assertLess(max(abs(r["position_seconds"]-r["input_seconds"]-offset) for r in tail), .10)
        self.assert_cleared([r for r in rows if not r["matched"]])

    def test_forward_and_backward_jump_still_searches_entire_reference(self):
        rows = self.match(np.r_[self.signals[1][:48000], self.signals[1][96000:144000], self.signals[1][16000:64000]])
        for lo, hi, offset in [(4.8, 6, 3), (7.8, 9, -5)]:
            tail = [r for r in rows if lo <= r["input_seconds"] <= hi]
            self.assertTrue(tail)
            self.assertTrue(all(r["matched"] and r["clip_id"] == "dialogue" for r in tail))
            self.assertLess(max(abs(r["position_seconds"]-r["input_seconds"]-offset) for r in tail), .10)

    def test_silence_and_unrelated_audio_clear_identity_and_caption(self):
        rows = self.match(np.r_[self.signals[0][:64000], np.zeros(32000)])
        self.assertTrue(any(r["text"] for r in rows))
        self.assert_cleared([r for r in rows if r["input_seconds"] >= 4.2])
        self.assert_cleared(self.match(np.random.default_rng(333).normal(0, .15, 48000)))

    def test_identical_and_nearly_identical_cross_clip_evidence_is_rejected(self):
        bank = self.work / "ambiguous.tsv"
        for feature in ("0.features", "near.features"):
            if feature.startswith("near"):
                payload = (self.work / "0.features").read_bytes()
                feats = np.frombuffer(payload[16:], "<f4").copy()
                feats += np.random.default_rng(14).normal(0, .01, feats.shape).astype("f4")
                (self.work / feature).write_bytes(payload[:16] + feats.tobytes())
            bank.write_text("a\tambulance_traffic\t0.features\t0.tsv\nb\tdialogue\t" + feature + "\t1.tsv\n", encoding="utf-8")
            rows = self.match(self.signals[0][:64000], bank)
            self.assertTrue(any(r["score"] > .99 for r in rows))
            self.assert_cleared(rows)

    def test_single_reference_matches_original_cli_behavior(self):
        single = self.work / "single.tsv"
        single.write_text(self.bank_rows.splitlines()[0] + "\n", encoding="utf-8")
        signal = np.r_[self.signals[0][:64000]*.4, np.zeros(16000), self.signals[0][96000:]]
        bank = self.match(signal, single)
        old = self.invoke(["--reference", native_path(self.work / "0.features"), "--cues", native_path(self.work / "0.tsv"),
                           "--pcm", native_path(self.work / (self._testMethodName + ".raw"))], binary=self.old_binary)
        self.assertEqual(len(old), len(bank))
        for a, b in zip(old, bank):
            for key in a:
                self.assertEqual(a[key], b[key], key)

    def test_weak_evidence_cannot_acquire_or_inherit_another_clips_lock(self):
        signal = np.random.default_rng(314).normal(0, .13, 16000*9)
        raw = self.work / "weak-reference.raw"
        self.write_pcm(raw, signal)
        feature = self.work / "weak-source.features"
        self.invoke(["--make-reference", native_path(raw), "--output", native_path(feature)])
        payload = feature.read_bytes()
        features = np.frombuffer(payload[16:], "<f4").reshape(-1, 32).copy()
        region = features[600:1400]
        level = float(np.sqrt(np.mean((region-region.mean(0))**2)))
        region += np.random.default_rng(315).normal(0, level*1.4, region.shape).astype("f4")
        degraded = self.work / "weak.features"
        degraded.write_bytes(payload[:16]+features.tobytes())
        manifest = self.work / "weak-bank.tsv"
        manifest.write_text("ambulance\tambulance_traffic\t0.features\t0.tsv\nweak\tdialogue\tweak.features\t1.tsv\n", encoding="utf-8")
        locked = self.match(signal, manifest)
        weak = [r for r in locked if 4.8 <= r["input_seconds"] <= 6.8]
        self.assertTrue(weak)
        self.assertTrue(all(.45 <= r["score"] < .65 and r["margin"] >= .10 for r in weak))
        self.assertTrue(all(r["matched"] and r["clip_id"] == "weak" for r in weak))
        self.assert_cleared(self.match(signal[64000:96000], manifest))
        # Arrange the new clip's weak position to look temporally continuous
        # with the old clip, so only clip-scoped history prevents a false lock.
        switched = self.match(np.r_[self.signals[0][:64000], signal[64000:96000]], manifest)
        self.assertTrue(any(r["matched"] for r in switched if r["input_seconds"] <= 4))
        self.assert_cleared([r for r in switched if r["input_seconds"] >= 5.6])
        lost = self.match(np.r_[signal[:48000], np.zeros(32000), signal[64000:96000]], manifest)
        self.assert_cleared([r for r in lost if r["input_seconds"] >= 5])

    def test_cross_clip_ambiguity_breaks_an_existing_lock(self):
        # The second clip shares only the middle of the first. Unique opening
        # audio acquires a lock, but the identical middle must revoke it.
        payload = (self.work / "0.features").read_bytes()
        first = np.frombuffer(payload[16:], "<f4").reshape(-1, 32)
        second = np.frombuffer((self.work / "1.features").read_bytes()[16:], "<f4").reshape(-1, 32).copy()
        second[600:1400] = first[600:1400]
        (self.work / "shared.features").write_bytes(payload[:16]+second.tobytes())
        manifest = self.work / "shared-bank.tsv"
        manifest.write_text("a\tambulance_traffic\t0.features\t0.tsv\nb\tdialogue\tshared.features\t1.tsv\n", encoding="utf-8")
        rows = self.match(self.signals[0], manifest)
        self.assertTrue(any(r["matched"] and r["clip_id"] == "a" for r in rows if r["input_seconds"] <= 3))
        self.assert_cleared([r for r in rows if 4.8 <= r["input_seconds"] <= 6.8])
        self.assertTrue(any(r["matched"] and r["clip_id"] == "a" for r in rows if r["input_seconds"] >= 8.8))

    def test_direct_api_validation_reset_and_no_timer_progression(self):
        proc = native_run([native_path(self.api_binary), native_path(self.manifest), native_path(self.work / "0.raw")])
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))

    def test_malformed_bank_paths_counts_and_sizes_are_rejected(self):
        bad = self.work / "bad-bank.tsv"
        base = "a\tdialogue\t0.features\t0.tsv\n"
        invalid = ["", base*2, base.replace("dialogue", "unknown"), "a\tdialogue\t0.features\n",
                   base.rstrip()+"\textra\n", "x"*4097,
                   "".join(base.replace("a\t", str(i)+"\t") for i in range(5))]
        for path in ("../0.features", "/etc/passwd", "C:/ref", "dir/../0.features", "dir\\ref", "./0.features", "missing", "0.features\x00", "0.features/", "dir//ref"):
            invalid.append(base.replace("0.features", path))
        for content in invalid:
            with self.subTest(content=repr(content[:90])):
                bad.write_text(content, encoding="utf-8")
                proc = self.invoke(["--bank", native_path(bad), "--pcm", native_path(self.work / "0.raw")], ok=False)
                self.assertNotEqual(proc.returncode, 0)
                self.assertFalse(proc.stdout)
        large = self.work / "large-cues.tsv"
        large.write_bytes(b"\n" * 65537)
        bad.write_text(base.replace("0.tsv", large.name), encoding="utf-8")
        self.assertNotEqual(self.invoke(["--bank", native_path(bad), "--pcm", native_path(self.work / "0.raw")], ok=False).returncode, 0)

    def test_bad_cues_and_corrupt_features_are_rejected_before_audio(self):
        bad = self.work / "bad-data-bank.tsv"
        cues = self.work / "bad-cues.tsv"
        invalid = [b"q\t0\t1\tD\tbad\n", b"q\tnan\t1\tA\tbad\n", b"q\t0\t999\tA\tbad\n",
                   b"q\t0\t1\tA\t\xc3\n", b"q\t0\t1\tA\tbad\nq\t2\t3\tB\tduplicate\n",
                   b"q\t0\t2\tA\tx\nz\t1\t3\tB\toverlap\n"]
        bad.write_text("a\tdialogue\t0.features\tbad-cues.tsv\n", encoding="utf-8")
        for content in invalid:
            cues.write_bytes(content)
            self.assertNotEqual(self.invoke(["--bank", native_path(bad), "--pcm", native_path(self.work / "0.raw")], ok=False).returncode, 0)
        broken = self.work / "broken.features"
        bad.write_text("a\tdialogue\tbroken.features\t0.tsv\n", encoding="utf-8")
        for payload in (b"SJTVF1\0\0"+b"\xff"*8, (self.work / "0.features").read_bytes()[:-1], b"x"*(16+24000*128+1)):
            broken.write_bytes(payload)
            self.assertNotEqual(self.invoke(["--bank", native_path(bad), "--pcm", native_path(self.work / "0.raw")], ok=False).returncode, 0)

    def test_four_reference_bound_and_nonregular_file_rejection(self):
        four = self.work / "four.tsv"
        four.write_text("".join(str(i)+"\tdialogue\t0.features\t0.tsv\n" for i in range(4)), encoding="utf-8")
        self.assert_cleared(self.match(self.signals[0][:32000], four))
        bad = self.work / "nonregular.tsv"
        bad.write_text("a\tdialogue\tnot-a-file\t0.tsv\n", encoding="utf-8")
        (self.work / "not-a-file").mkdir()
        self.assertNotEqual(self.invoke(["--bank", native_path(bad), "--pcm", native_path(self.work / "0.raw")], ok=False).returncode, 0)

    def test_symlink_escape_is_rejected_where_supported(self):
        nested = self.work / "nested"
        nested.mkdir()
        manifest = nested / "bank.tsv"
        manifest.write_text("a\tdialogue\tescape.features\tempty.tsv\n", encoding="utf-8")
        (nested / "empty.tsv").write_text("", encoding="utf-8")
        proc = native_run(["ln", "-s", native_path(self.work / "0.features"), native_path(nested / "escape.features")])
        message = proc.stderr.decode("utf-8", "replace")
        if proc.returncode and any(reason in message for reason in ("Permission denied", "Operation not permitted", "not supported")):
            self.skipTest("artifact filesystem does not permit symlinks; no OS settings changed")
        self.assertEqual(proc.returncode, 0, message)
        rejected = self.invoke(["--bank", native_path(manifest), "--pcm", native_path(self.work / "0.raw")], ok=False)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse(rejected.stdout)

    def test_prepare_does_not_overwrite_and_cli_rejects_bad_options(self):
        feat = self.work / "0.features"
        before = feat.read_bytes()
        self.assertNotEqual(self.invoke(["--make-reference", native_path(self.work / "0.raw"), "--output", native_path(feat)], ok=False).returncode, 0)
        self.assertEqual(before, feat.read_bytes())
        for option, value in (("--cross-clip-margin", "nan"), ("--threshold", "0.65junk"), ("--margin", "-1"), ("--typo", "1")):
            self.assertNotEqual(self.invoke(["--bank", native_path(self.manifest), "--pcm", native_path(self.work / "0.raw"), option, value], ok=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
