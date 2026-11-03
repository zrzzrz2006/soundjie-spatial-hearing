"""A UI-only release must preserve every other business payload and refuse overwrite."""
import hashlib
import importlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.host.assemble_soundjie_alert4_release import read_archive, write_archive


class UnifiedReleaseTests(unittest.TestCase):
    def module(self):
        name='scripts.host.build_soundjie_unified_ui_release'
        self.assertIsNotNone(importlib.util.find_spec(name),'UI-only release builder is not implemented')
        return importlib.import_module(name)

    def inputs(self,root):
        data={'runtime/soundjie_hdmi_ui':b'old-ui',
              'assets/hdmi/soundjie_glyph_atlas.bin':b'old-atlas',
              'model/dpu_soundjie_d3.elf':b'exact-model',
              'runtime/soundjie_d4_runtime':b'exact-runtime',
              'runtime/soundjie_d4_classifier':b'exact-classifier',
              'config/alert-thresholds.txt':b'exact-thresholds',
              'tv/reference.features':b'exact-reference','tv/traffic.bin':b'exact-traffic',
              'tv/cues.tsv':b'exact-captions','board/run_soundjie_tv_session.sh':b'exact-supervisor',
              'manifest/release.json':b'{"schema":1}', 'SHA256SUMS':b'old-sums'}
        base=root/'base.tgz';write_archive(base,data)
        return base,hashlib.sha256(base.read_bytes()).hexdigest(),data

    def test_only_ui_atlas_and_declared_metadata_change_and_tar_modes_remain_valid(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base,sha,original=self.inputs(root)
            result=m.assemble(base,sha,b'new-ui',b'new-atlas',root/'out',{'audit':'test'})
            payload=read_archive(root/'out/SJ4U2.TGZ',result['archive_sha256'],'sj-alert4-release/')
            allowed={'runtime/soundjie_hdmi_ui','assets/hdmi/soundjie_glyph_atlas.bin','SHA256SUMS','manifest/release.json'}
            changed={n for n in original if original[n]!=payload[n]}
            self.assertEqual(changed,allowed)
            self.assertEqual(set(payload)-set(original),{'manifest/ui-update.json'})
            self.assertEqual(payload['tv/traffic.bin'],b'exact-traffic')
            self.assertEqual(result['unchanged_payload_count'],len(original)-len(allowed))
            self.assertEqual(result['changed_business_files'],sorted(allowed-{'SHA256SUMS','manifest/release.json'}))
            self.assertFalse(result['board_tested'])
            for line in payload['SHA256SUMS'].decode().splitlines():
                digest,name=line.split('  ',1)
                self.assertEqual(digest,hashlib.sha256(payload[name]).hexdigest())
            with tarfile.open(root/'out/SJ4U2.TGZ') as t:
                self.assertEqual(t.getmember('sj-alert4-release/runtime/soundjie_hdmi_ui').mode,0o755)
                self.assertEqual(t.getmember('sj-alert4-release/assets/hdmi/soundjie_glyph_atlas.bin').mode,0o644)
            patch=read_archive(root/'out/SJU2.TGZ',result['patch_sha256'],'sj-unified-ui-v2/')
            self.assertEqual(set(patch),{'runtime/soundjie_hdmi_ui','assets/hdmi/soundjie_glyph_atlas.bin','manifest/ui-update.json','SHA256SUMS'})
            self.assertEqual(patch['runtime/soundjie_hdmi_ui'],b'new-ui')

    def test_hash_mismatch_and_existing_destination_refused(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base,sha,_=self.inputs(root);out=root/'out'
            with self.assertRaises(ValueError):m.assemble(base,'0'*64,b'ui',b'atlas',out,{})
            self.assertFalse(out.exists())
            out.mkdir();(out/'mine.txt').write_bytes(b'keep')
            with self.assertRaises(FileExistsError):m.assemble(base,sha,b'ui',b'atlas',out,{})
            self.assertEqual((out/'mine.txt').read_bytes(),b'keep')

    def test_missing_ui_target_refused_before_output_creation(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'base.tgz';write_archive(base,{'model/only':b'model'})
            sha=hashlib.sha256(base.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):m.assemble(base,sha,b'ui',b'atlas',root/'out',{})
            self.assertFalse((root/'out').exists())

    def test_abi_audit_refuses_newer_glibc_wrong_architecture_and_dynamic_libstdcxx(self):
        m=self.module()
        good='Machine: ARM\nTag_CPU_arch: v7\nTag_ABI_VFP_args: VFP registers\nRequesting program interpreter: /lib/ld-linux-armhf.so.3\nGLIBC_2.17\n'
        self.assertEqual(m.audit_abi(good),'2.17')
        for bad in (good+'GLIBC_2.34',good.replace('Machine: ARM','Machine: X86-64'),
                    good.replace('v7','v8'),good+'Shared library: [libstdc++.so.6]'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):m.audit_abi(bad)


if __name__=='__main__':unittest.main()
