import importlib
import importlib.util
import unittest


class D4QuantizeTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.host.quantize_soundjie_d4'),
                             'D4 quantization adapter missing')
        return importlib.import_module('scripts.host.quantize_soundjie_d4')

    def records(self):
        rows = []
        for c in range(8):
            rows.append(dict(purpose='validation', target_index=c, source_group='val'+str(c)))
            for i in range(8):
                for variant in range(2):
                    rows.append(dict(purpose='train', target_index=c, source_group='%s-%s' % (c,i)))
        return rows

    def test_selects_64_distinct_train_sources(self):
        m = self.module()
        rows = self.records()
        selected = m.select_calibration(rows)
        self.assertEqual(len(selected), 64)
        self.assertEqual(selected[:8], [1,3,5,7,9,11,13,15])
        self.assertTrue(all(rows[i]['purpose']=='train' for i in selected))
        self.assertEqual(len({rows[i]['source_group'] for i in selected}), 64)

    def test_rejects_insufficient_sources(self):
        m = self.module()
        rows = self.records()
        rows = [r for r in rows if r['source_group'] != '0-7']
        with self.assertRaises(ValueError):
            m.select_calibration(rows)

    def test_verifier_rejects_changed_artifact_and_unsafe_path(self):
        import hashlib
        import tempfile
        from pathlib import Path
        m = self.module()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'a').write_bytes(b'abc')
            m.verify_files(p, {'a': hashlib.sha256(b'abc').hexdigest()})
            for bindings in ({'a': '0'*64}, {'../a': '0'*64}):
                with self.assertRaises(ValueError):
                    m.verify_files(p, bindings)

    def test_elf_requires_arm_32(self):
        m = self.module()
        header = bytearray(64)
        header[:6] = b'\x7fELF\x01\x01'
        header[18:20] = b'\x28\x00'
        m.require_arm32(header)
        for pos, value in ((4, 2), (18, 62), (0, 0)):
            wrong = header.copy()
            wrong[pos] = value
            with self.assertRaises(ValueError):
                m.require_arm32(wrong)
