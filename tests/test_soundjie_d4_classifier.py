"""Real PCM/frontend/protocol tests, with only unavailable DPU hardware doubled."""
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/soundjie_d4_classifier.cpp'
CONFIG = ROOT / 'config/logmel_frontend_v1.json'


def linux(path):
    p = Path(path).resolve().as_posix()
    return '/mnt/' + p[0].lower() + p[2:]


class D4ClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.is_file():
            raise AssertionError('missing real PCM to DPU classifier adapter')
        cls.temp = tempfile.TemporaryDirectory(prefix='d4-classifier-test-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.folder = Path(cls.temp.name)
        cls.binary = cls.folder / 'classifier'
        fixture = ROOT / 'tests/fixtures/d4_classifier'
        cmd = ['wsl.exe', '-d', 'Ubuntu', '--', 'g++', '-std=c++11', '-O2', '-Wall', '-Wextra', '-Werror',
               '-DSOUNDJIE_LINKED_MODEL_SHA256="' + hashlib.sha256(b'test-model').hexdigest() + '"',
               '-DSOUNDJIE_FRONTEND_SHA256="' + hashlib.sha256(CONFIG.read_bytes()).hexdigest() + '"',
               '-DSOUNDJIE_PHYSICAL_OUTPUTS=' + str(getattr(cls,'outputs',8)),
               '-DSOUNDJIE_STORAGE_OUTPUTS=' + str(getattr(cls,'storage',getattr(cls,'outputs',8))),
               '-I' + linux(fixture), '-I' + linux(ROOT / 'src')]
        cmd += [linux(p) for p in (SOURCE, ROOT / 'src/logmel_frontend.cpp', ROOT / 'src/fft_radix2.cpp',
                                   ROOT / 'src/sha256.cpp', fixture / 'n2cube_stub.cpp')]
        cmd += ['-o', linux(cls.binary)]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode:
            raise AssertionError(result.stderr.decode('utf-8', errors='replace'))
        spec = importlib.util.spec_from_file_location('d4_classifier_reference', ROOT / 'scripts/host/logmel_reference.py')
        cls.reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.reference)

    def setUp(self):
        self.case = tempfile.TemporaryDirectory(dir=self.folder)
        self.addCleanup(self.case.cleanup)
        self.root = Path(self.case.name)
        for name in ('config', 'model'):
            (self.root / name).mkdir()
        shutil.copyfile(CONFIG, self.root / 'config/logmel_frontend_v1.json')
        (self.root / 'model/dpu_soundjie_d3.elf').write_bytes(b'test-model')
        self.samples = np.rint(12000 * np.sin(2 * np.pi * 733 * np.arange(16000) / 16000)).astype('<i2')
        self.pcm = self.root / 'window.s16'
        self.pcm.write_bytes(self.samples.tobytes())

    def run_case(self, **changes):
        env = {'SOUNDJIE_BUNDLE_DIR': linux(self.root), 'SJ_TEST_TRACE': linux(self.root / 'trace'),
               'SJ_TEST_TENSOR': linux(self.root / 'tensor'), 'SJ_TEST_INPUT_SCALE': '32', 'SJ_TEST_OUTPUT_SCALE': '0.5'}
        env.update(changes)
        return subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'env',
                               *[k + '=' + v for k, v in env.items()], linux(self.binary), linux(self.pcm)],
                              capture_output=True)

    def test_pcm_uses_real_frontend_dynamic_scales_and_clean_protocol(self):
        result = self.run_case()
        self.assertEqual(result.returncode, 0, result.stderr)
        fields = result.stdout.decode().split()
        self.assertEqual(len(fields), 11)
        self.assertEqual(int(fields[0]), 3)
        expected_conf = math.exp(2) / sum(math.exp(x * 0.5) for x in [-2, -1, 0, 4, 1, -3, 2, 0])
        self.assertAlmostEqual(float(fields[1]), expected_conf, places=7)
        scores = np.asarray([float(x) for x in fields[3:]])
        expected_scores = np.exp(np.array([-2, -1, 0, 4, 1, -3, 2, 0]) * .5)
        expected_scores /= expected_scores.sum()
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-7)
        self.assertAlmostEqual(scores.sum(), 1.)
        self.assertGreaterEqual(float(fields[2]), 0)
        config = self.reference.load_frontend_config(CONFIG)
        expected = self.reference.quantize_int8(self.reference.compute_logmel(self.samples.astype(float) / 32768.0, config), 32)
        actual = np.frombuffer((self.root / 'tensor').read_bytes(), dtype=np.int8)
        np.testing.assert_array_equal(actual, expected.reshape(-1))
        self.assertEqual((self.root / 'trace').read_text().splitlines(),
                         ['open', 'load', 'create', 'run', 'destroy_task', 'destroy_kernel', 'close'])

    def test_rejects_truncated_pcm_and_changed_model_before_dpu(self):
        for content in (b'', self.samples.tobytes()[:-1], self.samples.tobytes() + b'xx'):
            self.pcm.write_bytes(content)
            result = self.run_case()
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b'')
            self.assertFalse((self.root / 'trace').exists())
        self.pcm.write_bytes(self.samples.tobytes())
        (self.root / 'model/dpu_soundjie_d3.elf').write_bytes(b'wrong model')
        self.assertNotEqual(self.run_case().returncode, 0)
        self.assertFalse((self.root / 'trace').exists())

    def test_changed_frontend_rejected_before_dpu(self):
        with (self.root / 'config/logmel_frontend_v1.json').open('ab') as stream:
            stream.write(b' ')
        self.assertNotEqual(self.run_case().returncode, 0)
        self.assertFalse((self.root / 'trace').exists())

    def test_invalid_scales_do_not_execute_inference(self):
        for key in ('SJ_TEST_INPUT_SCALE', 'SJ_TEST_OUTPUT_SCALE'):
            for value in ('0', '-1', 'nan', 'inf'):
                with self.subTest(key=key, value=value):
                    result = self.run_case(**{key: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stdout, b'')
        self.assertNotIn('run', (self.root / 'trace').read_text().splitlines())

    def test_dpu_faults_never_publish_a_prediction(self):
        for failure in ('open', 'load', 'create', 'input_size', 'output_size', 'input_null', 'output_null', 'run', 'zero'):
            with self.subTest(failure=failure):
                result = self.run_case(SJ_TEST_FAIL=failure)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b'')

    def test_runtime_calls_real_adapter_with_two_audio_windows(self):
        runtime = self.root / 'runtime'
        build = subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'g++', '-std=c++11', '-O2',
                                '-I' + linux(ROOT / 'src'),
                                *[linux(ROOT / 'src' / name) for name in
                                  ('pcm_ring_buffer.cpp', 'soundjie_d4_state.cpp', 'soundjie_d4_runtime.cpp',
                                   'fft_radix2.cpp','known_clip_matcher.cpp','traffic_fine_model.cpp')],
                                '-o', linux(runtime)], capture_output=True)
        self.assertEqual(build.returncode, 0, build.stderr)
        stereo = np.repeat(np.tile(self.samples, 2)[:, None], 2, axis=1).astype('<i2')
        stream = self.root / 'stereo.raw'
        stream.write_bytes(stereo.tobytes())
        result = subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'env',
                                 'SOUNDJIE_BUNDLE_DIR=' + linux(self.root),
                                 'SJ_TEST_TRACE=' + linux(self.root / 'trace'),
                                 'SJ_TEST_TENSOR=' + linux(self.root / 'tensor'),
                                 'SJ_TEST_INPUT_SCALE=32', 'SJ_TEST_OUTPUT_SCALE=0.5',
                                 linux(runtime), '--input', linux(stream),
                                 '--output-dir', linux(self.root / 'session'),
                                 '--current', linux(self.root / 'current.json'),
                                 '--classifier-command', linux(self.binary)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['dpu_calls'], 2)
        rows = [json.loads(line) for line in (self.root / 'session/timeline.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([row['class_id'] for row in rows], [3, 3, -1])
        self.assertEqual(rows[0]['label_zh'], '玻璃破碎')
        self.assertFalse(rows[-1]['usb_ok'])
        self.assertFalse(list((self.root / 'session').glob('window-*.s16')))
        self.assertEqual((self.root / 'trace').read_text().splitlines().count('run'), 2)


class D4ClassifierBuildTests(unittest.TestCase):
    def test_build_refuses_existing_output_before_compiler_or_model_access(self):
        script = ROOT / 'scripts/guest/build_soundjie_d4_classifier.sh'
        self.assertTrue(script.is_file(), 'missing reproducible ARM classifier build')
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / 'sentinel').write_bytes(b'preserve')
            result = subprocess.run(['wsl.exe', '-d', 'Ubuntu', '--', 'bash', linux(script),
                                     '--repo-root', linux(ROOT), '--dnndk-root', '/missing',
                                     '--model-elf', '/missing/model', '--output-dir', linux(folder)],
                                    capture_output=True)
            self.assertEqual(result.returncode, 3, result.stderr)
            self.assertEqual((folder / 'sentinel').read_bytes(), b'preserve')
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ['sentinel'])


if __name__ == '__main__':
    unittest.main()
