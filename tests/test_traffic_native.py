import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import torch
from scripts.host.train_soundjie_traffic import feature,export_model
ROOT=Path(__file__).resolve().parents[1]
def lin(p):return '/mnt/'+str(p)[0].lower()+str(p)[2:].replace('\\','/')

class TrafficNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(dir=ROOT/'artifacts');cls.work=Path(cls.tmp.name);cls.binary=cls.work/'traffic'
        sources=['fft_radix2.cpp','known_clip_matcher.cpp','traffic_fine_model.cpp','traffic_fine_cli.cpp']
        r=subprocess.run(['wsl','-d','Ubuntu','--','g++','-std=c++11','-O2','-Wall','-Wextra','-Werror','-I'+lin(ROOT/'src'),*[lin(ROOT/'src'/s) for s in sources],'-o',lin(cls.binary)],capture_output=True)
        if r.returncode:raise AssertionError(r.stderr.decode('utf-8','replace'))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_python_cpp_feature_and_probability_agree(self):
        torch.manual_seed(8);m=torch.nn.Sequential(torch.nn.Linear(96,48),torch.nn.ReLU(),torch.nn.Linear(48,3))
        mean=np.linspace(-1,1,96).astype('f4');scale=np.linspace(.3,2,96).astype('f4')
        model=self.work/'model.bin';export_model(model,mean,scale,m)
        x=np.random.default_rng(8).integers(-12000,12000,16000,dtype=np.int16);raw=self.work/'input.raw';raw.write_bytes(x.astype('<i2').tobytes())
        r=subprocess.run(['wsl','-d','Ubuntu','--',lin(self.binary),'--model',lin(model),'--pcm',lin(raw)],capture_output=True);self.assertEqual(r.returncode,0,r.stderr)
        got=json.loads(r.stdout);f=feature(x/32768)
        with torch.no_grad():expected=torch.sigmoid(m(torch.from_numpy((f-mean)/scale))).numpy()
        np.testing.assert_allclose(got['features'],f,atol=2e-5);np.testing.assert_allclose(got['scores'],expected,atol=2e-5)
    def test_truncated_model_is_rejected(self):
        model=self.work/'bad.bin';model.write_bytes(b'SJTF1\0\0\0')
        r=subprocess.run(['wsl','-d','Ubuntu','--',lin(self.binary),'--model',lin(model),'--pcm','/dev/null'],capture_output=True)
        self.assertNotEqual(r.returncode,0)

if __name__=='__main__':unittest.main()
