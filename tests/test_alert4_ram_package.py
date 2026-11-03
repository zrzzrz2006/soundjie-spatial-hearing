import tempfile
from pathlib import Path
import unittest
import numpy as np
import soundfile as sf
from scripts.host import package_soundjie_alert4 as pack

class RamPackageTests(unittest.TestCase):
    def test_pcm_reconstruction_preserves_center_padding_and_tensor(self):
        frontend=pack.load_frontend_config(pack.ROOT/'config/logmel_frontend_v1.json')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'audio.wav';pcm=np.rint(11000*np.sin(np.arange(8000)*.33)).astype('<i2');sf.write(p,pcm,16000,subtype='PCM_16')
            digest=pack.sha(p.read_bytes());src=dict(path=str(p),sha256=digest);start=-4000
            expected=np.zeros(16000,dtype='<i2');expected[4000:12000]=pcm
            row=dict(recording_id='test',parent_sha256=digest,pcm_sha256_16k=pack.sha(pcm.tobytes()),start_sample_16k=start,
                feature_sha256=pack.sha(pack.compute_feature(expected.astype(np.float64)/32768.,frontend).tobytes()))
            actual,_=pack.pcm_fixture(row,{('test',digest):src},frontend);self.assertEqual(actual,expected.tobytes())
            row['feature_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'exported tensor'):pack.pcm_fixture(row,{('test',digest):src},frontend)

    def test_existing_package_is_refused_before_any_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=p/'user.txt';f.write_text('keep')
            with self.assertRaises(FileExistsError):pack.build(Path('/missing'),p,Path('/missing'),4000)
            self.assertEqual(f.read_text(),'keep')

if __name__=='__main__':unittest.main()
