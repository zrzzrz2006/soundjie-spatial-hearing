import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "guest" / "prepare_quantized_deploy.py"


class PrepareQuantizedDeployTests(unittest.TestCase):
    def test_replaces_image_data_with_fixed_data_input(self):
        source = '''name: "probe"
layer {
  name: "calibration_input"
  type: "ImageData"
  top: "data"
  top: "label"
  image_data_param { source: "list.txt" batch_size: 1 }
}
layer {
  name: "data_fixed"
  type: "FixedNeuron"
  bottom: "data"
  top: "data"
  fixed_param { fixed_method: OVER_FLOW bit_width: 8 }
}
'''
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "quantized.prototxt"
            output_path = root / "deploy.prototxt"
            input_path.write_text(source, encoding="utf-8")
            subprocess.run(
                ["py", "-3", str(SCRIPT), "--input", str(input_path),
                 "--output", str(output_path)],
                check=True, capture_output=True, text=True,
            )
            result = output_path.read_text(encoding="utf-8")

        self.assertIn('input: "data"', result)
        self.assertIn("input_shape { dim: 1 dim: 1 dim: 64 dim: 96 }", result)
        self.assertNotIn("ImageData", result)
        self.assertNotIn('top: "label"', result)
        self.assertIn('type: "FixedNeuron"', result)

    def test_rejects_missing_or_multiple_image_data_layers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.prototxt"
            for text in ('name: "none"\n', 'layer { type: "ImageData" }\nlayer { type: "ImageData" }\n'):
                input_path.write_text(text, encoding="utf-8")
                result = subprocess.run(
                    ["py", "-3", str(SCRIPT), "--input", str(input_path),
                     "--output", str(root / "out.prototxt")],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
