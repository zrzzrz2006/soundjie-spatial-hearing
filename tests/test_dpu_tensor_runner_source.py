import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "dpu_tensor_runner.cpp"


class DpuTensorRunnerSourceTests(unittest.TestCase):
    def test_repopulates_input_before_every_dpu_run(self):
        source = RUNNER.read_text(encoding="utf-8")
        loop = source.index("for (int iteration = 1;")
        run = source.index("dpuRunTask(handles.task)", loop)
        populate = source.find("populate_input(", loop, run)

        self.assertNotEqual(
            populate,
            -1,
            "the input tensor must be rewritten inside the iteration loop before dpuRunTask",
        )


if __name__ == "__main__":
    unittest.main()
