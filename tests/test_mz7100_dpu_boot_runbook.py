import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs" / "runbooks" / "mz7100-dpu-uboot-one-time.md"


class Mz7100DpuBootRunbookTests(unittest.TestCase):
    def test_ps_pl_clock_level_shifter_and_reset_are_required_in_order(self):
        text = RUNBOOK.read_text(encoding="utf-8")
        commands = [
            "fpga loadb 0 0x10000000 ${filesize}",
            "mw.l 0xF8000008 0x0000DF0D",
            "mw.l 0xF8000170 0x00200500",
            "mw.l 0xF800A000 0x00000001",
            "mw.l 0xF800A014 0x00000F01",
            "mw.l 0xF8000004 0x0000767B",
            "md.l 0xF8000170 1",
            "md.l 0xF800A000 1",
            "md.l 0xF800A014 1",
            "bootm 0x02000000 - 0x03000000",
        ]

        offsets = [text.index(command) for command in commands]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("Core 0 Run timeout", text)
        self.assertIn("interrupt counter: 0", text)


if __name__ == "__main__":
    unittest.main()
