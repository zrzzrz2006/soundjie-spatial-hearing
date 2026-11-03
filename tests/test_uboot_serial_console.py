import base64
import unittest

from scripts.host.uboot_serial_console import (
    AUTOBOOT_INTERRUPT_BYTE,
    UbootInterruptDetector,
    encode_base64_lines,
)


class UbootInterruptDetectorTests(unittest.TestCase):
    def test_detects_banner_across_serial_chunks_once(self):
        detector = UbootInterruptDetector()

        self.assertFalse(detector.feed(b"FSBL output\r\nU-Bo"))
        self.assertTrue(detector.feed(b"ot 2018.01\r\n"))
        self.assertFalse(detector.feed(b"U-Boot repeated\r\n"))

    def test_uses_ctrl_c_for_zero_delay_autoboot(self):
        self.assertEqual(AUTOBOOT_INTERRUPT_BYTE, b"\x03")

    def test_base64_transfer_lines_round_trip_and_fit_terminal_width(self):
        payload = bytes(range(256)) * 5
        lines = encode_base64_lines(payload)

        self.assertTrue(all(1 <= len(line) <= 76 for line in lines))
        self.assertEqual(base64.b64decode(b"".join(lines)), payload)


if __name__ == "__main__":
    unittest.main()
