"""All mutable fixtures stay beneath skill/tests and are automatically removed."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parent
SCRIPT = TESTS.parent / "package-manifest-verify/scripts/verify_manifest.py"
SPEC = importlib.util.spec_from_file_location("verify_manifest", SCRIPT)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="manifest-", dir=TESTS)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.root = self.work / "package"
        self.root.mkdir()
        self.payload = self.root / "file.bin"
        self.payload.write_bytes(b"reproducible payload\x00\xff")
        self.digest = hashlib.sha256(self.payload.read_bytes()).hexdigest()
        self.manifest = self.work / "sha256sums.txt"
        self.manifest.write_text(f"{self.digest}  file.bin\n", encoding="utf-8")

    def cli(self):
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), str(self.root), str(self.manifest)],
                                capture_output=True, text=True, check=False)
        return result.returncode, json.loads(result.stdout)

    def test_verified_repeat_is_deterministic_and_read_only(self):
        before = {path: path.read_bytes() for path in self.work.rglob("*") if path.is_file()}
        first = self.cli()
        self.assertEqual(first, self.cli())
        self.assertEqual(first[0], 0)
        self.assertEqual(first[1]["status"], "VERIFIED")
        self.assertEqual(first[1]["checked"], 1)
        after = {path: path.read_bytes() for path in self.work.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_tamper_and_missing_payload(self):
        self.payload.write_bytes(b"tampered")
        code, result = self.cli()
        self.assertEqual(code, 1)
        self.assertEqual(result["files"][0]["status"], "MISMATCH")
        self.payload.unlink()
        self.assertEqual(self.cli()[0], 2)

    def test_unsafe_paths_are_rejected_before_hashing(self):
        outside = self.work / "outside.bin"
        outside.write_bytes(self.payload.read_bytes())
        names = ["../outside.bin", "sub/../../outside.bin", "/outside.bin", "C:/outside.bin",
                 "C:outside.bin", "\\\\server\\share\\file", "//server/share/file", "sub\\file",
                 "file.bin:stream", "./file.bin", "sub//file", "nul", "CON.txt", "dir/COM1",
                 "file.bin.", "file.bin ", "sub/../file.bin", "file\x00.bin"]
        for name in names:
            with self.subTest(name=name):
                self.manifest.write_text(f"{self.digest}  {name}\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    VERIFY.verify(self.root, self.manifest)

    def test_malformed_empty_duplicate_and_missing_manifest(self):
        for content in ("", "wrong  file.bin\n", f"{self.digest}  file.bin\n{self.digest}  FILE.BIN\n"):
            with self.subTest(content=content):
                self.manifest.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    VERIFY.verify(self.root, self.manifest)
        self.manifest.unlink()
        self.assertEqual(self.cli()[0], 2)

    def test_binary_marker_and_relative_subdirectory(self):
        nested = self.root / "sub"
        nested.mkdir()
        (nested / "file with space.bin").write_bytes(self.payload.read_bytes())
        self.manifest.write_text(f"{self.digest.upper()} *sub/file with space.bin\n", encoding="utf-8")
        self.assertEqual(self.cli()[0], 0)

    def test_link_escape_is_rejected(self):
        outside = self.work / "outside.bin"
        outside.write_bytes(self.payload.read_bytes())
        link = self.root / "link.bin"
        try:
            os.symlink(outside, link)
        except OSError as exc:
            self.skipTest("Symlink creation unavailable: " + str(exc))
        self.manifest.write_text(f"{self.digest}  link.bin\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Links/reparse"):
            VERIFY.verify(self.root, self.manifest)


if __name__ == "__main__":
    unittest.main()
