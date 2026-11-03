import hashlib
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src" / "sha256.h"
SOURCE = ROOT / "src" / "sha256.cpp"


def wsl_path(path):
    path = Path(path).resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


HARNESS = r"""
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "sha256.h"

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "vectors") {
      soundjie::Sha256 empty;
      std::cout << empty.final() << "\n";
      soundjie::Sha256 abc;
      abc.update("a", 1);
      abc.update("bc", 2);
      std::cout << abc.final() << "\n";
      std::vector<uint8_t> fixture(1000);
      for (std::size_t i = 0; i < fixture.size(); ++i) fixture[i] = static_cast<uint8_t>(i % 251);
      soundjie::Sha256 chunked;
      for (std::size_t offset = 0; offset < fixture.size();) {
        const std::size_t length = std::min<std::size_t>(37, fixture.size() - offset);
        chunked.update(&fixture[offset], length);
        offset += length;
      }
      std::cout << chunked.final() << "\n";
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "file") {
      std::cout << soundjie::sha256_file(argv[2]) << "\n";
      return 0;
    }
    return 64;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
"""


class Sha256CppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        harness = cls.root / "sha256_harness.cpp"
        cls.binary = cls.root / "sha256_harness"
        harness.write_text(textwrap.dedent(HARNESS), encoding="utf-8", newline="\n")
        result = subprocess.run([
            "wsl.exe", "g++", "-std=c++11", "-Wall", "-Wextra", "-Werror",
            f"-I{wsl_path(ROOT / 'src')}", wsl_path(harness), wsl_path(SOURCE),
            "-o", wsl_path(cls.binary),
        ], text=True, encoding="utf-8", errors="replace", capture_output=True)
        if result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_harness(self, *args):
        return subprocess.run(
            ["wsl.exe", wsl_path(self.binary), *args], text=True,
            encoding="utf-8", errors="replace", capture_output=True)

    def test_streaming_vectors_match_hashlib_across_blocks(self):
        result = self.run_harness("vectors")
        self.assertEqual(result.returncode, 0, result.stderr)
        fixture = bytes(index % 251 for index in range(1000))
        self.assertEqual(result.stdout.splitlines(), [
            hashlib.sha256(b"").hexdigest(),
            hashlib.sha256(b"abc").hexdigest(),
            hashlib.sha256(fixture).hexdigest(),
        ])

    def test_length_guard_uses_sha256_bit_length_limit(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("UINT64_MAX / 8", text)

    def test_file_hash_matches_hashlib_and_read_errors_fail(self):
        fixture = self.root / "fixture.bin"
        fixture.write_bytes(bytes(range(256)) * 19 + b"tail")
        result = self.run_harness("file", wsl_path(fixture))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), hashlib.sha256(fixture.read_bytes()).hexdigest())
        missing = self.run_harness("file", wsl_path(self.root / "missing.bin"))
        self.assertNotEqual(missing.returncode, 0)
        directory = self.run_harness("file", wsl_path(self.root))
        self.assertNotEqual(directory.returncode, 0)


if __name__ == "__main__":
    unittest.main()
