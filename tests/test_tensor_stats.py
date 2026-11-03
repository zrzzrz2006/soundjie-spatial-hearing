import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src" / "tensor_stats.h"
SOURCE = ROOT / "src" / "tensor_stats.cpp"


def to_wsl(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    suffix = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


HARNESS = r"""
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include "tensor_stats.h"

int main(int argc, char** argv) {
  if (argc != 2) return 64;
  const std::string mode(argv[1]);
  if (mode == "stats") {
    const int8_t values[] = {-128, -1, 0, 2, 127};
    const TensorStats result = compute_tensor_stats(values, 5);
    std::cout << result.minimum << "," << result.maximum << "," << result.nonzero << "\n";
    return 0;
  }
  if (mode == "equal") {
    const int8_t left[] = {1, 2, 3, 4};
    const int8_t right[] = {1, 2, 3, 4};
    size_t mismatch = 999;
    const bool equal = tensors_equal(left, right, 4, &mismatch);
    std::cout << equal << "," << mismatch << "\n";
    return 0;
  }
  if (mode == "mismatch") {
    const int8_t left[] = {1, 2, 3, 4};
    const int8_t right[] = {1, 2, 9, 4};
    size_t mismatch = 999;
    const bool equal = tensors_equal(left, right, 4, &mismatch);
    std::cout << equal << "," << mismatch << "\n";
    return 0;
  }
  if (mode == "pattern") {
    int8_t values[16] = {};
    fill_deterministic_pattern(values, 16, 0x12345678u);
    for (size_t i = 0; i < 16; ++i) {
      if (i) std::cout << ',';
      std::cout << static_cast<int>(values[i]);
    }
    std::cout << "\n";
    return 0;
  }
  return 64;
}
"""


class TensorStatsTests(unittest.TestCase):
    def build_harness(self, root: Path) -> Path:
        self.assertTrue(HEADER.is_file(), f"missing production header: {HEADER}")
        self.assertTrue(SOURCE.is_file(), f"missing production source: {SOURCE}")
        harness = root / "tensor_stats_harness.cpp"
        binary = root / "tensor_stats_harness"
        harness.write_text(textwrap.dedent(HARNESS), encoding="utf-8", newline="\n")
        command = (
            f"g++ -std=c++11 -Wall -Wextra -Werror "
            f"-I{to_wsl(HEADER.parent)} {to_wsl(harness)} {to_wsl(SOURCE)} "
            f"-o {to_wsl(binary)}"
        )
        result = subprocess.run(
            ["wsl.exe", "bash", "-lc", command],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return binary

    def run_mode(self, mode: str) -> str:
        with tempfile.TemporaryDirectory() as temp:
            binary = self.build_harness(Path(temp))
            result = subprocess.run(
                ["wsl.exe", "bash", "-lc", f"{to_wsl(binary)} {mode}"],
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout.strip().splitlines()[-1]

    def test_computes_signed_min_max_and_nonzero_count(self):
        self.assertEqual(self.run_mode("stats"), "-128,127,4")

    def test_equal_tensors_report_length_as_mismatch_sentinel(self):
        self.assertEqual(self.run_mode("equal"), "1,4")

    def test_mismatch_reports_first_different_byte(self):
        self.assertEqual(self.run_mode("mismatch"), "0,2")

    def test_fixed_seed_pattern_is_reproducible(self):
        self.assertEqual(
            self.run_mode("pattern"),
            "37,35,68,24,8,-51,-99,-87,39,-111,120,120,32,-107,70,-23",
        )


if __name__ == "__main__":
    unittest.main()
