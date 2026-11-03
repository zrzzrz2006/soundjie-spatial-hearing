"""One elevated run: finish the pinned second sector, then reopen SD READ ONLY.

No retries. Names are deliberately fixed and create-only. Any repair failure
exits immediately. Both children revalidate the exact disk identity themselves.
"""
from pathlib import Path
import subprocess
import sys


def main():
    folder = Path(__file__).resolve().parent
    repair = subprocess.run([
        sys.executable, "-B", str(folder / "repair_sd_boot_two_sectors.py"),
        "--apply-confirmed", "--resume-second",
        "--run-name", "sd-boot-repair-20261024-remaining",
    ], check=False)
    if repair.returncode:
        return repair.returncode
    capture = subprocess.run([
        sys.executable, "-B", str(folder / "backup_sd_boot_readonly.py"),
        "--capture-name", "sd-boot-readonly-20261024-final",
    ], check=False)
    return capture.returncode


if __name__ == "__main__":
    raise SystemExit(main())
