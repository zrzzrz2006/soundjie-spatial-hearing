import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "board" / "xvf3800_probe.sh"
RECORD = ROOT / "scripts" / "board" / "xvf3800_record.sh"


def to_wsl(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    suffix = path.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{suffix}"


class BoardScriptTests(unittest.TestCase):
    def make_fixture(self, card: int, channels: str, formats: str = "S16_LE"):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        asound = root / "proc" / "asound"
        card_dir = asound / f"card{card}"
        card_dir.mkdir(parents=True)
        (asound / "cards").write_text(
            f" {card} [XVF3800       ]: USB-Audio - ReSpeaker XVF3800\n"
            "                      Seeed Studio ReSpeaker XVF3800\n",
            encoding="utf-8",
        )
        (card_dir / "usbid").write_text("2886:0018\n", encoding="utf-8")
        (card_dir / "id").write_text("XVF3800\n", encoding="utf-8")
        (card_dir / "stream0").write_text(
            "Seeed Studio ReSpeaker XVF3800 at usb-e0002000, high speed\n",
            encoding="utf-8",
        )
        fake = root / "arecord"
        fake.write_text(
            textwrap.dedent(
                f"""\
                #!/bin/sh
                case " $* " in
                  *" -l "*)
                    echo "card {card}: XVF3800 [ReSpeaker XVF3800], device 0: USB Audio [USB Audio]"
                    ;;
                  *" --dump-hw-params "*)
                    cat <<'EOF'
                ACCESS:  RW_INTERLEAVED
                FORMAT:  {formats}
                SUBFORMAT:  STD
                CHANNELS: {channels}
                RATE: [16000 48000]
                EOF
                    ;;
                  *)
                    out=""
                    for arg in "$@"; do
                      if [ "$arg" = "--file" ]; then
                        echo "--file is parsed as --file-type by ALSA 1.1.3" >&2
                        exit 64
                      fi
                      out=$arg
                    done
                    [ -n "$out" ] && printf 'fake-wave' > "$out"
                    ;;
                esac
                """
            ),
            encoding="utf-8",
            newline="\n",
        )
        fake.chmod(0o755)
        return temp, root, fake

    def run_probe(self, root: Path, fake: Path, report: Path):
        command = (
            f"bash {to_wsl(PROBE)} --proc-root {to_wsl(root / 'proc')} "
            f"--arecord-bin {to_wsl(fake)} --report {to_wsl(report)}"
        )
        return subprocess.run(
            ["wsl.exe", "bash", "-lc", command],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )

    def test_no_matching_device_returns_not_found(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "proc" / "asound").mkdir(parents=True)
            (root / "proc" / "asound" / "cards").write_text(
                " 0 [Codec]: Generic - Onboard Codec\n", encoding="utf-8"
            )
            fake = root / "arecord"
            fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8", newline="\n")
            fake.chmod(0o755)
            result = self.run_probe(root, fake, root / "report.env")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("DEVICE_FOUND=no", result.stdout)

    def test_identity_detection_survives_card_number_change(self):
        fixture, root, fake = self.make_fixture(3, "[2 6]")
        with fixture:
            result = self.run_probe(root, fake, root / "report.env")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("CARD_INDEX=3", result.stdout)
            self.assertIn("PCM_DEVICE=hw:3,0", result.stdout)
            self.assertIn("USB_ID=2886:0018", result.stdout)

    def test_two_channel_capabilities_are_reported(self):
        fixture, root, fake = self.make_fixture(1, "2")
        with fixture:
            result = self.run_probe(root, fake, root / "report.env")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("CHANNELS=2", result.stdout)
            self.assertIn("SELECTED_CHANNELS=2", result.stdout)

    def test_six_channel_capabilities_prefer_six(self):
        fixture, root, fake = self.make_fixture(5, "[2 6]")
        with fixture:
            result = self.run_probe(root, fake, root / "report.env")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("CHANNELS=[2 6]", result.stdout)
            self.assertIn("SELECTED_CHANNELS=6", result.stdout)

    def test_unsupported_format_is_rejected(self):
        fixture, root, fake = self.make_fixture(2, "2", "S24_3LE")
        with fixture:
            result = self.run_probe(root, fake, root / "report.env")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SUPPORTED_CAPTURE=no", result.stdout)

    def test_record_uses_discovered_card_and_capabilities(self):
        fixture, root, fake = self.make_fixture(4, "[2 6]")
        with fixture:
            output = root / "capture.wav"
            report = root / "report.env"
            command = (
                f"bash {to_wsl(RECORD)} --seconds 1 --output {to_wsl(output)} "
                f"--proc-root {to_wsl(root / 'proc')} --arecord-bin {to_wsl(fake)} "
                f"--report {to_wsl(report)}"
            )
            result = subprocess.run(
                ["wsl.exe", "bash", "-lc", command],
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_file())
            self.assertIn("CAPTURE_DEVICE=hw:4,0", result.stdout)
            self.assertIn("CAPTURE_CHANNELS=6", result.stdout)


if __name__ == "__main__":
    unittest.main()
