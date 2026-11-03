import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "guest" / "build_sound_shape_runner.sh"
HOST_SCRIPT = ROOT / "scripts" / "host" / "build_sound_shape_runner.ps1"


class SoundShapeGuestRunnerBuildTests(unittest.TestCase):
    def test_cross_build_and_abi_audit_are_fail_closed(self):
        text = SCRIPT.read_text(encoding="utf-8")
        for token in (
            "arm-linux-gnueabihf-g++", "-static-libgcc", "-static-libstdc++",
            "libn2cube.so", "libhineon.so", "readelf -h", "readelf -A",
            "Version5 EABI", "Tag_ABI_VFP_args", "GLIBC_2.17", "sha256sum",
            "output already exists", "src/sha256.cpp", "src/sha256.h",
            "SOUNDJIE_LINKED_MODEL_SHA256", 'ln "$temporary" "$output"',
            "model_snapshot", "mktemp -d", 'chmod 400 "$model_snapshot"',
        ):
            self.assertIn(token, text)
        self.assertIn('"$model_snapshot" -L', text)
        self.assertNotIn('"$model_elf" -L', text)

    def test_host_build_cleans_temporary_output_and_binds_model_hash(self):
        text = HOST_SCRIPT.read_text(encoding="utf-8")
        for token in (
            "SOUNDJIE_LINKED_MODEL_SHA256", "try {", "finally {",
            "Remove-Item -LiteralPath $temporary", "[System.IO.File]::Move",
            "$modelSnapshot", "[System.IO.File]::Copy",
        ):
            self.assertIn(token, text)
        self.assertIn("$modelSnapshot ('-L'", text)
        self.assertNotIn("$ModelElf ('-L'", text)
        self.assertIn("'-DSOUNDJIE_LINKED_MODEL_SHA256=\"{0}\"'", text)
        self.assertNotIn("LINKED_MODEL_SHA256=\\\"", text)


if __name__ == "__main__":
    unittest.main()
