import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "host" / "stage_sound_shape_probe_sd.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sound_shape_stager", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SoundShapeSdStagingTests(unittest.TestCase):
    def test_wrapper_pins_create_only_sound_namespace(self):
        module = load_module()
        self.assertEqual(module.DESTINATIONS, ("SJD0.TGZ", "SJD0.SHA256"))
        self.assertEqual(module.MANIFEST_NAME, "SJD0.SHA256")

    def test_preflight_delegates_exact_files_and_serial(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sd = root / "sd"
            source = root / "source"
            sd.mkdir()
            source.mkdir()
            manifest = source / module.MANIFEST_NAME
            manifest.write_text("manifest", encoding="ascii")
            sentinel = object()
            with mock.patch.object(module.STAGER, "preflight", return_value=sentinel) as call:
                result = module.preflight(sd, source, "54D7-D258")
        self.assertIs(result, sentinel)
        call.assert_called_once_with(
            sd, source, manifest, module.DESTINATIONS,
            expected_volume_serial="54D7-D258",
        )

    def test_cli_has_no_overwrite_or_arbitrary_destination_option(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("--destination", text)
        self.assertNotIn("overwrite", text.lower())
        self.assertIn("dry_run=args.dry_run", text)
        self.assertIn('with args.report.open("x"', text)


if __name__ == "__main__":
    unittest.main()
