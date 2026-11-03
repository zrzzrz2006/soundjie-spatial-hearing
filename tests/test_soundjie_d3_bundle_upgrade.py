import importlib
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module("scripts.host.upgrade_soundjie_d3_bundle_config")
    finally:
        sys.path.pop(0)


class SoundjieD3BundleUpgradeTests(unittest.TestCase):
    def test_parses_recording_specific_esc50_attribution(self):
        module = load_module()
        text = (
            "- [1-13613-A.ogg]: clip derived from smoke.wav "
            "(http://www.freesound.org/people/hookhead/sounds/13613/) "
            "by hookhead [CC0]\n"
        )
        value = module.parse_esc50_attribution(text, "1-13613-A-37.wav")
        self.assertEqual(value["creator"], "hookhead")
        self.assertEqual(value["license"], "CC0-1.0")
        self.assertEqual(value["source_url"], "https://freesound.org/people/hookhead/sounds/13613/")
        self.assertIn("smoke.wav", value["source_citation"])

    def test_parses_librispeech_reader_name(self):
        module = load_module()
        text = ";ID |SEX|SUBSET|MINUTES|NAME\n1272 | M | dev-clean | 8.0 | John Doe\n"
        readers = module.parse_librispeech_readers(text)
        self.assertEqual(readers["1272"], "John Doe")

    def test_rejects_missing_recording_specific_metadata(self):
        module = load_module()
        with self.assertRaisesRegex(ValueError, "ESC-50 attribution"):
            module.parse_esc50_attribution("unrelated", "1-13613-A-37.wav")

    def test_upgrade_is_create_only_and_binds_demo_git_and_licenses(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.json"
            demo = root / "demo.jsonl"
            esc = root / "ESC-LICENSE"
            libri = root / "LIBRI-LICENSE"
            speakers = root / "SPEAKERS.TXT"
            model.write_text('{"upstream": {}}\n', encoding="utf-8")
            demo.write_text("{}\n", encoding="utf-8")
            esc.write_text(
                "- [1-13613-A.ogg]: clip derived from smoke.wav "
                "(http://www.freesound.org/people/hookhead/sounds/13613/) "
                "by hookhead [CC0]\n",
                encoding="utf-8",
            )
            libri.write_text("LibriSpeech license\n", encoding="utf-8")
            speakers.write_text("1272 | M | dev-clean | 8.0 | John Doe\n", encoding="utf-8")

            def member(name, path):
                payload = path.read_bytes()
                return {
                    "name": name,
                    "source": str(path),
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "mode": "0644",
                }

            config = {
                "archive_members": [
                    member("manifest/soundjie-d3-model-manifest.json", model),
                    member("manifest/demo.jsonl", demo),
                    member("probe/run_soundjie_d3.sh", demo),
                ],
                "cases": [
                    {
                        "split": "demo",
                        "recording_id": "esc50:x/1-13613-A-37.wav",
                    },
                    {
                        "split": "demo",
                        "recording_id": (
                            "librispeech_dev_clean:x/1272-128104-0000.flac"
                        ),
                    },
                ],
            }
            source = root / "input.json"
            source.write_text(json.dumps(config), encoding="utf-8")
            output = root / "output"
            commit = "a" * 40
            result = module.upgrade_config(
                source, esc, libri, speakers, commit, output
            )
            upgraded = json.loads(result.read_text(encoding="utf-8"))
            names = [item["name"] for item in upgraded["archive_members"]]
            self.assertEqual(upgraded["git_commit"], commit)
            self.assertIn("licenses/ESC-50-LICENSE", names)
            self.assertIn("licenses/LibriSpeech-LICENSE.TXT", names)
            derived = json.loads(
                (output / "soundjie-d3-model-manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(derived["upstream"]["git_commit"], commit)
            self.assertEqual(
                derived["upstream"]["demo_manifest_sha256"],
                hashlib.sha256(demo.read_bytes()).hexdigest(),
            )
            with self.assertRaises(FileExistsError):
                module.upgrade_config(source, esc, libri, speakers, commit, output)


if __name__ == "__main__":
    unittest.main()
