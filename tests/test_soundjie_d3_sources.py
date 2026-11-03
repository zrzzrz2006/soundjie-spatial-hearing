import hashlib
import json
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_modules(testcase: unittest.TestCase):
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.host import download_soundjie_d3 as downloader
        from scripts.host import soundjie_d3_manifest as manifest
    except (ImportError, ModuleNotFoundError) as exc:
        testcase.fail(f"D3 source modules are missing: {exc}")
    finally:
        try:
            sys.path.remove(str(REPO_ROOT))
        except ValueError:
            pass
    return manifest, downloader


def source_entry(*, source_id="fixture", sha256=None, size_bytes=None, **overrides):
    payload = {
        "id": source_id,
        "status": "active",
        "version": "1",
        "official_page": "https://example.invalid/dataset",
        "urls": ["https://example.invalid/archive.zip"],
        "archive_name": f"{source_id}.zip",
        "sha256": sha256 or ("0" * 64),
        "size_bytes": 123 if size_bytes is None else size_bytes,
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "license_decision": "approved",
        "license_decision_rationale": "fixture permits attributed competition use",
        "license_evidence": [
            {"path": "LICENSE.txt", "sha256": "1" * 64},
        ],
        "competition_use": "approved",
        "allowed_classes": ["background_unknown"],
        "fallback_source_id": None,
    }
    payload.update(overrides)
    return payload


def contract_payload(sources):
    return {
        "schema": 1,
        "dataset_root": "F:\\MLK\\DPU_ZYNQ7000\\datasets\\soundjie_v1",
        "sources": sources,
    }


class PayloadServer:
    def __init__(self, payload: bytes, *, honor_range: bool):
        self.payload = payload
        self.honor_range = honor_range
        self.requests = []

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                range_value = self.headers.get("Range")
                owner.requests.append(range_value)
                body = owner.payload
                status = 200
                start = 0
                if range_value and owner.honor_range:
                    start = int(range_value.removeprefix("bytes=").removesuffix("-"))
                    body = owner.payload[start:]
                    status = 206
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                if status == 206:
                    self.send_header(
                        "Content-Range",
                        f"bytes {start}-{len(owner.payload) - 1}/{len(owner.payload)}",
                    )
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}/archive.zip"

    def __exit__(self, _exc_type, _exc, _tb):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


class SoundjieD3SourceTests(unittest.TestCase):
    def write_contract(self, root: Path, payload) -> Path:
        path = root / "sources.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_rejects_unknown_keys_or_duplicate_source_ids(self):
        manifest, _ = load_modules(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unknown = contract_payload([source_entry(unexpected="not allowed")])
            with self.assertRaisesRegex(manifest.ContractError, "unknown.*unexpected"):
                manifest.load_source_contract(self.write_contract(root, unknown))

            duplicate = contract_payload(
                [source_entry(source_id="same"), source_entry(source_id="same")]
            )
            with self.assertRaisesRegex(manifest.ContractError, "duplicate.*same"):
                manifest.load_source_contract(self.write_contract(root, duplicate))

    def test_rejects_missing_license_or_unapproved_competition_use(self):
        manifest, _ = load_modules(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_license = contract_payload([source_entry(license="")])
            with self.assertRaisesRegex(manifest.ContractError, "license"):
                manifest.load_source_contract(self.write_contract(root, missing_license))

            unapproved = contract_payload([source_entry(competition_use="unresolved")])
            with self.assertRaisesRegex(manifest.ContractError, "competition_use"):
                manifest.load_source_contract(self.write_contract(root, unapproved))

    def test_dataset_root_is_exact_f_drive_path(self):
        manifest, _ = load_modules(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = contract_payload([source_entry()])
            wrong["dataset_root"] = "C:\\datasets\\soundjie_v1"
            with self.assertRaisesRegex(manifest.ContractError, "dataset_root"):
                manifest.load_source_contract(self.write_contract(root, wrong))

            valid = manifest.load_source_contract(
                self.write_contract(root, contract_payload([source_entry()]))
            )
            self.assertEqual(
                str(valid.dataset_root),
                "F:\\MLK\\DPU_ZYNQ7000\\datasets\\soundjie_v1",
            )

    def test_repository_contract_activates_only_two_audited_archives(self):
        manifest, _ = load_modules(self)
        contract = manifest.load_source_contract(
            REPO_ROOT / "config" / "soundjie_d3_sources.json"
        )
        self.assertEqual(
            [(source.id, source.status) for source in contract.sources],
            [("esc50", "active"), ("librispeech_dev_clean", "active")],
        )
        for source in contract.sources:
            self.assertEqual(source.license_decision, "approved")
            self.assertEqual(source.competition_use, "approved")
            self.assertRegex(source.sha256 or "", r"^[0-9a-f]{64}$")
            self.assertGreater(source.size_bytes or 0, 0)
            self.assertTrue(source.license_url.startswith("https://"))
            self.assertTrue(source.license_decision_rationale)
            self.assertTrue(source.license_evidence)

    def test_license_audit_is_bound_to_declared_file_hashes(self):
        manifest, _ = load_modules(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "LICENSE.txt"
            evidence.write_text("CC-BY-4.0 fixture", encoding="utf-8")
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            payload = source_entry(
                license_evidence=[{"path": "LICENSE.txt", "sha256": digest}]
            )
            source = manifest.load_source_contract(
                self.write_contract(root, contract_payload([payload]))
            ).sources[0]
            decision = manifest.audit_license(root, source)
            self.assertTrue(decision.approved)
            self.assertEqual(decision.evidence_sha256, (("LICENSE.txt", digest),))

            evidence.write_text("unrelated or modified text", encoding="utf-8")
            decision = manifest.audit_license(root, source)
            self.assertFalse(decision.approved)
            self.assertIn("hash", decision.reason)

    def test_partial_resume_requires_matching_range(self):
        manifest, downloader = load_modules(self)
        payload = (b"soundjie-d3-range-fixture-" * 1000) + b"end"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloads = root / "downloads"
            downloads.mkdir()
            partial = downloads / "fixture.zip.partial"
            partial.write_bytes(payload[:311])

            with PayloadServer(payload, honor_range=False) as url:
                contract = manifest.load_source_contract(
                    self.write_contract(
                        root,
                        contract_payload(
                            [
                                source_entry(
                                    urls=[url], sha256=digest, size_bytes=len(payload)
                                )
                            ]
                        ),
                    )
                )
                record = downloader.download_source(contract.sources[0], root)

            final_path = downloads / "fixture.zip"
            self.assertEqual(final_path.read_bytes(), payload)
            self.assertEqual(record.sha256, digest)
            self.assertFalse(record.resumed)
            self.assertGreaterEqual(len(record.attempted_urls), 1)
            self.assertFalse(partial.exists())

    def test_hash_mismatch_never_replaces_verified_archive(self):
        manifest, downloader = load_modules(self)
        original = b"already-present-user-data"
        incoming = b"different-network-payload"
        expected = hashlib.sha256(incoming).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloads = root / "downloads"
            downloads.mkdir()
            final_path = downloads / "fixture.zip"
            final_path.write_bytes(original)

            with PayloadServer(incoming, honor_range=True) as url:
                contract = manifest.load_source_contract(
                    self.write_contract(
                        root,
                        contract_payload(
                            [
                                source_entry(
                                    urls=[url], sha256=expected, size_bytes=len(incoming)
                                )
                            ]
                        ),
                    )
                )
                with self.assertRaisesRegex(downloader.DownloadError, "existing target"):
                    downloader.download_source(contract.sources[0], root)

            self.assertEqual(final_path.read_bytes(), original)
            self.assertFalse((downloads / "fixture.zip.partial").exists())

    def test_extract_rejects_traversal_links_and_existing_destination(self):
        _, downloader = load_modules(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traversal = root / "traversal.zip"
            with zipfile.ZipFile(traversal, "w") as archive:
                archive.writestr("../escaped.txt", b"escape")
            with self.assertRaisesRegex(downloader.DownloadError, "unsafe archive"):
                downloader.extract_archive(traversal, root / "out-traversal")
            self.assertFalse((root / "escaped.txt").exists())

            linked = root / "linked.zip"
            info = zipfile.ZipInfo("link")
            info.external_attr = (0o120777 << 16)
            with zipfile.ZipFile(linked, "w") as archive:
                archive.writestr(info, "target")
            with self.assertRaisesRegex(downloader.DownloadError, "link"):
                downloader.extract_archive(linked, root / "out-link")

            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(downloader.DownloadError, "already exists"):
                downloader.extract_archive(linked, existing)

    def test_extract_publishes_safe_zip_create_only(self):
        _, downloader = load_modules(self)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "safe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("dataset/LICENSE.txt", "CC-BY-4.0")
                archive.writestr("dataset/audio/example.wav", b"RIFF-fixture")
            destination = root / "unpacked"
            result = downloader.extract_archive(archive_path, destination)
            self.assertEqual(result, destination)
            self.assertEqual(
                (destination / "dataset" / "LICENSE.txt").read_text(encoding="utf-8"),
                "CC-BY-4.0",
            )
            self.assertEqual(
                (destination / "dataset" / "audio" / "example.wav").read_bytes(),
                b"RIFF-fixture",
            )


if __name__ == "__main__":
    unittest.main()
