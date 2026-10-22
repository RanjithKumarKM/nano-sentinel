import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from process_incoming import process_incoming


class FakeInspectionApi:
    def __init__(self):
        self.records = []
        self.uploads = []

    def get(self, url, timeout):
        response = Mock()
        response.json.return_value = {"inspections": list(self.records)}
        return response

    def post(self, url, files, timeout):
        field_name, upload = next(iter(files.items()))
        filename, source, _content_type = upload
        self.uploads.append((field_name, filename, source.read()))
        record = {
            "inspection_id": f"NS-TEST-{len(self.uploads)}",
            "filename": filename,
            "model": {"predicted_class": "none", "confidence_percent": 99.0},
        }
        self.records.append(record)
        response = Mock()
        response.json.return_value = record
        return response


class ProcessIncomingTests(unittest.TestCase):
    def test_import_skips_already_saved_names_and_keeps_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            incoming = Path(directory)
            first = incoming / "demo_wafer_0001.npy"
            second = incoming / "demo_wafer_0002.npy"
            first.write_bytes(b"first map")
            second.write_bytes(b"second map")
            api = FakeInspectionApi()

            result = process_incoming(incoming, http=api)

            self.assertEqual(result, {
                "found": 2, "imported": 2, "skipped": 0, "failed": 0
            })
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            self.assertEqual(len(api.uploads), 2)

            result = process_incoming(incoming, http=api)

            self.assertEqual(result, {
                "found": 2, "imported": 0, "skipped": 2, "failed": 0
            })
            self.assertEqual(len(api.uploads), 2)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())

    def test_dry_run_lists_files_without_uploading_or_moving_them(self):
        with tempfile.TemporaryDirectory() as directory:
            incoming = Path(directory)
            source = incoming / "demo_wafer_0001.npy"
            source.write_bytes(b"wafer map")
            api = FakeInspectionApi()

            result = process_incoming(incoming, dry_run=True, http=api)

            self.assertEqual(result, {
                "found": 1, "imported": 0, "skipped": 0, "failed": 0
            })
            self.assertEqual(api.uploads, [])
            self.assertTrue(source.is_file())

    def test_failed_upload_does_not_mark_file_as_processed(self):
        with tempfile.TemporaryDirectory() as directory:
            incoming = Path(directory)
            source = incoming / "bad.npy"
            source.write_bytes(b"invalid map")
            api = Mock()
            api.get.return_value.json.return_value = {"inspections": []}
            api.post.side_effect = OSError("offline")

            result = process_incoming(incoming, http=api)

            self.assertEqual(result, {
                "found": 1, "imported": 0, "skipped": 0, "failed": 1
            })
            self.assertTrue(source.is_file())


if __name__ == "__main__":
    unittest.main()
