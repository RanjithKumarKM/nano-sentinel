import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from inference import CLASS_NAMES
import main


class InspectionGradCamTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.maps_dir = self.root / "wafer_maps"
        self.maps_dir.mkdir()
        self.legacy_dir = self.root / "Incoming"
        self.legacy_dir.mkdir()
        self.record = {
            "inspection_id": "NS-TEST123456",
            "filename": "wafer.npy",
            "source": {"filename": "wafer.npy"},
        }
        self.inspection_patch = patch.object(
            main, "get_inspection", return_value=self.record
        )
        self.maps_patch = patch.object(main, "WAFER_MAPS", self.maps_dir)
        self.legacy_patch = patch.object(
            main, "LEGACY_WAFER_DIRS", (self.legacy_dir,)
        )
        self.inspection_patch.start()
        self.maps_patch.start()
        self.legacy_patch.start()
        self.addCleanup(self.inspection_patch.stop)
        self.addCleanup(self.maps_patch.stop)
        self.addCleanup(self.legacy_patch.stop)
        self.client = TestClient(main.app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_history_explanation_uses_saved_map_and_returns_heatmap(self):
        wafer = np.array([[0, 1, 2], [1, 1, 0]], dtype=np.uint8)
        np.save(self.maps_dir / f"{self.record['inspection_id']}.npy", wafer)
        result = {
            "predicted_class": "Scratch",
            "confidence_percent": 82.5,
            "target_layer": "block_13_expand",
            "heatmap": np.full(wafer.shape, 0.5, dtype=np.float32),
        }

        with patch.object(main, "generate_gradcam", return_value=result) as generate:
            response = self.client.get(
                f"/inspections/{self.record['inspection_id']}/explain"
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["predicted_class"], "Scratch")
        self.assertEqual(payload["target_layer"], "block_13_expand")
        self.assertEqual(payload["original_map"], wafer.tolist())
        self.assertEqual(payload["heatmap"], result["heatmap"].tolist())
        np.testing.assert_array_equal(generate.call_args.args[0], wafer)

    def test_history_explanation_falls_back_to_existing_input_map(self):
        wafer = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        np.save(self.legacy_dir / "wafer.npy", wafer)
        result = {
            "predicted_class": "Donut",
            "confidence_percent": 99.0,
            "target_layer": "block_13_expand",
            "heatmap": np.zeros(wafer.shape, dtype=np.float32),
        }

        with patch.object(main, "generate_gradcam", return_value=result) as generate:
            response = self.client.get(
                f"/inspections/{self.record['inspection_id']}/explain"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["original_map"], wafer.tolist())
        np.testing.assert_array_equal(generate.call_args.args[0], wafer)

    def test_history_without_source_map_returns_actionable_not_found(self):
        with patch.object(main, "generate_gradcam") as generate:
            response = self.client.get(
                f"/inspections/{self.record['inspection_id']}/explain"
            )

        self.assertEqual(response.status_code, 404)
        self.assertIn("original wafer map is unavailable", response.json()["detail"])
        generate.assert_not_called()

    def test_inspection_saves_original_map_for_later_explanation(self):
        input_path = Path(main.__file__).resolve().parent / "Incoming" / "demo_wafer_0001.npy"
        wafer = np.load(input_path, allow_pickle=False)
        contents = io.BytesIO()
        np.save(contents, wafer, allow_pickle=False)
        contents.seek(0)
        records_dir = self.root / "records"
        reports_dir = self.root / "reports"
        records_dir.mkdir()
        reports_dir.mkdir()
        probabilities = {name: 1 / 9 for name in CLASS_NAMES}

        with patch.object(main, "WAFER_MAPS", self.maps_dir), \
                patch.object(main, "RECORDS", records_dir), \
                patch.object(main, "REPORTS", reports_dir), \
                patch.object(main, "predict_wafer", return_value={
                    "predicted_class": "none",
                    "confidence_percent": 11.11,
                    "probabilities": probabilities,
                }), \
                patch.object(main, "generate_inspection_pdf"), \
                patch.object(main, "save_inspection"):
            response = self.client.post(
                "/inspect",
                files={"file": ("uploaded_wafer.npy", contents, "application/octet-stream")},
            )

        self.assertEqual(response.status_code, 200)
        saved_map = self.maps_dir / f"{response.json()['inspection_id']}.npy"
        self.assertTrue(saved_map.is_file())
        np.testing.assert_array_equal(np.load(saved_map, allow_pickle=False), wafer)


if __name__ == "__main__":
    unittest.main()
