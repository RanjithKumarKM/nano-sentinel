import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException

import assistant
import history
import inference
from inference import CLASS_NAMES


def probabilities():
    return {name: (0.6 if name == "Scratch" else 0.05) for name in CLASS_NAMES}


class InspectionNormalizationTests(unittest.TestCase):
    def test_nested_record_exposes_compatible_aliases(self):
        record = history.normalize_inspection_record({
            "inspection_id": "NS-ABC123",
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "source": {"filename": "wafer.npy"},
            "model": {
                "predicted_class": "Scratch",
                "confidence_percent": 60,
                "probabilities": probabilities(),
            },
            "severity": {"severity_category": "LOW"},
            "decision": {"inspection_status": "MANUAL_REVIEW"},
        })

        self.assertEqual(record["predicted_class"], "Scratch")
        self.assertEqual(record["model"]["predicted_class"], "Scratch")
        self.assertEqual(record["confidence_percent"], 60)
        self.assertEqual(record["status"], "MANUAL_REVIEW")
        self.assertEqual(record["filename"], "wafer.npy")
        self.assertEqual(record["timestamp"], record["timestamp_utc"])

    def test_flat_database_row_exposes_nested_sections(self):
        record = history.normalize_inspection_record({
            "inspection_id": "NS-ABC123",
            "filename": "wafer.npy",
            "predicted_class": "Scratch",
            "confidence_percent": 60,
            "probabilities": probabilities(),
            "severity_category": "LOW",
            "inspection_status": "MANUAL_REVIEW",
        })

        self.assertEqual(record["model"]["predicted_class"], "Scratch")
        self.assertEqual(record["source"]["filename"], "wafer.npy")
        self.assertEqual(record["severity"]["severity_category"], "LOW")
        self.assertEqual(record["decision"]["inspection_status"], "MANUAL_REVIEW")

    def test_save_requires_actual_prediction_fields(self):
        with self.assertRaisesRegex(ValueError, "class or confidence"):
            history.save_inspection({"model": {}, "severity": {}, "decision": {}})

    def test_history_returns_more_than_default_page_size(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(history, "DB_PATH", Path(directory) / "history.db"):
                history.initialize_database()
                for index in range(101):
                    history.save_inspection({
                        "inspection_id": f"NS-TEST-{index:03d}",
                        "timestamp_utc": f"2026-01-{index // 24 + 1:02d}T{index % 24:02d}:00:00+00:00",
                        "source": {"filename": f"wafer-{index:03d}.npy"},
                        "model": {
                            "predicted_class": "none",
                            "confidence_percent": 99,
                            "probabilities": probabilities(),
                        },
                        "severity": {"severity_category": "LOW"},
                        "decision": {"inspection_status": "HIGH_CONFIDENCE_PREDICTION"},
                    })

                self.assertEqual(len(history.list_inspections()), 101)
                self.assertEqual(len(history.list_inspections(limit=10)), 10)

    def test_prediction_uses_ninth_output_for_none_and_probability_confidence(self):
        outputs = [0.001] * 8 + [0.992]
        model = Mock()
        model.predict.return_value = [outputs]
        wafer = [[0, 1], [2, 1]]

        with patch.object(inference, "get_model", return_value=model):
            prediction = inference.predict_wafer(wafer)

        self.assertEqual(prediction["predicted_class"], "none")
        self.assertEqual(prediction["confidence_percent"], 99.2)
        self.assertEqual(len(prediction["probabilities"]), 9)
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0)


class LocalAssistantTests(unittest.TestCase):
    def test_probability_analysis_ranks_stored_distribution(self):
        analysis = assistant._probability_analysis({
            "model": {"probabilities": probabilities()}
        })

        self.assertEqual(analysis["top_classes"][0]["class"], "Scratch")
        self.assertEqual(analysis["top_classes"][0]["probability_percent"], 60)
        self.assertEqual(analysis["top_two_margin_percentage_points"], 55)

    def test_probability_analysis_rejects_invalid_distribution(self):
        invalid = probabilities()
        invalid["Scratch"] = 2
        self.assertIsNone(assistant._probability_analysis({
            "model": {"probabilities": invalid}
        }))

    def test_assistant_uses_local_ollama_model_and_context(self):
        request = assistant.QuestionRequest(
            question="Explain this inspection",
            conversation_history=[{"role": "user", "content": "Earlier question"}],
        )
        context = {"inspection_count": 1, "recent_inspections": []}
        response = Mock(ok=True)
        response.json.return_value = {"message": {"content": "Local answer"}}

        with patch.object(assistant, "build_assistant_context", return_value=context), \
                patch.object(assistant.requests, "post", return_value=response) as post:
            result = assistant.ask_assistant(request)

        self.assertEqual(result["answer"], "Local answer")
        self.assertEqual(result["model"], "qwen2:7b")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://127.0.0.1:11434/api/chat")
        payload = kwargs["json"]
        self.assertEqual(payload["model"], "qwen2:7b")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn(json.dumps(context, ensure_ascii=False, separators=(",", ":")),
                      payload["messages"][0]["content"])
        self.assertIn("LOW/MODERATE/HIGH category assigned from the pixel-map",
                      payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][-2]["content"], "Earlier question")
        self.assertEqual(payload["messages"][-1]["content"], "Explain this inspection")

    def test_assistant_rejects_non_loopback_provider_url(self):
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://example.com:11434"}):
            with self.assertRaises(HTTPException) as raised:
                assistant._ollama_chat_url()
        self.assertEqual(raised.exception.status_code, 503)

    def test_unknown_inspection_is_reported_without_model_request(self):
        with patch.object(assistant, "build_assistant_context", return_value=None), \
                patch.object(assistant.requests, "post") as post:
            result = assistant.ask_assistant(
                assistant.QuestionRequest(question="Explain NS-MISSING")
            )

        self.assertIn("NS-MISSING", result["answer"])
        post.assert_not_called()

    def test_missing_ollama_model_has_actionable_error(self):
        response = Mock(ok=False, status_code=404, text="model not found")
        response.json.return_value = {"error": "model not found"}
        with patch.object(assistant, "build_assistant_context", return_value={}), \
                patch.object(assistant.requests, "post", return_value=response):
            with self.assertRaises(HTTPException) as raised:
                assistant.ask_assistant(
                    assistant.QuestionRequest(question="Explain model confidence")
                )
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("qwen2:7b", raised.exception.detail)

    def test_ollama_timeout_is_reported(self):
        with patch.object(assistant, "build_assistant_context", return_value={}), \
                patch.object(
                    assistant.requests, "post", side_effect=assistant.requests.Timeout
                ):
            with self.assertRaises(HTTPException) as raised:
                assistant.ask_assistant(
                    assistant.QuestionRequest(question="Explain model confidence")
                )
        self.assertEqual(raised.exception.status_code, 504)

    def test_database_delete_removes_only_requested_record(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(history, "DB_PATH", Path(directory) / "history.db"):
                history.initialize_database()
                for inspection_id in ("NS-FIRST", "NS-SECOND"):
                    history.save_inspection({
                        "inspection_id": inspection_id,
                        "timestamp_utc": "2026-01-01T00:00:00+00:00",
                        "source": {"filename": f"{inspection_id}.npy"},
                        "model": {
                            "predicted_class": "Scratch",
                            "confidence_percent": 60,
                            "probabilities": probabilities(),
                        },
                        "severity": {"severity_category": "LOW"},
                        "decision": {"inspection_status": "MANUAL_REVIEW"},
                    })

                self.assertTrue(history.delete_inspection("NS-FIRST"))
                self.assertIsNone(history.get_inspection("NS-FIRST"))
                self.assertEqual(
                    [record["inspection_id"] for record in history.list_inspections()],
                    ["NS-SECOND"],
                )
                self.assertFalse(history.delete_inspection("NS-FIRST"))


if __name__ == "__main__":
    unittest.main()
