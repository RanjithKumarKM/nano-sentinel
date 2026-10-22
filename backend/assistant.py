import json
import logging
import math
import os
import re
from typing import Literal
from urllib.parse import urlsplit

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from history import (
    count_inspections,
    count_inspections_needing_review,
    get_inspection,
    list_inspections,
)
from inference import CLASS_NAMES

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/assistant", tags=["Engineer Assistant"])
QUESTION_MAX_LENGTH = 1000
CONVERSATION_TURNS = 8
CONVERSATION_TEXT_LIMIT = 1200
OLLAMA_TIMEOUT_SECONDS = 90
DEFAULT_MODEL = "qwen2:7b"


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=CONVERSATION_TEXT_LIMIT)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=QUESTION_MAX_LENGTH)
    conversation_history: list[ConversationTurn] = Field(
        default_factory=list, max_items=CONVERSATION_TURNS
    )


def _probability_analysis(record):
    model = record.get("model")
    model = model if isinstance(model, dict) else {}
    probabilities = model.get("probabilities", record.get("probabilities"))
    if not isinstance(probabilities, dict) or set(probabilities) != set(CLASS_NAMES):
        return None

    try:
        values = {name: float(probabilities[name]) for name in CLASS_NAMES}
    except (TypeError, ValueError):
        return None
    if (
        any(not math.isfinite(value) or value < 0 for value in values.values())
        or not math.isclose(sum(values.values()), 1.0, abs_tol=1e-3)
    ):
        return None

    ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
    return {
        "probabilities_percent": {
            name: round(value * 100, 2) for name, value in values.items()
        },
        "top_classes": [
            {"class": name, "probability_percent": round(value * 100, 2)}
            for name, value in ranked[:3]
        ],
        "top_two_margin_percentage_points": (
            round((ranked[0][1] - ranked[1][1]) * 100, 2)
            if len(ranked) > 1 else None
        ),
    }


def build_inspection_context(record):
    """Keep only stored facts and calculations derived from a stored probability vector."""
    model = record.get("model")
    model = model if isinstance(model, dict) else {}
    source = record.get("source")
    source = source if isinstance(source, dict) else {}
    severity = record.get("severity")
    severity = severity if isinstance(severity, dict) else {}
    decision = record.get("decision")
    decision = decision if isinstance(decision, dict) else {}

    context = {}
    fields = {
        "inspection_id": record.get("inspection_id"),
        "timestamp_utc": record.get("timestamp_utc"),
        "filename": source.get("filename", record.get("filename")),
        "predicted_class": model.get(
            "predicted_class", record.get("predicted_class")
        ),
        "confidence_percent": model.get(
            "confidence_percent", record.get("confidence_percent")
        ),
        "severity_category": severity.get(
            "severity_category", record.get("severity_category")
        ),
        "defect_ratio_percent": severity.get(
            "defect_ratio_percent", record.get("defect_ratio_percent")
        ),
        "inspection_status": decision.get(
            "inspection_status",
            record.get("inspection_status", record.get("status")),
        ),
        "recommendation": decision.get(
            "recommendation", record.get("recommendation")
        ),
    }
    context.update({key: value for key, value in fields.items() if value is not None})

    analysis = _probability_analysis(record)
    if analysis is not None:
        context.update(analysis)

    gradcam = record.get("gradcam")
    if isinstance(gradcam, dict) and gradcam:
        context["gradcam"] = gradcam
    return context


def build_assistant_context(inspection_id=None):
    context = {
        "inspection_count": count_inspections(),
        "manual_review_count": count_inspections_needing_review(),
    }
    if inspection_id:
        record = get_inspection(inspection_id)
        if record is None:
            return None
        context["selected_inspection"] = build_inspection_context(record)
        return context

    recent = list_inspections(limit=10)
    context["recent_inspections"] = [
        build_inspection_context(record) for record in recent
    ]
    return context


def _ollama_chat_url():
    provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
    if provider != "ollama":
        raise HTTPException(
            status_code=503,
            detail="Only the local Ollama provider is supported. Set LLM_PROVIDER=ollama.",
        )

    base_url = os.environ.get(
        "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
    ).strip().rstrip("/")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise HTTPException(
            status_code=503,
            detail="Ollama must use a local loopback URL, such as http://127.0.0.1:11434.",
        )
    return f"{base_url}/api/chat"


def _conversation_messages(turns):
    messages = []
    total_characters = 0
    for turn in turns:
        content = turn.content.strip()
        total_characters += len(content)
        if not content or total_characters > 5000:
            raise HTTPException(
                status_code=422,
                detail="Conversation history must contain at most 5,000 characters.",
            )
        messages.append({"role": turn.role, "content": content})
    return messages


@router.post("/ask")
def ask_assistant(request: QuestionRequest):
    question = request.question.strip()
    if len(question) < 2:
        raise HTTPException(status_code=422, detail="Enter a question with at least 2 characters.")

    id_match = re.search(r"\bNS-[A-Za-z0-9-]+\b", question, re.IGNORECASE)
    inspection_id = id_match.group(0).upper() if id_match else None
    context = build_assistant_context(inspection_id)
    if context is None:
        return {"answer": f"I couldn't find inspection {inspection_id} in the database."}

    model = os.environ.get("LLM_MODEL", DEFAULT_MODEL).strip()
    if not model or len(model) > 128:
        raise HTTPException(
            status_code=503,
            detail="Set LLM_MODEL to the exact tag of a locally installed Ollama model.",
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are Nano Sentinel's local wafer-inspection assistant. "
                "Answer clearly and concisely. The JSON context below is the only "
                "source of inspection-specific facts. Use only values present there; "
                "do not invent classes, percentages, inspections, severity scores, or "
                "recommendations. The probability percentages, top-three ranking, and "
                "top-two margin are computed from stored model output; quote them rather "
                "than estimating them. If a field is absent, say it is unavailable. "
                "Distinguish stored facts from general explanations. Class probabilities "
                "do not establish a physical cause or prove correctness; model confidence "
                "is not necessarily calibrated. Severity is only this application's "
                "provisional LOW/MODERATE/HIGH category assigned from the pixel-map "
                "defect ratio. The code thresholds are LOW below 1%, MODERATE from 1% "
                "to below 5%, and HIGH at 5% or above; these are not validated "
                "manufacturing standards. When explaining severity, use that definition "
                "and do not call it concern, risk, impact, criticality, functionality, "
                "quality, or urgency, or infer physical consequences. Only describe "
                "Grad-CAM if Grad-CAM data is present "
                "in the context; otherwise explain that the saved record does not include "
                "a heatmap. Never claim to have seen a wafer image."
                "\n\nInspection context (JSON):\n"
                + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            ),
        },
        *_conversation_messages(request.conversation_history),
        {"role": "user", "content": question},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 350},
    }

    try:
        response = requests.post(
            _ollama_chat_url(),
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        logger.warning("Local Ollama request timed out for model %s", model)
        raise HTTPException(
            status_code=504,
            detail="The local Ollama model did not respond in time. Check Ollama and try again.",
        ) from exc
    except requests.RequestException as exc:
        logger.warning("Local Ollama request failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "The local Ollama service is unavailable. Start Ollama "
                "(ollama serve) and confirm the configured model is installed."
            ),
        ) from exc

    if not response.ok:
        try:
            detail = response.json().get("error", response.text)
        except ValueError:
            detail = response.text
        logger.warning("Ollama returned HTTP %s: %s", response.status_code, detail)
        if response.status_code == 404:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Ollama does not have model '{model}'. Install that exact model "
                    "tag locally with `ollama pull <model-tag>` or set LLM_MODEL to "
                    "an installed tag."
                ),
            )
        raise HTTPException(
            status_code=502,
            detail=f"The local Ollama service returned an error: {detail}",
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="The local Ollama service returned an invalid response.",
        ) from exc
    answer = data.get("message", {}).get("content", "")
    if not isinstance(answer, str) or not answer.strip():
        raise HTTPException(
            status_code=502,
            detail="The local Ollama model returned an empty answer. Please try again.",
        )
    return {"answer": answer.strip(), "provider": "ollama", "model": model}
