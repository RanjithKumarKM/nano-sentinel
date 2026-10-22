import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from inference import predict_wafer, get_model
from gradcam import generate_gradcam
from history import (
    initialize_database,
    save_inspection,
    list_inspections,
    get_inspection,
    delete_inspection,
    delete_all_inspections,
    normalize_inspection_record,
)
from report_generator import generate_inspection_pdf
from assistant import router as assistant_router

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RECORDS = RESULTS / "inspection_records"
REPORTS = ROOT / "reports"
WAFER_MAPS = RESULTS / "wafer_maps"
RECORDS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
WAFER_MAPS.mkdir(parents=True, exist_ok=True)
LEGACY_WAFER_DIRS = (
    Path(__file__).resolve().parent / "Incoming",
    ROOT / "incoming" / "processed",
    ROOT / "incoming" / "failed",
)


def validate_wafer_map(wafer):
    if wafer.ndim != 2 or wafer.size == 0:
        raise ValueError("Expected a non-empty 2D wafer map.")
    if not np.issubdtype(wafer.dtype, np.number) or not np.isfinite(wafer).all():
        raise ValueError("Wafer map must contain finite numeric values.")
    if not np.isin(wafer, [0, 1, 2]).all():
        raise ValueError("Wafer map must contain only values 0, 1, and 2.")
    return wafer


def find_inspection_wafer_map(record):
    inspection_id = record.get("inspection_id")
    if not isinstance(inspection_id, str) or Path(inspection_id).name != inspection_id:
        return None

    saved_map = WAFER_MAPS / f"{inspection_id}.npy"
    if saved_map.is_file():
        return saved_map

    source = record.get("source")
    filename = record.get("filename")
    if not filename and isinstance(source, dict):
        filename = source.get("filename")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        return None

    for directory in LEGACY_WAFER_DIRS:
        candidate = directory / filename
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        if resolved.is_file() and resolved.parent == directory.resolve():
            return resolved
    return None

app = FastAPI(title="Nano Sentinel API",
              description="AI-assisted wafer map classification and inspection workflow",
              version="1.1.0")
initialize_database()
app.include_router(assistant_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
)


@app.get("/")
def home():
    return {"project": "NANO SENTINEL", "status": "running",
            "model": "MobileNetV2 V2", "features": ["inspection", "history", "pdf reports"]}


@app.get("/health")
def health():
    try:
        model = get_model()
        return {"status": "healthy", "model_loaded": True,
                "input_shape": str(model.input_shape), "output_shape": str(model.output_shape)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/inspect")
async def inspect_wafer(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".npy"):
        raise HTTPException(
            status_code=400, detail="Upload a .npy file containing a 2D categorical wafer map.")
    try:
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413, detail="File is too large. Maximum upload size is 10 MB.")
        wafer = np.load(io.BytesIO(contents), allow_pickle=False)
        validate_wafer_map(wafer)
        prediction = predict_wafer(wafer)
        die_pixels = int(np.count_nonzero(wafer > 0))
        defective_pixels = int(np.count_nonzero(wafer == 2))
        defect_ratio = round(defective_pixels / die_pixels *
                             100, 2) if die_pixels else 0.0
        severity = "HIGH" if defect_ratio >= 5 else "MODERATE" if defect_ratio >= 1 else "LOW"
        confidence = prediction["confidence_percent"]
        status = "MANUAL_REVIEW" if confidence < 70 else (
            "REVIEW_RECOMMENDED" if confidence < 90 else "HIGH_CONFIDENCE_PREDICTION"
        )
        inspection_id = "NS-" + uuid.uuid4().hex[:10].upper()
        record = {
            "inspection_id": inspection_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": {"type": "uploaded_npy", "filename": Path(file.filename).name},
            "model": {"name": "MobileNetV2", "version": "V2", **prediction},
            "severity": {
                "total_die_pixels": die_pixels,
                "defective_die_pixels": defective_pixels,
                "defect_ratio_percent": defect_ratio,
                "severity_category": severity,
                "note": "Provisional pixel-map heuristic; not validated against manufacturing standards."
            },
            "decision": {
                "inspection_status": status,
                "recommendation": "Review the wafer map and model output before taking corrective action."
            }
        }
        record = normalize_inspection_record(record)
        np.save(
            WAFER_MAPS / f"{inspection_id}.npy",
            wafer,
            allow_pickle=False,
        )
        record_path = RECORDS / f"{inspection_id}.json"
        report_path = REPORTS / f"{inspection_id}_inspection_report.pdf"
        record["record_path"] = str(record_path)
        try:
            generate_inspection_pdf(record, report_path)
            record["report_path"] = str(report_path)
        except Exception as report_exc:
            record["report_path"] = None
            record["report_generation_error"] = str(report_exc)

        with record_path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        save_inspection(record)
        return record
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        await file.close()


@app.get("/inspections")
def inspection_history():
    return {"inspections": list_inspections()}


@app.get("/inspections/{inspection_id}")
def inspection_details(inspection_id: str):
    record = get_inspection(inspection_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail="Inspection record not found.")
    return record


@app.delete("/inspections/{inspection_id}")
def remove_inspection(inspection_id: str):
    if not delete_inspection(inspection_id):
        raise HTTPException(
            status_code=404, detail="Inspection record not found.")
    return {"message": "Inspection deleted", "inspection_id": inspection_id}


@app.get("/inspections/{inspection_id}/report")
def inspection_report(inspection_id: str):
    record = get_inspection(inspection_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail="Inspection record not found.")
    report_path = record.get("report_path")
    if not report_path or not Path(report_path).is_file():
        raise HTTPException(
            status_code=404, detail="PDF report is not available for this inspection.")
    return FileResponse(str(report_path), media_type="application/pdf",
                        filename=f"{inspection_id}_inspection_report.pdf")


@app.post("/explain")
async def explain_wafer(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="Upload a .npy wafer map."
        )

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum upload size is 10 MB."
            )

        wafer = np.load(io.BytesIO(contents), allow_pickle=False)

        validate_wafer_map(wafer)

        result = generate_gradcam(wafer)

        return {
            "predicted_class": result["predicted_class"],
            "confidence_percent": result["confidence_percent"],
            "target_layer": result["target_layer"],
            "original_map": wafer.astype(int).tolist(),
            "heatmap": result["heatmap"].astype(float).tolist(),
            "explanation_note": (
                "Grad-CAM highlights regions influencing the model prediction. "
                "It does not establish the physical cause of a defect."
            )
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation failed: {exc}"
        )
    finally:
        await file.close()


@app.get("/inspections/{inspection_id}/explain")
def explain_inspection(inspection_id: str):
    record = get_inspection(inspection_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail="Inspection record not found.")

    wafer_path = find_inspection_wafer_map(record)
    if wafer_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "The original wafer map is unavailable for this inspection. "
                "Grad-CAM requires the saved map or its original file in an "
                "incoming or processed folder."
            ),
        )

    try:
        wafer = np.load(wafer_path, allow_pickle=False)
        validate_wafer_map(wafer)
        result = generate_gradcam(wafer)
        return {
            "predicted_class": result["predicted_class"],
            "confidence_percent": result["confidence_percent"],
            "target_layer": result["target_layer"],
            "original_map": wafer.astype(int).tolist(),
            "heatmap": result["heatmap"].astype(float).tolist(),
            "explanation_note": (
                "Grad-CAM highlights regions influencing the model prediction. "
                "It does not establish the physical cause of a defect."
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Grad-CAM generation failed: {exc}",
        ) from exc
