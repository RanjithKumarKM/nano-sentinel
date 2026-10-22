
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"

FILES = {
    "requirements.txt": r'''
tensorflow>=2.16,<2.22
numpy>=1.26,<2.2
opencv-python-headless>=4.9
fastapi>=0.110
uvicorn[standard]>=0.27
python-multipart>=0.0.9
streamlit>=1.32
pillow>=10.0
reportlab>=4.0
requests>=2.31
''',

    "preprocessing.py": r'''
import cv2
import numpy as np


def preprocess_wafer_map(wafer, target_size=96):
    """Convert a categorical 2D wafer map into a model input."""
    wafer = np.asarray(wafer)

    if wafer.ndim != 2 or wafer.size == 0:
        raise ValueError("Expected a non-empty 2D wafer map.")

    if not np.isfinite(wafer).all():
        raise ValueError("Wafer map contains invalid values.")

    # WM-811K categorical values: 0=background, 1=normal, 2=defective.
    if not np.isin(wafer, [0, 1, 2]).all():
        raise ValueError("Wafer map must contain only values 0, 1, and 2.")

    wafer = wafer.astype(np.uint8)
    height, width = wafer.shape

    scale = target_size / max(height, width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))

    resized = cv2.resize(
        wafer,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST
    )

    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized

    rgb = np.repeat(canvas[..., None], 3, axis=-1).astype(np.float32)

    # Must match the preprocessing used to train the saved model.
    return (rgb / 2.0) * 2.0 - 1.0
''',

    "inference.py": r'''
import os
import numpy as np
import tensorflow as tf

from preprocessing import preprocess_wafer_map

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc",
    "Near-full", "Random", "Scratch", "none"
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(
    ROOT, "models", "nano_sentinel_mobilenetv2_v2.keras"
)
MODEL_PATH = os.environ.get("NANO_SENTINEL_MODEL", DEFAULT_MODEL)

_model = None


def get_model():
    """Load the model only once per backend process."""
    global _model

    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}. "
                "Copy the .keras model into the project's models folder."
            )
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    return _model


def predict_wafer(wafer):
    processed = preprocess_wafer_map(wafer)
    batch = np.expand_dims(processed, axis=0)

    model = get_model()
    outputs = np.asarray(model.predict(batch, verbose=0))[0]

    if len(outputs) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected 9 model outputs, received {len(outputs)}."
        )

    # Keras classifier outputs are expected to be probabilities.
    # Validate rather than silently applying a second softmax.
    if (
        not np.isfinite(outputs).all()
        or np.any(outputs < 0)
        or not np.isclose(outputs.sum(), 1.0, atol=1e-3)
    ):
        raise ValueError(
            "Model outputs are not valid probabilities. "
            "Check the saved model's final activation."
        )

    best_index = int(np.argmax(outputs))

    return {
        "predicted_class": CLASS_NAMES[best_index],
        "confidence_percent": round(float(outputs[best_index]) * 100, 2),
        "probabilities": {
            name: round(float(value), 8)
            for name, value in zip(CLASS_NAMES, outputs)
        }
    }
''',

    "main.py": r'''
import io
import os
import json
import uuid
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import predict_wafer, get_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RECORDS = os.path.join(RESULTS, "inspection_records")
os.makedirs(RECORDS, exist_ok=True)

app = FastAPI(
    title="Nano Sentinel API",
    description="AI-assisted wafer map classification",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "project": "NANO SENTINEL",
        "status": "running",
        "model": "MobileNetV2 V2"
    }


@app.get("/health")
def health():
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": True,
            "input_shape": str(model.input_shape),
            "output_shape": str(model.output_shape)
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/inspect")
async def inspect_wafer(file: UploadFile = File(...)):
    """
    Upload a NumPy .npy file containing a 2D categorical wafer map.
    Values must be 0, 1, or 2.
    """
    if not file.filename or not file.filename.lower().endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="Upload a .npy file containing a 2D categorical wafer map."
        )

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum upload size is 10 MB."
            )

        wafer = np.load(io.BytesIO(contents), allow_pickle=False)

        if wafer.ndim != 2 or wafer.size == 0:
            raise ValueError("Expected a non-empty 2D wafer map.")

        prediction = predict_wafer(wafer)

        wafer = np.asarray(wafer)
        die_pixels = int(np.count_nonzero(wafer > 0))
        defective_pixels = int(np.count_nonzero(wafer == 2))
        defect_ratio = (
            round(defective_pixels / die_pixels * 100, 2)
            if die_pixels else 0.0
        )

        # Provisional map statistic, not validated manufacturing severity.
        severity = (
            "HIGH" if defect_ratio >= 5
            else "MODERATE" if defect_ratio >= 1
            else "LOW"
        )

        confidence = prediction["confidence_percent"]
        if confidence < 70:
            status = "MANUAL_REVIEW"
        elif confidence < 90:
            status = "REVIEW_RECOMMENDED"
        else:
            status = "HIGH_CONFIDENCE_PREDICTION"

        # A high confidence score does not guarantee correctness.
        inspection_id = "NS-" + uuid.uuid4().hex[:10].upper()

        record = {
            "inspection_id": inspection_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "type": "uploaded_npy",
                "filename": os.path.basename(file.filename)
            },
            "model": {
                "name": "MobileNetV2",
                "version": "V2",
                **prediction
            },
            "severity": {
                "total_die_pixels": die_pixels,
                "defective_die_pixels": defective_pixels,
                "defect_ratio_percent": defect_ratio,
                "severity_category": severity,
                "note": "Provisional pixel-map heuristic; not validated against manufacturing standards."
            },
            "decision": {
                "inspection_status": status,
                "recommendation": (
                    "Review the wafer map and model output before "
                    "taking corrective action."
                )
            }
        }

        record_path = os.path.join(RECORDS, inspection_id + ".json")
        with open(record_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        record["record_path"] = record_path
        return record

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        await file.close()
''',

    "app.py": r'''
import requests
import numpy as np
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Nano Sentinel",
    page_icon="🔬",
    layout="wide"
)

st.title("NANO SENTINEL")
st.subheader("Intelligent Automated Wafer Inspection")
st.caption("AI-assisted wafer-map classification and inspection records")

with st.sidebar:
    st.header("System status")
    try:
        health = requests.get(API_URL + "/health", timeout=5)
        if health.ok:
            st.success("Backend connected")
        else:
            st.warning("Backend returned an error")
    except requests.RequestException:
        st.error("Backend is not running")
        st.code("uvicorn main:app --reload", language="bash")

st.markdown(
    "Upload a **NumPy `.npy` wafer map** with categorical values "
    "`0` (background), `1` (normal die), and `2` (defective die)."
)

uploaded = st.file_uploader(
    "Choose wafer map",
    type=["npy"]
)

if uploaded is not None:
    try:
        wafer = np.load(uploaded, allow_pickle=False)

        if wafer.ndim != 2 or wafer.size == 0:
            st.error("The file must contain a non-empty 2D array.")
            st.stop()

        st.write("Wafer map shape:", wafer.shape)
        st.write("Unique values:", np.unique(wafer).tolist())

        if not np.isin(wafer, [0, 1, 2]).all():
            st.error("Only categorical values 0, 1, and 2 are supported.")
            st.stop()

        st.image(
            (wafer.astype(np.float32) / 2.0),
            caption="Uploaded categorical wafer map",
            clamp=True,
            width="stretch"
        )

        if st.button("Inspect wafer", type="primary"):
            with st.spinner("Running model inference..."):
                response = requests.post(
                    API_URL + "/inspect",
                    files={
                        "file": (
                            uploaded.name,
                            uploaded.getvalue(),
                            "application/octet-stream"
                        )
                    },
                    timeout=120
                )

            if response.ok:
                result = response.json()
                model = result["model"]
                severity = result["severity"]
                decision = result["decision"]

                st.success("Inspection completed")
                col1, col2, col3 = st.columns(3)
                col1.metric("Predicted class", model["predicted_class"])
                col2.metric(
                    "Confidence",
                    f'{model["confidence_percent"]:.2f}%'
                )
                col3.metric(
                    "Inspection status",
                    decision["inspection_status"].replace("_", " ")
                )

                st.subheader("Class probabilities")
                st.bar_chart(model["probabilities"])

                st.subheader("Map-based statistics")
                st.write({
                    "Total die pixels": severity["total_die_pixels"],
                    "Defective die pixels": severity["defective_die_pixels"],
                    "Defect ratio": (
                        f'{severity["defect_ratio_percent"]:.2f}%'
                    ),
                    "Provisional category": severity["severity_category"]
                })

                st.info(decision["recommendation"])
                st.warning(
                    "Confidence does not guarantee correctness. "
                    "The severity category is provisional. This initial "
                    "backend does not yet generate Grad-CAM or PDF reports."
                )

                st.download_button(
                    "Download JSON inspection record",
                    data=__import__("json").dumps(result, indent=2),
                    file_name=result["inspection_id"] + ".json",
                    mime="application/json"
                )
            else:
                st.error(f"Inspection failed: {response.text}")

    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
''',

    "README.md": r'''
# Nano Sentinel Backend

## Project layout
- `main.py`: FastAPI service
- `inference.py`: saved model loading and prediction
- `preprocessing.py`: wafer-map preprocessing
- `app.py`: Streamlit dashboard
- `requirements.txt`: Python dependencies

## Model file
Place the saved model at:
`../models/nano_sentinel_mobilenetv2_v2.keras`

## Setup (Windows PowerShell)
From the `backend` directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Start the API
```powershell
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs

## Start the dashboard
Open a second terminal, activate the same environment, and run:
```powershell
streamlit run app.py
```
Open http://localhost:8501

## Upload format
The initial API accepts `.npy` files containing 2D arrays with values:
0 = background, 1 = normal die, 2 = defective die.

## Important
Verify preprocessing against the exact saved-model training pipeline before
using deployment predictions. This initial scaffold does not yet implement
Grad-CAM, automatic folder monitoring, or PDF report generation.
'''
}

# Create backend and output directories
BACKEND.mkdir(parents=True, exist_ok=True)

for folder in [
    ROOT / "models",
    ROOT / "results" / "inspection_records",
    ROOT / "results" / "gradcam",
    ROOT / "results" / "batch_inspection",
    ROOT / "reports",
]:
    folder.mkdir(parents=True, exist_ok=True)

# Write all backend files
for filename, content in FILES.items():
    destination = BACKEND / filename
    destination.write_text(content.strip() + "\n", encoding="utf-8")
    print("Created:", destination)

model_path = ROOT / "models" / "nano_sentinel_mobilenetv2_v2.keras"
print("\nModel file present:", model_path.is_file())
if not model_path.is_file():
    print("Copy your saved .keras model into:", model_path)

print("\nBackend scaffold created successfully.")
print("Next: open a terminal in the backend folder and install requirements.")
