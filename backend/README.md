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

## Local inspection assistant
The assistant uses a local Ollama service; it does not send inspection data to a
hosted LLM. Install Ollama, then download the model tag used by default:

```powershell
ollama pull qwen2:7b
ollama serve
```

Keep the Ollama service on this machine. In another PowerShell window, start
FastAPI with the local provider and model (these are also the defaults):

```powershell
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_BASE_URL = "http://127.0.0.1:11434"
$env:LLM_MODEL = "qwen2:7b"
uvicorn main:app --reload
```

`LLM_MODEL` must exactly match an installed Ollama tag. The API accepts only a
loopback Ollama URL and reports a useful error if Ollama or the model is
unavailable; there is no cloud-provider fallback.

## Start the dashboard
Open a second terminal, activate the same environment, and run:
```powershell
streamlit run app.py
```
Open http://localhost:8501

## Upload format
The initial API accepts `.npy` files containing 2D arrays with values:
0 = background, 1 = normal die, 2 = defective die.

## Optional folder watcher
`POST /inspect` accepts one uploaded wafer map; it does not scan directories.
To process queued maps automatically, run this separate polling watcher while
the FastAPI server is running. From the project root:

```powershell
.\backend\.venv\Scripts\python.exe .\incoming\failed\watcher.py
```

The watcher reads `backend\Incoming\*.npy`, posts each file to `/inspect`,
moves successful inputs to `incoming\processed\`, and moves failed inputs to
`incoming\failed\`. Those folders are created when the watcher starts. It
processes files sequentially; it does not run automatically when FastAPI starts.

For a one-time, non-moving import of the queued files, use the importer from
the project root instead:

```powershell
.\backend\.venv\Scripts\python.exe .\backend\process_incoming.py --dry-run
.\backend\.venv\Scripts\python.exe .\backend\process_incoming.py
```

It uses the existing `POST /inspect` prediction and SQLite persistence path,
skips filenames already present in inspection history, and leaves every
original input in `backend\Incoming`. After it finishes, refresh Inspection
History in the frontend. Keep FastAPI running while importing.

## Important
Verify preprocessing against the exact saved-model training pipeline before
using deployment predictions. Model confidence is not necessarily calibrated,
and severity is a provisional pixel-map heuristic rather than a manufacturing
standard. Validate results through the approved engineering workflow.
