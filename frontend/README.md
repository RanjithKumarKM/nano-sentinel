# Nano Sentinel React Frontend

Dark engineering-console frontend for the existing Nano Sentinel FastAPI backend.

## Requirements
- Windows 10/11
- Node.js 20 LTS or newer (includes npm)
- Existing Nano Sentinel backend and model files

## Install
1. Extract this folder into `C:\Users\ranji\Downloads\NanoSentinel\frontend` (or another location).
2. Open PowerShell in that `frontend` folder.
3. Run:
   ```powershell
   Copy-Item .env.example .env
   npm install
   npm run dev
   ```
4. Keep the Vite terminal open and visit `http://localhost:5173`.
5. In a second PowerShell terminal, start your existing backend:
   ```powershell
   cd C:\Users\ranji\Downloads\NanoSentinel\backend
   .\.venv\Scripts\Activate.ps1
   python -m uvicorn main:app --reload
   ```
   If your virtual environment is in the project root, use `..\.venv\Scripts\Activate.ps1` instead.

## Backend requirements
The existing API should expose:
- `GET /health`
- `POST /inspect` (multipart form field named `file`, a `.npy` file)
- `POST /explain` (multipart form field named `file`, a `.npy` file)
- `GET /inspections`
- `DELETE /inspections/{inspection_id}` (removes one inspection history record)
- `GET /inspections/{inspection_id}/report`
- `POST /assistant/ask` with JSON `{ "question": "..." }`

The frontend integrates with the existing API and does not replace the inference, preprocessing, Grad-CAM, SQLite, or report-generation architecture.

## CORS
Your FastAPI app must allow the Vite origin `http://localhost:5173` and optionally `http://127.0.0.1:5173`. Add these to the existing `CORSMiddleware` allow_origins list. Keep your current Streamlit origins if you still use that app.

Example:
```python
allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]
```
Restart FastAPI after changing CORS.

## Important compatibility notes
- The UI is designed around the response fields described in the project integration notes. If `/inspections` wraps records in a different response shape, `src/api.js` already handles common `items` and `inspections` wrappers.
- The assistant page expects `/assistant/ask` to accept `{ "question": "..." }` and return `answer`, `response`, `message`, or `reply`. Adjust only `askAssistant` in `src/api.js` if your assistant uses different names.
- Grad-CAM endpoint must return `original_map`, `heatmap`, `predicted_class`, `target_layer`, and optionally `explanation_note`.
- This frontend visualizes stored predictions and does not claim that model predictions are ground truth or that Grad-CAM proves physical defect causes.
