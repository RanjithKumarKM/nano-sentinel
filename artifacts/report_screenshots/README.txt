Nano Sentinel report screenshots

Captured 24 genuine application screenshots at 1440 x 1000 desktop viewport (the existing PDF is previewed in Microsoft Edge).
Source: the existing local React/Vite application, FastAPI API, inspection history, and generated PDF report.
No model inference was submitted and no SQLite inspection records were created, edited, or deleted during capture.
The existing sample .npy is selected locally in the upload screen only; it is not uploaded.

Folders correspond to dashboard, inspection, explainability, history, analytics/reports, and assistant/system screens.
screenshot_manifest.csv records each capture and every feature that was not captured, with the reason.
contact_sheet.png is a thumbnail index of the captured application screenshots.

Not captured:
- Grad-CAM: The selected history record has no stored Grad-CAM result. Generating one through the UI requires a new inspection, which would write to the existing SQLite database; this workflow avoids that write.

Rerun from the project root after starting FastAPI and Vite:
  node .\artifacts\capture_report_screenshots.mjs
Start FastAPI in one PowerShell window if needed:
  cd .\backend; ..\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
Start Vite in another PowerShell window if needed:
  cd .\frontend; npm run dev -- --host 127.0.0.1
Optional URL overrides:
  $env:NANO_SENTINEL_FRONTEND_URL='http://localhost:5173/'
  $env:NANO_SENTINEL_API_URL='http://127.0.0.1:8000'
Playwright must be installed in frontend/node_modules; install its Chromium browser once:
  cd .\frontend; npm install --no-save --no-package-lock playwright; npx playwright install chromium
The existing Microsoft Edge installation is used only to display the generated PDF report.

This archive contains original browser captures; no screenshot values or pixels have been edited.
