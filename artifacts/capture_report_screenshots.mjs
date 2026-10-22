import { createRequire } from "node:module";
import path from "node:path";
import os from "node:os";
import fs from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { inflateSync } from "node:zlib";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectDir = path.resolve(scriptDir, "..");
const outputDir = path.join(scriptDir, "report_screenshots");
const require = createRequire(path.join(projectDir, "frontend", "package.json"));
const { chromium } = require("playwright");
const frontendUrl = process.env.NANO_SENTINEL_FRONTEND_URL || "http://127.0.0.1:5173/";
const apiUrl = process.env.NANO_SENTINEL_API_URL || "http://127.0.0.1:8000";
const viewport = { width: 1440, height: 1000 };
const captures = [];
const errors = [];
const failedResources = [];

function csvEscape(value) {
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
}

async function hasRenderedPixels(imagePath, minimumRatio = 0.01) {
  const png = await fs.readFile(imagePath);
  if (png.subarray(0, 8).toString("hex") !== "89504e470d0a1a0a") {
    throw new Error(`Browser screenshot is not a PNG: ${imagePath}`);
  }
  let width;
  let height;
  let bitDepth;
  let colorType;
  const compressed = [];
  for (let offset = 8; offset < png.length;) {
    const length = png.readUInt32BE(offset);
    const type = png.toString("ascii", offset + 4, offset + 8);
    const data = png.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
    } else if (type === "IDAT") {
      compressed.push(data);
    } else if (type === "IEND") {
      break;
    }
    offset += length + 12;
  }
  if (bitDepth !== 8 || ![2, 6].includes(colorType)) {
    throw new Error(`Unsupported screenshot PNG format (bit depth ${bitDepth}, color type ${colorType}).`);
  }
  const bytesPerPixel = colorType === 6 ? 4 : 3;
  const stride = width * bytesPerPixel;
  const raw = inflateSync(Buffer.concat(compressed));
  let previous = Buffer.alloc(stride);
  let darkContentPixels = 0;
  let paperPixels = 0;
  for (let y = 0, offset = 0; y < height; y++) {
    const filter = raw[offset++];
    const row = Buffer.from(raw.subarray(offset, offset + stride));
    offset += stride;
    for (let x = 0; x < stride; x++) {
      const left = x >= bytesPerPixel ? row[x - bytesPerPixel] : 0;
      const up = previous[x];
      const upperLeft = x >= bytesPerPixel ? previous[x - bytesPerPixel] : 0;
      if (filter === 1) row[x] = (row[x] + left) & 255;
      else if (filter === 2) row[x] = (row[x] + up) & 255;
      else if (filter === 3) row[x] = (row[x] + Math.floor((left + up) / 2)) & 255;
      else if (filter === 4) {
        const base = left + up - upperLeft;
        const leftDistance = Math.abs(base - left);
        const upDistance = Math.abs(base - up);
        const upperLeftDistance = Math.abs(base - upperLeft);
        const predictor = leftDistance <= upDistance && leftDistance <= upperLeftDistance
          ? left : upDistance <= upperLeftDistance ? up : upperLeft;
        row[x] = (row[x] + predictor) & 255;
      } else if (filter !== 0) {
        throw new Error(`Unsupported PNG row filter ${filter}.`);
      }
    }
    const channels = colorType === 6 ? 4 : 3;
    if (y > height * 0.08 && y < height * 0.9) {
      for (let x = Math.floor(width * 0.2); x < Math.floor(width * 0.8); x++) {
      const pixel = x * bytesPerPixel;
      const alpha = channels === 4 ? row[pixel + 3] : 255;
        if (alpha > 0 && row[pixel] < 140 && row[pixel + 1] < 140 && row[pixel + 2] < 140) darkContentPixels++;
      if (alpha > 0 && row[pixel] > 240 && row[pixel + 1] > 240 && row[pixel + 2] > 240) paperPixels++;
      }
    }
    previous = row;
  }
  return darkContentPixels / (width * height) >= minimumRatio && paperPixels / (width * height) >= 0.05;
}

async function saveCapture(page, filename, feature, screen, notes = "", options = {}) {
  const target = path.join(outputDir, filename);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await page.screenshot({ path: target, animations: "disabled", ...options });
  captures.push({ filename: filename.replaceAll(path.sep, "/"), feature, screen, status: "captured", notes });
}

async function saveElement(page, selector, filename, feature, screen, notes = "") {
  const element = page.locator(selector).first();
  await element.waitFor({ state: "visible", timeout: 15000 });
  const target = path.join(outputDir, filename);
  await fs.mkdir(path.dirname(target), { recursive: true });
  await element.screenshot({ path: path.join(outputDir, filename), animations: "disabled" });
  captures.push({ filename: filename.replaceAll(path.sep, "/"), feature, screen, status: "captured", notes });
}

async function openWorkspace(page, name) {
  const heading = {
    "Overview": "Operations overview",
    "New inspection": "Wafer inspection",
    "Inspection history": "Inspection history",
    "Analytics": "Inspection analytics",
    "AI assistant": "Nano Sentinel Assistant",
    "System health": "System health"
  }[name];
  if (!heading) throw new Error(`Unknown workspace screen: ${name}`);
  await page.locator(".sidebar .nav-item").filter({ hasText: name }).click();
  await page.getByRole("heading", { name: heading, exact: true }).waitFor({ state: "visible", timeout: 15000 });
}

async function createContactSheet(browser) {
  const tiles = captures.filter(item => item.status === "captured").map(({ filename, feature }) => {
    const imagePath = pathToFileURL(path.join(outputDir, filename)).href;
    return `<figure><img src="${imagePath}" alt=""><figcaption>${filename}<small>${feature}</small></figcaption></figure>`;
  }).join("");
  const html = `<!doctype html><html><head><meta charset="utf-8"><style>
    *{box-sizing:border-box}body{margin:0;padding:24px;background:#0a0f18;color:#e6edf7;font:13px Segoe UI,Arial,sans-serif}
    h1{margin:0 0 8px;font-size:22px}p{margin:0 0 20px;color:#9aa9bf}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}
    figure{min-width:0;margin:0;padding:8px;border:1px solid #29384b;border-radius:8px;background:#111a27}
    img{display:block;width:100%;height:150px;object-fit:contain;background:#080c13}
    figcaption{padding:8px 2px 1px;overflow-wrap:anywhere;font-weight:700}small{display:block;margin-top:4px;color:#91a3bc;font-weight:400}
  </style></head><body><h1>Nano Sentinel · Report Screenshot Contact Sheet</h1>
  <p>${captures.filter(item => item.status === "captured").length} genuine application screenshots · ${new Date().toLocaleString()}</p><div class="grid">${tiles}</div></body></html>`;
  const htmlPath = path.join(os.tmpdir(), `nano-sentinel-contact-${process.pid}.html`);
  await fs.writeFile(htmlPath, html, "utf8");
  try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 1 });
    await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "load" });
    await page.waitForFunction(() => [...document.querySelectorAll("figure img")].every(image => image.complete && image.naturalWidth > 0), { timeout: 30000 });
    await page.screenshot({ path: path.join(outputDir, "contact_sheet.png"), fullPage: true, animations: "disabled" });
    await page.close();
  } finally {
    await fs.rm(htmlPath, { force: true });
  }
}

async function main() {
  await fs.mkdir(outputDir, { recursive: true });
  const response = await fetch(`${apiUrl}/health`);
  if (!response.ok) throw new Error(`Backend health returned HTTP ${response.status}. Start FastAPI and rerun.`);
  const health = await response.json();
  if (health.status !== "healthy" || health.model_loaded !== true) {
    throw new Error(`Backend/model is not ready: ${JSON.stringify(health)}`);
  }

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport, deviceScaleFactor: 1, acceptDownloads: true });
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => {
    if (message.type() === "error") errors.push(`${message.location().url || "browser"}: ${message.text()}`);
  });
  page.on("response", response => {
    if (response.status() >= 400) failedResources.push(`${response.status()} ${response.url()}`);
  });

  try {
    const appResponse = await page.goto(frontendUrl, { waitUntil: "networkidle", timeout: 45000 });
    if (!appResponse?.ok()) throw new Error(`Frontend returned HTTP ${appResponse?.status() ?? "no response"}.`);
    await page.getByRole("heading", { name: "Operations overview" }).waitFor({ state: "visible", timeout: 30000 });
    await page.getByText("Total inspections", { exact: true }).waitFor({ state: "visible" });
    await page.locator(".metric-grid").first().waitFor({ state: "visible" });
    await page.locator("tbody tr").first().waitFor({ state: "visible", timeout: 30000 });

    await saveCapture(page, "01_dashboard/01_dashboard_main.png", "Dashboard", "Overview · /", "Live dashboard with records fetched from the local API.", { fullPage: true });
    await saveElement(page, ".metric-grid", "01_dashboard/02_dashboard_system_metrics.png", "Dashboard", "Overview · /", "Shows the actual API status and saved-record counters.");
    await saveElement(page, ".overview-charts", "01_dashboard/03_dashboard_activity_and_classes.png", "Dashboard", "Overview · /", "Recorded inspection activity and predicted-class distribution.");
    await saveElement(page, ".page-content > .panel", "01_dashboard/04_dashboard_recent_inspections.png", "Dashboard", "Overview · /", "Recent persisted inspection rows.");

    await openWorkspace(page, "New inspection");
    await saveCapture(page, "02_inspection/05_inspection_upload.png", "Inspection", "New inspection · /", "Actual upload UI; accepted format is a 2D .npy array with values 0, 1, and 2.", { fullPage: true });
    const sample = path.join(projectDir, "backend", "Incoming", "demo_wafer_0001.npy");
    await fs.access(sample);
    await page.locator('input[type="file"][accept=".npy"]').setInputFiles(sample);
    await page.locator(".upload-zone").getByText("demo_wafer_0001.npy", { exact: true }).waitFor({ state: "visible" });
    await saveCapture(page, "02_inspection/06_existing_wafer_map_selected.png", "Inspection", "New inspection · /", "An existing input file is selected locally; it is not submitted, and no inspection record is created.", { fullPage: true });

    await openWorkspace(page, "Inspection history");
    const rows = page.locator("tbody tr");
    const recordCount = await rows.count();
    if (recordCount < 1) throw new Error("Inspection history is empty; no existing result is available to capture.");
    await saveCapture(page, "04_history/12_inspection_history.png", "Inspection history", "Inspection history · /", `${recordCount} rows were rendered by the application.`, { fullPage: true });
    await page.locator("select").selectOption("Review");
    await page.locator("tbody tr").first().waitFor({ state: "visible" });
    await saveCapture(page, "04_history/13_history_review_filter.png", "Inspection history", "Inspection history · Review filter", "Uses the built-in review-status filter.");
    await page.locator("select").selectOption("All");
    const search = page.getByPlaceholder("Search class, filename, ID…");
    await search.fill("demo_wafer_0001.npy");
    await page.getByText("1 records", { exact: true }).waitFor({ state: "visible" });
    await saveCapture(page, "04_history/14_history_search.png", "Inspection history", "Inspection history · search", "Filters the persisted history using the built-in filename search.");
    await search.fill("");
    await page.getByText(`${recordCount} records`, { exact: true }).waitFor({ state: "visible" });

    const targetRow = page.locator("tbody tr").first();
    await targetRow.scrollIntoViewIfNeeded();
    await targetRow.click({ button: "right" });
    const menu = page.getByRole("menu", { name: /Actions for inspection/ });
    await menu.waitFor({ state: "visible" });
    await saveCapture(page, "04_history/15_inspection_context_menu.png", "Inspection history", "Inspection history · row context menu", "Right-clicked a real history row; both implemented actions are visible.");
    await page.keyboard.press("Escape");
    await menu.waitFor({ state: "hidden" });
    await targetRow.click({ button: "right" });
    await page.getByRole("menuitem", { name: "View Details" }).click();
    await page.getByRole("heading", { name: "Wafer inspection" }).waitFor({ state: "visible" });
    await page.getByText("PREDICTED CLASS", { exact: true }).waitFor({ state: "visible" });
    await saveCapture(page, "02_inspection/07_existing_inspection_prediction.png", "Inspection", "Existing inspection detail · /", "Prediction shown is an existing persisted model result; no new inference was submitted.", { fullPage: true });
    await saveElement(page, ".result-banner", "02_inspection/08_prediction_confidence.png", "Inspection", "Existing inspection detail · /", "Actual class and confidence from the selected database record.");
    await saveElement(page, ".result-stats", "03_explainability/09_severity_status_recommendation.png", "Explainability and review", "Existing inspection detail · /", "Severity, status, and available inspection metadata from the persisted result.");
    const confidenceAnalysis = page.locator(".confidence-analysis");
    if (await confidenceAnalysis.count()) {
      await saveElement(page, ".confidence-analysis", "02_inspection/10_confidence_analysis.png", "Inspection", "Existing inspection detail · /", "Confidence analysis rendered from stored model probabilities.");
    } else {
      captures.push({ filename: "02_inspection/10_confidence_analysis.png", feature: "Inspection", screen: "Existing inspection detail · /", status: "not captured", notes: "No confidence-analysis panel was available for the selected record." });
    }
    const probabilityPanel = page.locator(".results-panel .panel").filter({ has: page.getByRole("heading", { name: "Class probabilities" }) });
    if (await probabilityPanel.count()) {
      const target = probabilityPanel.first();
      await target.scrollIntoViewIfNeeded();
      await target.screenshot({ path: path.join(outputDir, "02_inspection", "11_class_probabilities.png"), animations: "disabled" });
      captures.push({ filename: "02_inspection/11_class_probabilities.png", feature: "Inspection", screen: "Existing inspection detail · /", status: "captured", notes: "Actual class probability distribution from the persisted model result." });
    } else {
      captures.push({ filename: "02_inspection/11_class_probabilities.png", feature: "Inspection", screen: "Existing inspection detail · /", status: "not captured", notes: "The selected record did not include a valid probability distribution." });
    }
    const gradcam = page.locator(".gradcam-grid");
    if (await gradcam.count()) {
      await gradcam.scrollIntoViewIfNeeded();
      await gradcam.screenshot({ path: path.join(outputDir, "03_explainability", "16_gradcam_explanation.png"), animations: "disabled" });
      captures.push({ filename: "03_explainability/16_gradcam_explanation.png", feature: "Grad-CAM", screen: "Existing inspection detail · /", status: "captured", notes: "Actual Grad-CAM output rendered by the application." });
    } else {
      captures.push({ filename: "03_explainability/16_gradcam_explanation.png", feature: "Grad-CAM", screen: "Existing inspection detail · /", status: "not captured", notes: "The selected history record has no stored Grad-CAM result. Generating one through the UI requires a new inspection, which would write to the existing SQLite database; this workflow avoids that write." });
    }

    await openWorkspace(page, "Overview");
    await openWorkspace(page, "Analytics");
    await saveCapture(page, "05_reports_analytics/17_analytics_dashboard.png", "Analytics", "Analytics · /", "Descriptive metrics and charts from records actually saved in SQLite.", { fullPage: true });
    await saveElement(page, ".content-grid", "05_reports_analytics/18_analytics_class_distribution.png", "Analytics", "Analytics · /", "Class distribution and date-based activity charts.");
    const analyticsCharts = page.locator(".content-grid .panel");
    if (await analyticsCharts.count() > 1) {
      await analyticsCharts.nth(1).screenshot({ path: path.join(outputDir, "05_reports_analytics", "19_analytics_activity_trend.png"), animations: "disabled" });
      captures.push({ filename: "05_reports_analytics/19_analytics_activity_trend.png", feature: "Analytics", screen: "Analytics · /", status: "captured", notes: "Activity trend from saved timestamps." });
    }

    await openWorkspace(page, "System health");
    await page.getByText("BACKEND CONNECTIVITY", { exact: true }).waitFor({ state: "visible" });
    await saveCapture(page, "06_assistant_system/20_system_health.png", "System health", "System health · /", `Health response: ${health.status}; model loaded: ${health.model_loaded}.`, { fullPage: true });
    await saveElement(page, ".health-grid", "06_assistant_system/21_integrations_status.png", "System health", "System health · /", "Shows API health and the configured application integration endpoints.");
    await saveElement(page, ".json-view", "06_assistant_system/22_backend_health_response.png", "System health", "System health · /", `Live /health JSON: ${JSON.stringify(health)}.`);

    await openWorkspace(page, "AI assistant");
    await page.getByText("Hello. I’m Nano Sentinel Assistant.", { exact: false }).waitFor({ state: "visible" });
    await saveCapture(page, "06_assistant_system/23_assistant_workspace.png", "AI assistant", "AI assistant · /", "Genuine assistant interface and local-model configuration; no prompt has been sent yet.", { fullPage: true });
    const ollamaResponse = await fetch("http://127.0.0.1:11434/api/tags").catch(() => null);
    if (ollamaResponse?.ok) {
      const input = page.getByRole("textbox", { name: "Ask the local inspection assistant" });
      await input.fill("Briefly explain what the recorded inspection history screen contains.");
      await page.getByRole("button", { name: "Send" }).click();
      try {
        await page.locator(".chat-row.assistant").last().locator(".chat-bubble p").waitFor({ state: "visible", timeout: 120000 });
        const answer = (await page.locator(".chat-row.assistant").last().locator(".chat-bubble p").innerText()).trim();
        if (answer && !answer.startsWith("Assistant request failed:")) {
          await saveCapture(page, "06_assistant_system/24_assistant_local_response.png", "AI assistant", "AI assistant · /", "Live response from the configured local assistant. No inspection data or history was changed.", { fullPage: true });
        } else {
          captures.push({ filename: "06_assistant_system/24_assistant_local_response.png", feature: "AI assistant", screen: "AI assistant · /", status: "not captured", notes: answer || "The assistant returned no response." });
        }
      } catch (error) {
        captures.push({ filename: "06_assistant_system/24_assistant_local_response.png", feature: "AI assistant", screen: "AI assistant · /", status: "not captured", notes: `No genuine assistant response arrived before timeout: ${error.message}` });
      }
    } else {
      captures.push({ filename: "06_assistant_system/24_assistant_local_response.png", feature: "AI assistant", screen: "AI assistant · /", status: "not captured", notes: "Local Ollama is not reachable at 127.0.0.1:11434; no assistant response was fabricated." });
    }

    const historyResponse = await context.request.get(`${apiUrl}/inspections`);
    if (!historyResponse.ok()) throw new Error(`History check failed with HTTP ${historyResponse.status()}.`);
    const historyData = await historyResponse.json();
    const records = historyData.inspections || historyData.items || historyData;
    const record = records.find(item => item.report_path);
    if (record) {
      const reportResponse = await context.request.get(`${apiUrl}/inspections/${encodeURIComponent(record.inspection_id)}/report`);
      const pdf = await reportResponse.body();
      if (!reportResponse.ok() || pdf.subarray(0, 4).toString() !== "%PDF") {
        throw new Error(`Existing PDF report could not be verified (HTTP ${reportResponse.status()}).`);
      }
      const reportServer = createServer((_request, serverResponse) => {
        serverResponse.writeHead(200, {
          "Content-Type": "application/pdf",
          "Content-Disposition": 'inline; filename="nano-sentinel-report.pdf"'
        });
        serverResponse.end(pdf);
      });
      await new Promise((resolve, reject) => {
        reportServer.once("error", reject);
        reportServer.listen(0, "127.0.0.1", resolve);
      });
      const pdfBrowser = await chromium.launch({ channel: "msedge", headless: false });
      try {
        const pdfPage = await pdfBrowser.newPage({ viewport, deviceScaleFactor: 1 });
        await pdfPage.goto(`http://127.0.0.1:${reportServer.address().port}/report.pdf`, { waitUntil: "load", timeout: 30000 });
        await pdfPage.locator('embed[type="application/pdf"]').waitFor({ state: "attached", timeout: 15000 });
        const pdfScreenshot = path.join(outputDir, "05_reports_analytics", "25_existing_pdf_report.png");
        await fs.mkdir(path.dirname(pdfScreenshot), { recursive: true });
        const renderDeadline = Date.now() + 20000;
        let rendered = false;
        while (Date.now() < renderDeadline) {
          await pdfPage.screenshot({ path: pdfScreenshot, animations: "disabled" });
          if (await hasRenderedPixels(pdfScreenshot)) {
            rendered = true;
            break;
          }
          await new Promise(resolve => setTimeout(resolve, 250));
        }
        if (!rendered) throw new Error("The Edge PDF viewer did not render visible report content before timeout.");
        captures.push({ filename: "05_reports_analytics/25_existing_pdf_report.png", feature: "PDF report", screen: `Existing report for ${record.inspection_id} · local PDF`, status: "captured", notes: "Genuine PDF bytes returned by the existing report endpoint and rendered in Edge's PDF viewer." });
        await pdfPage.close();
      } finally {
        await pdfBrowser.close();
        await new Promise(resolve => reportServer.close(resolve));
      }
    } else {
      captures.push({ filename: "05_reports_analytics/25_existing_pdf_report.png", feature: "PDF report", screen: "Existing inspection report", status: "not captured", notes: "No saved record with an available report path was returned by the API." });
    }

    if (errors.length) {
      captures.push({ filename: "", feature: "Validation", screen: "Browser run", status: "warning", notes: [...new Set(errors)].join(" | ") });
    }
    if (failedResources.length) {
      captures.push({ filename: "", feature: "Network", screen: "Browser run", status: "warning", notes: [...new Set(failedResources)].join(" | ") });
    }

    await createContactSheet(browser);
    const manifestHeader = ["filename", "feature", "actual_route_or_screen", "capture_status", "notes"];
    const manifest = [manifestHeader, ...captures.map(item => [item.filename, item.feature, item.screen, item.status, item.notes])]
      .map(row => row.map(csvEscape).join(",")).join("\r\n");
    await fs.writeFile(path.join(outputDir, "screenshot_manifest.csv"), `${manifest}\r\n`, "utf8");
    const captured = captures.filter(item => item.status === "captured").length;
    const unavailable = captures.filter(item => item.status === "not captured");
    const readme = [
      "Nano Sentinel report screenshots",
      "",
      `Captured ${captured} genuine application screenshots at ${viewport.width} x ${viewport.height} desktop viewport (the existing PDF is previewed in Microsoft Edge).`,
      "Source: the existing local React/Vite application, FastAPI API, inspection history, and generated PDF report.",
      "No model inference was submitted and no SQLite inspection records were created, edited, or deleted during capture.",
      "The existing sample .npy is selected locally in the upload screen only; it is not uploaded.",
      "",
      "Folders correspond to dashboard, inspection, explainability, history, analytics/reports, and assistant/system screens.",
      "screenshot_manifest.csv records each capture and every feature that was not captured, with the reason.",
      "contact_sheet.png is a thumbnail index of the captured application screenshots.",
      "",
      "Not captured:",
      ...(unavailable.length ? unavailable.map(item => `- ${item.feature}: ${item.notes}`) : ["- None"]),
      ...(failedResources.length ? ["", "Non-blocking browser resource failures:", ...Array.from(new Set(failedResources), resource => `- ${resource}`)] : []),
      ...(errors.length ? ["", "Non-blocking browser console messages:", ...Array.from(new Set(errors), error => `- ${error}`)] : []),
      "",
      "Rerun from the project root after starting FastAPI and Vite:",
      "  node .\\artifacts\\capture_report_screenshots.mjs",
      "Start FastAPI in one PowerShell window if needed:",
      "  cd .\\backend; ..\\.venv\\Scripts\\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000",
      "Start Vite in another PowerShell window if needed:",
      "  cd .\\frontend; npm run dev -- --host 127.0.0.1",
      "Optional URL overrides:",
      "  $env:NANO_SENTINEL_FRONTEND_URL='http://localhost:5173/'",
      "  $env:NANO_SENTINEL_API_URL='http://127.0.0.1:8000'",
      "Playwright must be installed in frontend/node_modules; install its Chromium browser once:",
      "  cd .\\frontend; npm install --no-save --no-package-lock playwright; npx playwright install chromium",
      "The existing Microsoft Edge installation is used only to display the generated PDF report.",
      "",
      "This archive contains original browser captures; no screenshot values or pixels have been edited."
    ].join("\r\n");
    await fs.writeFile(path.join(outputDir, "README.txt"), `${readme}\r\n`, "utf8");

    const archivePath = path.join(scriptDir, "NanoSentinel_Report_Screenshots.zip");
    const archivedFiles = [
      ...captures.filter(item => item.status === "captured").map(item => item.filename),
      "contact_sheet.png",
      "screenshot_manifest.csv",
      "README.txt"
    ];
    const archivedPaths = archivedFiles.map(filename => `'${filename.replaceAll("'", "''")}'`).join(",");
    const psCommand = `$ErrorActionPreference='Stop'; Add-Type -AssemblyName System.IO.Compression; Add-Type -AssemblyName System.IO.Compression.FileSystem; $source='${outputDir.replaceAll("'", "''")}'; $archivePath='${archivePath.replaceAll("'", "''")}'; $files=@(${archivedPaths}); if(Test-Path -LiteralPath $archivePath) { Remove-Item -LiteralPath $archivePath -Force }; $zip=[IO.Compression.ZipFile]::Open($archivePath,[IO.Compression.ZipArchiveMode]::Create); try { foreach($name in $files) { $full=Join-Path $source $name; if(-not (Test-Path -LiteralPath $full)) { throw "Missing archive file: $name" }; [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip,$full,$name.Replace('/','\\'),[IO.Compression.CompressionLevel]::Optimal) | Out-Null } } finally { $zip.Dispose() }`;
    execFileSync("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", psCommand], { stdio: "inherit" });

    const zipListing = execFileSync("powershell.exe", [
      "-NoProfile", "-NonInteractive", "-Command",
      `Add-Type -AssemblyName System.IO.Compression.FileSystem; $z=[IO.Compression.ZipFile]::OpenRead('${archivePath.replaceAll("'", "''")}'); try { $z.Entries | ForEach-Object { $_.FullName } } finally { $z.Dispose() }`
    ], { encoding: "utf8" }).trim().split(/\r?\n/).filter(Boolean);
    if (!zipListing.some(name => name.endsWith("screenshot_manifest.csv")) ||
        !zipListing.some(name => name.endsWith("README.txt")) ||
        !zipListing.some(name => name.endsWith("contact_sheet.png"))) {
      throw new Error("ZIP verification failed: required manifest, README, or contact sheet is missing.");
    }
    console.log(JSON.stringify({
      frontendUrl,
      apiUrl,
      health,
      capturedScreenshots: captured,
      unavailableCaptures: unavailable,
      applicationErrors: [...new Set(errors)],
      failedResources: [...new Set(failedResources)],
      outputDir,
      archivePath,
      archiveEntries: zipListing.length,
      archiveVerified: true
    }, null, 2));
  } finally {
    await context.close();
    await browser.close();
  }
}

main().catch(error => {
  console.error(`Screenshot capture failed: ${error.stack || error.message}`);
  process.exitCode = 1;
});
