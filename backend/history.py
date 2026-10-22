import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATABASE_DIR / "inspection_history.db"


def normalize_inspection_record(record):
    """Expose a consistent shape for new JSON records and flattened DB rows."""
    item = dict(record)
    model = item.get("model")
    model = dict(model) if isinstance(model, dict) else {}
    source = item.get("source")
    source = dict(source) if isinstance(source, dict) else {}
    severity = item.get("severity")
    severity = dict(severity) if isinstance(severity, dict) else {}
    decision = item.get("decision")
    decision = dict(decision) if isinstance(decision, dict) else {}

    aliases = {
        "predicted_class": model.get("predicted_class", item.get("predicted_class")),
        "confidence_percent": model.get("confidence_percent", item.get("confidence_percent")),
        "probabilities": model.get("probabilities", item.get("probabilities")),
    }
    for key, value in aliases.items():
        if value is not None:
            model[key] = value
            item[key] = value
    item["model"] = model

    if source.get("filename") is None and item.get("filename") is not None:
        source["filename"] = item["filename"]
    if source:
        item["source"] = source
        if source.get("filename") is not None:
            item["filename"] = source["filename"]

    severity_fields = (
        "total_die_pixels",
        "defective_die_pixels",
        "defect_ratio_percent",
        "severity_category",
        "note",
    )
    if isinstance(item.get("severity"), str):
        severity.setdefault("severity_category", item["severity"])
    for key in severity_fields:
        if severity.get(key) is None and item.get(key) is not None:
            severity[key] = item[key]
        if severity.get(key) is not None:
            item[key] = severity[key]
    item["severity"] = severity

    status = decision.get(
        "inspection_status",
        item.get("inspection_status", item.get("status")),
    )
    recommendation = decision.get("recommendation", item.get("recommendation"))
    if status is not None:
        decision["inspection_status"] = status
        item["inspection_status"] = status
        item["status"] = status
    if recommendation is not None:
        decision["recommendation"] = recommendation
        item["recommendation"] = recommendation
    item["decision"] = decision

    if item.get("timestamp") is None and item.get("timestamp_utc") is not None:
        item["timestamp"] = item["timestamp_utc"]
    return item


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                inspection_id TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence_percent REAL NOT NULL,
                probabilities_json TEXT NOT NULL,
                total_die_pixels INTEGER NOT NULL,
                defective_die_pixels INTEGER NOT NULL,
                defect_ratio_percent REAL NOT NULL,
                severity_category TEXT NOT NULL,
                inspection_status TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                record_path TEXT,
                report_path TEXT
            )
        """)
        columns = {r["name"]
                   for r in conn.execute("PRAGMA table_info(inspections)")}
        if "report_path" not in columns:
            conn.execute("ALTER TABLE inspections ADD COLUMN report_path TEXT")


def save_inspection(record):
    model, severity, decision = record.get("model", {}), record.get(
        "severity", {}), record.get("decision", {})
    source = record.get("source", {})
    predicted_class = model.get("predicted_class")
    confidence_percent = model.get("confidence_percent")
    probabilities = model.get("probabilities")
    if predicted_class is None or confidence_percent is None:
        raise ValueError("Inspection prediction is missing its class or confidence.")
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError("Inspection prediction is missing class probabilities.")
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO inspections (
                inspection_id, timestamp_utc, filename, predicted_class, confidence_percent,
                probabilities_json, total_die_pixels, defective_die_pixels, defect_ratio_percent,
                severity_category, inspection_status, recommendation, record_path, report_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["inspection_id"], record["timestamp_utc"], source.get(
                "filename", "unknown"),
            predicted_class, float(confidence_percent),
            json.dumps(probabilities), int(
                severity.get("total_die_pixels", 0)),
            int(severity.get("defective_die_pixels", 0)), float(
                severity.get("defect_ratio_percent", 0)),
            severity.get("severity_category", "UNKNOWN"), decision.get(
                "inspection_status", "MANUAL_REVIEW"),
            decision.get("recommendation", ""), record.get(
                "record_path"), record.get("report_path")
        ))


def list_inspections(limit=None):
    with get_connection() as conn:
        if limit is None:
            rows = conn.execute(
                "SELECT * FROM inspections ORDER BY timestamp_utc DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM inspections ORDER BY timestamp_utc DESC LIMIT ?",
                (max(1, min(int(limit), 500)),)
            ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["probabilities"] = json.loads(item.pop("probabilities_json"))
        items.append(normalize_inspection_record(item))
    return items


def count_inspections():
    with get_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]


def count_inspections_needing_review():
    with get_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM inspections "
            "WHERE inspection_status IN ('MANUAL_REVIEW', 'REVIEW_RECOMMENDED')"
        ).fetchone()[0]


def get_inspection(inspection_id):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM inspections WHERE inspection_id = ?", (inspection_id,)).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["probabilities"] = json.loads(item.pop("probabilities_json"))
    return normalize_inspection_record(item)


def delete_inspection(inspection_id):
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM inspections WHERE inspection_id = ?",
            (inspection_id,)
        )
        return cursor.rowcount > 0


def delete_all_inspections():
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM inspections")
        return cursor.rowcount
