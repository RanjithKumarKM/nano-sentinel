"""Import queued wafer maps through the running FastAPI inspection pipeline."""

import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
INCOMING = Path(__file__).resolve().parent / "Incoming"
DEFAULT_API_URL = "http://127.0.0.1:8000"
MAX_FILE_BYTES = 10 * 1024 * 1024


def process_incoming(incoming=INCOMING, api_url=DEFAULT_API_URL, dry_run=False,
                     http=requests):
    incoming = Path(incoming)
    if not incoming.is_dir():
        raise FileNotFoundError(f"Incoming folder does not exist: {incoming}")

    response = http.get(f"{api_url.rstrip('/')}/inspections", timeout=30)
    response.raise_for_status()
    payload = response.json()
    records = payload if isinstance(payload, list) else payload.get(
        "inspections", payload.get("items", [])
    )
    existing_names = {
        str(record.get("filename") or record.get("source", {}).get("filename"))
        for record in records
        if record.get("filename") or record.get("source", {}).get("filename")
    }

    totals = {"found": 0, "imported": 0, "skipped": 0, "failed": 0}
    for path in sorted(incoming.glob("*.npy")):
        totals["found"] += 1
        if path.name in existing_names:
            totals["skipped"] += 1
            print(f"Already in inspection history; skipping: {path.name}")
            continue

        if dry_run:
            print(f"Would inspect (original file will be kept): {path.name}")
            continue

        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                raise ValueError("File exceeds the 10 MB API upload limit.")
            with path.open("rb") as wafer_file:
                upload = http.post(
                    f"{api_url.rstrip('/')}/inspect",
                    files={
                        "file": (
                            path.name,
                            wafer_file,
                            "application/octet-stream",
                        )
                    },
                    timeout=180,
                )
            upload.raise_for_status()
            record = upload.json()
            inspection_id = record.get("inspection_id")
            if not inspection_id:
                raise ValueError("Inspection API response did not include an inspection ID.")
            existing_names.add(path.name)
            totals["imported"] += 1
            print(
                f"Saved {path.name}: {record['model']['predicted_class']} "
                f"({record['model']['confidence_percent']:.2f}%), "
                f"ID {inspection_id}"
            )
        except (OSError, ValueError, KeyError, TypeError, requests.RequestException) as exc:
            totals["failed"] += 1
            print(f"Failed {path.name}: {exc}", file=sys.stderr)

    return totals


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect backend\\Incoming .npy files through POST /inspect. "
            "Original inputs are never moved or deleted."
        )
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--incoming",
        type=Path,
        default=INCOMING,
        help=f"Input folder (default: {INCOMING})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List new files without uploading or changing them.",
    )
    args = parser.parse_args()

    try:
        totals = process_incoming(
            incoming=args.incoming,
            api_url=args.api_url,
            dry_run=args.dry_run,
        )
    except (OSError, requests.RequestException, ValueError) as exc:
        parser.exit(1, f"Could not import incoming files: {exc}\n")

    print(
        "Summary: "
        f"{totals['found']} found, {totals['imported']} imported, "
        f"{totals['skipped']} already present, {totals['failed']} failed"
    )
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
