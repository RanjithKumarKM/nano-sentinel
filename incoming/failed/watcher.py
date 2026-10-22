
import time
import shutil
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCOMING = PROJECT_ROOT / "backend" / "Incoming"
PROCESSED = PROJECT_ROOT / "incoming" / "processed"
FAILED = PROJECT_ROOT / "incoming" / "failed"

API_URL = "http://127.0.0.1:8000/inspect"
POLL_SECONDS = 2
MAX_WAIT_SECONDS = 60
MAX_FILE_BYTES = 10 * 1024 * 1024

def wait_until_stable(path):
    """Wait until a file's size stops changing."""
    previous_size = -1
    stable_checks = 0
    deadline = time.monotonic() + MAX_WAIT_SECONDS

    while time.monotonic() < deadline:
        if not path.exists():
            return False

        size = path.stat().st_size

        if size > 0 and size == previous_size:
            stable_checks += 1
            if stable_checks >= 2:
                return True
        else:
            stable_checks = 0

        previous_size = size
        time.sleep(1)

    return False


def unique_destination(folder, filename):
    destination = folder / filename
    if not destination.exists():
        return destination

    stem, suffix = destination.stem, destination.suffix
    counter = 1

    while True:
        candidate = folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def inspect_file(path):
    if path.suffix.lower() != ".npy":
        print(f"Skipping unsupported file: {path.name}")
        return

    print(f"\nNew wafer detected: {path.name}")

    if not wait_until_stable(path):
        print("File did not become stable or was removed.")
        if path.exists():
            destination = unique_destination(FAILED, path.name)
            shutil.move(str(path), str(destination))
            print("Moved to:", destination)
        return

    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("File exceeds the 10 MB API limit.")

        with path.open("rb") as wafer_file:
            response = requests.post(
                API_URL,
                files={
                    "file": (
                        path.name,
                        wafer_file,
                        "application/octet-stream",
                    )
                },
                timeout=180,
            )

        if not response.ok:
            raise RuntimeError(
                f"API returned {response.status_code}: {response.text}"
            )

        result = response.json()
        prediction = result.get("model", {})

        print("Inspection completed")
        print("Inspection ID:", result.get("inspection_id"))
        print("Predicted class:", prediction.get("predicted_class"))
        print(
            "Confidence:",
            f'{prediction.get("confidence_percent", 0):.2f}%',
        )
        print("PDF:", result.get("report_path", "Not generated"))

        destination = unique_destination(PROCESSED, path.name)
        shutil.move(str(path), str(destination))
        print("Moved to:", destination)

    except Exception as exc:
        print("Inspection failed:", exc)

        if path.exists():
            destination = unique_destination(FAILED, path.name)
            shutil.move(str(path), str(destination))
            print("Moved to:", destination)


def main():
    for folder in (INCOMING, PROCESSED, FAILED):
        folder.mkdir(parents=True, exist_ok=True)

    print("NANO SENTINEL - AUTOMATIC WAFER MONITOR")
    print("Watching:", INCOMING)
    print("Press Ctrl+C to stop.")

    seen = set()

    try:
        while True:
            files = sorted(
                p for p in INCOMING.iterdir()
                if p.is_file() and p.suffix.lower() == ".npy"
            )

            for path in files:
                # Do not repeatedly retry the same file in this run
                # if it cannot be moved or disappears during processing.
                key = str(path.resolve())

                if key in seen:
                    continue

                seen.add(key)
                inspect_file(path)

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("\nWafer monitor stopped.")


if __name__ == "__main__":
    main()
