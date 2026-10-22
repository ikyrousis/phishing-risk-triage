"""Capture the reference screenshots for the report from the live application.

Each sample is uploaded through the Flask test client, so the rendered page is
produced by the same code path as a manual upload, and captured with a headless
Chromium browser (Microsoft Edge, Chrome or Chromium). The capture is cropped to
the page content so no screenshot ships blank filler.

Run after ``python train_model.py`` and ``python scripts/analyze_results.py``:

    python scripts/capture_screenshots.py

The browser is found from the PHISHING_BROWSER environment variable, then from
the PATH, then from the standard Windows install locations. All three captures
use a 1440 pixel desktop viewport, one per risk band.

Outputs results/screenshot-high-risk.png, results/screenshot-medium-risk.png and
results/screenshot-low-risk.png directly in results/.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The Flask app stays at the project root, alongside the lecturer's core modules.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import app

OUTPUT_DIR = ROOT / "results"
# One sample per risk band of the deployed detector: high, medium (review), low.
SAMPLES = ("phishing_example.eml", "phishing_1.html", "legitimate_internal_notice.eml")
OUTPUTS = ("screenshot-high-risk.png", "screenshot-medium-risk.png", "screenshot-low-risk.png")
WIDTHS = (1440, 1440, 1440)
MAX_HEIGHT = 3200
PADDING = 32

BROWSER_COMMANDS = (
    "msedge", "chrome", "google-chrome", "google-chrome-stable", "chromium",
    "chromium-browser",
)
BROWSERS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)


def find_browser() -> str:
    override = os.environ.get("PHISHING_BROWSER")
    if override:
        if Path(override).exists():
            return override
        raise SystemExit(f"PHISHING_BROWSER points to a missing file: {override}")
    for command in BROWSER_COMMANDS:
        found = shutil.which(command)
        if found:
            return found
    for path in BROWSERS:
        if Path(path).exists():
            return path
    raise SystemExit(
        "No headless-capable Chromium browser found. Set PHISHING_BROWSER to the "
        "executable path."
    )


def render(sample: str, directory: Path) -> Path:
    client = app.test_client()
    payload = (ROOT / "sample_inputs" / sample).read_bytes()
    response = client.post(
        "/",
        data={"file": (io.BytesIO(payload), sample)},
        content_type="multipart/form-data",
    )
    if response.status_code != 200:
        raise SystemExit(f"Rendering {sample} failed with status {response.status_code}")
    page = directory / f"{sample}.html"
    page.write_bytes(response.data)
    return page


def capture(browser: str, page: Path, output: Path, width: int) -> None:
    shot = output.with_suffix(".raw.png")
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--window-size={width},{MAX_HEIGHT}",
            f"--screenshot={shot}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    try:
        from PIL import Image

        with Image.open(shot) as image:
            rgb = image.convert("RGB")
            bottom = rgb.getpixel((2, rgb.height - 2))
            pixels = rgb.load()
            last = 0
            for y in range(rgb.height - 1, -1, -1):
                if any(pixels[x, y] != bottom for x in range(0, rgb.width, 8)):
                    last = min(y + PADDING, rgb.height)
                    break
            rgb.crop((0, 0, rgb.width, last)).save(output)
    finally:
        shot.unlink(missing_ok=True)


def main() -> None:
    browser = find_browser()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        for sample, output, width in zip(SAMPLES, OUTPUTS, WIDTHS):
            page = render(sample, work)
            capture(browser, page, OUTPUT_DIR / output, width)
            print(f"captured {output} from {sample} at {width}px width")


if __name__ == "__main__":
    main()
