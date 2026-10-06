"""Screenshots 01, 02 and 04 of webapp/screenshots/ with the paper defaults.

usage:  python scripts/webapp_screenshots.py [--base http://127.0.0.1:8000]
                                            [--out webapp/screenshots] [--payload FILE]

Drives the web app with Playwright/Chromium (an installed browser, e.g.
PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers): 01 the page as it opens (every
field at the cantilever's paper value), 02 the cantilever example loaded,
04 the finished run with the default settings (the paper's QUBO-SA, block
Hessian, seed 0, MeshControl 36: a few minutes) showing the evaluated binary
design and the result card.  Without --base a server is started on a free
port (one BLAS thread, like the study).  --payload saves the JSON body the
page posted to /api/jobs.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(base, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(base + "/api/health", timeout=1).close()
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    raise RuntimeError("server did not start")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--out", default=str(ROOT / "webapp" / "screenshots"))
    ap.add_argument("--payload")
    ap.add_argument("--timeout", type=float, default=1800, help="run timeout (s)")
    a = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = None
    base = a.base
    if not base:
        port = _free_port()
        base = f"http://127.0.0.1:{port}"
        env = dict(os.environ, FREETO_WEBAPP_DIR=tempfile.mkdtemp(prefix="qff3d_shots_"),
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        proc = subprocess.Popen([sys.executable, "-m", "webapp.server", "--port", str(port)],
                                cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    try:
        _wait(base)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            posted = []
            page.on("request", lambda r: posted.append(r.post_data)
                    if r.url.endswith("/api/jobs") and r.method == "POST" else None)
            page.goto(base, wait_until="networkidle")
            page.wait_for_function(
                "document.getElementById('paper-method').options.length > 1", timeout=20000)
            page.wait_for_timeout(500)
            page.screenshot(path=str(out / "01_initial.png"))
            page.select_option("#example-select", "cantilever_beam")
            page.click("#example-load-btn")
            page.wait_for_function(
                "document.getElementById('load-table-body').children.length >= 1", timeout=20000)
            page.wait_for_timeout(1200)
            # show the parameter card (paper values) below the example and files
            page.evaluate("""() => {
              const box = document.querySelector('#left-panel .panel-scroll');
              const card = document.getElementById('p-optimizer').closest('fieldset');
              box.scrollTop = card.offsetTop - box.offsetTop - 260;
            }""")
            page.wait_for_timeout(300)
            page.screenshot(path=str(out / "02_example_loaded.png"))
            page.click("#run-btn")
            page.wait_for_function(
                "['Done', 'Error', 'Stopped'].some(s => "
                "document.getElementById('run-message').textContent.includes(s))",
                timeout=a.timeout * 1000)
            page.wait_for_timeout(2500)  # evaluated design + audit card
            page.evaluate("document.getElementById('result-card').scrollIntoView()")
            page.wait_for_timeout(300)
            page.screenshot(path=str(out / "04_done.png"))
            msg = page.text_content("#run-message")
            browser.close()
        if a.payload and posted:
            Path(a.payload).write_text(json.dumps(json.loads(posted[-1]), indent=1,
                                                  sort_keys=True))
        print("run message:", msg)
        print("screenshots:", ", ".join(str(out / n) for n in
                                        ("01_initial.png", "02_example_loaded.png", "04_done.png")))
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    main()
