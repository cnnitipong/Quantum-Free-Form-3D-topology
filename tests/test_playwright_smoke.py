"""Headless-browser smoke test for the QFF-3D web app.

Starts a real server subprocess, drives it with Playwright/Chromium exactly
like a person would (load page -> load GE example -> shrink the job to a
tiny/fast configuration -> Run -> wait for completion), and saves screenshots
for visual review to a temporary directory (FREETO_SMOKE_SCREENSHOTS=<dir> to
keep them).  The committed webapp/screenshots/01, 02 and 04 show a finished
paper-default run and come from scripts/webapp_screenshots.py.

Requires the Chromium build already installed at PLAYWRIGHT_BROWSERS_PATH
(do NOT run `playwright install` — see task instructions); this test skips
itself gracefully if Playwright or a browser isn't available.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = REPO_ROOT / "webapp" / "screenshots"

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except Exception:  # noqa: BLE001
    HAVE_PLAYWRIGHT = False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, timeout: float = 20.0):
    import urllib.request

    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.3)
    raise RuntimeError(f"Server did not become healthy in time: {last_err}")


@pytest.mark.skipif(not HAVE_PLAYWRIGHT, reason="playwright not installed")
def test_browser_smoke(tmp_path):
    SCREENSHOT_DIR = Path(os.environ.get("FREETO_SMOKE_SCREENSHOTS") or tmp_path / "screenshots")
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    workdir = tmp_path / "freeto_web_playwright"
    base_url = f"http://127.0.0.1:{port}"

    env = dict(os.environ)
    env["FREETO_WEBAPP_DIR"] = str(workdir)
    proc = subprocess.Popen(
        [sys.executable, "-m", "webapp.server", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO_ROOT),
        env=env,
        # m7: never pipe the server's stdout/stderr without draining it. A
        # PIPE fills up (Windows anonymous pipes default to just 4 KB, vs.
        # 64 KB on Linux) once uvicorn logs enough polling requests, which
        # then blocks the server on its own stderr.write() and hangs this
        # test. Discard it instead -- this test only cares about the HTTP
        # responses and screenshots, not the server's console output.
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(base_url)

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            console_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

            page.goto(base_url, wait_until="networkidle")
            page.wait_for_selector(
                "#example-select option[value='GE_bracket']", state="attached", timeout=10000
            )
            page.screenshot(path=str(SCREENSHOT_DIR / "01_initial.png"))

            # Load the GE bracket example.
            page.select_option("#example-select", "GE_bracket")
            page.click("#example-load-btn")
            page.wait_for_function(
                "document.getElementById('load-table-body').children.length >= 2",
                timeout=10000,
            )
            page.wait_for_timeout(800)  # let three.js finish loading/rendering STL meshes
            page.screenshot(path=str(SCREENSHOT_DIR / "02_example_loaded.png"))

            # Layout regression guard: no input/select of the side panel may be
            # clipped by the panel's right edge (at 1440 and at 1024 px wide),
            # with the Quantum settings card (and its Advanced grid) visible.
            def clipped_controls():
                return page.evaluate("""() => {
                  const pr = document.getElementById('left-panel').getBoundingClientRect();
                  const bad = [];
                  document.querySelectorAll('#left-panel input, #left-panel select').forEach(e => {
                    const r = e.getBoundingClientRect();
                    if (r.width > 0 && r.right > pr.right - 2) bad.push(e.id || e.type);
                  });
                  return bad;
                }""")

            has_qubo = page.eval_on_selector(
                "#p-optimizer option[value='QUBO']", "o => !o.disabled")
            if has_qubo:
                # QUBO is listed first and selected by default (also after
                # loading an example), with its settings card visible and
                # preset to the paper's QUBO-SA (block) options.
                assert page.eval_on_selector("#p-optimizer", "e => e.options[0].value") == "QUBO"
                assert page.eval_on_selector("#p-optimizer", "e => e.value") == "QUBO"
                assert page.is_visible("#quantum-settings-card")
                assert page.eval_on_selector("#qubo-backend", "e => e.value") == "sa"
                assert page.eval_on_selector("#qubo-hessian", "e => e.value") == "block"
                page.select_option("#p-optimizer", "QUBO")
                page.evaluate("document.getElementById('quantum-advanced-details').open = true")
            for width in (1440, 1024):
                page.set_viewport_size({"width": width, "height": 800})
                page.wait_for_timeout(150)
                assert clipped_controls() == [], f"clipped controls at {width}px"
            page.set_viewport_size({"width": 1440, "height": 900})
            if has_qubo:
                page.select_option("#p-optimizer", "OC")

            # Shrink the job so it runs fast in CI. NB: the real core needs
            # enough mesh resolution for the (thin) GE_force load region to
            # contain at least one grid node; 12 is fine for the stub core
            # but the real solver needs ~30 for this particular example.
            page.fill("#p-mesh_control", "30")
            page.fill("#p-max_iter", "3")

            page.click("#run-btn")
            page.wait_for_timeout(500)
            page.screenshot(path=str(SCREENSHOT_DIR / "03_running.png"))

            page.wait_for_function(
                "document.getElementById('run-message').textContent.includes('Done')",
                timeout=30000,
            )
            page.wait_for_timeout(500)
            page.screenshot(path=str(SCREENSHOT_DIR / "04_done.png"))

            assert not page.is_disabled("#download-stl-btn")
            assert not page.is_disabled("#download-npz-btn")

            # -- Truss tab -------------------------------------------------
            # Open it and try the smallest exact-enumeration benchmark;
            # feature-detects gracefully (like the rest of the app) if
            # freeto.truss isn't installed in this environment.
            page.click('.tab-btn[data-tab="truss"]')
            page.wait_for_timeout(400)  # let /api/truss/benchmarks populate the select
            page.screenshot(path=str(SCREENSHOT_DIR / "05_truss_tab.png"))

            truss_options = page.eval_on_selector_all(
                "#truss-benchmark-select option",
                "opts => opts.map(o => o.value).filter(v => v)",
            )
            if truss_options:
                # T2s (gs_3x2, 12 bars) is the smallest exact benchmark
                # documented in docs/QUANTUM_DESIGN.md §D; fall back to
                # whatever the first real option is otherwise.
                bench_value = "gs_3x2" if "gs_3x2" in truss_options else truss_options[0]
                page.select_option("#truss-benchmark-select", bench_value)
                page.wait_for_timeout(500)  # ground-structure detail fetch + canvas draw
                page.screenshot(path=str(SCREENSHOT_DIR / "06_truss_benchmark_loaded.png"))

                exact_disabled = page.eval_on_selector(
                    '#truss-method-select option[value="exact"]', "o => o.disabled"
                )
                if not exact_disabled:
                    page.select_option("#truss-method-select", "exact")

                page.click("#truss-run-btn")
                page.wait_for_function(
                    "['Done.', 'Error'].some(s => "
                    "document.getElementById('truss-run-message').textContent.includes(s))",
                    timeout=60000,
                )
                page.wait_for_timeout(400)
                page.screenshot(path=str(SCREENSHOT_DIR / "07_truss_done.png"))
                assert "Error" not in page.text_content("#truss-run-message")
            else:
                # No benchmarks listed -> freeto.truss isn't installed here;
                # the tab must show a clear notice instead of a silently
                # broken/empty control (see webapp/core_loader.py's
                # TRUSS_USABLE feature-detection).
                unavailable_hidden = page.eval_on_selector("#truss-unavailable-card", "el => el.hidden")
                assert unavailable_hidden is False, (
                    "Truss tab has no benchmarks and does not show the 'unavailable' notice"
                )

            browser.close()

        # Print (rather than assert on) any console errors so they show up
        # in test output for a human to inspect; a few benign ones (e.g. from
        # third-party libs) are fine, but this makes real bugs visible.
        if console_errors:
            print("Browser console errors:\n" + "\n".join(console_errors))

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
