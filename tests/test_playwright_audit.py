"""Playwright smoke for the Physics-check UI (Setup tab card, status bar, "audit any result",
Study tab custom builder / past studies).  Screenshots go to webapp/screenshots/audit_*.png.

Uses the real ``freeto.audit`` when importable, otherwise the test fake
(tests/_serve_for_ui.py); skips without Playwright/Chromium."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SHOTS = REPO_ROOT / "webapp" / "screenshots"

try:
    from playwright.sync_api import sync_playwright
    HAVE_PLAYWRIGHT = True
except Exception:  # noqa: BLE001
    HAVE_PLAYWRIGHT = False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_health(base, timeout=30.0):
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=1.0) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    raise RuntimeError("server did not start")


@pytest.mark.skipif(not HAVE_PLAYWRIGHT, reason="playwright not installed")
def test_physics_check_card(tmp_path):
    SHOTS.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    workdir = tmp_path / "wd"
    # a finished study on disk, as left behind by an earlier server session
    sd = workdir / "study" / "deadbeef0001"
    sd.mkdir(parents=True)
    (sd / "results.json").write_text(json.dumps({
        "suite": "quick", "created": "2026-10-02 10:00:00", "records": [
            {"run_id": "S3-0-cantilever_beam-MMA-s0", "study": "S3", "kind": "continuum",
             "problem": "cantilever_beam", "label": "MMA", "compliance": 0.147, "crisp_compliance": 0.145,
             "audit_ok": True, "n_components": 1, "floating_frac": 0.0, "loads_solid": 1.0, "gap": 0.0},
            {"run_id": "S4-5-bridge_deck-QUBO-s0", "study": "S4", "kind": "continuum",
             "problem": "bridge_deck", "label": "QUBO-sa (block)", "backend": "sa", "compliance": 0.02,
             "crisp_compliance": 179.5, "audit_ok": False, "n_components": 22, "floating_frac": 0.19,
             "loads_solid": 0.956, "gap": 1.2}]}))
    (sd / "summary.md").write_text("# demo summary\n")
    env = dict(os.environ, FREETO_WEBAPP_DIR=str(workdir))
    proc = subprocess.Popen([sys.executable, str(REPO_ROOT / "tests" / "_serve_for_ui.py"), str(port)],
                            cwd=str(REPO_ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait_health(base)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(base, wait_until="networkidle")
            page.wait_for_selector("#example-select option[value='cantilever_beam']", state="attached", timeout=10000)
            page.select_option("#example-select", "cantilever_beam")
            page.click("#example-load-btn")
            page.wait_for_function("document.getElementById('load-table-body').children.length >= 1", timeout=10000)
            page.wait_for_timeout(500)
            page.fill("#p-mesh_control", "20")
            page.fill("#p-max_iter", "40")
            page.select_option("#p-optimizer", "MMA")
            assert page.is_checked("#p-audit")
            assert page.is_hidden("#audit-card")
            page.click("#run-btn")
            page.wait_for_function("document.getElementById('run-message').textContent.includes('Done')", timeout=60000)
            page.wait_for_selector("#audit-card:not([hidden])", timeout=15000)

            pill = page.text_content("#audit-pill").strip()
            assert pill in ("PASS", "FAIL")
            rows = page.locator("#audit-checks-body tr")
            assert rows.count() >= 7
            names = page.eval_on_selector_all("#audit-checks-body .audit-name", "e => e.map(x => x.textContent)")
            assert "single_grounded_body" in names
            # overlay image loads (server renders it lazily)
            page.wait_for_function("document.getElementById('audit-img').naturalWidth > 0", timeout=60000)
            assert "components" in page.text_content("#status-audit")
            page.locator("#audit-card").scroll_into_view_if_needed()
            page.locator("#audit-card").screenshot(path=str(SHOTS / "audit_01_card.png"))
            page.screenshot(path=str(SHOTS / "audit_02_setup_done.png"))

            # layout guard: nothing in the card is clipped by the side panel
            clipped = page.evaluate("""() => {
              const pr = document.getElementById('left-panel').getBoundingClientRect();
              const bad = [];
              document.querySelectorAll('#audit-card *, #audit-any-card *').forEach(e => {
                const r = e.getBoundingClientRect();
                if (r.width > 0 && r.right > pr.right + 1 && !e.closest('.table-scroll')) bad.push(e.tagName + '#' + (e.id || e.className));
              });
              return bad;
            }""")
            assert clipped == [], clipped

            if os.environ.get("FREETO_UI_EXPECT_PASS", "") == "1":
                assert pill == "PASS"

            # click to enlarge + Esc
            page.click("#audit-img-link")
            page.wait_for_selector("#lightbox:not([hidden])")
            page.screenshot(path=str(SHOTS / "audit_03_lightbox.png"))
            page.keyboard.press("Escape")
            page.wait_for_selector("#lightbox", state="hidden")

            # JSON download
            with page.expect_download() as dl:
                page.click("#audit-download-json-btn")
            data = json.loads(Path(dl.value.path()).read_text())
            assert "checks" in data and "ok" in data

            # QUBO run: live "components: N, floating: x %" in the status bar while running
            if page.eval_on_selector("#p-optimizer option[value='QUBO']", "o => !o.disabled"):
                page.select_option("#p-optimizer", "QUBO")
                page.select_option("#qubo-backend", "sa")
                page.fill("#p-max_iter", "25")
                page.click("#run-btn")
                page.wait_for_function(
                    "/components: \\d+, floating: [\\d.]+ %/.test(document.getElementById('status-audit').textContent)",
                    timeout=60000)
                assert page.is_visible("#status-audit")
                page.screenshot(path=str(SHOTS / "audit_06_qubo_live.png"), clip={"x": 0, "y": 0, "width": 1440, "height": 60})
                page.wait_for_function("document.getElementById('run-message').textContent.includes('Done')", timeout=120000)
                page.wait_for_function("document.getElementById('audit-summary').textContent.length > 0")
                page.locator("#audit-card").scroll_into_view_if_needed()
                page.wait_for_function("document.getElementById('audit-img').naturalWidth > 0", timeout=60000)
                page.screenshot(path=str(SHOTS / "audit_07_qubo_done.png"))

            # audit any result: the finished job is listed; re-audit it
            page.click("#audit-job-refresh-btn")
            page.wait_for_function("document.getElementById('audit-job-select').options.length >= 2")
            page.select_option("#audit-job-select", index=1)
            page.click("#audit-job-run-btn")
            page.wait_for_function("document.getElementById('audit-any-message').textContent.includes('Audit ')", timeout=60000)

            # -- Study tab ----------------------------------------------------
            page.click('.tab-btn[data-tab="study"]')
            page.wait_for_selector("#study-past-list .past-item", timeout=10000)
            page.select_option("#study-suite-select", "custom")
            page.wait_for_selector("#study-cproblems-list input", timeout=10000)
            page.screenshot(path=str(SHOTS / "audit_04_study_custom.png"))
            page.click("#study-past-list .past-item")
            page.wait_for_selector("#study-results-table-wrap table", timeout=10000)
            body = page.text_content("#study-results-table-wrap")
            assert "physically invalid" in body and "PASS" in body and "FAIL" in body
            assert page.locator("#study-results-table-wrap tr.row-invalid").count() == 1
            page.screenshot(path=str(SHOTS / "audit_05_study_past.png"))
            assert errors == [], errors
            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
