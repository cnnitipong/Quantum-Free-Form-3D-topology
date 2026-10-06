"""Web-app tests for the physics-audit endpoints and the study list/custom builder
(docs/AUDIT_API.md "Web app").

Two layers:

* ``TestAuditPlumbing`` runs against a *fake* ``freeto.audit`` (monkeypatched into
  ``webapp.core_loader``) that returns the documented dict shape. It checks the web app's
  own logic -- routing, status codes, caching, status fields, 503 fallback -- and works
  before / without the real module.
* ``TestRealAudit`` runs the real ``freeto.audit`` through the API and is skipped when the
  module is not importable.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))
os.environ.setdefault("FREETO_WEBAPP_DIR", str(REPO_ROOT / ".pytest_freeto_web"))

from fastapi.testclient import TestClient  # noqa: E402

import webapp.server as server_module  # noqa: E402
from webapp import core_loader  # noqa: E402
import _fake_audit as _fake  # noqa: E402

pytestmark = pytest.mark.skipif(not core_loader.CORE_USABLE, reason="freeto core not importable")

REAL_AUDIT = core_loader._load_audit()  # True if freeto.audit imports
PNG_MAGIC = _fake.PNG_MAGIC
CHECK_NAMES = set(_fake.CHECK_NAMES)
_tiny_png = _fake.tiny_png


@pytest.fixture()
def workdir(tmp_path):
    return tmp_path / "web"


@pytest.fixture()
def client(workdir):
    app = server_module.create_app(workdir)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def fake_audit(monkeypatch):
    """Stand-in for freeto.audit with the AUDIT_API.md dict shape (counts calls)."""
    calls, restore = _fake.install(core_loader)
    yield calls
    restore()


def _wait(client, url, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(url).json()
        if s["status"] in ("done", "error", "stopped"):
            return s
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {url}")


def _tiny_job(client, example="GE_bracket", **over):
    ex = client.post(f"/api/examples/{example}/load").json()["prefill"]
    payload = {"label": "audit-test", "domain_id": ex["domain_id"], "loads": ex["loads"],
               "fixed_id": ex["fixed_id"], "mesh_control": 30, "volfrac": 0.3,
               "youngs_modulus": 210e9, "max_iter": 3, "optimizer": "MMA"}
    payload.update(over)
    r = client.post("/api/jobs", json=payload)
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    s = _wait(client, f"/api/jobs/{jid}")
    assert s["status"] == "done", s.get("error")
    return jid, s


# ---------------------------------------------------------------------------
class TestAuditPlumbing:
    def test_audit_default_on_and_dict_shape(self, client, fake_audit):
        jid, status = _tiny_job(client)
        assert status["audit_enabled"] is True
        assert status["audit_ok"] is True and status["n_components"] == 1
        assert status["floating_frac"] == 0.0
        r = client.get(f"/api/jobs/{jid}/audit")
        assert r.status_code == 200, r.text
        a = r.json()
        assert a["ok"] is True
        assert set(a["checks"]) == CHECK_NAMES
        assert all({"pass", "value", "detail"} <= set(c) for c in a["checks"].values())
        assert isinstance(a["components"], list) and a["components"][0]["grounded"] is True
        assert (client.app.state.workdir / "jobs" / jid / "audit.json").is_file()

    def test_audit_disabled_gives_409_then_run(self, client, fake_audit):
        jid, status = _tiny_job(client, audit=False)
        assert status["audit_enabled"] is False and status["audit_ok"] is None
        assert client.get(f"/api/jobs/{jid}/audit").status_code == 409
        assert client.get(f"/api/jobs/{jid}/audit.png").status_code == 409
        r = client.post(f"/api/jobs/{jid}/audit/run")
        assert r.status_code == 200, r.text
        assert r.json()["ok"] is True and fake_audit["report"] == 1
        assert client.get(f"/api/jobs/{jid}/audit").status_code == 200
        st = client.get(f"/api/jobs/{jid}").json()
        assert st["audit_ok"] is True and st["audit_enabled"] is True

    def test_audit_run_recomputes_and_invalidates_png(self, client, fake_audit):
        jid, _ = _tiny_job(client)
        assert client.get(f"/api/jobs/{jid}/audit.png").status_code == 200
        assert fake_audit["figure"] == 1
        fake_audit["ok"] = False
        a = client.post(f"/api/jobs/{jid}/audit/run").json()
        assert a["ok"] is False and a["n_components"] == 3
        assert client.get(f"/api/jobs/{jid}").json()["audit_ok"] is False
        assert client.get(f"/api/jobs/{jid}/audit.png").status_code == 200
        assert fake_audit["figure"] == 2  # cache dropped by the re-run

    def test_audit_png_is_png_and_cached(self, client, fake_audit):
        jid, _ = _tiny_job(client)
        r = client.get(f"/api/jobs/{jid}/audit.png")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content.startswith(PNG_MAGIC)
        r2 = client.get(f"/api/jobs/{jid}/audit.png")
        assert r2.content == r.content and fake_audit["figure"] == 1

    def test_unknown_and_unfinished_jobs(self, client, fake_audit):
        assert client.get("/api/jobs/nope/audit").status_code == 404
        assert client.get("/api/jobs/nope/audit.png").status_code == 404
        assert client.post("/api/jobs/nope/audit/run").status_code == 404

    def test_503_when_module_missing(self, client, monkeypatch):
        jid, status = _tiny_job(client)  # the real module may or may not be present
        monkeypatch.setattr(core_loader, "AUDIT_USABLE", False)
        monkeypatch.setattr(core_loader, "_load_audit", lambda: False)
        monkeypatch.setattr(core_loader, "AUDIT_IMPORT_ERROR", "No module named 'freeto.audit'")
        r = client.post(f"/api/jobs/{jid}/audit/run")
        assert r.status_code == 503
        assert "audit module not available" in r.json()["detail"]
        r = client.get(f"/api/jobs/{jid}/audit.png")
        # either a cached figure from a previous real audit or the 503
        assert r.status_code in (200, 503, 409)
        assert client.get("/api/health").json()["audit"]["usable"] is False

    def test_failed_audit_never_fails_the_job(self, client, monkeypatch, request):
        def boom(res, cfg):
            raise RuntimeError("kaboom")
        _calls, restore = _fake.install(core_loader)
        request.addfinalizer(restore)
        monkeypatch.setattr(sys.modules["freeto.audit"], "audit_result", boom)
        monkeypatch.setattr(core_loader, "audit_result", boom)
        jid, status = _tiny_job(client)
        assert status["status"] == "done" and status["audit_ok"] is None
        assert "kaboom" in (status["audit_error"] or "")
        assert client.get(f"/api/jobs/{jid}/audit").status_code == 404

    def test_audit_live_in_status(self, client):
        """audit_live reported by the core's iteration callback shows up in the status."""
        from webapp.jobs import Job
        job = Job(id="x", label="x", config_dict={}, cfg=None)
        job.history.setdefault("audit_live", []).append({"n_components": 4, "floating_frac": 0.12})
        snap = job.snapshot_status()
        assert snap["audit_live"] == {"n_components": 4, "floating_frac": 0.12}
        assert snap["audit_ok"] is None


# ---------------------------------------------------------------------------
class TestStudyListAndCustom:
    def _fake_study(self, workdir, sid="abc123def456", audit=True, complete=True):
        d = workdir / "study" / sid
        d.mkdir(parents=True)
        rec = {"run_id": "S3-0-x-MMA-s0", "kind": "continuum", "problem": "cantilever_beam",
               "label": "MMA", "compliance": 1.0, "crisp_compliance": 1.1, "error": None}
        rec2 = dict(rec, run_id="S3-1-x-Q-s0", label="QUBO-sa")
        if audit:
            rec.update(audit_ok=True, n_components=1, floating_frac=0.0, loads_solid=1.0)
            rec2.update(audit_ok=False, n_components=5, floating_frac=0.2, loads_solid=0.9)
        if complete:
            (d / "results.json").write_text(json.dumps(
                {"suite": "quick", "created": "2026-10-02 12:00:00", "elapsed": 5.0,
                 "records": [rec, rec2], "summary": [], "figures": []}))
        (d / "summary.md").write_text("# summary\n")
        (d / "F1_fake.png").write_bytes(_tiny_png())
        return d

    def test_list_empty(self, client):
        r = client.get("/api/study/list")
        assert r.status_code == 200
        assert [s for s in r.json()["studies"] if s["source"] == "web"] == []

    def test_list_and_open_past_study_after_restart(self, workdir):
        self._fake_study(workdir)
        self._fake_study(workdir, sid="old000000001", audit=False)
        self._fake_study(workdir, sid="part00000001", complete=False)
        # a *new* app instance on the same workdir == a restarted server
        with TestClient(server_module.create_app(workdir)) as c:
            rows = {s["id"]: s for s in c.get("/api/study/list").json()["studies"] if s["source"] == "web"}
            assert set(rows) == {"abc123def456", "old000000001", "part00000001"}
            r = rows["abc123def456"]
            assert r["suite"] == "quick" and r["n_records"] == 2 and r["n_invalid"] == 1
            assert r["has_audit"] is True and r["complete"] is True and r["n_figures"] == 1
            assert rows["old000000001"]["has_audit"] is False
            assert rows["part00000001"]["complete"] is False

            res = c.get("/api/study/results/abc123def456")
            assert res.status_code == 200
            body = res.json()
            assert body["status"] == "done" and len(body["results"]["records"]) == 2
            assert body["summary_md"].startswith("# summary")
            names = {f["name"] for f in body["files"]}
            assert {"results.json", "F1_fake.png", "summary.md"} <= names
            png = c.get("/api/study/file/abc123def456/F1_fake.png")
            assert png.status_code == 200 and png.content.startswith(PNG_MAGIC)

            part = c.get("/api/study/results/part00000001").json()
            assert part["status"] == "incomplete" and part["results"] is None

    def test_study_file_traversal_and_unknown(self, workdir):
        self._fake_study(workdir)
        with TestClient(server_module.create_app(workdir)) as c:
            assert c.get("/api/study/file/abc123def456/..%2F..%2Fsecret").status_code == 404
            assert c.get("/api/study/file/abc123def456/nope.png").status_code == 404
            assert c.get("/api/study/results/..").status_code == 404
            assert c.get("/api/study/results/does-not-exist").status_code == 404

    def test_custom_options_and_spec_builder(self, client):
        opts = client.get("/api/study/custom_options").json()
        assert {p["id"] for p in opts["problems"]} >= {"cantilever_beam", "bridge_deck"}
        assert "MMA" in {m["id"] for m in opts["methods"]}
        meshes = {p["id"]: p["default_mesh_control"] for p in opts["problems"]}
        assert meshes["bridge_deck"] == 41  # corrected example; from freeto.study's quick spec
        assert client.post("/api/examples/bridge_deck/load").json()["prefill"]["volfrac"] == 0.35
        c = server_module.StudyCustomIn(problems=["cantilever_beam", "bridge_deck"],
                                        methods=["QUBO-sa (block)", "BESO-sort"],
                                        seeds=[0, 1], mesh_control=20, max_iter=7)
        spec = server_module.build_custom_study_spec(c)
        runs = spec["runs"]
        assert spec["name"] == "custom" and len(runs) == 2 * 3  # MMA baseline added
        assert all(r["mesh_control"] == 20 and r["max_iter"] == 7 and r["kind"] == "continuum" for r in runs)
        by = {(r["problem"], r["label"]): r for r in runs}
        assert by[("cantilever_beam", "QUBO-sa (block)")]["seeds"] == [0, 1]
        assert by[("cantilever_beam", "MMA")]["seeds"] == [0]
        assert by[("bridge_deck", "BESO-sort")]["seeds"] == [0]
        assert by[("cantilever_beam", "QUBO-sa (block)")]["options"] == {"backend": "sa", "hessian": "block"}
        with pytest.raises(ValueError):
            server_module.build_custom_study_spec(
                server_module.StudyCustomIn(problems=["cantilever_beam"], methods=["nope"]))

    def test_custom_study_validation_via_api(self, client):
        if not core_loader.STUDY_USABLE:
            pytest.skip("freeto.study not importable")
        r = client.post("/api/study/run", json={"custom": {"problems": ["cantilever_beam"], "methods": ["nope"]}})
        assert r.status_code == 400
        r = client.post("/api/study/run", json={})
        assert r.status_code == 400

    @pytest.mark.skipif(not core_loader.STUDY_USABLE, reason="freeto.study not importable")
    def test_tiny_custom_study_end_to_end(self, client):
        r = client.post("/api/study/run", json={
            "custom": {"problems": ["cantilever_beam"], "methods": ["MMA"], "mesh_control": 20,
                       "max_iter": 4, "include_baseline": False}, "figures": False})
        assert r.status_code == 200, r.text
        jid, sid = r.json()["job_id"], r.json()["study_id"]
        s = _wait(client, f"/api/study/status/{jid}", timeout=240)
        assert s["status"] == "done", s.get("error")
        assert s["study_id"] == sid
        rows = {x["id"]: x for x in client.get("/api/study/list").json()["studies"]}
        assert sid in rows and rows[sid]["n_records"] == 1 and rows[sid]["n_errors"] == 0
        by_dir = client.get(f"/api/study/results/{sid}").json()
        by_job = client.get(f"/api/study/results/{jid}").json()
        assert by_dir["results"]["records"][0]["run_id"] == by_job["results"]["records"][0]["run_id"]


# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_AUDIT, reason="freeto.audit not available yet")
class TestRealAudit:
    def test_real_audit_contract(self, client):
        jid, status = _tiny_job(client, "cantilever_beam", mesh_control=20, max_iter=40)
        r = client.get(f"/api/jobs/{jid}/audit")
        assert r.status_code == 200, r.text
        a = r.json()
        assert isinstance(a["ok"], bool)
        assert CHECK_NAMES <= set(a["checks"])
        for name, c in a["checks"].items():
            assert {"pass", "value", "detail"} <= set(c), name
        assert isinstance(a["n_components"], int) and a["n_components"] >= 1
        assert 0.0 <= a["floating_frac"] <= 1.0
        assert isinstance(a["components"], list) and a["components"]
        assert {"label", "n_elements", "grounded", "has_load", "bbox_mm"} <= set(a["components"][0])
        st = client.get(f"/api/jobs/{jid}").json()
        assert st["audit_ok"] == a["ok"] and st["n_components"] == a["n_components"]
        json.dumps(a)  # JSON-serialisable

    def test_real_audit_passes_on_tiny_mma_job(self, client):
        jid, _ = _tiny_job(client, "cantilever_beam", mesh_control=20, max_iter=40)
        a = client.get(f"/api/jobs/{jid}/audit").json()
        failed = {k: v for k, v in a["checks"].items() if not v["pass"]}
        assert a["ok"] is True, failed
        assert a["n_components"] == 1 and a["floating_frac"] < 0.01

    def test_real_audit_png_and_run(self, client):
        jid, _ = _tiny_job(client, "cantilever_beam", mesh_control=20, max_iter=40, audit=False)
        assert client.get(f"/api/jobs/{jid}/audit").status_code == 409
        r = client.post(f"/api/jobs/{jid}/audit/run")
        assert r.status_code == 200, r.text
        assert "checks" in r.json()
        png = client.get(f"/api/jobs/{jid}/audit.png")
        assert png.status_code == 200
        assert png.headers["content-type"] == "image/png" and png.content.startswith(PNG_MAGIC)
        assert (client.app.state.workdir / "jobs" / jid / "audit.png").is_file()
