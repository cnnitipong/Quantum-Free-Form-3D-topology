"""Public (hosted) mode limits (webapp/public.py) and sub-path safety of the
frontend.  Unit / TestClient tests only: no job here runs the optimiser (the
job manager's worker is replaced by a stub that waits on an event)."""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi.testclient import TestClient  # noqa: E402

import webapp.server as server_module  # noqa: E402
from webapp.jobs import JobManager  # noqa: E402
from webapp.public import (  # noqa: E402
    LimitError,
    PublicConfig,
    check_capacity,
    check_job_request,
    check_truss_request,
    client_ip,
)

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "webapp" / "static"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_public_mode_is_off_by_default():
    cfg = PublicConfig.from_env({})
    assert cfg.enabled is False
    assert cfg.n_workers == 1 and cfg.max_job_seconds is None


def test_defaults_and_env_overrides():
    cfg = PublicConfig.from_env({"QFF3D_PUBLIC": "1"})
    assert cfg.enabled
    assert (cfg.max_running, cfg.max_queued, cfg.max_jobs_per_client) == (1, 8, 1)
    assert (cfg.max_mesh_control, cfg.max_iter, cfg.max_qaoa_block) == (50, 300, 12)
    assert (cfg.max_upload_mb, cfg.max_upload_files) == (20.0, 6)
    assert cfg.max_job_seconds == 20 * 60 and cfg.ttl_seconds == 2 * 3600
    assert "deleted after 2 hours" in cfg.notice()
    assert cfg.to_dict()["study_enabled"] is False

    cfg = PublicConfig.from_env({"QFF3D_PUBLIC": "true", "QFF3D_MAX_QUEUED": "3",
                                 "QFF3D_MAX_MESH_CONTROL": "40", "QFF3D_TTL_MINUTES": "30",
                                 "QFF3D_MAX_UPLOAD_MB": "4.5", "QFF3D_DIRECT_URL": "https://x.hf.space/"})
    assert cfg.max_queued == 3 and cfg.max_mesh_control == 40 and cfg.max_upload_mb == 4.5
    assert cfg.direct_url == "https://x.hf.space/" and "30 minutes" in cfg.notice()
    assert PublicConfig.from_env({"QFF3D_PUBLIC": "0"}).enabled is False
    with pytest.raises(ValueError):
        PublicConfig.from_env({"QFF3D_MAX_QUEUED": "lots"})
    with pytest.raises(ValueError):
        PublicConfig.from_env({"QFF3D_MAX_RUNNING": "0"})


def test_client_ip_prefers_first_forwarded_hop():
    assert client_ip({"x-forwarded-for": "203.0.113.7, 10.0.0.1"}, "10.0.0.2") == "203.0.113.7"
    assert client_ip({}, "10.0.0.2") == "10.0.0.2"
    assert client_ip({"x-forwarded-for": " "}, None) == "unknown"


# ---------------------------------------------------------------------------
# request checks
# ---------------------------------------------------------------------------

PUB = PublicConfig(enabled=True)


def _req(**kw):
    base = dict(mesh_control=36, max_iter=300, optimizer="QUBO", qubo_backend="sa", qubo_block_size=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_paper_settings_fit_inside_the_limits():
    for mc in (24, 30, 36, 41, 46, 50):
        check_job_request(PUB, _req(mesh_control=mc))
    check_job_request(PUB, _req(qubo_backend="qaoa", qubo_block_size=8))


@pytest.mark.parametrize("kw, words", [
    (dict(mesh_control=51), "mesh_control = 51"),
    (dict(max_iter=301), "max_iter = 301"),
    (dict(qubo_backend="qaoa", qubo_block_size=13), "block size 13"),
    (dict(qubo_backend="qiskit_aer", qubo_block_size=None), "at most 12"),
])
def test_requests_above_a_limit_are_rejected(kw, words):
    with pytest.raises(LimitError) as ei:
        check_job_request(PUB, _req(**kw))
    assert words in ei.value.message and ei.value.status == 400


def test_limits_inactive_locally():
    check_job_request(PublicConfig(), _req(mesh_control=200, max_iter=5000,
                                           qubo_backend="qaoa", qubo_block_size=20))
    check_truss_request(PublicConfig(), SimpleNamespace(backend="qaoa", options={"block_size": 20}))


def test_truss_qaoa_block_cap():
    with pytest.raises(LimitError):
        check_truss_request(PUB, SimpleNamespace(backend="qaoa", options={"block_size": 16}))
    check_truss_request(PUB, SimpleNamespace(backend="qaoa", options={"block_size": 8}))
    check_truss_request(PUB, SimpleNamespace(backend="sa", options={"block_size": 64}))


class _FakeManager:
    def __init__(self, active, queued):
        self.active, self.queued = active, queued

    def active_count(self, client=None):
        return self.active.get(client, 0) if client is not None else sum(self.active.values())

    def queued_count(self):
        return self.queued


def test_capacity_per_client_and_queue():
    check_capacity(PUB, _FakeManager({}, 0), "a")
    with pytest.raises(LimitError) as ei:
        check_capacity(PUB, _FakeManager({"a": 1}, 0), "a")
    assert ei.value.status == 429
    with pytest.raises(LimitError) as ei:
        check_capacity(PUB, _FakeManager({}, 8), "b")
    assert ei.value.status == 429 and "busy" in ei.value.message
    check_capacity(PublicConfig(), _FakeManager({"a": 5}, 99), "a")


# ---------------------------------------------------------------------------
# job manager: wall time, counts, retention
# ---------------------------------------------------------------------------

def _wait(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_wall_time_limit_stops_the_job_cleanly(tmp_path):
    mgr = JobManager(tmp_path / "jobs", max_wall_s=0.3)

    def fake_continuum(job):
        assert job.stop_event.wait(10), "the wall-time timer never fired"
        job.status = "stopped"
        job.stage = "stopped"

    mgr._run_continuum = fake_continuum
    job = mgr.submit("t", {}, cfg=None)
    assert _wait(lambda: job.status == "stopped")
    assert _wait(lambda: job.finished_at is not None)
    assert job.extra.get("wall_time_exceeded") is True
    assert "wall-time limit" in (job.error or "")
    assert any("Wall-time limit" in line for line in job.log_lines)


def test_no_wall_time_limit_locally(tmp_path):
    mgr = JobManager(tmp_path / "jobs")
    assert mgr.max_wall_s is None
    done = threading.Event()

    def fake_continuum(job):
        time.sleep(0.2)
        job.status = "done"
        done.set()

    mgr._run_continuum = fake_continuum
    job = mgr.submit("t", {}, cfg=None)
    assert done.wait(5) and _wait(lambda: job.finished_at is not None)
    assert job.status == "done" and not job.stop_event.is_set()


def test_purge_expired_deletes_finished_jobs_only(tmp_path):
    mgr = JobManager(tmp_path / "jobs")
    release = threading.Event()

    def fake_continuum(job):
        if job.label == "block":
            release.wait(10)
        job.status = "done"

    mgr._run_continuum = fake_continuum
    old = mgr.submit("old", {}, cfg=None)
    assert _wait(lambda: old.status == "done" and old.finished_at is not None)
    (tmp_path / "jobs" / old.id).mkdir(parents=True)
    (tmp_path / "jobs" / old.id / "result.stl").write_bytes(b"x")
    stray = tmp_path / "jobs" / "leftover0000"
    stray.mkdir()
    os.utime(stray, (time.time() - 10_000, time.time() - 10_000))
    running = mgr.submit("block", {}, cfg=None)
    queued = mgr.submit("queued", {}, cfg=None)
    assert _wait(lambda: running.status == "running")
    assert mgr.active_count() == 2 and mgr.queued_count() == 1 and mgr.running_count() == 1
    assert mgr.jobs_ahead(queued.id) == 1 and mgr.jobs_ahead(running.id) is None

    deleted = mgr.purge_expired(ttl_s=3600, now=time.time() + 7200)
    assert old.id in deleted and "leftover0000" in deleted
    assert mgr.get(old.id) is None and not (tmp_path / "jobs" / old.id).exists()
    assert not stray.exists()
    assert mgr.get(running.id) is not None and mgr.get(queued.id) is not None
    assert mgr.delete_job(running.id) is False
    release.set()
    assert _wait(lambda: queued.status == "done")


# ---------------------------------------------------------------------------
# the API in public mode
# ---------------------------------------------------------------------------

def _public_app(tmp_path, **kw):
    cfg = PublicConfig(enabled=True, **kw)
    app = server_module.create_app(tmp_path / "work", public=cfg)
    return app


def _block_worker(app):
    """Replace the worker's job body with one that waits for `release`."""
    release = threading.Event()
    mgr = app.state.manager

    def fake_run(job):
        job.status = "running"
        release.wait(20)
        job.status = "stopped"
        job.finished_at = time.time()

    mgr._run_job = fake_run
    return release


def test_health_reports_public_limits(tmp_path):
    with TestClient(_public_app(tmp_path)) as c:
        h = c.get("/api/health").json()
    assert h["public"]["enabled"] is True and h["public"]["max_mesh_control"] == 50
    assert "Shared demo server" in h["public"]["notice"] and h["workdir"] is None


def test_health_local_mode(tmp_path):
    with TestClient(server_module.create_app(tmp_path, public=PublicConfig())) as c:
        h = c.get("/api/health").json()
    assert h["public"]["enabled"] is False and h["workdir"]


def test_study_runs_disabled_in_public_mode(tmp_path):
    with TestClient(_public_app(tmp_path)) as c:
        r = c.post("/api/study/run", json={"suite": "smoke"})
    assert r.status_code == 403 and "local version" in r.json()["detail"]


def test_job_limits_through_the_api(tmp_path):
    app = _public_app(tmp_path, max_queued=1)
    release = _block_worker(app)
    try:
        with TestClient(app) as c:
            r = c.post("/api/examples/cantilever_beam/load")
            assert r.status_code == 200
            cantilever_prefill = r.json()["prefill"]
            too_fine = dict(cantilever_prefill, mesh_control=60)
            r = c.post("/api/jobs", json=too_fine)
            assert r.status_code == 400 and "mesh_control = 60" in r.json()["detail"]
            r = c.post("/api/jobs", json=dict(cantilever_prefill, max_iter=1000))
            assert r.status_code == 400 and "max_iter" in r.json()["detail"]

            a = {"X-Forwarded-For": "198.51.100.1"}
            r1 = c.post("/api/jobs", json=cantilever_prefill, headers=a)
            assert r1.status_code == 200, r1.text
            assert _wait(lambda: app.state.manager.running_count() == 1)
            r = c.post("/api/jobs", json=cantilever_prefill, headers=a)
            assert r.status_code == 429 and "already have 1 job" in r.json()["detail"]

            r2 = c.post("/api/jobs", json=cantilever_prefill, headers={"X-Forwarded-For": "198.51.100.2"})
            assert r2.status_code == 200
            assert r2.json()["jobs_ahead"] == 1 and r2.json()["queue_position"] == 1
            st = c.get(f"/api/jobs/{r2.json()['job_id']}").json()
            assert st["status"] == "queued" and st["jobs_ahead"] == 1

            r = c.post("/api/jobs", json=cantilever_prefill, headers={"X-Forwarded-For": "198.51.100.3"})
            assert r.status_code == 429 and "busy" in r.json()["detail"]

            # each visitor only lists their own jobs
            mine = c.get("/api/jobs", headers=a).json()
            assert [j["id"] for j in mine] == [r1.json()["job_id"]]
    finally:
        release.set()


def test_example_with_a_finer_mesh_is_prefilled_at_the_cap(tmp_path):
    with TestClient(_public_app(tmp_path)) as c:
        body = c.post("/api/examples/air_bracket/load").json()
        assert body["prefill"]["mesh_control"] == 50 and "Mesh reduced from 90" in body["limits_note"]
        paper = c.post("/api/examples/cantilever_beam/load").json()
        assert paper["prefill"]["mesh_control"] == 36 and "limits_note" not in paper


def _stl_bytes():
    return (REPO / "examples" / "STLs" / "lever_force1.STL").read_bytes()


def test_upload_limits(tmp_path):
    data = _stl_bytes()
    with TestClient(_public_app(tmp_path, max_upload_mb=len(data) / 1024 / 1024 / 2,
                                max_upload_files=2)) as c:
        r = c.post("/api/upload", files=[("files", ("a.stl", data, "model/stl"))])
        assert r.status_code == 413 and "accepts STL files up to" in r.json()["detail"]
        small = b"solid t\nendsolid t\n"
        r = c.post("/api/upload", files=[("files", (f"{i}.stl", small, "model/stl")) for i in range(3)])
        assert r.status_code == 413 and "At most 2 files" in r.json()["detail"]
        # a body far above files x size is refused before it is parsed
        big = b"0" * (3 * 1024 * 1024)
        r = c.post("/api/upload", files=[("files", ("big.stl", big, "model/stl"))])
        assert r.status_code == 413 and "per request" in r.json()["detail"]


def test_uploads_are_deleted_after_the_ttl(tmp_path):
    app = _public_app(tmp_path)
    with TestClient(app) as c:
        rec = c.post("/api/upload", files=[("files", ("lever.stl", _stl_bytes(), "model/stl"))]).json()[0]
        c.post("/api/examples/cantilever_beam/load")
        reg = app.state.registry
        path = Path(reg.get(rec["id"]).path)
        assert path.is_file()
        assert reg.purge_expired(3600, keep_ids={rec["id"]}, now=time.time() + 7200) == []
        assert reg.purge_expired(3600, now=time.time() + 7200) == [rec["id"]]
        assert not path.exists() and reg.get(rec["id"]) is None
        # bundled example files are never deleted
        assert any(r.source == "example" for r in reg._records.values())
        assert c.get(f"/api/files/{rec['id']}").status_code == 404
        assert set(app.state.cleanup()) == {"jobs", "files"}


def test_data_stays_inside_the_workdir(tmp_path):
    app = _public_app(tmp_path)
    with TestClient(app) as c:
        rec = c.post("/api/upload", files=[("files", ("../../evil name.stl", _stl_bytes(),
                                                       "model/stl"))]).json()[0]
    p = Path(app.state.registry.get(rec["id"]).path).resolve()
    assert p.parent == (tmp_path / "work" / "uploads").resolve()


# ---------------------------------------------------------------------------
# sub-path safety
# ---------------------------------------------------------------------------

def test_frontend_uses_no_root_absolute_urls():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert not re.search(r'(?:src|href)="/(?!/)', html), "absolute src/href in index.html"
    imap = re.search(r'<script type="importmap">(.*?)</script>', html, re.S).group(1)
    assert '"/' not in imap.replace('"imports"', "")
    css = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
    assert not re.search(r"url\(\s*['\"]?/(?!/)", css)
    js = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    code = "\n".join(line for line in js.splitlines() if not line.lstrip().startswith("//"))
    assert not re.search(r"""["'`]/(api|static)/""", code), "root-absolute URL in app.js"
    assert "function apiUrl(" in js


def test_index_served_with_relative_assets_and_forwarded_prefix(tmp_path):
    with TestClient(server_module.create_app(tmp_path, public=PublicConfig())) as c:
        r = c.get("/")
        assert r.status_code == 200 and "./static/js/app.js" in r.text
        assert c.get("/static/js/app.js").status_code == 200
        docs = c.get("/docs", headers={"X-Forwarded-Prefix": "/qff3d"})
        assert docs.status_code == 200 and "/qff3d/openapi.json" in docs.text
        assert c.get("/api/health", headers={"X-Forwarded-Prefix": "/qff3d"}).status_code == 200
