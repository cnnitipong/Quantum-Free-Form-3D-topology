"""FastAPI TestClient tests for the FreeTO-Python web app.

These exercise the whole backend surface (upload, examples, job lifecycle,
downloads, validation) against whichever core is active — the real
`freeto` package when it is importable, otherwise the built-in stub. Either
way the web app's own logic (routing, file registry, job queueing,
validation plumbing, STL/NPZ serving) is fully verified.
"""

from __future__ import annotations

import importlib
import io
import os
import struct
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("FREETO_WEBAPP_DIR", str(REPO_ROOT / ".pytest_freeto_web"))

from fastapi.testclient import TestClient  # noqa: E402

import webapp.server as server_module  # noqa: E402
from webapp.core_loader import CORE_SOURCE  # noqa: E402

EXAMPLES_STL_DIR = REPO_ROOT / "examples" / "STLs"


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("freeto_web_test")


@pytest.fixture(scope="module")
def client(workdir):
    app = server_module.create_app(workdir)
    with TestClient(app) as c:
        yield c


def _tiny_binary_stl_bytes(offset=(0, 0, 0)):
    """A minimal valid, watertight, non-degenerate binary STL: a unit
    tetrahedron (4 triangles, non-zero extent on all 3 axes), offset in
    space so several calls can produce distinguishable bboxes."""
    ox, oy, oz = offset
    p0 = (0 + ox, 0 + oy, 0 + oz)
    p1 = (1 + ox, 0 + oy, 0 + oz)
    p2 = (0 + ox, 1 + oy, 0 + oz)
    p3 = (0 + ox, 0 + oy, 1 + oz)
    # Outward-wound faces of a tetrahedron (winding doesn't matter for these
    # tests — only triangle count / bbox / watertightness are checked).
    faces = [
        (p0, p2, p1),
        (p0, p1, p3),
        (p1, p2, p3),
        (p2, p0, p3),
    ]
    buf = io.BytesIO()
    buf.write(b"\x00" * 80)
    buf.write(struct.pack("<I", len(faces)))
    rec = struct.Struct("<12fH")
    for a, b, c in faces:
        buf.write(rec.pack(0, 0, 0, *a, *b, *c, 0))
    return buf.getvalue()


def wait_for_terminal(client, job_id, timeout=30.0):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        last = resp.json()
        if last["status"] in ("done", "error", "stopped"):
            return last
        time.sleep(0.1)
    raise AssertionError(f"Job {job_id} did not finish in time; last={last}")


class TestHealth:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["core"]["source"] in ("real", "stub")


class TestUpload:
    def test_upload_single(self, client):
        data = _tiny_binary_stl_bytes()
        resp = client.post(
            "/api/upload",
            files={"files": ("cube.stl", data, "application/octet-stream")},
        )
        assert resp.status_code == 200, resp.text
        records = resp.json()
        assert len(records) == 1
        rec = records[0]
        assert rec["name"] == "cube.stl"
        assert rec["triangles"] == 4
        assert rec["bbox"]["min"] == [0.0, 0.0, 0.0]
        assert rec["bbox"]["max"] == [1.0, 1.0, 1.0]
        assert rec["watertight"] is True

        # the file should be servable back
        get_resp = client.get(f"/api/files/{rec['id']}")
        assert get_resp.status_code == 200
        assert get_resp.content == data

    def test_upload_flat_stl_rejected(self, client):
        """A degenerate (zero-thickness) STL is caught at upload time with a
        clear message, not silently registered as a 0-triangle file."""
        ox, oy, oz = 20, 20, 20
        p0, p1, p2 = (0 + ox, 0 + oy, 0 + oz), (1 + ox, 0 + oy, 0 + oz), (0 + ox, 1 + oy, 0 + oz)
        buf = io.BytesIO()
        buf.write(b"\x00" * 80)
        buf.write(struct.pack("<I", 1))
        buf.write(struct.Struct("<12fH").pack(0, 0, 1, *p0, *p1, *p2, 0))
        resp = client.post(
            "/api/upload",
            files={"files": ("flat.stl", buf.getvalue(), "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "flat" in resp.json()["detail"].lower() or "degenerate" in resp.json()["detail"].lower()

    def test_upload_garbage_rejected(self, client):
        resp = client.post(
            "/api/upload",
            files={"files": ("garbage.stl", b"this is not an stl file at all, just text", "text/plain")},
        )
        assert resp.status_code == 400

    def test_upload_multiple(self, client):
        d1 = _tiny_binary_stl_bytes((0, 0, 0))
        d2 = _tiny_binary_stl_bytes((5, 5, 5))
        resp = client.post(
            "/api/upload",
            files=[
                ("files", ("a.stl", d1, "application/octet-stream")),
                ("files", ("b.stl", d2, "application/octet-stream")),
            ],
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_upload_empty_file_rejected(self, client):
        resp = client.post(
            "/api/upload",
            files={"files": ("empty.stl", b"", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_unknown_file_id_404(self, client):
        resp = client.get("/api/files/does-not-exist")
        assert resp.status_code == 404


class TestExamples:
    def test_list_examples(self, client):
        resp = client.get("/api/examples")
        assert resp.status_code == 200
        names = {e["name"] for e in resp.json()}
        assert {"GE_bracket", "air_bracket", "hand", "quadcopter"} <= names

    def test_load_example_prefill_shape(self, client):
        resp = client.post("/api/examples/GE_bracket/load")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        prefill = data["prefill"]
        assert prefill["domain_id"]
        assert len(prefill["loads"]) == 2
        # GE example reuses the same force STL for both load cases
        assert prefill["loads"][0]["file_id"] == prefill["loads"][1]["file_id"]
        assert prefill["loads"][0]["fz"] == 1500.0
        assert prefill["loads"][1]["fy"] == -2000.0
        assert prefill["fixed_id"]
        assert data["files"]["domain"]["triangles"] > 0

    def test_unknown_example_404(self, client):
        resp = client.post("/api/examples/does_not_exist/load")
        assert resp.status_code == 404


class TestValidation:
    def test_missing_domain(self, client):
        resp = client.post("/api/jobs", json={"domain_id": None, "loads": []})
        assert resp.status_code == 400
        assert "domain" in resp.json()["detail"].lower()

    def test_missing_support(self, client):
        ex = client.post("/api/examples/GE_bracket/load").json()
        prefill = ex["prefill"]
        payload = {
            "domain_id": prefill["domain_id"],
            "loads": prefill["loads"],
            # no fixed_id / xfixed_id / yfixed_id / zfixed_id
        }
        resp = client.post("/api/jobs", json=payload)
        assert resp.status_code == 400
        assert "support" in resp.json()["detail"].lower()

    def test_all_zero_loads(self, client):
        ex = client.post("/api/examples/GE_bracket/load").json()
        prefill = ex["prefill"]
        payload = {
            "domain_id": prefill["domain_id"],
            "fixed_id": prefill["fixed_id"],
            "loads": [{"file_id": prefill["loads"][0]["file_id"], "fx": 0, "fy": 0, "fz": 0}],
        }
        resp = client.post("/api/jobs", json=payload)
        assert resp.status_code == 400
        assert "load" in resp.json()["detail"].lower()

    def test_no_loads_at_all(self, client):
        ex = client.post("/api/examples/GE_bracket/load").json()
        prefill = ex["prefill"]
        payload = {
            "domain_id": prefill["domain_id"],
            "fixed_id": prefill["fixed_id"],
            "loads": [],
        }
        resp = client.post("/api/jobs", json=payload)
        assert resp.status_code == 400


class TestJobLifecycle:
    def test_tiny_job_runs_to_completion(self, client):
        ex = client.post("/api/examples/GE_bracket/load").json()
        prefill = ex["prefill"]
        payload = {
            "label": "pytest-tiny",
            "domain_id": prefill["domain_id"],
            "loads": prefill["loads"],
            "fixed_id": prefill["fixed_id"],
            # NB: the real core needs enough mesh resolution for the (thin)
            # GE_force load region to contain at least one grid node; 12 is
            # fine for the stub but the real solver raises a clear
            # "contains no grid nodes" ValueError below ~30 for this example.
            "mesh_control": 30,
            "volfrac": 0.3,
            "youngs_modulus": 210e9,
            "max_iter": 3,
        }
        resp = client.post("/api/jobs", json=payload)
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]

        final = wait_for_terminal(client, job_id)
        assert final["status"] == "done"
        assert final["setup"] is not None
        assert final["setup"]["nelx"] > 0
        assert len(final["history"]["compliance"]) == final["last_iter"] == 3
        assert final["has_result"] is True

        stl_resp = client.get(f"/api/jobs/{job_id}/result.stl")
        assert stl_resp.status_code == 200
        assert stl_resp.content[:80] != b""
        ntri = struct.unpack("<I", stl_resp.content[80:84])[0]
        assert 84 + ntri * 50 == len(stl_resp.content)
        assert ntri > 0

        npz_resp = client.get(f"/api/jobs/{job_id}/result.npz")
        assert npz_resp.status_code == 200
        assert npz_resp.content[:2] == b"PK"  # zip magic (npz is a zip)

    def test_preview_available_during_or_after_run(self, client):
        ex = client.post("/api/examples/GE_bracket/load").json()
        prefill = ex["prefill"]
        payload = {
            "label": "pytest-preview",
            "domain_id": prefill["domain_id"],
            "loads": prefill["loads"],
            "fixed_id": prefill["fixed_id"],
            "mesh_control": 30,
            "max_iter": 4,
        }
        job_id = client.post("/api/jobs", json=payload).json()["job_id"]
        wait_for_terminal(client, job_id, timeout=60.0)
        resp = client.get(f"/api/jobs/{job_id}/preview.stl")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("model/stl") or len(resp.content) > 84

    def test_stop_job(self, client):
        ex = client.post("/api/examples/GE_bracket/load").json()
        prefill = ex["prefill"]
        payload = {
            "label": "pytest-stop",
            "domain_id": prefill["domain_id"],
            "loads": prefill["loads"],
            "fixed_id": prefill["fixed_id"],
            "mesh_control": 60,
            "max_iter": 500,
        }
        job_id = client.post("/api/jobs", json=payload).json()["job_id"]
        time.sleep(0.2)
        stop_resp = client.post(f"/api/jobs/{job_id}/stop")
        assert stop_resp.status_code == 200
        final = wait_for_terminal(client, job_id, timeout=15.0)
        assert final["status"] == "stopped"

    def test_job_list(self, client):
        resp = client.get("/api/jobs")
        assert resp.status_code == 200
        jobs = resp.json()
        assert isinstance(jobs, list)
        assert len(jobs) >= 1

    def test_unknown_job_404(self, client):
        assert client.get("/api/jobs/does-not-exist").status_code == 404
        assert client.post("/api/jobs/does-not-exist/stop").status_code == 404


class TestFrontendServed:
    def test_index_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "FreeTO" in resp.text

    def test_static_vendor_files_present(self, client):
        for path in (
            "/static/vendor/three/three.module.js",
            "/static/vendor/three/OrbitControls.js",
            "/static/vendor/three/STLLoader.js",
            "/static/vendor/chartjs/chart.umd.min.js",
            "/static/js/app.js",
            "/static/css/style.css",
        ):
            resp = client.get(path)
            assert resp.status_code == 200, path

    def test_static_js_mime_type(self, client):
        """B2 regression: browsers refuse to load an ES module (or resolve an
        importmap entry) served as text/plain. Some Windows machines have a
        stray `HKEY_CLASSES_ROOT\\.js` registry override that makes stdlib
        `mimetypes` return exactly that unless webapp/server.py forces the
        correct type at import time (see the `mimetypes.add_type` block)."""
        for path, allowed in (
            ("/static/js/app.js", ("text/javascript", "application/javascript")),
            ("/static/vendor/three/three.module.js", ("text/javascript", "application/javascript")),
            ("/static/css/style.css", ("text/css",)),
        ):
            resp = client.get(path)
            content_type = resp.headers["content-type"].split(";")[0].strip()
            assert content_type in allowed, f"{path} served as {content_type!r}"

    def test_windows_registry_style_override_is_defeated(self):
        """Simulate the actual failure mode: something (an installer, a
        stale HKCR entry) registers '.js' -> 'text/plain' in the stdlib
        `mimetypes` table *after* our module-level fix already ran. Reimport
        webapp.server and confirm it re-asserts the correct type rather than
        relying on import order / assuming it only ever runs once."""
        import mimetypes

        mimetypes.add_type("text/plain", ".js")  # simulate a bad registry entry
        assert mimetypes.guess_type("app.js")[0] == "text/plain"

        import importlib as _importlib

        import webapp.server as _server_module

        _importlib.reload(_server_module)
        assert mimetypes.guess_type("app.js")[0] == "text/javascript"


class TestWorkdirIsLazy:
    def test_workdir_not_created_until_used(self, tmp_path):
        workdir = tmp_path / "not_yet_created"
        assert not workdir.exists()
        app = server_module.create_app(workdir)
        # Constructing the app (as happens at import time for the module-
        # level `app = create_app()`) must not touch the filesystem.
        assert not workdir.exists()
        with TestClient(app) as c:
            assert not workdir.exists()
            data = _tiny_binary_stl_bytes()
            resp = c.post("/api/upload", files={"files": ("x.stl", data, "application/octet-stream")})
            assert resp.status_code == 200
            # Only now (an actual upload) should the workdir appear.
            assert workdir.exists()
            assert (workdir / "uploads").exists()


class TestCoreUnavailable:
    def test_job_creation_refused_when_core_unusable(self, client, monkeypatch):
        """No silent fallback to the stub: if the real core isn't usable,
        job creation must fail loudly (503), never silently run on a fake
        solver."""
        monkeypatch.setattr(server_module, "CORE_USABLE", False)
        ex = client.post("/api/examples/GE_bracket/load").json()
        prefill = ex["prefill"]
        payload = {
            "domain_id": prefill["domain_id"],
            "loads": prefill["loads"],
            "fixed_id": prefill["fixed_id"],
            "mesh_control": 30,
        }
        resp = client.post("/api/jobs", json=payload)
        assert resp.status_code == 503
        assert "core" in resp.json()["detail"].lower()

    def test_health_reports_usable_flag(self, client):
        resp = client.get("/api/health")
        core = resp.json()["core"]
        assert "usable" in core
        assert "import_error" in core


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
