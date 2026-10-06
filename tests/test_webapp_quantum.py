"""FastAPI TestClient tests for the QUANTUM / Truss / Study web app extension.

Everything here is written against docs/QUANTUM_API.md and feature-detects
the real modules exactly like webapp/core_loader.py does: a test that needs
freeto.quantum / freeto.truss / freeto.study / a working continuum
optimizer="QUBO" is skipped (not failed) when that piece isn't installed yet,
so this file stays green whether it runs against the fully-implemented
Opus core or an earlier snapshot of it. The web app's own logic (routing,
validation, job-kind dispatch, feature-detection/graceful-degradation) is
exercised unconditionally.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("FREETO_WEBAPP_DIR", str(REPO_ROOT / ".pytest_freeto_web_quantum"))

from fastapi.testclient import TestClient  # noqa: E402

import webapp.server as server_module  # noqa: E402
from webapp import core_loader  # noqa: E402


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("freeto_web_quantum_test")


@pytest.fixture(scope="module")
def client(workdir):
    app = server_module.create_app(workdir)
    with TestClient(app) as c:
        yield c


def wait_for_terminal(client, url, timeout=60.0, poll=0.2):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        resp = client.get(url)
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["status"] in ("done", "error", "stopped"):
            return last
        time.sleep(poll)
    raise AssertionError(f"{url} did not finish in time; last={last}")


# ---------------------------------------------------------------------------
# /api/quantum/backends — must always answer 200, real or feature-detected off
# ---------------------------------------------------------------------------
class TestQuantumBackendsEndpoint:
    def test_always_200_and_shape(self, client):
        resp = client.get("/api/quantum/backends")
        assert resp.status_code == 200
        data = resp.json()
        assert "quantum_available" in data
        assert "backends" in data
        assert "continuum_qubo_supported" in data
        assert isinstance(data["backends"], list)

    @pytest.mark.skipif(not core_loader.QUANTUM_USABLE, reason="freeto.quantum not installed")
    def test_backend_entries_shape_when_available(self, client):
        resp = client.get("/api/quantum/backends")
        data = resp.json()
        assert data["quantum_available"] is True
        names = {b["name"] for b in data["backends"]}
        # exact/sa/tabu/greedy/qaoa are documented as always available (no
        # optional SDK needed) — docs/QUANTUM_API.md §2.
        assert {"exact", "sa", "tabu", "greedy", "qaoa"} <= names
        for b in data["backends"]:
            for key in ("name", "available", "description", "needs_token", "kind", "reason"):
                assert key in b, f"backend entry missing {key!r}: {b}"
            if not b["available"]:
                assert b["reason"], f"unavailable backend {b['name']} has no hint/reason"

    def test_graceful_degradation_when_unavailable(self, client, monkeypatch):
        """Simulates freeto.quantum missing entirely, regardless of whether
        it's actually installed in this environment — the endpoint must
        still answer 200 with an empty, clearly-labelled backend list (the
        frontend uses this to hide the QUBO controls)."""
        monkeypatch.setattr(
            server_module, "quantum_backends_payload",
            lambda: {"quantum_available": False, "import_error": "simulated: not installed", "backends": []},
        )
        resp = client.get("/api/quantum/backends")
        assert resp.status_code == 200
        data = resp.json()
        assert data["quantum_available"] is False
        assert data["backends"] == []
        assert "not installed" in data["import_error"]


# ---------------------------------------------------------------------------
# /api/truss/* — same always-200-for-listing, 503-for-actions-when-missing pattern
# ---------------------------------------------------------------------------
class TestTrussEndpoints:
    def test_benchmarks_always_200_and_shape(self, client):
        resp = client.get("/api/truss/benchmarks")
        assert resp.status_code == 200
        data = resp.json()
        assert "truss_available" in data
        assert "benchmarks" in data
        assert isinstance(data["benchmarks"], list)

    @pytest.mark.skipif(core_loader.TRUSS_USABLE, reason="only checks the not-installed path")
    def test_run_refused_with_503_when_truss_unavailable(self, client):
        resp = client.post("/api/truss/run", json={"benchmark_id": "ten_bar", "method": "exact"})
        assert resp.status_code == 503
        assert "truss" in resp.json()["detail"].lower()

    @pytest.mark.skipif(core_loader.TRUSS_USABLE, reason="only checks the not-installed path")
    def test_benchmark_detail_404_message_when_unavailable(self, client):
        resp = client.get("/api/truss/benchmarks/ten_bar")
        assert resp.status_code == 503

    @pytest.mark.skipif(not core_loader.TRUSS_USABLE, reason="freeto.truss not installed")
    def test_list_benchmarks_includes_documented_ids(self, client):
        resp = client.get("/api/truss/benchmarks")
        data = resp.json()
        assert data["truss_available"] is True
        ids = {b["id"] for b in data["benchmarks"]}
        # docs/QUANTUM_API.md §3: "ids: ten_bar (T1), gs_4x2 (T2), gs_3x2
        # (T2s), tower3d (T3), gs_9x3 (T4), column3d (T5)"
        assert {"ten_bar", "gs_4x2", "gs_3x2", "tower3d", "gs_9x3", "column3d"} <= ids
        assert set(data["methods"]) >= {"exact", "qubo", "sort", "oc", "oc_round", "oc_qubo"}

    @pytest.mark.skipif(not core_loader.TRUSS_USABLE, reason="freeto.truss not installed")
    def test_benchmark_detail_has_geometry(self, client):
        resp = client.get("/api/truss/benchmarks/gs_3x2")
        assert resp.status_code == 200, resp.text
        prob = resp.json()["problem"]
        assert prob["dim"] == 2
        assert len(prob["nodes"]) > 0
        assert len(prob["bars"]) > 0

    @pytest.mark.skipif(not core_loader.TRUSS_USABLE, reason="freeto.truss not installed")
    def test_unknown_benchmark_400(self, client):
        resp = client.post("/api/truss/run", json={"benchmark_id": "does_not_exist", "method": "exact"})
        assert resp.status_code == 400

    @pytest.mark.skipif(not core_loader.TRUSS_USABLE, reason="freeto.truss not installed")
    def test_unknown_method_400(self, client):
        resp = client.post("/api/truss/run", json={"benchmark_id": "gs_3x2", "method": "not_a_method"})
        assert resp.status_code == 400

    @pytest.mark.skipif(not core_loader.TRUSS_USABLE, reason="freeto.truss not installed")
    def test_exact_run_on_smallest_benchmark(self, client):
        """T2s (gs_3x2, 12 bars) is small enough for exact enumeration and is
        the benchmark docs/QUANTUM_DESIGN.md pins a known optimum for."""
        resp = client.post("/api/truss/run", json={"benchmark_id": "gs_3x2", "method": "exact"})
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]
        final = wait_for_terminal(client, f"/api/jobs/{job_id}", timeout=120.0)
        assert final["status"] == "done", final.get("error")
        assert final["kind"] == "truss"

        result = client.get(f"/api/jobs/{job_id}/truss_result")
        assert result.status_code == 200, result.text
        data = result.json()
        # the finished job reports what it was *submitted* with, so the UI can
        # label result rows without trusting its (mutable) form controls
        assert final["spec"]["benchmark_id"] == "gs_3x2"
        assert final["spec"]["method"] == "exact"
        assert data["method"] == "exact"
        assert data["compliance"] > 0
        assert data["feasible"] is True
        assert "on" in data and len(data["on"]) == len(data["bars"])

    @pytest.mark.skipif(not core_loader.TRUSS_USABLE, reason="freeto.truss not installed")
    def test_exact_refused_on_too_large_benchmark(self, client):
        benches = {b["id"]: b for b in client.get("/api/truss/benchmarks").json()["benchmarks"]}
        # T4 (gs_9x3, 116 bars since v2; 118 in docs/QUANTUM_DESIGN.md §D) is documented as
        # having no exact reference — exact_available must be False for it.
        too_big = next((bid for bid, b in benches.items() if not b.get("exact_available", True)), None)
        if too_big is None:
            pytest.skip("no benchmark in this build is flagged exact_available=False")
        resp = client.post("/api/truss/run", json={"benchmark_id": too_big, "method": "exact"})
        assert resp.status_code == 400
        assert "exact" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# /api/study/* — same pattern
# ---------------------------------------------------------------------------
class TestStudyEndpoints:
    def test_run_refused_without_suite_or_spec(self, client):
        if not core_loader.STUDY_USABLE:
            pytest.skip("freeto.study not installed (checked separately below)")
        resp = client.post("/api/study/run", json={})
        assert resp.status_code == 400

    @pytest.mark.skipif(core_loader.STUDY_USABLE, reason="only checks the not-installed path")
    def test_run_refused_with_503_when_study_unavailable(self, client):
        resp = client.post("/api/study/run", json={"suite": "smoke"})
        assert resp.status_code == 503
        assert "study" in resp.json()["detail"].lower()

    def test_unknown_job_id_status_404(self, client):
        assert client.get("/api/study/status/does-not-exist").status_code == 404
        assert client.get("/api/study/results/does-not-exist").status_code == 404

    @pytest.mark.skipif(not core_loader.STUDY_USABLE, reason="freeto.study not installed")
    def test_smoke_suite_runs_to_completion(self, client):
        """docs/QUANTUM_API.md §4: 'python -m freeto.study --suite smoke'
        (T2s + D1 at MC16, exact/sa) is documented to finish in well under
        2 minutes; give it generous headroom in CI."""
        resp = client.post("/api/study/run", json={"suite": "smoke"})
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]
        final = wait_for_terminal(client, f"/api/study/status/{job_id}", timeout=240.0)
        assert final["status"] == "done", final.get("error")

        results = client.get(f"/api/study/results/{job_id}")
        assert results.status_code == 200, results.text
        data = results.json()
        assert data["results"] is not None
        assert isinstance(data["files"], list)
        # results.json/csv should have been written by the study runner.
        names = {f["name"] for f in data["files"]}
        assert "results.json" in names or "results.csv" in names
        # fix-round record fields the Study table renders (crisp vs native)
        recs = data["results"]["records"]
        cont = [r for r in recs if r.get("kind") == "continuum" and not r.get("error")]
        assert cont and all("crisp_compliance" in r and "volfrac_target" in r for r in cont)
        # the log never prints Python's "None" (gap=None / "F2 None - starting")
        log = client.get(f"/api/jobs/{job_id}").json()["log_lines"]
        assert not any("gap=None" in line or " None " in line for line in log), log
        # run positions are 1-based "i/n"
        assert any(line.startswith("[1/") for line in log)


# ---------------------------------------------------------------------------
# Continuum optimizer="QUBO" — through the ordinary /api/jobs pipeline
# ---------------------------------------------------------------------------
class TestContinuumQubo:
    def test_examples_expose_category(self, client):
        """Deliverable #2: the examples dropdown groups by `category`
        (paper/beam/truss-like/advanced), already present in freeto/examples.py."""
        resp = client.get("/api/examples")
        assert resp.status_code == 200
        cats = {e["category"] for e in resp.json()}
        assert cats  # non-empty
        assert cats <= {"paper", "beam", "truss-like", "advanced"}

    @pytest.mark.skipif(core_loader.CORE_SUPPORTS_QUBO, reason="only checks the not-yet-wired path")
    def test_qubo_optimizer_refused_with_clear_400_when_core_not_wired(self, client):
        """The freeto.quantum backends can exist before freeto/core.py's
        FreeTOConfig grows the `qubo` field (see docs/QUANTUM_API.md §1) —
        this must be a clear validation error, not a crash."""
        ex = client.post("/api/examples/GE_bracket/load").json()
        prefill = ex["prefill"]
        payload = {
            "domain_id": prefill["domain_id"],
            "loads": prefill["loads"],
            "fixed_id": prefill["fixed_id"],
            "mesh_control": 30,
            "max_iter": 2,
            "optimizer": "QUBO",
        }
        resp = client.post("/api/jobs", json=payload)
        assert resp.status_code == 400
        assert "qubo" in resp.json()["detail"].lower()

    @pytest.mark.skipif(not core_loader.CORE_SUPPORTS_QUBO, reason="freeto core does not support optimizer='QUBO' yet")
    def test_qubo_job_runs_and_reports_qubo_stats(self, client):
        ex = client.post("/api/examples/GE_bracket/load").json()
        prefill = ex["prefill"]
        payload = {
            "label": "pytest-qubo",
            "domain_id": prefill["domain_id"],
            "loads": prefill["loads"],
            "fixed_id": prefill["fixed_id"],
            "mesh_control": 30,
            "max_iter": 3,
            "optimizer": "QUBO",
            "qubo_backend": "sa",
            # fix-round QUBOOptions fields + common crisp evaluation
            "qubo_gamma": 0.0,
            "qubo_interp": "beso",
            "qubo_move_limit": 0.25,
            "qubo_move_limit_min": 0.02,
            "qubo_protect_loads": True,
            "qubo_guard": True,
            "qubo_guard_tol": 0.5,
            "qubo_guard_tol_target": 0.25,
            "qubo_max_rejects": 4,
            "qubo_qaoa_polish": True,
            "eval_crisp": True,
        }
        resp = client.post("/api/jobs", json=payload)
        assert resp.status_code == 200, resp.text
        job_id = resp.json()["job_id"]
        final = wait_for_terminal(client, f"/api/jobs/{job_id}", timeout=120.0)
        assert final["status"] == "done", final.get("error")
        # eval_crisp -> status.result_info carries the crisp metrics (None = inf)
        assert final["result_info"] is not None and "crisp_compliance" in final["result_info"]
        assert final["spec"] is None  # continuum jobs have no truss spec
        # docs/QUANTUM_API.md: every QUBO-mode `iter` callback dict gains a
        # "qubo" sub-dict with n_free/n_blocks/solver_time/... — the job's
        # continuum callback forwards this into history["qubo"].
        assert "qubo" in final["history"]
        assert len(final["history"]["qubo"]) >= 1
        q0 = final["history"]["qubo"][0]
        for key in ("n_free", "solver_time"):
            assert key in q0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
