"""Regression tests for the defects fixed in the 2026-10-06 code review
(BUG_REVIEW.md in the repository's parent folder)."""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from freeto import FreeTOError, example_config, run_freeto  # noqa: E402
from freeto.quantum import QUBOOptions  # noqa: E402
from freeto.quantum.qaoa import run_qaoa  # noqa: E402


# R1 -------------------------------------------------------------------------
@pytest.mark.parametrize("backend,bs", [("exact", 25), ("exact", 30), ("qaoa", 21)])
def test_qubo_block_size_above_backend_capacity_is_rejected(backend, bs):
    with pytest.raises(ValueError, match="block_size"):
        QUBOOptions(backend=backend, block_size=bs).validate()


@pytest.mark.parametrize("backend,bs", [("exact", 24), ("qaoa", 20), ("sa", 5000),
                                        ("auto", 5000), ("exact", None)])
def test_qubo_block_size_within_capacity_is_accepted(backend, bs):
    assert QUBOOptions(backend=backend, block_size=bs).validate()


def test_config_with_too_large_exact_block_fails_validation_not_the_run():
    cfg = example_config("cantilever_beam", mesh_control=24)
    cfg.optimizer = "QUBO"
    cfg.qubo = {"backend": "exact", "block_size": 30}
    with pytest.raises(FreeTOError, match="block_size"):
        cfg.validate()


# R2 -------------------------------------------------------------------------
def test_run_qaoa_constant_energy_reports_the_full_info_dict():
    general = run_qaoa(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([-1.0, 0.5]), 0.0,
                       p=2, shots=20, seed=0)["info"]
    r = run_qaoa(np.zeros((3, 3)), np.zeros(3), 2.5, p=2, shots=20, seed=0)
    info = r["info"]
    missing = sorted(set(general) - set(info))
    assert missing == []
    assert info["E_min"] == info["E_max"] == 2.5
    assert info["best_shot_optimal"] is True and info["approx_ratio"] == 1.0
    assert len(info["angles"]) == 4                     # 2 p angles (qiskit circuit needs them)
    assert r["samples"].shape == (20, 3) and r["energy"] == 2.5


# R3 -------------------------------------------------------------------------
def test_cli_reports_setup_errors_without_traceback(capsys):
    from freeto import cli
    # GE_bracket at MeshControl 20: the load region holds no grid node, which
    # is only detected inside run_freeto (after validate())
    rc = cli.main(["--example", "GE_bracket", "--mesh", "20", "--max-iter", "1", "--quiet"])
    err = capsys.readouterr().err
    assert rc == 2
    assert err.startswith("error: ") and "contains no grid nodes" in err
    assert "Traceback" not in err


# R4 -------------------------------------------------------------------------
@pytest.mark.parametrize("field,value", [
    ("rmin", math.inf), ("penal", math.inf), ("youngs_modulus", math.inf),
    ("tolx", math.nan), ("tol_thresh", math.nan), ("beta_init", 0.0), ("beta_init", -1.0),
    ("beta_step", -0.5), ("beta_max", 0.0), ("mma_move", 0.0), ("mma_move", 1.5),
    ("eval_crisp", 1.5), ("eval_crisp", 0.0), ("eval_beta", 0.0)])
def test_validate_rejects_non_finite_or_out_of_range_numbers(field, value):
    cfg = example_config("cantilever_beam", mesh_control=24)
    setattr(cfg, field, value)
    with pytest.raises(FreeTOError, match=field):
        cfg.validate()


@pytest.mark.parametrize("field,value", [
    ("tolx", 0.0), ("tolx", -1.0), ("beta_step", 0.0), ("beta_max", math.inf),
    ("mma_move", 1.0), ("eval_crisp", True), ("eval_crisp", False), ("eval_crisp", 0.25),
    ("eval_beta", 8.0)])
def test_validate_still_accepts_valid_extremes(field, value):
    cfg = example_config("cantilever_beam", mesh_control=24)
    setattr(cfg, field, value)
    assert cfg.validate()


def test_eval_crisp_false_means_off():
    cfg = example_config("cantilever_beam", mesh_control=24)
    cfg.max_iter, cfg.audit, cfg.eval_crisp = 2, False, False
    res = run_freeto(cfg, log=None)
    assert "crisp_compliance" not in res.extra and "crisp_target" not in res.extra


# R5 -------------------------------------------------------------------------
def test_job_snapshot_survives_history_keys_added_by_the_worker():
    from webapp.jobs import Job
    job = Job(id="x", label="x", config_dict={})

    class Racy(list):
        """Simulates the worker thread adding a history key while the
        snapshot copies the history (the GIL can switch between items)."""
        def __iter__(self):
            job.history.setdefault("qubo", []).append({"n_free": 1})
            return super().__iter__()

    job.history["compliance"] = Racy([1.0, 2.0])
    st = job.snapshot_status()              # raised "dictionary changed size" before
    assert st["history"]["compliance"] == [1.0, 2.0]


# R6 -------------------------------------------------------------------------
def test_truss_run_rejects_unknown_or_unavailable_backend_up_front():
    from fastapi.testclient import TestClient
    from webapp import core_loader
    import webapp.server as server_module
    if not (core_loader.TRUSS_USABLE and core_loader.QUANTUM_USABLE):
        pytest.skip("freeto.truss / freeto.quantum not importable")
    with TestClient(server_module.create_app(tempfile.mkdtemp())) as c:
        r = c.post("/api/truss/run", json={"benchmark_id": "gs_3x2", "method": "qubo",
                                           "backend": "no_such_backend"})
        assert r.status_code == 400 and "Unknown QUBO backend" in r.json()["detail"]
        unavailable = [b["name"] for b in c.get("/api/quantum/backends").json()["backends"]
                       if not b["available"]]
        if unavailable:
            r = c.post("/api/truss/run", json={"benchmark_id": "gs_3x2", "method": "qubo",
                                               "backend": unavailable[0]})
            assert r.status_code == 400 and "unavailable" in r.json()["detail"]
        # "auto" and methods that ignore the backend are still accepted
        r = c.post("/api/truss/run", json={"benchmark_id": "gs_3x2", "method": "exact",
                                           "backend": "no_such_backend"})
        assert r.status_code == 200


# R7 -------------------------------------------------------------------------
def test_refined_bc_map_drops_loads_on_inactive_coarse_nodes_instead_of_nan():
    from freeto.evaluate import _map_bcs
    nelx = nely = nelz = 2
    f = 2
    nnc = 27
    F = np.zeros((3 * nnc, 1))
    F[3 * 26 + 1, 0] = -5.0          # coarse corner node (2, 2, 2): outside the active part
    F[3 * 0 + 1, 0] = -1.0           # coarse node (0, 0, 0): active
    # active refined nodes: those of coarse element (0, 0, 0) only (R, C, P <= 2)
    ny = nx = nz = f * 2
    n = np.arange((ny + 1) * (nx + 1) * (nz + 1))
    R, C, P = n % (ny + 1), (n // (ny + 1)) % (nx + 1), n // ((ny + 1) * (nx + 1))
    act_nodes = n[(R <= 2) & (C <= 2) & (P <= 2)]
    Ff, _ = _map_bcs(F, np.array([], dtype=np.int64), nelx, nely, nelz, f, act_nodes, "interp")
    assert np.all(np.isfinite(Ff))
    assert Ff.sum() == pytest.approx(-1.0)   # the active node's load is conserved, the other dropped
