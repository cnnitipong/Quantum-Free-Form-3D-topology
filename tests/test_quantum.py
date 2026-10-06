"""Tests for the QUBO / quantum extension (freeto.quantum, optimizer="QUBO",
freeto.study)."""
from __future__ import annotations

import itertools
import json
import math
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from freeto import FreeTOConfig, FreeTOError, run_freeto, example_config        # noqa: E402
from freeto.quantum import (QUBOOptions, QuantumBackendUnavailable, available_backends,  # noqa
                            energy, from_ising, normalize_qubo, solve_qubo, to_ising)
from freeto.quantum import backends as qb                                         # noqa: E402
from freeto.quantum.anneal import simulated_annealing, greedy_coloring           # noqa: E402
from freeto.quantum.blockqubo import solve_volume_model                           # noqa: E402
from freeto.quantum.exact import bits_of, exact_solve, spectrum                   # noqa: E402
from freeto.quantum.qaoa import apply_mixer, qaoa_state, run_qaoa                 # noqa: E402
from freeto.quantum.qubo import to_bqm_dict                                       # noqa: E402


def rand_qubo(n, rng, dense=True):
    Q = rng.normal(size=(n, n))
    Q = 0.5 * (Q + Q.T)
    np.fill_diagonal(Q, 0.0)
    return Q, rng.normal(size=n)


def brute(Q, h, c=0.0):
    n = h.size
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    E = np.einsum("ij,jk,ik->i", X, Q, X) + X @ h + c
    return X, E


# ---------------------------------------------------------------------------
# QUBO containers
# ---------------------------------------------------------------------------
def test_normalize_energy_ising_bqm_roundtrips():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(7, 7))                  # asymmetric with diagonal
    h = rng.normal(size=7)
    Q, h2, c = normalize_qubo(A, h, 0.5)
    assert np.allclose(Q, Q.T) and np.allclose(np.diag(Q), 0)
    X = (rng.random((20, 7)) < 0.5).astype(float)
    E_ref = np.einsum("ij,jk,ik->i", X, A, X) + X @ h + 0.5
    assert np.allclose(energy(Q, h2, c, X), E_ref)
    Qs, hs, cs = normalize_qubo(sp.csr_matrix(A), h, 0.5)
    assert np.allclose(energy(Qs, hs, cs, X), E_ref)
    J, b, ci = to_ising(Q, h2, c)
    Z = 1 - 2 * X
    assert np.allclose(np.einsum("ij,jk,ik->i", Z, J, Z) + Z @ b + ci, E_ref)
    Q3, h3, c3 = from_ising(J, b, ci)
    assert np.allclose(Q3, Q) and np.allclose(h3, h2) and np.isclose(c3, c)
    lin, quad, off = to_bqm_dict(Q, h2, c)
    E_bqm = [off + sum(lin[i] * x[i] for i in lin) + sum(v * x[i] * x[j]
                                                          for (i, j), v in quad.items())
             for x in X]
    assert np.allclose(E_bqm, E_ref)


@pytest.mark.parametrize("n", [5, 10, 17])
def test_exact_equals_bruteforce(n):
    rng = np.random.default_rng(n)
    Q, h = rand_qubo(n, rng)
    E = spectrum(Q, h, 0.25)
    idx = np.arange(2 ** n)
    assert np.allclose(E, energy(Q, h, 0.25, bits_of(idx, n)))
    x, Emin, info = exact_solve(Q, h, 0.25)
    assert Emin == pytest.approx(E.min()) and energy(Q, h, 0.25, x) == pytest.approx(Emin)
    if n <= 10:
        _, Eb = brute(Q, h, 0.25)
        assert Emin == pytest.approx(Eb.min())


# ---------------------------------------------------------------------------
# heuristics vs exact
# ---------------------------------------------------------------------------
def test_sa_tabu_greedy_find_exact_optimum_on_random_qubos():
    rng = np.random.default_rng(42)
    hits = {"sa": 0, "tabu": 0, "greedy": 0}
    N = 24
    for k in range(N):
        n = 10 + (k % 5)                          # n = 10 .. 14
        Q, h = rand_qubo(n, rng)
        ex = solve_qubo(Q, h, backend="exact")
        for be in hits:
            r = solve_qubo(Q, h, backend=be, seed=k, num_reads=16 if be == "greedy" else None)
            assert r.energy >= ex.energy - 1e-9
            assert energy(*normalize_qubo(Q, h, 0.0), r.x) == pytest.approx(r.energy)
            hits[be] += r.energy <= ex.energy + 1e-9
    assert hits["tabu"] == N
    assert hits["sa"] >= 0.95 * N
    assert hits["greedy"] >= 0.6 * N


def test_sa_sparse_grid_colored_path():
    # 3-D grid couplings (n = 216): chromatic SA vs best of several runs
    nx = 6
    idx = np.arange(nx ** 3).reshape(nx, nx, nx)
    rng = np.random.default_rng(3)
    rows, cols = [], []
    for ax in range(3):
        a = np.take(idx, np.arange(nx - 1), axis=ax).ravel()
        b = np.take(idx, np.arange(1, nx), axis=ax).ravel()
        rows += [a]
        cols += [b]
    r, c = np.concatenate(rows), np.concatenate(cols)
    w = rng.normal(size=r.size)
    Q = sp.coo_matrix((np.concatenate([w, w]), (np.concatenate([r, c]),
                                                np.concatenate([c, r]))),
                      shape=(nx ** 3,) * 2).tocsr()
    h = rng.normal(size=nx ** 3)
    col = greedy_coloring(Q)
    assert np.all(col[r] != col[c]) and col.max() + 1 <= 7
    out = simulated_annealing(Q, h, num_reads=16, seed=0)
    assert out["info"]["path"] == "colored"
    best = min(simulated_annealing(Q, h, num_reads=16, seed=s)["energies"].min()
               for s in range(1, 6))
    Ebest = out["energies"].min()
    assert Ebest <= best + 0.01 * abs(best)
    assert np.allclose(energy(Q, h, 0.0, out["samples"]), out["energies"])


def test_sa_batched_linear_terms():
    rng = np.random.default_rng(5)
    Q, h = rand_qubo(9, rng)
    H = np.stack([h, h + 3.0, h - 3.0])
    out = simulated_annealing(Q, H, num_reads=24, seed=1)
    assert out["samples"].shape == (24, 9) and set(out["row"]) == {0, 1, 2}
    for k in range(3):
        sel = out["row"] == k
        ex = exact_solve(Q, H[k])[1]
        assert out["energies"][sel].min() == pytest.approx(ex, abs=1e-9)


# ---------------------------------------------------------------------------
# QAOA
# ---------------------------------------------------------------------------
def test_qaoa_two_qubit_closed_form():
    # E = J z1 z2 (x = (1-z)/2): <Z1 Z2> = sin(4 beta) sin(2 gamma J) for p = 1
    J = 0.7
    z = 1 - 2 * bits_of(np.arange(4), 2)
    E = J * z[:, 0] * z[:, 1]
    for gam, bet in ((0.3, 0.2), (1.1, -0.4), (0.05, 0.7)):
        psi = qaoa_state(E, np.array([gam, bet]), 2, 1)
        zz = float((np.abs(psi) ** 2) @ (z[:, 0] * z[:, 1]))
        assert zz == pytest.approx(np.sin(4 * bet) * np.sin(2 * gam * J), abs=1e-12)


def test_qaoa_mixer_unitary_and_matches_dense():
    from scipy.linalg import expm
    n = 3
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi /= np.linalg.norm(psi)
    X = np.array([[0, 1], [1, 0]])
    B = sum(_op(X, q, n) for q in range(n))
    ref = expm(-1j * 0.37 * B) @ psi
    out = apply_mixer(psi.copy(), 0.37, n)
    assert np.allclose(out, ref) and np.linalg.norm(out) == pytest.approx(1.0)


def _op(X, q, n):
    """Single-qubit operator on qubit q (little-endian: qubit 0 = least significant)."""
    out = np.array([[1.0]])
    for k in reversed(range(n)):
        out = np.kron(out, X if k == q else np.eye(2))
    return out


def test_qaoa_quality_n8():
    rng = np.random.default_rng(11)
    ratios, popts = [], []
    for _ in range(3):
        Q, h = rand_qubo(8, rng)
        r = run_qaoa(Q, h, p=3, shots=500, seed=0)
        ratios.append(r["info"]["approx_ratio"])
        popts.append(r["info"]["p_opt"])
        assert 0 <= r["info"]["approx_ratio"] <= 1
    assert np.mean(ratios) >= 0.6
    assert min(popts) > 5.0 / 2 ** 8
    res = solve_qubo(Q, h, backend="qaoa", p=2, shots=200, seed=0)
    assert "approx_ratio" in res.info and res.timing["wall"] > 0
    assert res.energy >= exact_solve(Q, h)[1] - 1e-9


def test_qaoa_size_limit():
    with pytest.raises(ValueError):
        run_qaoa(np.zeros((21, 21)), np.ones(21), p=1)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------
def test_available_backends_and_cloud_errors(monkeypatch):
    bs = available_backends()
    names = {b["name"] for b in bs}
    assert {"exact", "sa", "tabu", "greedy", "qaoa", "dwave_qpu", "dwave_hybrid",
            "qiskit_aer", "ibm", "dwave_sa", "dwave_tabu"} <= names
    for b in bs:
        assert set(b) >= {"name", "available", "description", "needs_token", "kind",
                          "reason"}
        if b["name"] in ("exact", "sa", "tabu", "greedy", "qaoa"):
            assert b["available"]
    monkeypatch.delenv("DWAVE_API_TOKEN", raising=False)
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    for name in ("dwave_qpu", "dwave_hybrid", "ibm"):
        with pytest.raises(QuantumBackendUnavailable):
            qb.check_backend(name)
        with pytest.raises(QuantumBackendUnavailable):
            solve_qubo(np.zeros((2, 2)), np.array([1.0, -1.0]), backend=name)
    with pytest.raises(ValueError):
        qb.check_backend("warp_drive")


class FakeBackend:
    """Records every block QUBO it is asked to solve and returns the exact optimum."""

    def __init__(self):
        self.calls = []

    def __call__(self, Q, h, const, **kw):
        x, E, info = exact_solve(Q, h, const)
        self.calls.append((np.array(Q.toarray() if sp.issparse(Q) else Q), h.copy(), x.copy()))
        return x, E, x[None], np.array([E]), {"solver": 0.0}, {"backend": "fake"}


def test_block_gauss_seidel_bookkeeping(monkeypatch):
    fake = FakeBackend()
    monkeypatch.setitem(qb.BACKENDS, "fake", ("fake", "classical", 24, (), None))
    monkeypatch.setitem(qb._DISPATCH, "fake", fake)
    rng = np.random.default_rng(7)
    m = 18
    Q, h = rand_qubo(m, rng)
    Q *= 0.3
    v = rng.uniform(0.5, 1.5, m)
    x0 = (rng.random(m) < 0.5).astype(float)
    blocks = [np.arange(0, 6), np.arange(6, 12), np.arange(12, 18)]
    lam0 = 0.4
    lq = 0.05                              # penalty mode: fixed lambda + lq (v.y - V)^2
    V = 0.5 * v.sum()
    y, info = solve_volume_model(Q, h, 0.0, v, 0.0, V, x0, backend="fake", blocks=blocks,
                                 volume="penalty", lam0=lam0, lambda_q=lq, sweeps=3, seed=1)
    assert len(fake.calls) >= 3 and info["stats"]["n_blocks"] == 3
    # replay: every block call must see Q_BB + lq (v v^T)_offdiag and
    # h_B + 2 Q[B, notB] y_notB + lam v_B + lq (v_B^2 + 2 r v_B), r = v_notB.y_notB - V;
    # the global energy must never increase
    ycur = x0.copy()
    Eglob = lambda z: float(z @ Q @ z + h @ z + lam0 * v @ z + lq * (v @ z - V) ** 2)  # noqa
    E_prev = Eglob(ycur)

    def pen(B):
        P = lq * np.outer(v[B], v[B])
        np.fill_diagonal(P, 0.0)
        return Q[np.ix_(B, B)] + P
    for QBB, hB, xB in fake.calls:
        B = next(b for b in blocks if np.allclose(QBB, pen(b)))
        nb = np.setdiff1d(np.arange(m), B)
        r = float(v[nb] @ ycur[nb]) - V
        expect = (h[B] + 2 * Q[np.ix_(B, nb)] @ ycur[nb] + lam0 * v[B]
                  + lq * (v[B] ** 2 + 2 * r * v[B]))
        assert np.allclose(hB, expect, atol=1e-10)
        old = float(ycur[B] @ QBB @ ycur[B] + hB @ ycur[B])
        new = float(xB @ QBB @ xB + hB @ xB)
        if new < old - 1e-12:
            ycur[B] = xB
        E_now = Eglob(ycur)
        assert E_now <= E_prev + 1e-9
        E_prev = E_now
    # the solver output is the replayed sweep followed by the greedy volume repair
    from freeto.quantum.blockqubo import _repair
    ycur, _, _ = _repair(Q, h, v, ycur, V)
    assert np.array_equal(ycur, y)
    assert v @ y <= V + 1e-9


def test_volume_model_bisection_is_feasible_and_exact_small():
    rng = np.random.default_rng(2)
    m = 12
    Q, h = rand_qubo(m, rng)
    Q = 0.1 * Q
    h = -np.abs(h)
    v = rng.uniform(0.5, 1.5, m)
    V = 0.4 * v.sum()
    for be in ("exact", "sa", "tabu"):
        y, info = solve_volume_model(Q, h, 0.0, v, 0.0, V, np.ones(m), backend=be, seed=0)
        assert v @ y <= V + 1e-9
    # separable model -> sorting (knapsack) path
    y, info = solve_volume_model(sp.csr_matrix((m, m)), h, 0.0, v, 0.0, V, np.ones(m),
                                 backend="sa")
    assert info["backend"] == "separable" and v @ y <= V + 1e-9


# ---------------------------------------------------------------------------
# continuum: options, validation, Hessian, end-to-end
# ---------------------------------------------------------------------------
def test_qubo_options_and_validation():
    o = QUBOOptions.from_dict({"backend": "SA", "hessian": "exact-block"}).normalized()
    assert o.backend == "sa" and o.hessian == "block"
    assert QUBOOptions.from_dict(o.to_dict()) == o
    with pytest.raises(ValueError):
        QUBOOptions.from_dict({"nonsense": 1})
    with pytest.raises(ValueError):
        QUBOOptions(hessian="full").validate()
    cfg = example_config("cantilever_beam", mesh_control=20, optimizer="qubo",
                         qubo={"backend": "sa"})
    assert cfg.validate()
    for bad in ({"volume": "exact"}, {"backend": "nope"}, {"frontier_fraction": 0}):
        with pytest.raises(FreeTOError):
            example_config("cantilever_beam", mesh_control=20, optimizer="QUBO",
                           qubo=bad).validate()
    os.environ.pop("DWAVE_API_TOKEN", None)
    with pytest.raises(FreeTOError):
        example_config("cantilever_beam", mesh_control=20, optimizer="QUBO",
                       qubo={"backend": "dwave_qpu"}).validate()


@pytest.fixture(scope="module")
def tiny_problem(tmp_path_factory):
    from freeto.geometry import box, face_slab, write_mesh
    d = tmp_path_factory.mktemp("tiny")
    dom = ((0, 0, 0), (12, 4, 4))
    write_mesh(d / "dom.stl", box(*dom))
    write_mesh(d / "fix.stl", face_slab(dom, "x-", 1.2))
    write_mesh(d / "load.stl", face_slab(dom, "x+", 1.2))
    return dict(domain=str(d / "dom.stl"), forces=[str(d / "load.stl")],
                fixed=str(d / "fix.stl"), mesh_control=13, volfrac=0.4, fmagy=[-1.0],
                youngs_modulus=1.0, solver="superlu")


def _internals(tp, penal):
    dbg = {}
    run_freeto(FreeTOConfig(**tp, max_iter=1, penal=penal), log=None, _debug=dbg)
    return dbg


@pytest.mark.parametrize("penal,interp", [(1.0, "beso"), (3.0, "simp")])
def test_block_hessian_matches_finite_differences(tiny_problem, penal, interp):
    """hessian='block': the exact design-space Hessian inside a patch (Gram term via
    a multi-RHS solve with the reused factorisation, + SIMP curvature for
    interp='simp') vs central finite differences of c(x), rho = J x."""
    from freeto.fe import Assembler, make_solver
    from freeto.quantum.update import QUBOUpdater
    d = _internals(tiny_problem, penal)
    ele, edof, F, free, KE, H, Hs = (d["ele"], d["edofMatn"], d["F"], d["freedofs"], d["KE"],
                                     d["H"], d["Hs"])
    dom = d["dom"]
    nn = ele.size
    ndof = F.shape[0]
    asm = Assembler(edof, KE, free, ndof)
    solver = make_solver("superlu", asm)
    E0, Emin = 1.0, 1e-3
    J = sp.diags(1.0 / Hs) @ H

    def comp(x):
        rho = J @ x
        E = Emin + rho ** penal * (E0 - Emin)
        data = asm.data(E)
        U = solver.solve(data, F[free], None)
        return float(F[free, 0] @ U[:, 0]), data, U

    rng = np.random.default_rng(0)
    x0 = rng.uniform(0.4, 0.9, nn)
    c0, data, Uf = comp(x0)
    U = np.zeros((ndof, 1))
    U[free] = np.asarray(Uf).reshape(-1, 1)
    upd = QUBOUpdater({"interp": interp, "hessian": "block"}, asm=asm, solver=solver, KE=KE,
                      edofMatn=edof, freedofs=free, H=H, Hs=Hs, ele=ele, MusD=d["MusD"],
                      nelx=dom.nelx, nely=dom.nely, nelz=dom.nelz, vol=0.4, nnele=nn, E0=E0,
                      Emin=Emin, penal=penal, simp=True, ndof=ndof)
    rho = J @ x0
    E, E1, E2, ces, KUs = upd._element_terms(rho, U)
    E1m = E1 / penal if interp == "beso" else E1
    C = np.array([10, 11, 25])
    comp(x0)                                        # refresh the factorisation
    Gram = upd._gram_blocks(C, [np.arange(3)], E1m, KUs, data)[0]
    Hd = Gram.copy()
    if interp == "simp":
        JC = sp.csc_matrix(J)[:, C]
        Hd += (JC.T @ sp.diags(-E2 * ces[0]) @ JC).toarray()
    eps = 1e-4
    for a, b in ((0, 0), (0, 1), (1, 2), (2, 2)):
        ea = np.zeros(nn)
        eb = np.zeros(nn)
        ea[C[a]] = eps
        eb[C[b]] = eps
        fd = (comp(x0 + ea + eb)[0] - comp(x0 + ea - eb)[0] - comp(x0 - ea + eb)[0]
              + comp(x0 - ea - eb)[0]) / (4 * eps * eps)
        assert Hd[a, b] == pytest.approx(fd, rel=2e-3, abs=1e-4 * abs(Hd).max())


def test_continuum_qubo_sa_end_to_end():
    """cantilever_beam at a small mesh (MC 25, 768 elements), solid start:
    feasible binary design, compliance improves once the volume target is
    reached, the binary design keeps its load path."""
    events = []
    cfg = example_config("cantilever_beam", mesh_control=25, optimizer="QUBO", max_iter=60,
                         qubo={"backend": "sa", "hessian": "diag", "seed": 0},
                         eval_binary=True)
    res = run_freeto(cfg, callback=events.append, log=None)
    setup = events[0]
    assert setup["stage"] == "setup" and setup["qubo"]["backend"] == "sa"
    it = [e for e in events if e["stage"] == "iter"]
    q = it[0]["qubo"]
    for k in ("n_free", "n_blocks", "n_solves", "backend", "solver_time", "qpu_time",
              "energy", "lambda", "volume_target", "n_flips", "approx_ratio", "wall_time",
              "hessian_time", "exact_match", "max_block"):
        assert k in q
    x = res.extra["binary_design"]
    assert set(np.unique(x)) <= {0.0, 1.0}
    assert abs(x.mean() - 0.3) < 0.02
    assert abs(res.finalvol - 0.3) < 0.03
    assert math.isfinite(res.comp) and res.comp == res.extra["final_compliance"]
    # + final solve, + one re-solve per accept-if-improves rejection (the SA
    # trajectory, hence the rejections, depends on the BLAS thread count)
    n_rej = sum(1 for q in res.extra["qubo_history"] if q.get("guard"))
    assert res.extra["fe_solves"] == res.iterations + 1 + n_rej
    vt = [e["qubo"]["volume_target"] for e in it]
    assert vt[0] == pytest.approx(0.95) and vt[-1] == pytest.approx(0.3)
    k0 = next(i for i, v in enumerate(vt) if v <= 0.3 + 1e-12)
    c_first_at_target = res.history["compliance"][k0 + 1]
    assert res.comp <= c_first_at_target
    assert res.extra["binary_compliance"] < 5 * res.comp      # load path intact
    assert json.dumps(res.extra["qubo_history"])


def test_continuum_qubo_improves_thresholded_oc_start(tiny_problem):
    cfg = FreeTOConfig(**tiny_problem, optimizer="QUBO", max_iter=30,
                       qubo={"backend": "sa", "hessian": "block", "init": "oc",
                             "n_warm": 8, "seed": 1})
    res = run_freeto(cfg, log=None)
    qh = res.extra["qubo_history"]
    assert qh[0]["backend"] == "threshold" and qh[0]["iter"] == 9
    c_init = res.history["compliance"][9]        # FE of the thresholded OC design
    assert res.comp < c_init
    assert abs(res.extra["binary_design"].mean() - 0.4) < 0.03
    assert {q["hessian"] for q in qh} == {"block"}


def test_continuum_qaoa_blocks_runs_quickly():
    import time
    t0 = time.perf_counter()
    cfg = example_config("cantilever_beam", mesh_control=20, optimizer="QUBO", max_iter=4,
                         qubo={"backend": "qaoa", "block_size": 8, "qaoa_p": 1,
                               "qaoa_shots": 200, "volume": "penalty", "init": "oc",
                               "n_warm": 2, "sweeps": 1, "verify_exact": True, "seed": 0})
    res = run_freeto(cfg, log=None)
    assert time.perf_counter() - t0 < 60
    qh = res.extra["qubo_history"]
    assert qh and all(q["max_block"] <= 8 for q in qh)
    assert qh[-1]["approx_ratio"] is not None and 0 < qh[-1]["approx_ratio"] <= 1
    assert qh[-1]["exact_match"] is not None
    assert math.isfinite(res.comp)


# ---------------------------------------------------------------------------
# study runner smoke
# ---------------------------------------------------------------------------
def test_study_smoke(tmp_path):
    from freeto.study import run_study, suite_spec
    events = []
    res = run_study(suite_spec("smoke"), callback=events.append, out_dir=str(tmp_path),
                    log=None)
    kinds = [e["event"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "done" and "run_end" in kinds
    errs = [r for r in res["records"] if r.get("error")]
    assert not errs, errs
    for f in ("results.json", "results.csv", "summary.md"):
        assert (tmp_path / f).is_file()
    assert len(res["figures"]) >= 5
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    ex = [r for r in data["records"] if r["label"] == "exact"][0]
    assert ex["gap"] == pytest.approx(0.0, abs=1e-9)
    qe = [r for r in data["records"] if r["label"] == "QUBO-exact"][0]
    # 12-bar T2s (v2, drop_fixed=True): the QUBO update ends on a mechanism
    # (tests/test_truss.py) -> infeasible, gap 'n/a'
    assert not qe["feasible"] and qe["gap"] is None
    # every continuum record has its field file (v2)
    for r in [r for r in data["records"] if r["kind"] == "continuum"]:
        assert (tmp_path / r["fields_file"]).is_file()
    co = [r for r in data["records"] if r["kind"] == "continuum"]
    assert all(r["crisp_compliance"] is not None for r in co)
    mma = [r for r in co if r["label"] == "MMA"][0]
    assert mma["gap"] == pytest.approx(0.0) and mma["beta_final"] > 2
    # native (c, V) of OC/MMA = the reported iterate (hist volfrac[-2]), not the
    # twice-smoothed returned field
    assert mma["volume_fraction"] == pytest.approx(mma["history_volfrac"][-2])
    assert mma["volume_fraction_returned"] < mma["volume_fraction"]
    assert data["machine"]["threads"]["requested"] == 1
    oc = [r for r in co if r["label"].startswith("OC")][0]
    assert oc["beta_final"] <= 2.0
    txt = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "does not show" in txt and "common crisp design" in txt
    # the summary must be writable in a legacy 8-bit locale (Windows cp1252)
    # only through an explicit utf-8 encoding: it contains non-cp1252 characters
    with pytest.raises(UnicodeEncodeError):
        txt.encode("cp1252")
    # reprocess from stored records (no QUBO re-runs), summary/figures rebuilt
    from freeto.study import reprocess
    (tmp_path / "summary.md").unlink()
    rr = reprocess(str(tmp_path), log=None)
    assert (tmp_path / "summary.md").is_file() and len(rr["figures"]) >= 5
    assert "reprocessed" in rr["note"]


def test_crisp_evaluation_equal_volume_and_failed_eval_keeps_run(monkeypatch):
    """eval_crisp: every method evaluated at the same crisp volume; a failing
    post-run evaluation (solver exception) must not discard a finished run."""
    import freeto.core as core
    cfgs = {o: example_config("cantilever_beam", mesh_control=20, optimizer=o, max_iter=25,
                              eval_crisp=True, qubo={"hessian": "none"} if o == "QUBO"
                              else None)
            for o in ("MMA", "QUBO")}
    vols = []
    for o, cfg in cfgs.items():
        r = run_freeto(cfg, log=None)
        assert math.isfinite(r.extra["crisp_compliance"]) and r.extra["crisp_compliance"] > 0
        assert abs(r.extra["crisp_volfrac"] - 0.3) < 2e-3
        vols.append(r.extra["crisp_volfrac"])
    assert abs(vols[0] - vols[1]) < 2e-3

    def boom(*a, **k):
        raise RuntimeError("Factor is exactly singular")
    monkeypatch.setattr(core, "crisp_projection", boom)
    msgs = []
    r = run_freeto(cfgs["MMA"], log=msgs.append)
    assert r.extra["crisp_compliance"] == math.inf
    assert "eval_crisp" in r.extra["eval_errors"] and math.isfinite(r.comp)
    assert any("post-run evaluation" in m for m in msgs)


def test_qaoa_backend_reports_raw_and_polished():
    rng = np.random.default_rng(4)
    Q, h = rand_qubo(10, rng)
    raw = solve_qubo(Q, h, backend="qaoa", p=1, shots=50, seed=0, polish=False)
    pol = solve_qubo(Q, h, backend="qaoa", p=1, shots=50, seed=0, polish=True)
    assert raw.energy == pytest.approx(raw.info["best_shot_energy"])
    assert pol.energy == pytest.approx(pol.info["polished_energy"])
    assert pol.energy <= raw.energy + 1e-12
    assert raw.info["best_shot_energy"] == pytest.approx(pol.info["best_shot_energy"])
    for k in ("uniform_ratio", "uniform_p_opt", "best_shot_ratio", "polish_time"):
        assert k in raw.info
    assert 0 < raw.info["uniform_ratio"] < 1
    # dimod is a required module of the D-Wave backends
    assert "dimod" in qb.BACKENDS["dwave_sa"][3] and "dimod" in qb.BACKENDS["dwave_qpu"][3]


def test_guard_and_load_protection_bridge_deck():
    """bridge_deck at MC 31: distributed load on every top node.  Without the
    stabilisation the binary update leaves loaded nodes in void (c ~ 1e2);
    with load protection + move limit + guard it stays within a few x MMA.

    Run on the *pre-correction* setup (nothing kept, V = 0.2: keepdom=None,
    keep_bc=False reproduce MusD = 0 and the same 248 loaded / 32 fixed nodes at
    MC 31), which is the hard case for the binary update; the corrected example
    keeps the deck (docs/NOTES_quantum.md §8)."""
    cfg = example_config("bridge_deck", mesh_control=31, optimizer="QUBO", max_iter=40,
                         keepdom=None, keep_bc=False, volfrac=0.2,
                         qubo={"backend": "sa", "hessian": "diag", "seed": 0})
    res = run_freeto(cfg, log=None)
    qh = res.extra["qubo_history"]
    assert all(q["n_protected"] > 0 for q in qh)
    assert res.comp < 0.1                     # MMA: 0.0096; unprotected: 1.6 - 80
    assert res.extra["returned_design"] in ("best", "last")
    # connectivity repair (default on): after every step no solid element is left
    # in a component that touches no support (without the repair: 20-30
    # components, 30-55 % of the binary design floating, crisp c ~ 1e2)
    rep = [q["connectivity_repair"] for q in qh if q["backend"] != "threshold"]
    assert len(rep) == len(qh) and all(r["floating_after"] == 0 for r in rep)
    assert sum(r["floating_before"] for r in rep) > 0          # the repair was needed
    assert abs(res.extra["binary_design"].mean() - 0.2) < 0.02
