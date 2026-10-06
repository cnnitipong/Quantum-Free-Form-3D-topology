"""Tests for the truss ground-structure module (freeto.truss)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from freeto.truss import (TrussProblem, TrussFE, candidate_bars, enumerate_exact,  # noqa: E402
                          get_benchmark, list_benchmarks, solve_truss, oc_continuous,
                          round_sorted)
from freeto.truss.benchmarks import C_EXACT                                         # noqa: E402


# ---------------------------------------------------------------------------
def test_benchmark_bar_counts():
    counts = {b["id"]: b["n_bars"] for b in list_benchmarks()}
    # v2 (2026-10-04): drop_fixed=True for every ground structure (no bar between
    # two fully fixed nodes): gs_3x2 13 -> 12, gs_4x2 22 -> 21, gs_9x3 118 -> 116
    assert counts == {"ten_bar": 10, "gs_4x2": 21, "gs_3x2": 12, "tower3d": 22,
                      "gs_9x3": 116, "column3d": 66}
    for b in list_benchmarks():
        p = get_benchmark(b["id"])
        fx = [n for n in range(p.n_nodes)
              if sum(1 for m, _ in p.supports if m == n) >= p.dim]
        assert not np.any(np.isin(p.bars[:, 0], fx) & np.isin(p.bars[:, 1], fx)), b["id"]
    assert get_benchmark("T2s").id == "gs_3x2"
    with pytest.raises(ValueError):
        get_benchmark("nope")


def test_overlap_filter_and_lmax():
    nodes = np.array([[0, 0], [1, 0], [2, 0], [1, 1]], float)
    bars = candidate_bars(nodes).tolist()
    assert [0, 2] not in bars                      # passes through node 1
    assert [0, 1] in bars and [1, 2] in bars and [0, 3] in bars
    assert len(candidate_bars(nodes, lmax=1.0)) == 3
    # bars between two fully fixed nodes are dropped
    b2 = candidate_bars(nodes, fixed_nodes=[0, 1])
    assert [0, 1] not in b2.tolist()


def _two_bar():
    # symmetric two-bar truss: supports at (-1, 0) and (1, 0), apex (0, -1), load P down
    nodes = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0]])
    return TrussProblem(nodes, [[0, 2], [1, 2]], [(0, 0), (0, 1), (1, 0), (1, 1)],
                        [(2, 0.0, -2.0)], E=3.0, A_full=0.5, vmax_fraction=1.0)


def test_two_bar_analytic():
    p = _two_bar()
    fe = TrussFE(p)
    # each bar: axial force N = P / (2 cos 45) = sqrt(2); c = sum N^2 L / (E A)
    N = 2.0 / (2 * np.cos(np.pi / 4))
    L = np.sqrt(2.0)
    c_exact = 2 * N ** 2 * L / (3.0 * 0.5)
    c, g, U, ok = fe.grad(np.ones(2))
    assert ok and c == pytest.approx(c_exact, rel=1e-12)
    # sensitivity: dc/dx_b = -N_b^2 L/(E A) (x = relative area)
    assert np.allclose(g, -N ** 2 * L / (3.0 * 0.5), rtol=1e-10)


def test_batched_equals_single_and_mechanism_detection():
    p = get_benchmark("gs_3x2")
    fe = TrussFE(p)
    rng = np.random.default_rng(0)
    X = rng.uniform(0.1, 1.0, (7, p.n_bars))
    U, c, ok = fe.solve(X)
    for r in range(7):
        c1 = fe.solve(X[r:r + 1])[1][0]
        assert c[r] == pytest.approx(c1, rel=1e-12)
        K = sum(X[r, b] * fe.Kb[b] for b in range(p.n_bars))
        assert c[r] == pytest.approx(float(fe.F[:, 0] @ np.linalg.solve(K, fe.F[:, 0])),
                                     rel=1e-10)
    # remove every bar touching the loaded node -> disconnected / mechanism
    x = np.ones(p.n_bars)
    tip = p.loaded_nodes[0]
    x[(p.bars == tip).any(axis=1)] = 0
    ev = fe.evaluate(x)
    assert not ev["stable"]
    assert fe.evaluate(np.ones(p.n_bars))["stable"]


def test_truss_hessian_vs_finite_differences():
    p = get_benchmark("tower3d")
    fe = TrussFE(p)
    rng = np.random.default_rng(1)
    x = rng.uniform(0.3, 1.0, p.n_bars)
    c0, g, U, _ = fe.grad(x)
    H = fe.hessian(x, U)
    eps = 1e-4
    for i, j in ((0, 3), (5, 5), (7, 12)):
        ei, ej = np.eye(p.n_bars)[i] * eps, np.eye(p.n_bars)[j] * eps
        gi = (fe.grad(x + ej)[1][i] - fe.grad(x - ej)[1][i]) / (2 * eps)
        assert H[i, j] == pytest.approx(gi, rel=1e-5, abs=1e-8 * abs(H).max())
    # gradient vs finite differences
    e = np.eye(p.n_bars)[4] * 1e-6
    fd = (fe.solve((x + e)[None])[1][0] - fe.solve((x - e)[None])[1][0]) / 2e-6
    assert g[4] == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("name", ["ten_bar", "gs_3x2", "gs_4x2", "tower3d"])
def test_exact_enumeration_pinned(name):
    p = get_benchmark(name)
    r = enumerate_exact(p)
    assert r.compliance == pytest.approx(C_EXACT[name], rel=1e-9)
    assert r.feasible and r.volume <= p.vmax * (1 + 1e-12)
    if name == "gs_3x2":
        # rigorous (kinematically stable) optimum of the 12-bar structure (v2,
        # drop_fixed=True: the budget 0.5 sum(L) fell from 8.5645 to 8.0645, so the
        # 13-bar optimum 9.96376 {1, 2, 4, 5, 7, 8, 10} (volume 8.243) is no longer
        # feasible); the 13-bar mechanism 9.0897 is still rejected (test below)
        assert r.compliance == pytest.approx(11.656854, abs=5e-6)
        assert np.flatnonzero(r.on).tolist() == [0, 1, 3, 4, 7, 9]
        assert r.info["n_feasible"] == 23
    if name == "gs_4x2":
        # same optimal bar set as the 22-bar structure (indices shifted by one)
        assert np.flatnonzero(r.on).tolist() == [0, 1, 4, 5, 9, 10, 12, 13, 15, 16, 18]
        assert r.info["n_feasible"] == 629
    if name == "ten_bar":
        assert np.flatnonzero(r.on).tolist() == [0, 2, 3, 6, 7, 8]
    assert r.info["n_feasible"] >= 1
    assert r.info["top"][0][0] == pytest.approx(r.compliance)


def test_kinematic_stability_rejects_collinear_hinge_and_prune():
    p = get_benchmark("gs_3x2")
    fe = TrussFE(p)
    # bar indices of the 12-bar structure (v2) = 13-bar indices - 1
    old = np.zeros(p.n_bars)
    old[[0, 1, 4, 5, 7, 9]] = 1.0       # node (1, 0) held only by two collinear bars
    ev = fe.evaluate(old)
    assert np.isfinite(ev["compliance"]) and ev["compliance"] < 10.0   # linear analysis ok ...
    assert not ev["stable"]                                            # ... but a mechanism
    new = np.zeros(p.n_bars)
    new[[0, 1, 3, 4, 6, 7, 9]] = 1.0
    assert fe.evaluate(new)["stable"]
    # a dangling bar (unloaded node touched once) is pruned, compliance unchanged
    dang = new.copy()
    dang[10] = 1.0                      # bar 3-5: node 5 is free, unloaded, degree 1
    assert not fe.evaluate(dang)["stable"]
    pr = fe.prune(dang)
    assert np.array_equal(pr, new)
    assert fe.evaluate(pr)["compliance"] == pytest.approx(fe.evaluate(dang)["compliance"])


def test_qubo_iterative_t1_exact_and_t2s_regression():
    """Under the rigorous stability test (M4) the iterative QUBO update reaches
    the exact optimum on T1.  On the 12-bar T2s (v2, drop_fixed=True) it ends
    on a kinematic mechanism (collinear hinge at node (1, 0): bars 0-2 and 2-4
    only) whose linear-analysis compliance equals the optimum -> infeasible,
    gap inf; first-order sorting reaches an optimal design (bar 2-3 instead of
    the zero-force bar 1-2 of the enumerated optimum; the relative difference
    1.5e-9 comes from the 1e-9 void area)."""
    t1 = solve_truss(get_benchmark("ten_bar"), "qubo", "exact", seed=0)
    assert t1.feasible and t1.gap == pytest.approx(0.0, abs=1e-9)
    p = get_benchmark("gs_3x2")
    r = solve_truss(p, "qubo", "exact", seed=0)
    assert not r.feasible and not r.stable
    assert r.compliance == pytest.approx(C_EXACT["gs_3x2"], rel=1e-8)
    assert np.flatnonzero(r.on).tolist() == [0, 1, 4, 7, 9, 10, 11]
    s = solve_truss(p, "sort", seed=0)
    assert s.feasible and s.gap == pytest.approx(0.0, abs=1e-8)
    with pytest.raises(ValueError):
        solve_truss(p, "qubo", "sa", seed=0, nbits=0)


@pytest.mark.parametrize("backend", ["sa", "tabu", "greedy"])
def test_qubo_iterative_stochastic_backends_t2s(backend):
    # the solver does not matter at this size: same fixed point (a mechanism on the
    # 12-bar T2s, see above) as the exact block solver
    r = solve_truss(get_benchmark("gs_3x2"), "qubo", backend, seed=0)
    assert not r.feasible and np.flatnonzero(r.on).tolist() == [0, 1, 4, 7, 9, 10, 11]
    assert r.qubo_solves > 0 and len(r.qubo_stats) == r.iterations
    assert r.timing["wall"] > 0


def test_qubo_iterative_qaoa_whole_problem():
    r = solve_truss(get_benchmark("ten_bar"), "qubo", "qaoa", seed=0, volume="penalty",
                    qaoa_p=1, qaoa_shots=200)
    assert r.feasible and r.gap <= 0.05
    ar = [a for s in r.qubo_stats for a in s["approx_ratios"]]
    assert ar and all(0.0 <= a <= 1.0 for a in ar)


def test_blocks_and_multibit():
    p = get_benchmark("gs_4x2")
    r = solve_truss(p, "qubo", "exact", seed=0, block_size=8, max_iter=6)
    assert all(s["n_blocks"] == 3 for s in r.qubo_stats) and r.volume <= p.vmax * (1 + 1e-9)
    r2 = solve_truss(get_benchmark("gs_3x2"), "qubo", "sa", seed=0, nbits=2, max_iter=8)
    levels = np.array([0, 1 / 3, 2 / 3, 1])
    assert np.all(np.min(np.abs(r2.areas[:, None] - levels[None, :]), axis=1) < 1e-12)
    assert r2.volume <= get_benchmark("gs_3x2").vmax * (1 + 1e-9)


def test_oc_and_rounding():
    p = get_benchmark("gs_3x2")
    x, c, it, hist = oc_continuous(p)
    assert x @ p.lengths <= p.vmax * (1 + 1e-6)
    assert c <= C_EXACT["gs_3x2"] * (1 + 1e-9)       # relaxation is a lower bound
    y = round_sorted(p, x)
    assert y @ p.lengths <= p.vmax * (1 + 1e-12)
    for m in ("oc", "oc_round", "oc_qubo"):
        r = solve_truss(p, m, "auto", seed=0)
        assert r.compliance > 0 and r.method == m


def test_result_to_dict_json_and_callback():
    events = []
    r = solve_truss(get_benchmark("gs_3x2"), "qubo", "sa", seed=1, callback=events.append)
    d = r.to_dict()
    s = json.dumps(d)
    assert json.loads(s)["problem"] == "gs_3x2"
    for k in ("nodes", "bars", "on", "areas", "compliance", "gap", "history", "timing",
              "qubo_stats"):
        assert k in d
    assert events and events[0]["stage"] == "iter" and "qubo" in events[0]
    assert len(events[0]["on"]) == 12
    pd = get_benchmark("tower3d").to_dict()
    assert pd["dim"] == 3 and len(pd["bars"]) == 22 and json.dumps(pd)


# ---------------------------------------------------------------------------
# v2 (P0b): kinematic repair of the truss QUBO / sorting update
@pytest.mark.parametrize("backend", ["exact", "sa", "tabu", "greedy"])
def test_kinematic_repair_t2s_feasible(backend):
    """Without the repair the update ends on a mechanism on the 12-bar T2s (see
    test_qubo_iterative_t1_exact_and_t2s_regression); with it every backend ends
    on the exact optimum."""
    p = get_benchmark("gs_3x2")
    r = solve_truss(p, "qubo", backend, seed=0, kinematic_repair=True)
    assert r.feasible and r.stable
    assert r.gap == pytest.approx(0.0, abs=1e-6)
    assert np.flatnonzero(r.on).tolist() == [0, 1, 3, 4, 7, 9]
    add = [s["kin_repair_added"] for s in r.qubo_stats]
    rm = [s["kin_repair_removed"] for s in r.qubo_stats]
    assert sum(add) >= 1 and sum(rm) >= 1


def test_kinematic_repair_function_and_default_off():
    from freeto.truss.optimize import kinematic_repair, _kin_ok
    p = get_benchmark("gs_3x2")
    fe = TrussFE(p)
    mech = np.zeros(p.n_bars)
    mech[[0, 1, 4, 7, 9, 10, 11]] = 1.0            # collinear hinge at node (1, 0)
    assert not _kin_ok(fe, mech)
    x, na, nr = kinematic_repair(fe, mech, p.vmax)
    assert _kin_ok(fe, x) and fe.evaluate(x)["stable"] and na >= 1
    assert x @ p.lengths <= p.vmax * (1 + 1e-9)
    # a stable design is returned unchanged
    ok = np.zeros(p.n_bars)
    ok[[0, 1, 3, 4, 7, 9]] = 1.0
    x2, na2, nr2 = kinematic_repair(fe, ok, p.vmax)
    assert np.array_equal(x2, ok) and na2 == nr2 == 0
    # off by default: no repair counts
    r = solve_truss(p, "sort", seed=0)
    assert all(s["kin_repair_added"] == 0 for s in r.qubo_stats)


def test_kinematic_repair_gs4x2_tower3d_stable():
    """gs_4x2: feasible with the repair (QUBO-sa +42 %); tower3d: stable but above
    the volume budget (the greedy removal cannot reach V_max) -- documented in
    docs/NOTES_quantum.md §9.6."""
    r = solve_truss(get_benchmark("gs_4x2"), "qubo", "sa", seed=0, kinematic_repair=True)
    assert r.feasible and r.gap == pytest.approx(0.42259, abs=1e-4)
    t = solve_truss(get_benchmark("tower3d"), "qubo", "sa", seed=0, kinematic_repair=True)
    assert t.stable and not t.feasible and t.volume > t.vmax
