"""Truss optimisers: exact enumeration, continuous OC (+ rounding), iterative
QUBO update, sorting/BESO control (docs/QUANTUM_DESIGN.md §B)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..quantum.backends import default_block_size
from ..quantum.blockqubo import separable_knapsack, solve_volume_model
from ..quantum.update import morton_key
from .fe import KIN_RTOL, PRESENT_TOL, TrussFE, stability_loads

__all__ = ["TrussResult", "enumerate_exact", "oc_continuous", "round_sorted",
           "qubo_rounding", "qubo_iterative", "solve_truss", "TRUSS_METHODS",
           "kinematic_repair"]

TRUSS_METHODS = ["exact", "qubo", "sort", "oc", "oc_round", "oc_qubo"]
EXACT_MAX_BARS = 24


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


@dataclass
class TrussResult:
    problem: str
    method: str
    backend: Optional[str]
    seed: Optional[int]
    nodes: np.ndarray
    bars: np.ndarray
    on: np.ndarray
    areas: np.ndarray
    compliance: float
    volume: float
    vmax: float
    volume_fraction: float
    feasible: bool
    stable: bool
    c_exact: Optional[float] = None
    gap: Optional[float] = None
    iterations: int = 0
    fe_solves: int = 0
    qubo_solves: int = 0
    history: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    qubo_stats: list = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def to_dict(self):
        return _jsonable({
            "problem": self.problem, "method": self.method, "backend": self.backend,
            "seed": self.seed, "nodes": self.nodes, "bars": self.bars,
            "on": [bool(b) for b in self.on], "areas": self.areas,
            "compliance": self.compliance, "volume": self.volume, "vmax": self.vmax,
            "volume_fraction": self.volume_fraction, "feasible": self.feasible,
            "stable": self.stable, "c_exact": self.c_exact, "gap": self.gap,
            "iterations": self.iterations, "fe_solves": self.fe_solves,
            "qubo_solves": self.qubo_solves, "history": self.history,
            "timing": self.timing, "qubo_stats": self.qubo_stats, "info": self.info})


def _result(prob, fe, method, backend, seed, x, t0, c_ref=None, **kw):
    ev = fe.evaluate(x)
    vmax = prob.vmax
    feasible = ev["stable"] and ev["volume"] <= vmax * (1 + 1e-9)
    c = ev["compliance"]
    c_ref = c_ref if c_ref is not None else prob.c_exact
    gap = (c / c_ref - 1.0) if (c_ref and feasible) else (math.inf if c_ref and not feasible
                                                          else None)
    timing = kw.pop("timing", {})
    timing.setdefault("wall", time.perf_counter() - t0)
    timing.setdefault("solver", None)
    timing.setdefault("qpu_access", None)
    return TrussResult(problem=prob.id, method=method, backend=backend, seed=seed,
                       nodes=prob.nodes, bars=prob.bars, on=np.asarray(x) > 0.5,
                       areas=np.asarray(x, dtype=float), compliance=c, volume=ev["volume"],
                       vmax=vmax, volume_fraction=ev["volume_fraction"], feasible=feasible,
                       stable=ev["stable"], c_exact=c_ref, gap=gap, timing=timing,
                       fe_solves=kw.pop("fe_solves", fe.n_solves), **kw)


# ----------------------------------------------------------------------------
def enumerate_exact(prob, chunk=2 ** 14, top=10, fe=None, callback=None):
    """All 2^M binary designs (M <= 24) in chunks: volume and loaded-node
    pre-filters, batched solves, stability checks.  Returns TrussResult with
    info = {n_total, n_volume_ok, n_evaluated, n_feasible, top: [(c, bars_on)]}."""
    t0 = time.perf_counter()
    fe = fe or TrussFE(prob)
    M = prob.n_bars
    if M > EXACT_MAX_BARS:
        raise ValueError(f"exact enumeration limited to {EXACT_MAX_BARS} bars (got {M})")
    L = fe.L
    vmax = prob.vmax * (1 + 1e-12)
    inc_loaded = fe.inc[:, fe.loaded]                      # (M, nloaded)
    bits = np.arange(M, dtype=np.int64)
    best_c, best_x = math.inf, None
    top_c = np.full(0, np.inf)
    top_x = np.zeros((0, M))
    n_vol = n_eval = n_feas = 0
    total = 2 ** M
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        X = ((idx[:, None] >> bits) & 1).astype(np.float64)
        ok = X @ L <= vmax
        n_vol += int(ok.sum())
        ok &= np.all(X @ inc_loaded > 0, axis=1)
        X = X[ok]
        if X.shape[0] == 0:
            continue
        n_eval += X.shape[0]
        _, c, rok = fe.solve(X)
        good = fe.stable_mask(X, c, rok)
        n_feas += int(good.sum())
        if not good.any():
            continue
        cg, Xg = c[good], X[good]
        allc = np.concatenate([top_c, cg])
        allx = np.vstack([top_x, Xg])
        k = min(top, allc.size)
        sel = np.argpartition(allc, k - 1)[:k]
        sel = sel[np.argsort(allc[sel])]
        top_c, top_x = allc[sel], allx[sel]
        if callback is not None:
            callback({"stage": "enumerate", "done": int(idx[-1] + 1), "total": total,
                      "best": float(top_c[0])})
    if top_c.size == 0:
        raise RuntimeError("no feasible design found by enumeration")
    best_x = top_x[0]
    res = _result(prob, fe, "exact", "enumeration", None, best_x, t0,
                  c_ref=float(top_c[0]), iterations=1)
    res.info = {"n_total": total, "n_volume_ok": n_vol, "n_evaluated": n_eval,
                "n_feasible": n_feas,
                "top": [(float(c), np.flatnonzero(x).tolist()) for c, x in zip(top_c, top_x)]}
    return res


# ----------------------------------------------------------------------------
def oc_continuous(prob, move=0.2, max_iter=300, tol=1e-5, amin=1e-9, fe=None,
                  callback=None, stop_event=None):
    """Continuous OC on relative areas in [amin, 1], L.x <= Vmax."""
    t0 = time.perf_counter()
    fe = fe or TrussFE(prob, amin=amin)
    L = fe.L
    Vmax = prob.vmax
    x = np.full(fe.M, prob.vmax_fraction)
    hist = {"compliance": [], "volume": []}
    it = 0
    for it in range(1, max_iter + 1):
        if stop_event is not None and stop_event.is_set():
            break
        c, g, U, _ = fe.grad(x)
        hist["compliance"].append(c)
        hist["volume"].append(float(L @ x))
        l1, l2 = 0.0, 1e12 * max(1.0, float(np.max(-g / L)))
        xn = x
        while (l2 - l1) / (l1 + l2) > 1e-8:
            lm = 0.5 * (l1 + l2)
            xn = np.clip(x * np.sqrt(np.maximum(-g, 0.0) / (L * lm)),
                         np.maximum(amin, x - move), np.minimum(1.0, x + move))
            if xn @ L > Vmax:
                l1 = lm
            else:
                l2 = lm
        ch = float(np.max(np.abs(xn - x)))
        x = xn
        if callback is not None:
            callback({"stage": "iter", "iter": it, "compliance": c,
                      "volume": float(L @ x), "volume_fraction": float(L @ x / L.sum()),
                      "areas": x.tolist()})
        if ch < tol:
            break
    c_cont = float(fe.solve(x[None, :])[1][0])
    return x, c_cont, it, hist


def round_sorted(prob, x_cont, L=None):
    """Keep the largest areas while the volume budget allows."""
    L = prob.lengths if L is None else L
    order = np.argsort(-x_cont, kind="stable")
    y = np.zeros_like(x_cont)
    vol = 0.0
    for i in order:
        if vol + L[i] <= prob.vmax * (1 + 1e-12):
            y[i] = 1.0
            vol += L[i]
    return y


def _model(g, H, xk, s):
    """Scaled model E(y) = g.(y - xk) + 1/2 (y - xk)^T H (y - xk) in standard form."""
    g = g / s
    H = H / s
    d = np.diag(H).copy()
    Q = 0.5 * (H - np.diag(d))
    Hx = H @ xk
    h = g + 0.5 * d - Hx
    const = 0.5 * float(xk @ Hx) - float(g @ xk)
    return Q, h, const


def _bar_blocks(prob, kb, rank=None):
    mid = 0.5 * (prob.nodes[prob.bars[:, 0]] + prob.nodes[prob.bars[:, 1]])
    lo, hi = mid.min(axis=0), mid.max(axis=0)
    q = np.round((mid - lo) / np.maximum(hi - lo, 1e-12) * 1023).astype(np.int64)
    q = np.pad(q, ((0, 0), (0, 3 - q.shape[1])))
    key = morton_key(q[:, 0], q[:, 1], q[:, 2])
    order = np.argsort(key if rank is None else rank, kind="stable")
    return [order[i:i + kb] for i in range(0, order.size, kb)]


def qubo_rounding(prob, x_cont, backend="auto", seed=None, fe=None, num_reads=None,
                  backend_opts=None):
    """One-shot quadratic rounding of the continuous optimum: Taylor model at
    x_cont with the exact Hessian, lambda bisection, repair."""
    fe = fe or TrussFE(prob)
    c, g, U, _ = fe.grad(x_cont)
    H = fe.hessian(x_cont, U)
    s = float(np.max(np.abs(g))) or 1.0
    Q, h, const = _model(g, H, x_cont, s)
    be = backend if backend != "auto" else ("exact" if fe.M <= 20 else "sa")
    y, info = solve_volume_model(Q, h, const, fe.L, 0.0, prob.vmax, round_sorted(prob, x_cont),
                                 backend=be, volume="bisection", seed=seed,
                                 num_reads=num_reads, backend_opts=backend_opts)
    return y, info


# ----------------------------------------------------------------------------
def _kin_scores(fe, X):
    """Per design (rows of X): (ratio lambda_min/lambda_max of the kinematic
    test matrix, number of zero modes, number of loaded nodes connected to a
    support).  Same matrix as :meth:`TrussFE.kinematic` (unit-area stiffness of
    the present bars on the free DOFs of the touched nodes, identity on the
    untouched DOFs)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    P = (X > PRESENT_TOL).astype(float)
    nf = fe.nf
    K = (P @ fe.Kg_flat).reshape(-1, nf, nf)
    td = (P @ fe.bar_node)[:, fe.dof_node] > 0
    idx = np.arange(nf)
    K[:, idx, idx] += np.where(td, 0.0, 1.0)
    w = np.linalg.eigvalsh(K)
    wmax = np.max(w, axis=1)
    ratio = np.where(td.any(axis=1), w[:, 0] / wmax, 0.0)
    nzero = np.sum(w <= KIN_RTOL * wmax[:, None], axis=1)
    # loaded nodes reached from the supports through present bars
    N = fe.prob.n_nodes
    A = np.einsum("rb,bn,bm->rnm", P, fe.inc, fe.inc)
    reach = np.broadcast_to(fe.supp.astype(float), (P.shape[0], N)).copy()
    for _ in range(N):
        new = np.minimum(1.0, reach + np.einsum("rn,rnm->rm", reach, A))
        if np.array_equal(new, reach):
            break
        reach = new
    nconn = np.sum(reach[:, fe.loaded] > 0, axis=1)
    return ratio, nzero, nconn


def _kin_ok(fe, x):
    """Repair test: kinematically stable and every loaded node connected."""
    x = np.asarray(x, dtype=float)[None, :]
    return bool(fe.kinematic(x)[0] and fe.connected(x > PRESENT_TOL)[0])


def kinematic_repair(fe, x, V_target):
    """Kinematic repair of a binary truss design (analogue of the continuum
    connectivity repair, docs/NOTES_quantum.md §9.6).

    1. prune dangling bars (:meth:`TrussFE.prune`);
    2. while the design fails the test (kinematically stable on the free DOFs
       of the touched nodes, lambda_min > KIN_RTOL lambda_max, and every loaded
       node connected to a support): add the absent bar that maximises
       lambda_min/lambda_max of the kinematic test matrix (ties: smallest
       volume; when no single bar removes every zero mode, the bar leaving the
       fewest zero modes and the most loaded nodes connected is taken first);
    3. only if bars were added and the volume exceeds ``V_target``: remove the
       present bar with the smallest |g_b| / L_b (first-order sensitivity per
       volume, at the current design) whose removal (+ pruning) keeps the test
       passing; repeat until the volume is within target or no bar is removable.

    Returns (x, n_added, n_removed)."""
    x = fe.prune(np.asarray(x, dtype=float))
    L = fe.L
    n_add = n_rm = 0
    if _kin_ok(fe, x):
        return x, 0, 0
    while True:
        absent = np.flatnonzero(x <= PRESENT_TOL)
        if absent.size == 0:
            break
        X = np.repeat(x[None, :], absent.size, axis=0)
        X[np.arange(absent.size), absent] = 1.0
        ratio, nzero, nconn = _kin_scores(fe, X)
        ok = (nzero == 0) & (nconn == fe.loaded.size) & (ratio > KIN_RTOL)
        r_eff = np.where(ratio > KIN_RTOL, ratio, 0.0)
        # lexicographic: passes, fewer zero modes, more loaded nodes connected,
        # larger lambda_min/lambda_max, smaller volume
        key = np.lexsort((L[absent], -r_eff, -nconn, nzero, ~ok))
        b = absent[key[0]]
        x[b] = 1.0
        n_add += 1
        if _kin_ok(fe, x):
            break
    if n_add == 0 or not _kin_ok(fe, x):
        return x, n_add, n_rm
    while float(L @ x) > float(V_target) * (1 + 1e-9):
        _, g, _, _ = fe.grad(x)
        present = np.flatnonzero(x > PRESENT_TOL)
        order = present[np.argsort(np.abs(g[present]) / L[present], kind="stable")]
        done = False
        for b in order:
            xt = x.copy()
            xt[b] = 0.0
            xt = fe.prune(xt)
            if _kin_ok(fe, xt):
                n_rm += int(np.count_nonzero((x > PRESENT_TOL) & (xt <= PRESENT_TOL)))
                x = xt
                done = True
                break
        if not done:
            break
    return x, n_add, n_rm


def qubo_iterative(prob, backend="sa", hessian="exact", volume="bisection", er=0.15,
                   max_iter=60, patience=6, sweeps=2, block_size=None, num_reads=None,
                   seed=None, nbits=1, lambda_q=None, qaoa_p=3, qaoa_shots=1000,
                   qaoa_init="linear_ramp", qaoa_maxiter=None, blocks="morton",
                   verify_exact=False, backend_options=None, amin_model=1e-3,
                   trust_region=True, stability_load=0.0, prune=True, qaoa_polish=True,
                   kinematic_repair=False, fe=None,
                   callback=None, stop_event=None, log=None):
    """Iterative QUBO design update from the full ground structure (x_0 = 1),
    volume schedule V_k = max(V_{k-1}(1 - er), Vmax), Taylor model with the
    exact Hessian (or none = sorting/BESO control), block Gauss-Seidel when M
    exceeds the backend capacity.  ``kinematic_repair`` (binary designs, nbits =
    1): each updated design is passed through :func:`kinematic_repair` with the
    iteration's volume target.  Returns (best feasible x, info)."""
    fe = fe or TrussFE(prob)
    do_repair = bool(kinematic_repair)
    # the Taylor model is expanded with absent bars at amin_model (1e-3, as in
    # the design's scratch experiment); feasibility/compliance use fe.amin (1e-9)
    # stability_load > 0: nominal random load cases (weight x c_full each) are
    # added to the *model* so that it sees kinematic mechanisms (M4)
    aux = stability_loads(prob, float(stability_load)) if stability_load else None
    if aux is not None or (amin_model and amin_model > fe.amin):
        fem = TrussFE(prob, amin=max(float(amin_model or fe.amin), fe.amin), aux_loads=aux)
    else:
        fem = fe
    rng = np.random.default_rng(seed)
    M = fe.M
    L = fe.L
    nb = int(nbits)
    if nb < 1 or nb != nbits:
        raise ValueError(f"nbits must be an integer >= 1 (got {nbits!r})")
    levels = 2 ** nb - 1
    W = np.kron(np.eye(M), (2.0 ** np.arange(nb)) / levels) if nb > 1 else None  # (M, M*nb)
    x = np.ones(M)
    q = np.ones(M * nb) if nb > 1 else None
    Vk = float(L @ x)
    Vmax = prob.vmax
    kb = block_size if block_size is not None else default_block_size(backend)
    nvar = M * nb
    blk = None
    if kb is not None and nvar > int(kb):
        if nb == 1:
            rank = None
            blk = _bar_blocks(prob, int(kb)) if blocks == "morton" else None
        if blk is None:
            order = np.arange(nvar)
            blk = [order[i:i + int(kb)] for i in range(0, nvar, int(kb))]
    bopts = dict(backend_options or {})
    if backend in ("qaoa", "qiskit_aer", "ibm"):
        bopts.setdefault("p", int(qaoa_p))
        bopts.setdefault("shots", int(qaoa_shots))
        bopts.setdefault("init", qaoa_init)
        bopts.setdefault("polish", bool(qaoa_polish))
        if qaoa_maxiter:
            bopts.setdefault("maxiter", int(qaoa_maxiter))
    hist = {"compliance": [], "volume": [], "n_changed": []}
    stats_all = []
    best = (math.inf, None, -1)
    no_imp = 0
    n_qubo = 0
    t_solver = 0.0
    t_qpu = None
    qaoa_params = {}
    it = 0
    for it in range(1, int(max_iter) + 1):
        if stop_event is not None and stop_event.is_set():
            break
        c, _g, _U, ok = fe.grad(x)
        if fem is fe:
            g, U = _g, _U
        else:
            _, g, U, _ = fem.grad(x)
        stable = bool(fe.stable_mask(x[None, :], np.array([c]), np.array([ok]))[0])
        at_target = Vk <= Vmax * (1 + 1e-12)
        hist["compliance"].append(float(c))
        hist["volume"].append(float(L @ x))
        if at_target and stable and float(L @ x) <= Vmax * (1 + 1e-9):
            if c < best[0] * (1 - 1e-12):
                best = (float(c), x.copy(), it)
                no_imp = 0
            else:
                no_imp += 1
        elif at_target:
            no_imp += 1
        if at_target and no_imp >= int(patience):
            break
        Vk = max(Vk * (1.0 - float(er)), Vmax)
        s = float(np.max(np.abs(g))) or 1.0
        if hessian == "none":
            H = np.zeros((M, M))
        else:
            H = fem.hessian(x, U)
        Qx, hx, cx = _model(g, H, x, s)
        if nb > 1:
            # E = 1/2 x^T H x + (g - H x_k).x + const with x = W q (q binary)
            Hs_, gs_ = H / s, g / s
            Qfull = 0.5 * (W.T @ Hs_ @ W)
            hq = W.T @ (gs_ - Hs_ @ x)
            dq = np.diag(Qfull).copy()
            Qq = Qfull - np.diag(dq)
            hq = hq + dq
            v = W.T @ L
            Qm, hm, x0 = Qq, hq, q
        else:
            Qm, hm, x0, v = Qx, hx, x, L
        V_prev_design = float(L @ x)
        attempts = [(Vk, 0.0)]
        if trust_region:
            Vmid = max(0.5 * (V_prev_design + Vk), Vmax) if V_prev_design > Vk else Vk
            attempts += [(Vk, 0.1), (Vmid, 0.1), (Vk, 0.3), (Vmid, 0.3), (Vk, 1.0)]
        cand = None
        n_try = 0
        for Vtry, mu in attempts:
            n_try += 1
            hmu = hm + mu * (1.0 - 2.0 * x0) if mu > 0 else hm
            lam0 = None
            if volume == "penalty":
                _, lam0 = separable_knapsack(hmu, v, Vtry)
            y, info = solve_volume_model(
                Qm, hmu, cx, v, 0.0, Vtry, x0, backend=backend, blocks=blk, volume=volume,
                lam0=lam0, lambda_q=lambda_q, sweeps=sweeps, num_reads=num_reads,
                backend_opts=bopts, verify_exact=verify_exact, stop_event=stop_event,
                qaoa_params=qaoa_params, rng=rng, log=log)
            st = info["stats"]
            n_qubo += st["n_solves"]
            t_solver += st["solver_time"]
            if st["qpu_time"] is not None:
                t_qpu = (t_qpu or 0.0) + st["qpu_time"]
            xn = W @ y if nb > 1 else y
            if prune:
                xp = fe.prune(xn)
                if nb == 1:
                    y = xp
                xn = xp if nb == 1 else xn
            if not trust_region:
                cand = (y, info, xn)
                break
            _, cn, okn = fe.solve(xn[None, :])
            stab = bool(fe.stable_mask(xn[None, :], cn, okn)[0])
            if cand is None:
                cand = (y, info, xn)
            if stab:
                cand = (y, info, xn)
                break
        y, info, xn = cand
        n_kadd = n_krm = 0
        if do_repair and nb == 1:
            xn, n_kadd, n_krm = _kin_repair_fn(fe, xn, Vk)
            y = xn
        st = info["stats"]
        if nb > 1:
            q = y
        nchg = int(np.count_nonzero(np.abs(xn - x) > 1e-12))
        hist["n_changed"].append(nchg)
        stat = {"iter": it, "n_free": int(nvar), "n_blocks": int(st["n_blocks"]),
                "n_solves": int(st["n_solves"]), "backend": info["backend"],
                "hessian": hessian, "wall_time": info["time"],
                "solver_time": float(st["solver_time"]), "qpu_time": st["qpu_time"],
                "energy": info["energy"], "lambda": info["lambda"],
                "volume_target": Vk / float(L.sum()), "n_flips": nchg,
                "approx_ratio": st["approx_ratio"], "p_opt": st["p_opt"],
                "exact_match": st["exact_match"], "max_block": int(st["max_block"]),
                "approx_ratios": st["approx_ratios"], "p_opts": st["p_opts"],
                "exact_match_raw": st.get("exact_match_raw"),
                "best_shot_ratio": st.get("best_shot_ratio"),
                "polish_improved": st.get("polish_improved"),
                "attempts": n_try, "kin_repair_added": n_kadd, "kin_repair_removed": n_krm}
        stats_all.append(stat)
        if callback is not None:
            callback({"stage": "iter", "iter": it, "compliance": float(c),
                      "volume": float(L @ x), "volume_fraction": float(L @ x / L.sum()),
                      "on": [int(b) for b in (x > 0.5)], "areas": x.tolist(),
                      "qubo": {k: v for k, v in stat.items()
                               if k not in ("approx_ratios", "p_opts")}})
        if nchg == 0 and at_target:
            # fixed point: evaluate once more next loop for bookkeeping, then stop
            x = xn
            c2, _, _, ok2 = fe.grad(x)
            if best[1] is None or c2 < best[0]:
                if float(L @ x) <= Vmax * (1 + 1e-9) and bool(
                        fe.stable_mask(x[None, :], np.array([c2]), np.array([ok2]))[0]):
                    best = (float(c2), x.copy(), it + 1)
            break
        x = xn
    if best[1] is None:
        # never feasible at target: return the last design (flagged infeasible)
        best = (math.inf, x.copy(), it)
    return best[1], {"iterations": it, "history": hist, "qubo_stats": stats_all,
                     "qubo_solves": n_qubo, "solver_time": t_solver, "qpu_time": t_qpu,
                     "best_iteration": best[2], "blocks": None if blk is None else len(blk)}


# ----------------------------------------------------------------------------
def solve_truss(prob, method="qubo", backend="sa", seed=None, callback=None,
                stop_event=None, c_ref=None, log=None, **options):
    """Unified entry point (docs/QUANTUM_API.md §3)."""
    t0 = time.perf_counter()
    method = str(method).lower()
    if method not in TRUSS_METHODS:
        raise ValueError(f"unknown truss method {method!r} (known: {', '.join(TRUSS_METHODS)})")
    fe = TrussFE(prob)
    c_ref = c_ref if c_ref is not None else (prob.c_exact or prob.c_best_known)
    if method == "exact":
        r = enumerate_exact(prob, fe=fe, callback=callback)
        r.timing = {"wall": time.perf_counter() - t0, "solver": None, "qpu_access": None}
        if c_ref:
            r.c_exact = c_ref
            r.gap = r.compliance / c_ref - 1.0
        return r
    if method in ("oc", "oc_round", "oc_qubo"):
        xc, c_cont, it, hist = oc_continuous(prob, fe=fe, callback=callback,
                                             stop_event=stop_event,
                                             **{k: v for k, v in options.items()
                                                if k in ("move", "max_iter", "tol")})
        info = {"c_continuous": c_cont, "areas_continuous": xc.tolist()}
        if method == "oc":
            res = _result(prob, fe, "oc", None, seed, xc, t0, c_ref=c_ref, iterations=it,
                          history=hist, info=info)
            return res
        if method == "oc_round":
            y = fe.prune(round_sorted(prob, xc))
            return _result(prob, fe, "oc_round", None, seed, y, t0, c_ref=c_ref,
                           iterations=it, history=hist, info=info)
        be = backend if backend else "auto"
        y, qi = qubo_rounding(prob, xc, backend=be, seed=seed, fe=fe,
                              num_reads=options.get("num_reads"))
        y = fe.prune(y)
        st = qi["stats"]
        return _result(prob, fe, "oc_qubo", qi["backend"], seed, y, t0, c_ref=c_ref,
                       iterations=it, history=hist, info=info, qubo_solves=st["n_solves"],
                       timing={"solver": st["solver_time"], "qpu_access": st["qpu_time"]})
    # qubo / sort
    kw = dict(options)
    if method == "sort":
        kw["hessian"] = "none"
        kw["volume"] = "bisection"
    allowed = {"hessian", "volume", "er", "max_iter", "patience", "sweeps", "block_size",
               "num_reads", "nbits", "lambda_q", "qaoa_p", "qaoa_shots", "qaoa_init",
               "qaoa_maxiter", "blocks", "verify_exact", "backend_options", "amin_model",
               "trust_region", "stability_load", "qaoa_polish", "prune", "kinematic_repair"}
    bad = set(kw) - allowed
    if bad:
        raise ValueError(f"unknown truss QUBO option(s): {', '.join(sorted(bad))}")
    be = backend or "sa"
    if be == "auto":
        be = "exact" if prob.n_bars * int(kw.get("nbits", 1)) <= 20 else "sa"
    x, info = qubo_iterative(prob, backend=be, seed=seed, fe=fe, callback=callback,
                             stop_event=stop_event, log=log, **kw)
    res = _result(prob, fe, method, "separable" if method == "sort" else be, seed, x, t0,
                  c_ref=c_ref, iterations=info["iterations"], history=info["history"],
                  qubo_stats=info["qubo_stats"], qubo_solves=info["qubo_solves"],
                  timing={"solver": info["solver_time"], "qpu_access": info["qpu_time"]},
                  info={"best_iteration": info["best_iteration"], "blocks": info["blocks"],
                        "options": _jsonable(kw)})
    return res


_kin_repair_fn = kinematic_repair      # the qubo_iterative argument shadows the name
