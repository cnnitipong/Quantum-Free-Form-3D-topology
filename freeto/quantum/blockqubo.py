"""Volume-constrained binary model minimisation shared by the continuum and
the truss QUBO updates (docs/QUANTUM_DESIGN.md §A.2, §A.4).

Model over the free variables y in {0,1}^m:

    E(y) = y^T Q y + h.y + const        (Q symmetric, zero diagonal)
    s.t.  v.y + V_fixed <= V_target

Volume handling:

* ``volume="bisection"``: minimise E + lam v.y for the smallest lam >= 0
  whose solution is feasible.  Cheap single-block backends that accept
  several linear terms at once (``sa``, ``exact``, ``greedy``) use a
  *multisection* search (one geometric round over [1e-5, 1] lam_max, then
  linear rounds), all candidate lambdas solved in one vectorised call;
  other backends bisect sequentially (``bisection_steps`` solves of the
  whole block sweep).
* ``volume="penalty"``: lam fixed (``lam0``) plus lam_q (v.y + V_fixed -
  V_target)^2 (dense rank-1 coupling), one block sweep.

Both end with a greedy repair (remove the cheapest elements per volume until
feasible, then add elements with negative marginal energy while the volume
fits), so the returned design is always feasible if any is.

Blocks: block Gauss-Seidel -- for each block B the other blocks enter the
linear term h_B + 2 Q[B, not B] y_notB; a block result is accepted only if
it does not increase the block (hence the global) energy.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .backends import is_batch_backend, solve_qubo
from .qubo import energy as qenergy

__all__ = ["solve_volume_model", "separable_knapsack", "SolveStats"]


class SolveStats:
    def __init__(self):
        self.n_solves = 0
        self.solver_time = 0.0
        self.qpu_time = None
        self.approx = []
        self.p_opt = []
        self.exact_match = []
        self.exact_match_raw = []
        self.best_shot_ratio = []
        self.polished = []
        self.max_block = 0
        self.n_blocks = 0
        self.backend = None
        self.qaoa_angles = None

    def add(self, res, n, verify=None, verify_raw=None):
        self.n_solves += 1
        self.solver_time += float(res.timing.get("solver") or 0.0)
        q = res.timing.get("qpu_access")
        if q is not None:
            self.qpu_time = (self.qpu_time or 0.0) + float(q)
        if "approx_ratio" in res.info and res.info["approx_ratio"] is not None:
            self.approx.append(float(res.info["approx_ratio"]))
        if "p_opt" in res.info and res.info["p_opt"] is not None:
            self.p_opt.append(float(res.info["p_opt"]))
        if verify is not None:
            self.exact_match.append(bool(verify))
        if verify_raw is not None:
            self.exact_match_raw.append(bool(verify_raw))
        if res.info.get("best_shot_ratio") is not None:
            self.best_shot_ratio.append(float(res.info["best_shot_ratio"]))
        if res.info.get("polished") is not None:
            self.polished.append(bool(res.info["polished"]))
        if res.info.get("angles"):
            self.qaoa_angles = res.info["angles"]
        self.max_block = max(self.max_block, int(n))

    def as_dict(self):
        return {"n_solves": self.n_solves, "solver_time": self.solver_time,
                "qpu_time": self.qpu_time,
                "approx_ratio": float(np.mean(self.approx)) if self.approx else None,
                "p_opt": float(np.mean(self.p_opt)) if self.p_opt else None,
                "exact_match": float(np.mean(self.exact_match)) if self.exact_match else None,
                "exact_match_raw": (float(np.mean(self.exact_match_raw))
                                    if self.exact_match_raw else None),
                "best_shot_ratio": (float(np.mean(self.best_shot_ratio))
                                    if self.best_shot_ratio else None),
                "polish_improved": float(np.mean(self.polished)) if self.polished else None,
                "max_block": self.max_block, "n_blocks": self.n_blocks,
                "approx_ratios": list(self.approx), "p_opts": list(self.p_opt)}


def separable_knapsack(h, v, budget):
    """min h.y s.t. v.y <= budget for diagonal models: take the most negative
    h_i/v_i first (the lam-threshold solution; BESO sorting)."""
    m = h.size
    y = np.zeros(m)
    if m == 0:
        return y, 0.0
    cand = np.flatnonzero(h < 0)
    order = cand[np.argsort(h[cand] / v[cand], kind="stable")]
    cum = np.cumsum(v[order])
    k = int(np.searchsorted(cum, budget + 1e-9 * max(1.0, abs(budget)), side="right"))
    y[order[:k]] = 1.0
    lam = float(-h[order[k]] / v[order[k]]) if k < order.size else 0.0
    return y, max(lam, 0.0)


def _nnz(Q):
    return Q.nnz if sp.issparse(Q) else int(np.count_nonzero(Q))


def _sub(Q, B):
    if sp.issparse(Q):
        return Q[B][:, B]
    return Q[np.ix_(B, B)]


def _rowsdot(Q, B, y):
    if sp.issparse(Q):
        return Q[B] @ y
    return Q[B] @ y


def _lam_max(Q, h, v):
    if sp.issparse(Q):
        neg = np.asarray(Q.minimum(0).sum(axis=1)).ravel()
    else:
        neg = np.minimum(Q, 0).sum(axis=1)
    r = (-h - 2.0 * neg) / v
    return float(max(r.max(initial=0.0), 1e-12)) * 1.05 + 1e-12


def _repair(Q, h, v, y, budget, max_flips=None, fill=True):
    """Greedy volume repair on E = y^T Q y + h.y (no lam term)."""
    y = y.copy()
    Qc = Q.tocsc() if sp.issparse(Q) else Q
    f = Q @ y
    vol = float(v @ y)
    tol = 1e-9 * max(1.0, abs(budget))
    n_rm = n_add = 0
    lim = max_flips or 4 * y.size + 10

    def col(i):
        if sp.issparse(Qc):
            a, b = Qc.indptr[i], Qc.indptr[i + 1]
            return Qc.indices[a:b], Qc.data[a:b]
        return None, Qc[:, i]
    while vol > budget + tol and n_rm < lim:
        on = np.flatnonzero(y > 0.5)
        if on.size == 0:
            break
        dE = -(h[on] + 2.0 * f[on])
        i = on[int(np.argmin(dE / v[on]))]
        y[i] = 0.0
        vol -= v[i]
        idx, val = col(i)
        if idx is None:
            f -= val
        else:
            f[idx] -= val
        n_rm += 1
    while n_add < lim:
        off = np.flatnonzero((y < 0.5) & (v <= budget - vol + tol))
        if off.size == 0:
            break
        dE = h[off] + 2.0 * f[off]
        k = int(np.argmin(dE / v[off]))
        if not (fill or dE[k] < -1e-14):
            break
        i = off[k]
        y[i] = 1.0
        vol += v[i]
        idx, val = col(i)
        if idx is None:
            f += val
        else:
            f[idx] += val
        n_add += 1
    return y, n_rm, n_add


def solve_volume_model(Q, h, const, v, V_fixed, V_target, x0, *, backend="sa",
                       blocks=None, volume="bisection", lam0=None, lambda_q=None,
                       sweeps=2, num_reads=None, seed=None, bisection_steps=14,
                       backend_opts=None, verify_exact=False, stop_event=None,
                       qaoa_params=None, rng=None, log=None):
    """Minimise the volume-constrained model; returns (y, info dict).

    ``x0``: current design (starting point of the block sweep and the
    reference of the acceptance test).  ``blocks``: list of index arrays
    (None = one block).  ``qaoa_params``: dict used to warm-start QAOA angles
    across calls (updated in place)."""
    t0 = time.perf_counter()
    rng = rng if rng is not None else np.random.default_rng(seed)
    h = np.asarray(h, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    m = h.size
    budget = float(V_target - V_fixed)
    stats = SolveStats()
    backend_opts = dict(backend_opts or {})
    x0 = np.asarray(x0, dtype=np.float64).copy()
    if blocks is None or len(blocks) <= 1:
        blocks = [np.arange(m)]
    blocks = [np.asarray(b, dtype=np.int64) for b in blocks if len(b)]
    stats.n_blocks = len(blocks)
    lam_used = 0.0
    mode = volume
    if m == 0:
        return x0, {"lambda": 0.0, "energy": float(const), "stats": stats.as_dict(),
                    "backend": backend, "repair_removed": 0, "repair_added": 0,
                    "volume": 0.0, "time": 0.0, "mode": mode}
    if budget < -1e-9 * max(1.0, abs(V_target)):
        if log:
            log("QUBO: the fixed part already exceeds the volume target; "
                "all free variables set to 0")
    separable = _nnz(Q) == 0
    used_backend = backend
    if separable:
        # diagonal model: exact solution by sorting (BESO control)
        y, lam_used = separable_knapsack(h, v, max(budget, 0.0))
        used_backend = "separable"
        stats.backend = "separable"
    else:
        call_kw = dict(backend=backend, num_reads=num_reads, normalized=True,
                       **backend_opts)

        def block_solve(B, y, lam, pen):
            """Solve block B given the rest of y; returns new y_B (accepted or not)."""
            QBB = _sub(Q, B)
            hB = h[B] + 2.0 * (_rowsdot(Q, B, y) - (QBB @ y[B])) + lam * v[B]
            if pen is not None:
                lq, Vfix = pen
                vB = v[B]
                r = float(v @ y - vB @ y[B]) + Vfix - V_target
                P = lq * np.outer(vB, vB)
                np.fill_diagonal(P, 0.0)
                QBB = (QBB.toarray() if sp.issparse(QBB) else QBB) + P
                hB = hB + lq * (vB * vB + 2.0 * r * vB)
            kw = dict(call_kw)
            kw["seed"] = int(rng.integers(0, 2 ** 31 - 1))
            if backend in ("qaoa", "qiskit_aer", "ibm"):
                if qaoa_params is not None and qaoa_params.get("angles") is not None \
                        and len(qaoa_params["angles"]) == 2 * int(kw.get("p", 3)):
                    kw["initial_params"] = qaoa_params["angles"]
            if backend == "greedy":
                kw["initial_state"] = y[B]
            res = solve_qubo(QBB, hB, 0.0, **kw)
            ver = ver_raw = None
            if verify_exact and B.size <= 20 and backend != "exact":
                from .exact import exact_solve
                _, Eex, _ = exact_solve(QBB, hB, 0.0)
                tol_ = 1e-9 * max(1.0, abs(Eex))
                ver = res.energy <= Eex + tol_
                if res.info.get("best_shot_energy") is not None:
                    ver_raw = res.info["best_shot_energy"] <= Eex + tol_
            stats.add(res, B.size, ver, ver_raw)
            if qaoa_params is not None and res.info.get("angles"):
                qaoa_params["angles"] = res.info["angles"]
            E_old = qenergy(QBB, hB, 0.0, y[B])
            if res.energy < E_old - 1e-12 * max(1.0, abs(E_old)):
                return res.x, True
            return y[B], False

        def sweep_all(lam, pen):
            y = x0.copy()
            for _ in range(int(sweeps)):
                changed = False
                for bi in rng.permutation(len(blocks)):
                    if stop_event is not None and stop_event.is_set():
                        return y
                    B = blocks[bi]
                    yB, acc = block_solve(B, y, lam, pen)
                    if acc and not np.array_equal(yB, y[B]):
                        y[B] = yB
                        changed = True
                if not changed or len(blocks) == 1:
                    break
            return y

        def feasible(y):
            return float(v @ y) <= budget + 1e-9 * max(1.0, abs(budget))

        if mode == "penalty":
            lq = float(lambda_q) if lambda_q is not None else 2.0 / float(v.max()) ** 2
            lam_used = float(lam0 or 0.0)
            y = sweep_all(lam_used, (lq, V_fixed))
        else:
            y0 = sweep_all(0.0, None) if not (len(blocks) == 1 and is_batch_backend(backend)) else None
            lam_hi = _lam_max(Q, h, v)
            if y0 is not None and feasible(y0):
                y, lam_used = y0, 0.0
            elif len(blocks) == 1 and is_batch_backend(backend):
                y, lam_used = _multisection(Q, h, v, budget, lam_hi, backend, num_reads,
                                            rng, stats, backend_opts)
            else:
                lo, hi = 0.0, lam_hi
                y_hi = np.zeros(m)
                for s in range(int(bisection_steps)):
                    if stop_event is not None and stop_event.is_set():
                        break
                    if lo <= 0.0:
                        lam = hi * 0.1          # geometric search for a bracket
                    elif hi / lo > 2.0:
                        lam = float(np.sqrt(lo * hi))
                    else:
                        lam = 0.5 * (lo + hi)
                    ys = sweep_all(lam, None)
                    if feasible(ys):
                        hi, y_hi = lam, ys
                    else:
                        lo = lam
                y, lam_used = y_hi, hi
        used_backend = backend
        stats.backend = backend
    y, n_rm, n_add = _repair(Q, h, v, y, max(budget, 0.0))
    E = qenergy(Q, h, const, y) if m else const
    return y, {"lambda": float(lam_used), "energy": float(E), "stats": stats.as_dict(),
               "backend": used_backend, "repair_removed": n_rm, "repair_added": n_add,
               "volume": float(v @ y), "time": time.perf_counter() - t0, "mode": mode}


def _multisection(Q, h, v, budget, lam_hi, backend, num_reads, rng, stats, backend_opts):
    """Vectorised lambda search for batch-capable single-block backends."""
    tolb = 1e-9 * max(1.0, abs(budget))
    m = h.size
    if backend == "exact":
        from .exact import spectrum
        E0 = spectrum(Q, h, 0.0)
        Vall = spectrum(np.zeros((m, m)), v, 0.0)

        def run(lams):
            t = time.perf_counter()
            out = []
            for lam in lams:
                i = int(np.argmin(E0 + lam * Vall))
                out.append(((i >> np.arange(m)) & 1).astype(float))
            stats.n_solves += 1
            stats.solver_time += time.perf_counter() - t
            stats.max_block = max(stats.max_block, m)
            return out
    else:
        from .anneal import simulated_annealing, greedy_descent

        def run(lams):
            H = h[None, :] + np.asarray(lams)[:, None] * v[None, :]
            K = len(lams)
            t = time.perf_counter()
            if backend == "sa":
                reads = max(2, int(num_reads or 32) // 8)
                r = simulated_annealing(Q, H, 0.0, num_reads=K * reads,
                                        seed=int(rng.integers(0, 2 ** 31 - 1)),
                                        sweeps=backend_opts.get("sweeps"))
                S, En, row = r["samples"], r["energies"], r["row"]
            elif backend == "tabu":
                from .anneal import tabu_search
                reads = max(2, int(num_reads or 8) // 4)
                r = tabu_search(Q, H, 0.0, num_reads=K * reads,
                                seed=int(rng.integers(0, 2 ** 31 - 1)),
                                steps=backend_opts.get("steps",
                                                       int(min(max(300, 10 * m), 6000))))
                S, En, row = r["samples"], r["energies"], r["row"]
            else:
                S_l, E_l, row_l = [], [], []
                for k in range(K):
                    g = greedy_descent(Q, H[k], 0.0, num_reads=int(num_reads or 8),
                                       seed=int(rng.integers(0, 2 ** 31 - 1)))
                    S_l.append(g["samples"])
                    E_l.append(g["energies"])
                    row_l.append(np.full(g["energies"].size, k))
                S, En, row = np.vstack(S_l), np.concatenate(E_l), np.concatenate(row_l)
            dt = time.perf_counter() - t
            stats.n_solves += 1
            stats.solver_time += dt
            stats.max_block = max(stats.max_block, m)
            out = []
            for k in range(K):
                sel = np.flatnonzero(row == k)
                out.append(S[sel[int(np.argmin(En[sel]))]].copy())
            return out
    lams = np.concatenate([[0.0], np.geomspace(lam_hi * 1e-5, lam_hi, 11)])
    ys = run(lams)
    feas = [float(v @ y) <= budget + tolb for y in ys]
    if feas[0]:
        return ys[0], 0.0
    if not any(feas):
        return np.zeros(m), lam_hi
    j = feas.index(True)
    lo, hi, y_hi = lams[j - 1], lams[j], ys[j]
    for _ in range(3):
        lams = np.linspace(lo, hi, 9)[1:-1]
        ys = run(lams)
        feas = [float(v @ y) <= budget + tolb for y in ys]
        if any(feas):
            j = feas.index(True)
            hi, y_hi = lams[j], ys[j]
            lo = lams[j - 1] if j > 0 else lo
        else:
            lo = lams[-1]
    return y_hi, float(hi)
