"""Classical QUBO heuristics in vectorised numpy: simulated annealing (``sa``),
tabu search (``tabu``) and steepest-descent local search (``greedy``).

All work on the normalised form E(x) = x^T Q x + h.x + const (Q symmetric,
zero diagonal).  Flipping x_i changes the energy by

    dE_i = (1 - 2 x_i) (h_i + 2 f_i),   f = Q x,

and the local fields f are updated incrementally after every accepted flip.

Simulated annealing runs ``R`` replicas at once, stored as (n, R) arrays so
that one variable's replicas are contiguous.  Dense Q: random-order
single-variable Metropolis sweeps (one numpy step per variable).  Sparse Q:
the variables are greedily coloured (no two coupled variables share a colour)
and all variables of one colour are updated simultaneously -- an exact
parallel (chromatic) Metropolis scheme, one numpy step per colour.  The
linear term may differ per replica (``h`` of shape (K, n): replicas are split
evenly among the K rows); the continuum update uses this to scan several
volume multipliers lambda in one call.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

__all__ = ["simulated_annealing", "tabu_search", "greedy_descent",
           "greedy_coloring", "default_sweeps"]


def default_sweeps(n):
    """200 sweeps for n <= 200 ... 50 for n >= 5000 (log-linear in between)."""
    if n <= 200:
        return 200
    if n >= 5000:
        return 50
    t = (math.log(n) - math.log(200)) / (math.log(5000) - math.log(200))
    return int(round(200 - 150 * t))


def greedy_coloring(Q):
    """Greedy largest-degree-first colouring of the coupling graph of sparse Q.

    Returns colour per variable (int array)."""
    Q = sp.csr_matrix(Q)
    n = Q.shape[0]
    indptr, indices = Q.indptr, Q.indices
    deg = np.diff(indptr)
    order = np.argsort(-deg, kind="stable")
    color = np.full(n, -1, dtype=np.int64)
    maxc = 0
    mark = np.zeros(int(deg.max(initial=0)) + 2, dtype=bool)
    for v in order:
        nb = indices[indptr[v]:indptr[v + 1]]
        used = color[nb]
        used = used[used >= 0]
        if used.size == 0:
            c = 0
        else:
            mark[:] = False
            mark[used[used < mark.size]] = True
            c = int(np.argmin(mark))
        color[v] = c
        if c > maxc:
            maxc = c
    return color


def _as_rows(h, n):
    h = np.asarray(h, dtype=np.float64)
    if h.ndim == 1:
        h = h[None, :]
    if h.shape[1] != n:
        raise ValueError("h has the wrong length")
    return h


def _beta_range(Qabs_rowsum, Qmin_nz, hrows):
    hmax = np.abs(hrows).max(axis=0)
    max_delta = float(np.max(hmax + 2.0 * Qabs_rowsum)) if hmax.size else 1.0
    if not max_delta > 0:
        max_delta = 1.0
    hnz = np.abs(hrows[hrows != 0])
    cands = []
    if hnz.size:
        cands.append(float(hnz.min()))
    if Qmin_nz is not None and Qmin_nz > 0:
        cands.append(2.0 * Qmin_nz)
    min_delta = min(cands) if cands else max_delta
    min_delta = max(min_delta, 1e-4 * max_delta)
    return math.log(2.0) / max_delta, math.log(100.0) / min_delta


class _Colored:
    """Colour classes as contiguous ranges of a permuted variable order."""

    def __init__(self, Q):
        Q = sp.csr_matrix(Q)
        color = greedy_coloring(Q)
        perm = np.argsort(color, kind="stable")
        self.perm = perm
        self.inv = np.empty_like(perm)
        self.inv[perm] = np.arange(perm.size)
        Qp = Q[perm][:, perm].tocsc()
        cp = color[perm]
        bounds = np.flatnonzero(np.diff(cp)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [perm.size]])
        self.ranges = list(zip(starts.tolist(), ends.tolist()))
        self.sub = []
        for a, b in self.ranges:
            C = Qp[:, a:b]
            rows = np.unique(C.indices)
            self.sub.append((rows, sp.csr_matrix(C[rows])))
        self.Qp = Qp.tocsr()
        self.ncolors = len(self.ranges)


PAIR_POLISH_MAX_N = 256


def _pair_polish(Q, Xt, Ft, Ht, max_rounds=None):
    """Zero-temperature descent with single *and* pair flips (vectorised over
    replicas).  Pair moves let the search swap two variables, which single
    flips cannot do at low temperature when a quadratic penalty fixes the
    cardinality (e.g. volume='penalty')."""
    n, R = Xt.shape
    X = Xt.T.copy()
    F = Ft.T.copy()
    H = Ht.T
    ar = np.arange(R)
    iu = np.triu_indices(n, k=1)
    Qu = Q[iu]
    rounds = 0
    moves = 0
    for rounds in range(1, (max_rounds or 4 * n) + 1):
        S = 1.0 - 2.0 * X
        D = S * (H + 2.0 * F)                               # single-flip dE (R, n)
        P = D[:, iu[0]] + D[:, iu[1]] + 2.0 * Qu[None, :] * S[:, iu[0]] * S[:, iu[1]]
        k = np.argmin(P, axis=1)
        j1 = np.argmin(D, axis=1)
        bestp = P[ar, k]
        bests = D[ar, j1]
        tol = -1e-12 * np.maximum(1.0, np.abs(H).max(axis=1))
        use_pair = (bestp < bests) & (bestp < tol)
        use_one = ~use_pair & (bests < tol)
        if not (use_pair.any() or use_one.any()):
            break
        for r in np.flatnonzero(use_one):
            i = j1[r]
            d = S[r, i]
            X[r, i] += d
            F[r] += d * Q[i]
        for r in np.flatnonzero(use_pair):
            a, b = iu[0][k[r]], iu[1][k[r]]
            for i in (a, b):
                d = 1.0 - 2.0 * X[r, i]
                X[r, i] += d
                F[r] += d * Q[i]
        moves += int(use_pair.sum() + use_one.sum())
    return np.ascontiguousarray(X.T), np.ascontiguousarray(F.T), moves


def simulated_annealing(Q, h, const=0.0, num_reads=32, sweeps=None, seed=None,
                        initial_state=None, beta_range=None, polish=True,
                        force_dense=None, pair_polish=True):
    """Simulated annealing.  ``h``: (n,) or (K, n).

    Returns dict(samples (R, n) float, energies (R,), row (R,) index of the h
    row each replica used, info).
    """
    rng = np.random.default_rng(seed)
    hrows = _as_rows(h, Q.shape[0])
    n = Q.shape[0]
    K = hrows.shape[0]
    R = max(int(num_reads), K)
    R = int(math.ceil(R / K) * K)
    row = np.repeat(np.arange(K), R // K)
    if n == 0:
        return dict(samples=np.zeros((R, 0)), energies=np.full(R, const),
                    row=row, info={"sweeps": 0})
    sweeps = int(sweeps) if sweeps else default_sweeps(n)
    sparse_in = sp.issparse(Q)
    colored = None
    if sparse_in:
        Qc = sp.csr_matrix(Q)
        use_dense = force_dense if force_dense is not None else (n <= 64)
        if not use_dense:
            colored = _Colored(Qc)
            if force_dense is None and colored.ncolors > 0.35 * n and n <= 4000:
                colored = None
                use_dense = True
        Qd = Qc.toarray() if colored is None else None
        Qabs_rowsum = np.asarray(abs(Qc).sum(axis=1)).ravel()
        Qmin = float(abs(Qc.data).min()) if Qc.nnz else None
    else:
        Qd = np.asarray(Q, dtype=np.float64)
        Qabs_rowsum = np.abs(Qd).sum(axis=1)
        nz = np.abs(Qd[Qd != 0])
        Qmin = float(nz.min()) if nz.size else None
    b0, b1 = beta_range if beta_range is not None else _beta_range(Qabs_rowsum, Qmin, hrows)
    betas = np.geomspace(b0, b1, sweeps) if sweeps > 1 else np.array([b1])

    # state (n, R) in the working order
    if initial_state is not None:
        x0 = np.asarray(initial_state, dtype=np.float64).reshape(-1, n)
        Xt = np.empty((n, R))
        for r in range(R):
            Xt[:, r] = x0[r % x0.shape[0]]
    else:
        Xt = (rng.random((n, R)) < 0.5).astype(np.float64)
    Ht = np.ascontiguousarray(hrows[row].T)
    if colored is not None:
        P = colored.perm
        Xt = Xt[P]
        Ht = Ht[P]
        Ft = colored.Qp @ Xt
    else:
        Ft = Qd @ Xt
    Ft = np.ascontiguousarray(Ft)
    n_acc = 0

    def dense_sweep(Tsw, order):
        nonlocal Ft
        acc_count = 0
        for i in order:
            x = Xt[i]
            d = 1.0 - 2.0 * x
            dE = d * (Ht[i] + 2.0 * Ft[i])
            acc = dE < Tsw[i]
            if acc.any():
                d *= acc
                x += d
                Ft += np.outer(Qd[:, i], d)
                acc_count += 1
        return acc_count

    def colored_sweep(Tsw, corder):
        acc_count = 0
        for c in corder:
            a, b = colored.ranges[c]
            X = Xt[a:b]
            d = 1.0 - 2.0 * X
            dE = d * (Ht[a:b] + 2.0 * Ft[a:b])
            acc = dE < Tsw[a:b]
            if acc.any():
                d *= acc
                X += d
                rows, S = colored.sub[c]
                Ft[rows] += S @ d
                acc_count += 1
        return acc_count

    for s in range(sweeps):
        Tsw = -np.log(rng.random((n, R)) + 1e-300) / betas[s]
        if colored is None:
            n_acc += dense_sweep(Tsw, rng.permutation(n))
        else:
            n_acc += colored_sweep(Tsw, rng.permutation(colored.ncolors))
    n_polish = 0
    if polish:
        zero = np.zeros((n, R))
        for _ in range(4 * n + 10):
            n_polish += 1
            if colored is None:
                k = dense_sweep(zero, rng.permutation(n))
            else:
                k = colored_sweep(zero, rng.permutation(colored.ncolors))
            if k == 0:
                break
    if colored is not None:
        Xt = Xt[colored.inv]
        Ft = Ft[colored.inv]
        Ht = Ht[colored.inv]
    n_pair = 0
    if polish and pair_polish and 2 <= n <= PAIR_POLISH_MAX_N:
        Qp = Qd if Qd is not None else (Q.toarray() if sp.issparse(Q) else np.asarray(Q))
        Xt, Ft, n_pair = _pair_polish(Qp, Xt, Ft, Ht)
    E = np.einsum("ir,ir->r", Xt, Ft) + np.einsum("ir,ir->r", Xt, Ht) + const
    info = {"sweeps": sweeps, "beta_range": (float(b0), float(b1)),
            "path": "colored" if colored is not None else "dense",
            "n_colors": colored.ncolors if colored is not None else n,
            "polish_sweeps": n_polish, "pair_moves": n_pair}
    return dict(samples=np.ascontiguousarray(Xt.T), energies=E, row=row, info=info)


def _rows_dense(Q, idx):
    if sp.issparse(Q):
        return Q[idx].toarray()
    return Q[idx]


def tabu_search(Q, h, const=0.0, num_reads=8, steps=None, tenure=None, seed=None,
                initial_state=None):
    """1-flip tabu search, ``num_reads`` independent restarts run in parallel
    (vectorised over restarts).  Aspiration: a tabu move is allowed when it
    gives a new best energy for that restart."""
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    hrows = _as_rows(h, n)
    K = hrows.shape[0]
    R = max(1, int(num_reads), K)
    R = int(math.ceil(R / K) * K)
    row = np.repeat(np.arange(K), R // K)
    h = hrows[row]                                   # (R, n) linear term per restart
    if n == 0:
        return dict(samples=np.zeros((R, 0)), energies=np.full(R, const), row=row, info={})
    if sp.issparse(Q) and n <= 5000:
        Qw = Q.toarray()
    else:
        Qw = Q if sp.issparse(Q) else np.asarray(Q, dtype=np.float64)
    steps = int(steps) if steps else int(min(max(500, 30 * n), 30000))
    tenure = int(tenure) if tenure else int(max(3, min(n // 10 + 2, 50, n - 1 if n > 1 else 1)))
    X = (rng.random((R, n)) < 0.5).astype(np.float64)
    if initial_state is not None:
        x0 = np.asarray(initial_state, dtype=np.float64).reshape(-1, n)
        X[:x0.shape[0]] = x0[:R]
    F = (Qw @ X.T).T if sp.issparse(Qw) else X @ Qw
    E = np.einsum("rn,rn->r", X, F) + np.einsum("rn,rn->r", X, h) + const
    bestE = E.copy()
    bestX = X.copy()
    tabu_until = np.zeros((R, n), dtype=np.int64)
    ar = np.arange(R)
    eps = 1e-12 * max(1.0, float(np.abs(hrows).max()))
    last_improve = np.zeros(R, dtype=np.int64)
    stall = max(200, 10 * n)
    for step in range(steps):
        dE = (1.0 - 2.0 * X) * (h + 2.0 * F)
        allowed = (tabu_until <= step) | ((E[:, None] + dE) < bestE[:, None] - eps)
        dEm = np.where(allowed, dE, np.inf)
        j = np.argmin(dEm, axis=1)
        blocked = ~np.isfinite(dEm[ar, j])
        if blocked.any():
            j[blocked] = np.argmin(tabu_until[blocked], axis=1)
        d = 1.0 - 2.0 * X[ar, j]
        X[ar, j] += d
        F += d[:, None] * _rows_dense(Qw, j)
        E += dE[ar, j]
        tabu_until[ar, j] = step + 1 + tenure
        imp = E < bestE - eps
        if imp.any():
            bestE[imp] = E[imp]
            bestX[imp] = X[imp]
            last_improve[imp] = step
        # restart stalled searches from a perturbed best state
        st = (step - last_improve) > stall
        if st.any():
            k = np.flatnonzero(st)
            Xn = bestX[k].copy()
            flip = rng.random(Xn.shape) < 0.25
            Xn[flip] = 1.0 - Xn[flip]
            X[k] = Xn
            F[k] = (Qw @ Xn.T).T if sp.issparse(Qw) else Xn @ Qw
            E[k] = np.einsum("rn,rn->r", Xn, F[k]) + np.einsum("rn,rn->r", Xn, h[k]) + const
            tabu_until[k] = 0
            last_improve[k] = step
    # recompute exactly (drift-free)
    Fb = (Qw @ bestX.T).T if sp.issparse(Qw) else bestX @ Qw
    Eb = np.einsum("rn,rn->r", bestX, Fb) + np.einsum("rn,rn->r", bestX, h) + const
    return dict(samples=bestX, energies=Eb, row=row, info={"steps": steps, "tenure": tenure})


def greedy_descent(Q, h, const=0.0, initial_state=None, num_reads=1, seed=None,
                   max_steps=None):
    """Steepest-descent single flips until a local minimum (vectorised over
    replicas).  With ``initial_state`` the first replica starts there."""
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    h = np.asarray(h, dtype=np.float64)
    R = max(1, int(num_reads))
    if n == 0:
        return dict(samples=np.zeros((R, 0)), energies=np.full(R, const), info={})
    Qw = Q.toarray() if (sp.issparse(Q) and n <= 5000) else Q
    X = (rng.random((R, n)) < 0.5).astype(np.float64)
    if initial_state is not None:
        x0 = np.asarray(initial_state, dtype=np.float64).reshape(-1, n)
        X[:x0.shape[0]] = x0[:R]
    F = (Qw @ X.T).T if sp.issparse(Qw) else X @ Qw
    ar = np.arange(R)
    max_steps = int(max_steps) if max_steps else 4 * n + 10
    steps = 0
    for steps in range(1, max_steps + 1):
        dE = (1.0 - 2.0 * X) * (h + 2.0 * F)
        j = np.argmin(dE, axis=1)
        go = dE[ar, j] < -1e-14 * max(1.0, float(np.abs(h).max()))
        if not go.any():
            break
        r = ar[go]
        jj = j[go]
        d = 1.0 - 2.0 * X[r, jj]
        X[r, jj] += d
        F[r] += d[:, None] * _rows_dense(Qw, jj)
    E = np.einsum("rn,rn->r", X, F) + X @ h + const
    return dict(samples=X, energies=E, info={"steps": steps})
