"""Batched dense truss finite elements (docs/QUANTUM_DESIGN.md §B).

K(A) = sum_b A_b k_b restricted to the free DOFs, with unit-area bar
stiffnesses k_b = E/L t t^T precomputed as a dense (M, nf, nf) stack, so a
batch of R designs is assembled with one tensordot and solved with one
batched ``np.linalg.solve``.  Areas are relative (x in [0, 1],
A = A_full max(x, amin)); ``amin = 1e-9`` keeps K non-singular for absent
bars.  A design is *stable* (:meth:`TrussFE.stable_mask`) if

1. the solve is finite with relative residual <= 1e-8 and c <= 1e3 c(full),
2. every loaded node is connected to a support through present bars, and
3. it is *kinematically stable*: the geometric stiffness of the present bars
   (unit areas, so the test does not depend on the area values) restricted
   to the free DOFs of the nodes they touch is positive definite (smallest
   eigenvalue > ``KIN_RTOL`` x largest).  This rejects mechanisms whose
   zero-energy mode happens to be orthogonal to the load -- e.g. a node
   joined only by two collinear bars -- which linear analysis with the
   ``amin`` area floor cannot see (docs/VERIFICATION_quantum.md M4).
"""
from __future__ import annotations

import numpy as np

__all__ = ["TrussFE", "KIN_RTOL", "PRESENT_TOL", "stability_loads"]

KIN_RTOL = 1e-9      # relative eigenvalue threshold of the kinematic test
PRESENT_TOL = 1e-6   # relative area above which a bar counts as present


class TrussFE:
    def __init__(self, prob, amin=1e-9, aux_loads=None):
        """``aux_loads``: optional (ndof, k) extra load cases appended to the
        problem's loads (used for the nominal *stability loads* of the QUBO
        model, see :func:`stability_loads`); None = the problem's loads only."""
        self.prob = prob
        self.amin = float(amin)
        nodes, bars = prob.nodes, prob.bars
        d = prob.dim
        self.dim = d
        M = bars.shape[0]
        self.M = M
        vec = nodes[bars[:, 1]] - nodes[bars[:, 0]]
        L = np.sqrt(np.einsum("ij,ij->i", vec, vec))
        self.L = L
        c = vec / L[:, None]
        t = np.concatenate([-c, c], axis=1)                       # (M, 2d)
        kb = (prob.E / L)[:, None, None] * t[:, :, None] * t[:, None, :]
        dofs = np.concatenate([bars[:, [0]] * d + np.arange(d), bars[:, [1]] * d + np.arange(d)],
                              axis=1)
        self.dofs = dofs
        free = prob.free_dofs
        self.free = free
        nf = free.size
        self.nf = nf
        fmap = np.full(prob.ndof, -1, dtype=np.int64)
        fmap[free] = np.arange(nf)
        self.fmap = fmap
        Kb = np.zeros((M, nf, nf))
        ld = fmap[dofs]                                           # (M, 2d)
        for i in range(2 * d):
            for j in range(2 * d):
                ok = (ld[:, i] >= 0) & (ld[:, j] >= 0)
                b = np.flatnonzero(ok)
                Kb[b, ld[b, i], ld[b, j]] += kb[b, i, j]
        self.Kb = Kb * prob.A_full
        self.Kb_flat = self.Kb.reshape(M, nf * nf)
        # geometric (unit-area, unit-length-scaled) stiffness for the rank test
        Kg = Kb * L[:, None, None] / prob.E
        self.Kg_flat = Kg.reshape(M, nf * nf)
        self.dof_node = free // d
        self.bar_node = np.zeros((M, prob.n_nodes))
        self.bar_node[np.arange(M), bars[:, 0]] = 1.0
        self.bar_node[np.arange(M), bars[:, 1]] = 1.0
        self.F = prob.load_matrix[free]                            # (nf, nload)
        if aux_loads is not None:
            self.F = np.hstack([self.F, np.asarray(aux_loads, dtype=float)[free]])
        # graph data
        self.inc = np.zeros((M, prob.n_nodes))
        self.inc[np.arange(M), bars[:, 0]] = 1.0
        self.inc[np.arange(M), bars[:, 1]] = 1.0
        self.loaded = prob.loaded_nodes
        self.supp = np.zeros(prob.n_nodes, dtype=bool)
        self.supp[prob.support_nodes] = True
        self.n_solves = 0
        cf = self.solve(np.ones((1, M)))[1][0]
        self.c_full = float(cf)

    # ------------------------------------------------------------------
    def areas(self, X):
        return np.maximum(np.asarray(X, dtype=float), self.amin)

    def stiffness(self, X):
        X = np.atleast_2d(self.areas(X))
        return (X @ self.Kb_flat).reshape(-1, self.nf, self.nf)

    def solve(self, X):
        """Batched solve.  Returns (U (R, nf, nload), c (R,), residual_ok (R,))."""
        X = np.atleast_2d(X)
        K = self.stiffness(X)
        F = np.broadcast_to(self.F, (K.shape[0],) + self.F.shape)
        self.n_solves += K.shape[0]
        try:
            U = np.linalg.solve(K, F)
        except np.linalg.LinAlgError:
            U = np.empty_like(F)
            for r in range(K.shape[0]):
                try:
                    U[r] = np.linalg.solve(K[r], F[r])
                except np.linalg.LinAlgError:
                    U[r] = np.nan
        c = np.einsum("rik,ik->r", U, self.F)
        R = np.einsum("rij,rjk->rik", K, U) - F
        nf = np.linalg.norm(self.F)
        res = np.sqrt(np.einsum("rik,rik->r", R, R)) / max(nf, 1e-300)
        ok = np.isfinite(c) & (res <= 1e-8)
        c = np.where(np.isfinite(c), c, np.inf)
        return U, c, ok

    def connected(self, on):
        """Every loaded node connected to a support node through present bars."""
        on = np.atleast_2d(np.asarray(on, dtype=float) > 0.5).astype(float)
        R = on.shape[0]
        N = self.prob.n_nodes
        A = np.einsum("rb,bn,bm->rnm", on, self.inc, self.inc)
        reach = np.broadcast_to(self.supp.astype(float), (R, N)).copy()
        for _ in range(N):
            new = np.minimum(1.0, reach + np.einsum("rn,rnm->rm", reach, A))
            if np.array_equal(new, reach):
                break
            reach = new
        return np.all(reach[:, self.loaded] > 0, axis=1)

    def kinematic(self, X, chunk=4096):
        """Kinematic stability of designs X (R, M): the unit-area stiffness of
        the present bars (x > PRESENT_TOL) on the free DOFs of the touched
        nodes is positive definite (min eigenvalue > KIN_RTOL * max)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.zeros(X.shape[0], dtype=bool)
        nf = self.nf
        for a in range(0, X.shape[0], chunk):
            P = (X[a:a + chunk] > PRESENT_TOL).astype(float)
            K = (P @ self.Kg_flat).reshape(-1, nf, nf)
            touched = (P @ self.bar_node) > 0                   # (r, N)
            td = touched[:, self.dof_node]                      # (r, nf)
            idx = np.arange(nf)
            K[:, idx, idx] += np.where(td, 0.0, 1.0)            # untouched DOFs: identity
            w = np.linalg.eigvalsh(K)
            wmax = np.max(w, axis=1)
            out[a:a + chunk] = (w[:, 0] > KIN_RTOL * wmax) & td.any(axis=1)
        return out

    def prune(self, x):
        """Remove dangling bars: repeatedly drop the present bars incident to
        an unloaded, unsupported node that only one present bar touches (they
        carry no force, so compliance is unchanged and volume decreases; such a
        node is a trivial mechanism).  Collinear or out-of-plane hinges that
        *do* carry force are not touched -- the kinematic test rejects them."""
        x = np.array(x, dtype=float)
        keep = self.supp.copy()
        keep[self.loaded] = True
        bars = self.prob.bars
        while True:
            on = x > PRESENT_TOL
            deg = np.bincount(bars[on].ravel(), minlength=self.prob.n_nodes)
            bad = (deg == 1) & ~keep
            drop = on & (bad[bars[:, 0]] | bad[bars[:, 1]])
            if not drop.any():
                return x
            x[drop] = 0.0

    def stable_mask(self, X, c=None, res_ok=None):
        """Stability of designs X (R, M) (criteria 1-3 of the module docs)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if c is None or res_ok is None:
            _, c, res_ok = self.solve(X)
        ok = np.asarray(res_ok, dtype=bool) & (np.asarray(c) <= 1e3 * self.c_full)
        if ok.any():
            ok[ok] = self.connected(X[ok] > PRESENT_TOL)
        if ok.any():
            ok[ok] = self.kinematic(X[ok])
        return ok

    def evaluate(self, x):
        """Compliance and stability of one design (binary or continuous)."""
        x = np.asarray(x, dtype=float)
        U, c, ok = self.solve(x[None, :])
        stable = bool(self.stable_mask(x[None, :], c, ok)[0])
        c = float(c[0])
        vol = float(self.L @ x)
        return {"compliance": c, "stable": stable, "volume": vol,
                "volume_fraction": vol / float(self.L.sum()), "U": U[0]}

    def grad(self, x):
        """c, dc/dx (M,), U (nf, nload) at relative areas x."""
        U, c, ok = self.solve(np.asarray(x, dtype=float)[None, :])
        U = U[0]
        g = -np.einsum("bij,ik,jk->b", self.Kb, U, U)
        return float(c[0]), g, U, bool(ok[0])

    def hessian(self, x, U=None):
        """Exact Hessian d2c/dx_b dx_c = 2 sum_loads (k_b u)^T K^-1 (k_c u)."""
        x = np.asarray(x, dtype=float)
        if U is None:
            U = self.grad(x)[2]
        K = self.stiffness(x[None, :])[0]
        H = np.zeros((self.M, self.M))
        for l in range(U.shape[1]):
            B = np.einsum("bij,j->ib", self.Kb, U[:, l])          # (nf, M)
            Z = np.linalg.solve(K, B)
            H += 2.0 * B.T @ Z
        self.n_solves += 1
        return 0.5 * (H + H.T)


def stability_loads(prob, weight=0.01, n_cases=2, seed=12345):
    """Nominal stability load cases (ndof, n_cases): random unit directions at
    every free node, scaled so that each case's compliance on the full ground
    structure is ``weight`` x the problem's compliance.  Added to the QUBO
    *model* only (never to the reported compliance or the feasibility test),
    they make mechanisms whose zero-energy mode is orthogonal to the real load
    expensive in the Taylor model (cf. the nominal-force method)."""
    fe0 = TrussFE(prob)
    rng = np.random.default_rng(seed)
    d = prob.dim
    G = np.zeros((prob.ndof, int(n_cases)))
    for k in range(int(n_cases)):
        v = rng.normal(size=(prob.n_nodes, d))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        G[:, k] = v.ravel()
    G[prob.fixed_dofs] = 0.0
    K = fe0.stiffness(np.ones((1, fe0.M)))[0]
    for k in range(G.shape[1]):
        gk = G[fe0.free, k]
        ck = float(gk @ np.linalg.solve(K, gk))
        if ck > 0:
            G[:, k] *= np.sqrt(weight * fe0.c_full / ck)
    return G
