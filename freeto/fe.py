"""Finite elements: lk_H8, fast repeated assembly and linear solvers."""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .errors import FreeTOError, SINGULAR_MSG

__all__ = ["lk_H8", "Assembler", "make_solver", "available_solvers"]


# ----------------------------------------------------------------------------
# element stiffness (lk_H8.m, verbatim)
# ----------------------------------------------------------------------------
def lk_H8(nu):
    A = np.array([[32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
                  [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12]],
                 dtype=float)
    k = 1 / 144 * A.T @ np.array([1.0, nu])
    k = np.concatenate([[0.0], k])  # 1-based access k[1]..k[14]
    K1 = np.array([[k[1], k[2], k[2], k[3], k[5], k[5]],
                   [k[2], k[1], k[2], k[4], k[6], k[7]],
                   [k[2], k[2], k[1], k[4], k[7], k[6]],
                   [k[3], k[4], k[4], k[1], k[8], k[8]],
                   [k[5], k[6], k[7], k[8], k[1], k[2]],
                   [k[5], k[7], k[6], k[8], k[2], k[1]]])
    K2 = np.array([[k[9], k[8], k[12], k[6], k[4], k[7]],
                   [k[8], k[9], k[12], k[5], k[3], k[5]],
                   [k[10], k[10], k[13], k[7], k[4], k[6]],
                   [k[6], k[5], k[11], k[9], k[2], k[10]],
                   [k[4], k[3], k[5], k[2], k[9], k[12]],
                   [k[11], k[4], k[6], k[12], k[10], k[13]]])
    K3 = np.array([[k[6], k[7], k[4], k[9], k[12], k[8]],
                   [k[7], k[6], k[4], k[10], k[13], k[10]],
                   [k[5], k[5], k[3], k[8], k[12], k[9]],
                   [k[9], k[10], k[2], k[6], k[11], k[5]],
                   [k[12], k[13], k[10], k[11], k[6], k[4]],
                   [k[2], k[12], k[9], k[4], k[5], k[3]]])
    K4 = np.array([[k[14], k[11], k[11], k[13], k[10], k[10]],
                   [k[11], k[14], k[11], k[12], k[9], k[8]],
                   [k[11], k[11], k[14], k[12], k[8], k[9]],
                   [k[13], k[12], k[12], k[14], k[7], k[7]],
                   [k[10], k[9], k[8], k[7], k[14], k[11]],
                   [k[10], k[8], k[9], k[7], k[11], k[14]]])
    K5 = np.array([[k[1], k[2], k[8], k[3], k[5], k[4]],
                   [k[2], k[1], k[8], k[4], k[6], k[11]],
                   [k[8], k[8], k[1], k[5], k[11], k[6]],
                   [k[3], k[4], k[5], k[1], k[8], k[2]],
                   [k[5], k[6], k[11], k[8], k[1], k[8]],
                   [k[4], k[11], k[6], k[2], k[8], k[1]]])
    K6 = np.array([[k[14], k[11], k[7], k[13], k[10], k[12]],
                   [k[11], k[14], k[7], k[12], k[9], k[2]],
                   [k[7], k[7], k[14], k[10], k[2], k[9]],
                   [k[13], k[12], k[10], k[14], k[7], k[11]],
                   [k[10], k[9], k[2], k[7], k[14], k[7]],
                   [k[12], k[2], k[9], k[11], k[7], k[14]]])
    KE = 1 / ((nu + 1) * (1 - 2 * nu)) * np.block(
        [[K1, K2, K3, K4],
         [K2.T, K5, K6, K3.T],
         [K3.T, K6, K5.T, K2.T],
         [K4, K3, K2, K1.T]])
    return KE


# ----------------------------------------------------------------------------
# assembly
# ----------------------------------------------------------------------------
def _idx_dtype(n):
    return np.int32 if n < 2**31 - 1 else np.int64


class _SymView:
    """Square symmetric matrix defined by a subset of the lower-triangle
    entries of the master pattern (plus optional extra diagonal entries),
    with precomputed gathers for the full CSR and upper-triangle CSR."""

    def __init__(self, n, rows, cols, src):
        # rows >= cols (lower), src: index into the master data array
        # (or -1 - k for "extra" diagonal k)
        self.n = n
        off = rows != cols
        fr = np.concatenate([rows, cols[off]])
        fc = np.concatenate([cols, rows[off]])
        fs = np.concatenate([src, src[off]])
        order = np.lexsort((fc, fr))
        fr, fc, fs = fr[order], fc[order], fs[order]
        it = _idx_dtype(max(n, fr.size))
        indptr = np.zeros(n + 1, dtype=it)
        np.cumsum(np.bincount(fr, minlength=n), out=indptr[1:])
        self.full_indptr = indptr
        self.full_indices = fc.astype(it)
        self.full_src = fs
        # upper triangle CSR (row <= col): transpose of the lower entries
        order = np.lexsort((rows, cols))
        ur, uc, us = cols[order], rows[order], src[order]
        indptr = np.zeros(n + 1, dtype=it)
        np.cumsum(np.bincount(ur, minlength=n), out=indptr[1:])
        self.up_indptr = indptr
        self.up_indices = uc.astype(it)
        self.up_src = us

    @staticmethod
    def _gather(data, extra, src):
        if extra is None:
            return data[src]
        allv = np.concatenate([data, extra])
        return allv[np.where(src >= 0, src, data.size + (-1 - src))]

    def full(self, data, extra=None):
        return sp.csr_matrix((self._gather(data, extra, self.full_src),
                              self.full_indices, self.full_indptr),
                             shape=(self.n, self.n))

    def upper(self, data, extra=None):
        return sp.csr_matrix((self._gather(data, extra, self.up_src),
                              self.up_indices, self.up_indptr),
                             shape=(self.n, self.n))


class Assembler:
    """Repeated assembly of K = sum_e E_e KE over the active elements.

    The lower-triangle pattern over the active DOFs (``n_vec`` in the MATLAB
    code) is built once; each iteration only computes the data array as a
    sparse matrix-vector product ``M @ E`` (M maps element moduli to pattern
    entries).  K is exactly symmetric by construction (as MATLAB's
    ``(K+K')/2``).
    """

    def __init__(self, edofMatn, KE, freedofs, ndof):
        t0 = time.perf_counter()
        edof = np.asarray(edofMatn, dtype=np.int64)
        nnele = edof.shape[0]
        self.nnele = nnele
        self.ndof = ndof
        act = np.unique(edof)
        self.act = act
        nact = act.size
        gmap = np.full(ndof, -1, dtype=np.int64)
        gmap[act] = np.arange(nact)
        freedofs = np.asarray(freedofs, dtype=np.int64)
        self.freedofs = freedofs
        isfree = np.zeros(nact, dtype=bool)
        isfree[gmap[freedofs]] = True
        self.isfree_act = isfree
        la, lb = np.tril_indices(24)
        kev = KE[la, lb]
        # local element entries -> active dof pairs (lower)
        le = gmap[edof]                                   # (nnele, 24)
        ra = le[:, la]
        rb = le[:, lb]
        r = np.maximum(ra, rb).ravel()
        c = np.minimum(ra, rb).ravel()
        del ra, rb
        key = r * nact + c
        del r, c
        ukey, inv = np.unique(key, return_inverse=True)
        del key
        nnz = ukey.size
        self.rows = (ukey // nact).astype(np.int64)
        self.cols = (ukey % nact).astype(np.int64)
        del ukey
        it = _idx_dtype(max(nnz, nnele))
        # M in CSC: column e holds the 300 entries of element e
        indptr = np.arange(0, 300 * nnele + 1, 300, dtype=np.int64)
        self.M = sp.csc_matrix((np.tile(kev, nnele), inv.astype(it),
                                indptr.astype(it) if it == np.int32 else indptr),
                               shape=(nnz, nnele)).tocsr()
        del inv
        self.nnz = nnz
        self.nact = nact
        self._views = {}
        self.setup_time = time.perf_counter() - t0

    def data(self, Ee):
        """Lower-triangle pattern values for element moduli ``Ee``."""
        return self.M @ np.asarray(Ee, dtype=np.float64)

    # views --------------------------------------------------------------
    def free_view(self):
        """Symmetric matrix on the free DOFs (``K(freedofs,freedofs)``)."""
        if "free" not in self._views:
            fmap = np.full(self.nact, -1, dtype=np.int64)
            fmap[self.isfree_act] = np.arange(self.isfree_act.sum())
            fr = fmap[self.rows]
            fc = fmap[self.cols]
            keep = (fr >= 0) & (fc >= 0)
            src = np.nonzero(keep)[0]
            self._views["free"] = _SymView(int(self.isfree_act.sum()),
                                           fr[keep], fc[keep], src)
        return self._views["free"]

    def active_view(self):
        """Symmetric matrix on all active DOFs where fixed DOFs keep only
        their diagonal (identity-like rows).  Keeps complete node triplets,
        which lets AMG use 3x3 blocks."""
        if "act" not in self._views:
            fr = self.isfree_act[self.rows]
            fc = self.isfree_act[self.cols]
            diag = self.rows == self.cols
            keep = (fr & fc) | (diag & ~fr)
            src = np.nonzero(keep)[0]
            self._views["act"] = _SymView(self.nact, self.rows[keep],
                                          self.cols[keep], src)
        return self._views["act"]


# ----------------------------------------------------------------------------
# linear solvers
# ----------------------------------------------------------------------------
def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def available_solvers():
    out = ["superlu", "amg"] if _have("pyamg") else ["superlu"]
    if _have("sksparse.cholmod"):
        out.insert(0, "cholmod")
    try:
        import pypardiso  # noqa: F401
        out.insert(0 if "cholmod" not in out else 1, "pardiso")
    except Exception:
        pass
    return out


class _Base:
    name = "base"
    uses = "free"   # which view

    def __init__(self, asm, log=None):
        self.asm = asm
        self.log = log or (lambda s: None)
        self.stats = {}

    def solve(self, data, F_free, U0_free=None):
        raise NotImplementedError

    def close(self):
        """Release memory held outside Python (PARDISO factorisations); a
        no-op for the other backends.  Safe to call more than once."""
        return None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _verify(self, data, F, U, tol=1e-6):
        """Raise FreeTOError if U is not a valid solution (singular K)."""
        U = np.asarray(U)
        if not np.all(np.isfinite(U)):
            raise FreeTOError(SINGULAR_MSG + " (non-finite displacements)")
        K = self.asm.free_view().full(data)
        F = np.asarray(F, dtype=np.float64).reshape(U.shape)
        nf = np.linalg.norm(F, axis=0)
        res = np.linalg.norm(K @ U - F, axis=0) / np.where(nf > 0, nf, 1.0)
        if np.any(res > tol):
            raise FreeTOError(SINGULAR_MSG + " (relative residual %.1e)"
                              % float(res.max()))
        return U


class SuperLUSolver(_Base):
    name = "superlu"

    def solve(self, data, F, U0=None):
        v = self.asm.free_view()
        K = v.full(data).tocsc()
        lu = spla.splu(K, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
        U = lu.solve(np.asarray(F, dtype=np.float64))
        self.last_lu = lu     # reused for extra right-hand sides (QUBO block Hessian)
        return self._verify(data, F, U)


class CholmodSolver(_Base):
    name = "cholmod"

    def __init__(self, asm, log=None):
        super().__init__(asm, log)
        from sksparse.cholmod import analyze  # noqa: F401
        self._factor = None

    def solve(self, data, F, U0=None):
        from sksparse.cholmod import analyze
        v = self.asm.free_view()
        K = v.full(data).tocsc()
        if self._factor is None:
            self._factor = analyze(K)
        self._factor.cholesky_inplace(K)
        return self._verify(data, F, self._factor(np.asarray(F, dtype=np.float64)))


class PardisoSolver(_Base):
    name = "pardiso"

    def __init__(self, asm, log=None):
        super().__init__(asm, log)
        from pypardiso import PyPardisoSolver
        self._ps = PyPardisoSolver(mtype=2)
        self._analysed = False

    def solve(self, data, F, U0=None):
        v = self.asm.free_view()
        K = v.upper(data)
        ps = self._ps
        b = np.asfortranarray(np.asarray(F, dtype=np.float64))
        ps._check_A(K)
        if not self._analysed:
            ps.set_phase(11)
            ps._call_pardiso(K, b)
            self._analysed = True
        ps.set_phase(22)
        ps._call_pardiso(K, b)
        ps.set_phase(33)
        return self._verify(data, F, ps._call_pardiso(K, b))

    def close(self):
        # MKL keeps the factorisation until phase -1 is called; without this a
        # process that creates many solvers (study worker, refined evaluation
        # of many designs) grows by the factorisation size per solver
        ps = getattr(self, "_ps", None)
        if ps is not None:
            try:
                ps.free_memory(everything=True)
            except Exception:
                pass
            self._ps = None
            self._analysed = False


def _rigid_body_modes(coords):
    """6 rigid-body modes for DOFs with node coordinates (nnode,3)."""
    n = coords.shape[0]
    x, y, z = (coords - coords.mean(axis=0)).T
    B = np.zeros((n, 3, 6))
    B[:, 0, 0] = 1
    B[:, 1, 1] = 1
    B[:, 2, 2] = 1
    B[:, 0, 3] = -y
    B[:, 1, 3] = x
    B[:, 1, 4] = -z
    B[:, 2, 4] = y
    B[:, 0, 5] = z
    B[:, 2, 5] = -x
    return B.reshape(3 * n, 6)


class AMGSolver(_Base):
    """pyamg smoothed-aggregation preconditioned CG.

    Works on the active-DOF system (fixed DOFs keep only their diagonal) with
    the six rigid-body modes as near-nullspace, symmetric Gauss-Seidel
    smoothing, and a warm start from the previous displacement.

    Hierarchy reuse: rebuilding the hierarchy costs roughly 25 V-cycles, so
    once the design changes slowly the previous hierarchy is tried first with
    an iteration budget of ~1.3x the count seen right after the last rebuild;
    if the budget is exceeded the hierarchy is rebuilt and PCG continues
    from the current iterate (reuse is then paused for a few iterations).
    If PCG fails altogether a direct solver is used.
    """
    name = "amg"
    uses = "act"

    def __init__(self, asm, node_coords, log=None, rtol=1e-8, maxiter=3000):
        super().__init__(asm, log)
        import pyamg  # noqa: F401
        self.rtol = rtol
        self.maxiter = maxiter
        self.B = _rigid_body_modes(node_coords)
        self.ml = None
        self.M = None
        self.base_its = None
        self.fallback = None
        self.n_builds = 0
        self.pause = 1
        self.backoff = 2
        self.total_its = 0

    def _build(self, A):
        import pyamg
        gs = ("gauss_seidel", {"sweep": "symmetric"})
        self.ml = pyamg.smoothed_aggregation_solver(
            A, B=self.B, symmetry="symmetric",
            strength=("symmetric", {"theta": 0.0}),
            smooth=("jacobi", {"omega": 4.0 / 3.0}),
            presmoother=gs, postsmoother=gs, improve_candidates=None,
            max_coarse=500, max_levels=12, coarse_solver="splu", keep=False)
        self.M = self.ml.aspreconditioner(cycle="V")
        self.base_its = None
        self.n_builds += 1

    def solve(self, data, F, U0=None):
        asm = self.asm
        free = asm.isfree_act
        A = asm.active_view().full(data)
        nrhs = F.shape[1]
        stale = self.ml is not None
        if stale and self.pause > 0:
            self.pause -= 1
            self._build(A)
            stale = False
        elif not stale:
            self._build(A)
        X = np.zeros((asm.nact, nrhs))
        tot_its = 0
        worst = 0.0
        for k in range(nrhs):
            b = np.zeros(asm.nact)
            b[free] = F[:, k]
            x = np.zeros(asm.nact)
            if U0 is not None:
                x[free] = U0[:, k]
            res = np.inf
            if stale and self.base_its is not None:
                budget = int(1.3 * self.base_its) + 5
                x, its, res = self._pcg(A, b, x, budget)
                tot_its += its
                if res > self.rtol:
                    # stale hierarchy too weak: rebuild, back off reuse
                    self._build(A)
                    stale = False
                    self.backoff = min(2 * self.backoff, 16)
                    self.pause = self.backoff
            if res > self.rtol:
                x, its, res = self._pcg(A, b, x, self.maxiter)
                tot_its += its
                if self.base_its is None:
                    self.base_its = max(its, 1)
            worst = max(worst, res)
            X[:, k] = x
        if stale:
            self.backoff = 2      # reuse worked
        self.total_its += tot_its
        self.stats = {"pcg_iterations": tot_its, "amg_builds": self.n_builds}
        if not np.isfinite(worst) or worst > max(1e3 * self.rtol, 1e-6):
            self.log("AMG-PCG did not converge (rel. residual %.2e); falling "
                     "back to a direct solver" % worst)
            if self.fallback is None:
                self.fallback = _direct_fallback(asm, self.log)
            self.ml = None
            return self.fallback.solve(data, F)
        return X[free]

    def _pcg(self, A, b, x, maxiter):
        """Preconditioned CG; returns (x, iterations, true rel. residual)."""
        nb = np.linalg.norm(b)
        if nb == 0:
            return np.zeros_like(b), 0, 0.0
        M = self.M
        r = b - A @ x
        tol = self.rtol * nb
        its = 0
        if np.linalg.norm(r) > tol:
            z = M @ r
            p = z.copy()
            rz = r @ z
            while its < maxiter:
                its += 1
                Ap = A @ p
                pAp = p @ Ap
                if not (pAp > 0):
                    break
                alpha = rz / pAp
                x += alpha * p
                r -= alpha * Ap
                if np.linalg.norm(r) <= tol:
                    break
                z = M @ r
                rz_new = r @ z
                p *= rz_new / rz
                p += z
                rz = rz_new
        res = np.linalg.norm(b - A @ x) / nb
        return x, its, res


def _direct_fallback(asm, log):
    avail = available_solvers()
    if "cholmod" in avail:
        return CholmodSolver(asm, log)
    if "pardiso" in avail:
        return PardisoSolver(asm, log)
    return SuperLUSolver(asm, log)


#: Above this many free DOFs, a direct factorization (CHOLMOD/PARDISO) can
#: need tens of GB of RAM as problems approach the `max_dofs` guard elsewhere
#: in the code; AMG-preconditioned CG uses far less memory for the same
#: system. Only affects the *selection* made by solver='auto' — explicitly
#: requesting solver='cholmod'/'pardiso' is never overridden, and the linear
#: system, tolerances and result are identical regardless of which backend
#: solves it.
DIRECT_SOLVER_DOF_CAP = 1_500_000


def make_solver(name, asm, node_coords=None, log=None):
    """Create a solver backend.  ``name='auto'`` picks cholmod > pardiso >
    (superlu if nfree < 10k else amg), except above
    ``DIRECT_SOLVER_DOF_CAP`` free DOFs, where it prefers amg (if available)
    over a direct factorization to avoid exhausting memory on very large
    problems."""
    avail = available_solvers()
    nfree = asm.freedofs.size
    if name == "auto":
        if nfree > DIRECT_SOLVER_DOF_CAP and "amg" in avail:
            name = "amg"
        elif "cholmod" in avail:
            name = "cholmod"
        elif "pardiso" in avail:
            name = "pardiso"
        elif nfree < 10000 or "amg" not in avail:
            name = "superlu"
        else:
            name = "amg"
    if name not in avail:
        raise ValueError(f"linear solver '{name}' is not available "
                         f"(available: {', '.join(avail)})")
    if name == "superlu":
        return SuperLUSolver(asm, log)
    if name == "cholmod":
        return CholmodSolver(asm, log)
    if name == "pardiso":
        return PardisoSolver(asm, log)
    if name == "amg":
        return AMGSolver(asm, node_coords, log)
    raise ValueError(name)
