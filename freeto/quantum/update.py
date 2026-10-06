"""QUBOUpdater: the QUBO design update of ``optimizer="QUBO"`` in
:func:`freeto.core.run_freeto` (docs/QUANTUM_DESIGN.md §A).

Per iteration (called exactly where the OC update is):

1. scale: g = dc / s, s = max|dc|; v = dv (filtered element volumes).
2. volume target V_k = max(V_{k-1}(1 - er), vol*nnele) (init="solid"),
   in units of v.x = sum of the filtered densities.
3. free set C_k: grey band of the smoothed densities + the f*V_k solid
   elements with the smallest |g|/v + the f*V_k void elements with the
   largest |g|/v, minus keep (MusD) elements.
4. model over y = x[C]:
       g.y + 1/2 (y - x_k)^T Qc (y - x_k) + gamma sum_N (x_e - x_f)^2 + mu |y - x_k|
   with the design-space Hessian Qc = J^T H_rho J (J = H diag(1/Hs)):
     * "none"   : 0 (BESO/sorting control),
     * "diag"   : -E''_e c_e on the diagonal of H_rho (SIMP curvature, no solve),
     * "scalar" : "diag" with every off-diagonal of Qc zeroed (separable
                  control: same curvature, no couplings),
     * "block"  : "diag" + the exact Gram term 2 Y_P^T K^-1 Y_P inside Morton
                  patches P (one multi-RHS solve with the existing
                  factorisation; patches = the solver blocks, or
                  ``hessian_block_size`` for whole-set backends).
5. solve with :func:`freeto.quantum.blockqubo.solve_volume_model`
   (block Gauss-Seidel, lambda bisection or penalty, greedy repair).
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components, dijkstra

from .backends import default_block_size
from .blockqubo import solve_volume_model
from .options import QUBOOptions

__all__ = ["QUBOUpdater", "multi_rhs_solve", "robust_solve", "morton_key",
           "oc_multiplier"]


# ----------------------------------------------------------------------------
def _part1by2(x):
    x = x.astype(np.uint64) & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def morton_key(i, j, k):
    """Z-order (Morton) key of integer grid coordinates."""
    return (_part1by2(np.asarray(i)) | (_part1by2(np.asarray(j)) << np.uint64(1))
            | (_part1by2(np.asarray(k)) << np.uint64(2)))


def oc_multiplier(vxPhys, dc, dv, target, move=0.1):
    """Lagrange multiplier of the OC bisection (same loop as oc_update)."""
    l1, l2 = 0.0, 1e9
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = -dc / dv
    lo, hi = vxPhys - move, vxPhys + move
    lmid = 0.5 * (l1 + l2)
    while l1 + l2 > 0 and (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        with np.errstate(invalid="ignore"):
            cand = vxPhys * np.sqrt(ratio / lmid)
        x = np.maximum(0.0, np.maximum(lo, np.minimum(1.0, np.minimum(hi, cand))))
        if x.sum() > target:
            l1 = lmid
        else:
            l2 = lmid
    return lmid


def multi_rhs_solve(solver, asm, data, Y, log=None):
    """Z = K^-1 Y on the free DOFs reusing the factorisation of the current
    iteration's FE solve where the solver keeps one (SuperLU: ``last_lu``,
    CHOLMOD: ``_factor``, PARDISO: phase 33 with the existing factor).
    AMG (no factorisation) falls back to a SuperLU factorisation."""
    Y = np.asfortranarray(np.asarray(Y, dtype=np.float64))
    name = getattr(solver, "name", "")
    if name == "superlu" and getattr(solver, "last_lu", None) is not None:
        return solver.last_lu.solve(Y)
    if name == "cholmod" and getattr(solver, "_factor", None) is not None:
        return np.asarray(solver._factor(Y))
    if name == "pardiso" and getattr(solver, "_ps", None) is not None:
        ps = solver._ps
        K = asm.free_view().upper(data)
        ps.set_phase(33)
        return np.asarray(ps._call_pardiso(K, Y))
    K = asm.free_view().full(data).tocsc()
    lu = spla.splu(K, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                   options=dict(SymmetricMode=True))
    return lu.solve(Y)


def robust_solve(solver, asm, data, F, U0=None, tol=1e-6):
    """``solver.solve`` with iterative refinement for crisp 0/1 designs.

    Binary designs (void modulus ~1e-9 E0, possibly with floating islands)
    can leave a direct solver at a relative residual of 1e-6..1e-4, which the
    core's check rejects as "singular".  Here the existing factorisation is
    reused for a few refinement steps.  A residual that stays in
    (tol, 1e-3] is accepted: it indicates a load path through void material
    (compliance 1e5..1e8 x larger -- a legitimately terrible design, which the
    optimiser must be able to see); larger residuals / non-finite values still
    raise FreeTOError."""
    from ..errors import FreeTOError, SINGULAR_MSG
    try:
        return solver.solve(data, F, U0)
    except FreeTOError as e:
        if "relative residual" not in str(e):
            raise
    F = np.asarray(F, dtype=np.float64)
    K = asm.free_view().full(data)
    U = multi_rhs_solve(solver, asm, data, F)
    nf = np.linalg.norm(F, axis=0)
    nf = np.where(nf > 0, nf, 1.0)
    res = np.inf
    for _ in range(10):
        R = F - K @ U
        res = float((np.linalg.norm(R, axis=0) / nf).max())
        if res <= 1e-10:
            break
        U = U + multi_rhs_solve(solver, asm, data, R)
    if not np.all(np.isfinite(U)) or res > max(tol, 1e-3):
        raise FreeTOError(SINGULAR_MSG + " (relative residual %.1e after refinement)" % res)
    return U


# ----------------------------------------------------------------------------
class QUBOUpdater:
    """State and logic of the continuum QUBO update (see module docstring)."""

    #: optional callable receiving each iteration's model (used by the study
    #: to extract sub-QUBOs for the QAOA scans); None in normal use
    capture = None

    def __init__(self, opts, *, asm, solver, KE, edofMatn, freedofs, H, Hs, ele,
                 MusD, nelx, nely, nelz, vol, nnele, E0, Emin, penal, simp, ndof,
                 log=None, F=None):
        o = QUBOOptions.from_any(opts).normalized()
        self.opts = o
        self.log = log or (lambda s: None)
        self.asm, self.solver, self.KE = asm, solver, KE
        self.edofMatn = edofMatn
        self.nnele = int(nnele)
        self.vol = float(vol)
        self.E0, self.Emin, self.penal, self.simp = float(E0), float(Emin), float(penal), simp
        self.H, self.Hs = H, Hs
        self.J = sp.csr_matrix(sp.diags(1.0 / Hs) @ H)          # rho = J x
        self.Jc = self.J.tocsc()
        self.ele = ele
        self.musd_pos = np.flatnonzero(np.isin(ele, MusD))
        self.is_musd = np.zeros(nnele, dtype=bool)
        self.is_musd[self.musd_pos] = True
        # backend / hessian resolution
        be = o.backend
        if be == "auto":
            be = "sa"
        self.backend = be
        hess = o.hessian
        direct = getattr(solver, "name", "") in ("superlu", "cholmod", "pardiso")
        if hess == "auto":
            hess = "block" if direct else "diag"
        elif hess == "block" and not direct:
            self.log("QUBO: hessian='block' needs a factorised stiffness matrix; the AMG "
                     "solver has none -> using hessian='diag' (set solver='superlu'/"
                     "'pardiso'/'cholmod' for the exact block Hessian)")
            hess = "diag"
        self.hessian = hess
        self.block_size = o.block_size if o.block_size is not None else default_block_size(be)
        # grid coordinates and Morton keys of the active elements
        jj = ele % nely
        ii = (ele // nely) % nelx
        kk = ele // (nely * nelx)
        self.mkey = morton_key(ii, jj, kk)
        # face-neighbour pairs among active elements
        pos = np.full(nelx * nely * nelz, -1, dtype=np.int64)
        pos[ele] = np.arange(nnele)
        pairs = []
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            i2, j2, k2 = ii + di, jj + dj, kk + dk
            ok = (i2 < nelx) & (j2 < nely) & (k2 < nelz)
            e2 = k2[ok] * (nelx * nely) + i2[ok] * nely + j2[ok]
            p2 = pos[e2]
            p1 = np.flatnonzero(ok)
            sel = p2 >= 0
            pairs.append(np.stack([p1[sel], p2[sel]], axis=1))
        self.pairs = np.concatenate(pairs, axis=0) if pairs else np.zeros((0, 2), np.int64)
        # free-dof map for the Hessian right-hand sides
        fmap = np.full(ndof, -1, dtype=np.int64)
        fmap[freedofs] = np.arange(freedofs.size)
        self.fdof = fmap[edofMatn]                     # (nnele, 24), -1 = fixed
        # element <-> loaded-node incidence (load protection)
        self.load_inc = None
        if F is not None and o.protect_loads:
            Fa = np.abs(np.asarray(F)).reshape(-1, 3, np.asarray(F).shape[-1]).sum(axis=(1, 2))
            lnodes = np.flatnonzero(Fa > 0)
            enodes = edofMatn[:, 0::3] // 3                # (nnele, 8)
            pos_n = np.full(ndof // 3, -1, dtype=np.int64)
            pos_n[lnodes] = np.arange(lnodes.size)
            pn = pos_n[enodes]
            r_, c_ = np.nonzero(pn >= 0)
            if r_.size:
                self.load_inc = sp.csr_matrix((np.ones(r_.size), (r_, pn[r_, c_])),
                                              shape=(nnele, lnodes.size))
        # support elements (touch a fixed DOF) and element <-> loaded-node
        # incidence of all loads (connectivity diagnostics / repair)
        self.is_sup = np.any(self.fdof < 0, axis=1)
        self.load_inc_all = None
        if F is not None:
            Fa = np.abs(np.asarray(F)).reshape(-1, 3, np.asarray(F).shape[-1]).sum(axis=(1, 2))
            lnodes = np.flatnonzero(Fa > 0)
            enodes = edofMatn[:, 0::3] // 3
            pos_n = np.full(ndof // 3, -1, dtype=np.int64)
            pos_n[lnodes] = np.arange(lnodes.size)
            pn = pos_n[enodes]
            r_, c_ = np.nonzero(pn >= 0)
            if r_.size:
                self.load_inc_all = sp.csr_matrix((np.ones(r_.size), (r_, pn[r_, c_])),
                                                  shape=(nnele, lnodes.size))
        self.on_load = (np.asarray(self.load_inc_all.sum(axis=1)).ravel() > 0
                        if self.load_inc_all is not None else np.zeros(nnele, dtype=bool))
        self.nfree = int(freedofs.size)
        self.rng = np.random.default_rng(o.seed)
        self.V_k = None
        self.target = self.vol * self.nnele
        self.at_target = False
        self.history = []
        self.last = None
        self.qaoa_params = {}
        self.best_c = math.inf
        self.no_improve = 0
        self.iters_at_target = 0
        self.n_qubo_solves = 0
        self.warm_done = o.init != "oc"
        self._hessian_warned = False
        self._g_prev = None
        self.interp = o.interp
        self.S = (self.E0 - self.Emin) if simp else self.E0 * (1.0 - self.Emin)
        # stabilisation (docs/NOTES_quantum.md §3.14): adaptive move limit and
        # accept-if-improves guard
        self.move = float(o.move_limit) if o.move_limit else None
        self.n_rejects_row = 0
        self.n_rejects = 0
        self.pending_guard = None
        self._Vk_before = None
        self._last_kind = None
        self._last_at_target = False

    # ------------------------------------------------------------------
    def describe(self):
        o = self.opts
        return {"backend": self.backend, "hessian": self.hessian, "volume": o.volume,
                "block_size": self.block_size, "blocks": o.blocks, "sweeps": o.sweeps,
                "init": o.init, "er": o.er, "gamma": o.gamma,
                "frontier_fraction": o.frontier_fraction, "lambda_q": o.lambda_q,
                "move_penalty": o.move_penalty, "patience": o.patience,
                "num_reads": o.num_reads, "seed": o.seed, "qaoa_p": o.qaoa_p,
                "qaoa_shots": o.qaoa_shots, "qaoa_init": o.qaoa_init,
                "hessian_block_size": o.hessian_block_size, "interp": o.interp,
                "free_set": o.free_set, "history_average": o.history_average,
                "qaoa_polish": o.qaoa_polish, "move_limit": o.move_limit,
                "move_limit_min": o.move_limit_min, "guard": o.guard,
                "protect_loads": o.protect_loads,
                "guard_tol": o.guard_tol, "guard_tol_target": o.guard_tol_target,
                "max_rejects": o.max_rejects, "connectivity": o.connectivity,
                "diagnostics": o.diagnostics}

    def initial_design(self, vx):
        if self.opts.init == "solid":
            x = np.ones(self.nnele)
            return x
        return vx

    # ------------------------------------------------------------------
    def observe(self, cval):
        """Compliance of the design produced in the previous iteration."""
        if self.at_target and self.iters_at_target >= 1:
            if cval < self.best_c * (1 - 1e-9):
                self.best_c = cval
                self.no_improve = 0
            else:
                self.no_improve += 1

    def should_stop(self, change, tolx):
        if not (self.warm_done and self.at_target):
            return False
        if self.iters_at_target >= 2 and change <= tolx:
            return True
        return self.no_improve >= int(self.opts.patience)

    def guard_reject(self, c_new, c_acc):
        """Accept-if-improves guard, called with the compliance of the design
        produced by the last update (``c_new``) and of the design it was
        computed from (``c_acc``).  Returns True if the design must be
        rejected: the caller restores the previous design and calls
        :meth:`update` again; the move limit is halved and the volume target
        of the rejected step is restored.  Tolerance: ``guard_tol`` (relative
        rise) while the volume target is still decreasing (removal raises
        compliance), ``guard_tol_target`` once it is reached.  After
        ``max_rejects`` consecutive rejections the design is accepted."""
        o = self.opts
        if not o.guard or self._last_kind != "qubo" or c_acc is None:
            return False
        tol = float(o.guard_tol_target if self._last_at_target else o.guard_tol)
        ok = math.isfinite(c_new) and c_new <= c_acc * (1.0 + tol)
        if ok:
            self.n_rejects_row = 0
            if self.move is not None:
                self.move = min(float(o.move_limit), self.move * 1.25)
            return False
        if self.n_rejects_row >= int(o.max_rejects):
            self.log(f"QUBO guard: accepting c = {c_new:.4g} after {self.n_rejects_row} "
                     f"rejections (reference {c_acc:.4g})")
            self.n_rejects_row = 0
            return False
        self.n_rejects_row += 1
        self.n_rejects += 1
        if self.move is not None:
            self.move = max(float(o.move_limit_min), 0.5 * self.move)
        if self._Vk_before is not None:
            self.V_k = self._Vk_before
        if self._last_at_target:
            self.no_improve += 1
        self.pending_guard = {"rejected_compliance": float(c_new),
                              "reference_compliance": float(c_acc),
                              "consecutive": int(self.n_rejects_row)}
        self.log(f"QUBO guard: rejected design with c = {c_new:.4g} (reference "
                 f"{c_acc:.4g}, tolerance {100 * tol:.0f} %); retrying with move limit "
                 f"{self.move if self.move is not None else 'off'}")
        return True

    def _protected(self, xk, ratio):
        """Greedy set cover of the loaded nodes by elements (prefer solid,
        then high |g|/v): every loaded node keeps at least one solid adjacent
        element, so a binary move can never leave a load in void material
        (FreeTO's keep_bc misses thin load regions at coarse meshes)."""
        A = self.load_inc
        prot = np.zeros(self.nnele, dtype=bool)
        if A is None:
            return prot
        unc = np.ones(A.shape[1])
        rn = ratio / max(float(ratio.max()), 1e-300)
        tie = 0.5 * (xk > 0.5) + 0.25 * rn                 # < 1: counts dominate
        # nodes already covered by keep elements
        if self.musd_pos.size:
            unc[np.asarray(A[self.musd_pos].sum(axis=0)).ravel() > 0] = 0.0
        while unc.any():
            cnt = A @ unc
            score = np.where(cnt > 0, cnt + tie, -np.inf)
            e = int(np.argmax(score))
            if not np.isfinite(score[e]):
                break
            prot[e] = True
            a0, a1 = A.indptr[e], A.indptr[e + 1]
            unc[A.indices[a0:a1]] = 0.0
        return prot

    def components(self, x):
        """Face-connected components of the solid elements of a binary design.
        Returns (labels (nnele,), -1 = void; 0 = the component holding the most
        support elements (ties: the largest), 1.. = the others; stats dict)."""
        s = np.asarray(x) > 0.5
        lab = np.full(self.nnele, -1, dtype=np.int64)
        ns = int(s.sum())
        st = {"n_solid": ns, "n_components": 0, "float_frac": 0.0, "n_isolated": 0,
              "load_frac_main": None}
        if ns == 0:
            return lab, st
        idx = np.flatnonzero(s)
        pos = np.full(self.nnele, -1, dtype=np.int64)
        pos[idx] = np.arange(ns)
        p1, p2 = self.pairs[:, 0], self.pairs[:, 1]
        both = s[p1] & s[p2]
        G = sp.coo_matrix((np.ones(int(both.sum())), (pos[p1[both]], pos[p2[both]])),
                          shape=(ns, ns))
        nc, cl = connected_components(G, directed=False)
        size = np.bincount(cl, minlength=nc)
        nsup = np.bincount(cl, weights=self.is_sup[idx].astype(float), minlength=nc)
        order = np.lexsort((-size, -nsup))            # main first
        rank = np.empty(nc, dtype=np.int64)
        rank[order] = np.arange(nc)
        lab[idx] = rank[cl]
        deg = np.bincount(pos[p1[both]], minlength=ns) + np.bincount(pos[p2[both]], minlength=ns)
        st.update(n_components=int(nc), float_frac=float(1.0 - size[order[0]] / ns),
                  n_isolated=int(np.sum(deg == 0)))
        if self.load_inc_all is not None:
            A = self.load_inc_all
            cov_main = np.asarray(A[lab == 0].sum(axis=0)).ravel() > 0
            st["load_frac_main"] = float(cov_main.mean())
        return lab, st

    def n_ungrounded(self, x):
        """Number of solid elements in components that touch no support element."""
        lab, st = self.components(x)
        if st["n_solid"] == 0:
            return 0
        sol = lab >= 0
        grounded = np.zeros(int(lab.max()) + 1, dtype=bool)
        grounded[np.unique(lab[sol & self.is_sup])] = True
        return int(np.sum(sol & ~grounded[np.maximum(lab, 0)]))

    def _rebalance(self, x, removable, ratio, dv, vmax, max_tries=None):
        """Remove the volume the connectivity repair added: solid boundary
        elements in ``removable`` with the smallest |g|/v first, each only if
        the design keeps every required element grounded (a removal that
        disconnects is undone; islands without required elements that a removal
        cuts off are dropped with it).  Returns (x, n_removed)."""
        x = x.copy()
        vol = float(dv @ x)
        tol = 1e-9 * max(1.0, abs(vmax))
        if vol <= vmax + tol:
            return x, 0
        p1, p2 = self.pairs[:, 0], self.pairs[:, 1]
        n_rm = 0
        tries = 0
        n_ex = int(math.ceil((vol - vmax) / max(float(np.mean(dv)), 1e-300)))
        lim = int(max_tries) if max_tries else max(50, 20 * n_ex)
        while vol > vmax + tol and tries < lim:
            s_ = x > 0.5
            diff = s_[p1] != s_[p2]
            bnd = np.zeros(self.nnele, dtype=bool)
            bnd[p1[diff]] = True
            bnd[p2[diff]] = True
            cand = np.flatnonzero(s_ & bnd & removable & ~self._reb_tried)
            if cand.size == 0:
                break
            e = int(cand[np.argmin(ratio[cand] / dv[cand])])
            tries += 1
            self._reb_tried[e] = True
            x[e] = 0.0
            lab, _ = self.components(x)
            sol = lab >= 0
            grounded = np.zeros(int(lab.max()) + 1 if sol.any() else 1, dtype=bool)
            grounded[np.unique(lab[sol & self.is_sup])] = True
            fl = sol & ~grounded[np.maximum(lab, 0)]
            if np.any(fl & self._reb_required):
                x[e] = 1.0                      # would disconnect a required element
                continue
            x[fl] = 0.0                         # cut-off island without required elements
            n_rm += 1 + int(fl.sum())
            vol = float(dv @ x)
        return x, n_rm

    def connect(self, x, x_prev, required, ratio):
        """Connectivity repair of a binary design (``QUBOOptions.connectivity``).

        The FE model of the update sees the design only through the density
        filter and the Heaviside smoothing, which bridge one-element gaps, so a
        binary step can leave solid elements that are not face-connected to any
        support (dangling islands, a deck cut off from its piers, isolated
        load-protection elements) without the compliance noticing; the crisp
        design then has loads acting on void.  Here every face-connected solid
        component that touches no support element is either

        * removed, if it holds no required element (keep / protected / element
          on a loaded node): floating material carries nothing, or
        * reconnected to the nearest grounded component along the cheapest
          path of void elements (Dijkstra on the face graph; elements solid
          in ``x_prev`` -- i.e. removed by this step -- and elements with a
          large sensitivity ratio are cheaper), if it does.

        Returns (x, info dict)."""
        x = np.asarray(x, dtype=np.float64).copy()
        lab, st = self.components(x)
        info = {"floating_before": int(np.sum(lab > 0)),
                "components_before": st["n_components"],
                "removed": 0, "added": 0, "reconnected": 0}
        if st["n_components"] <= 1 or st["n_solid"] == 0:
            return x, info
        ncomp = st["n_components"]
        sol = lab >= 0
        grounded = np.zeros(ncomp, dtype=bool)
        grounded[np.unique(lab[sol & self.is_sup])] = True
        if not grounded.any():
            return x, info                 # nothing to attach to (no solid support)
        req = np.zeros(ncomp, dtype=bool)
        req[np.unique(lab[sol & required])] = True
        drop = ~grounded & ~req
        rm = sol & drop[np.maximum(lab, 0)]
        x[rm] = 0.0
        info["removed"] = int(rm.sum())
        n_todo = int(np.sum(~grounded & req))
        if n_todo == 0:
            return x, info
        rn = ratio / max(float(np.max(ratio)), 1e-300)
        p1, p2 = self.pairs[:, 0], self.pairs[:, 1]
        for _ in range(n_todo):
            lab, _st = self.components(x)
            sol = lab >= 0
            grounded = np.zeros(int(lab.max()) + 1, dtype=bool)
            grounded[np.unique(lab[sol & self.is_sup])] = True
            pending = [k for k in range(grounded.size) if not grounded[k]
                       and np.any(required & (lab == k))]
            if not pending:
                break
            src = np.flatnonzero(sol & grounded[np.maximum(lab, 0)])
            # cost of entering an element: ~0 if solid, 0.5 if it was solid
            # before this step, 1 + (1 - normalised ratio) otherwise
            w = np.where(sol, 1e-9, np.where(np.asarray(x_prev) > 0.5, 0.5, 2.0 - rn))
            G = sp.csr_matrix((np.concatenate([w[p2], w[p1]]),
                               (np.concatenate([p1, p2]), np.concatenate([p2, p1]))),
                              shape=(self.nnele, self.nnele))
            dist, pred = dijkstra(G, directed=True, indices=src, min_only=True,
                                  return_predecessors=True)[:2]
            k = pending[0]
            members = np.flatnonzero(lab == k)
            e = int(members[np.argmin(dist[members])])
            if not np.isfinite(dist[e]):
                break
            while e >= 0 and not (sol[e] and grounded[lab[e]]):
                if x[e] < 0.5:
                    x[e] = 1.0
                    info["added"] += 1
                e = int(pred[e])
            info["reconnected"] += 1
        return x, info

    def _truncate(self, Q, h, v, a, y, lam, cap, budget):
        """Keep at most ``cap`` of the flips y != a: greedy on the Lagrangian
        z^T Q z + (h + lam v).z over the proposed flips, then the volume repair."""
        from .blockqubo import _repair
        F = np.flatnonzero(y != a)
        z = a.copy()
        hl = h + lam * v
        Qc = Q.tocsc()
        f = np.asarray(Q @ z).ravel()
        cand = np.ones(F.size, dtype=bool)
        n = 0
        while n < cap and cand.any():
            idx = F[cand]
            d = (1.0 - 2.0 * z[idx]) * (hl[idx] + 2.0 * f[idx])
            k = int(np.argmin(d))
            if not d[k] < 0:
                break
            i = idx[k]
            sgn = 1.0 - 2.0 * z[i]
            z[i] = 1.0 - z[i]
            a0, a1 = Qc.indptr[i], Qc.indptr[i + 1]
            f[Qc.indices[a0:a1]] += sgn * Qc.data[a0:a1]
            cand[np.flatnonzero(F == i)[0]] = False
            n += 1
        z, n_rm, n_add = _repair(Q, h, v, z, max(budget, 0.0))
        return z, n_rm, n_add

    # ------------------------------------------------------------------
    def _element_terms(self, vxPhys, U):
        """E', E'', E and ce (per load case) of every active element."""
        rho = np.clip(vxPhys, 0.0, 1.0)
        if self.simp:
            p = self.penal
            d = self.E0 - self.Emin
            E = self.Emin + rho ** p * d
            E1 = p * rho ** (p - 1) * d
            E2 = p * (p - 1) * rho ** (p - 2) * d if p >= 2 else \
                p * (p - 1) * np.maximum(rho, 1e-3) ** (p - 2) * d
        else:
            E = rho * self.E0 + (1 - rho) * (self.Emin * self.E0)
            E1 = np.full_like(rho, self.E0 * (1 - self.Emin))
            E2 = np.zeros_like(rho)
        ces, KUs = [], []
        for l in range(U.shape[1]):
            Ue = U[self.edofMatn, l]
            KU = Ue @ self.KE
            ces.append(np.einsum("ij,ij->i", KU, Ue))
            KUs.append(KU)
        return E, E1, E2, ces, KUs

    def _patches(self, C, blocks):
        if blocks is not None:
            return blocks
        hb = int(self.opts.hessian_block_size)
        order = np.argsort(self.mkey[C], kind="stable")
        return [order[a:a + hb] for a in range(0, C.size, hb)]

    def _gram_blocks(self, C, patches, E1, KUs, data):
        """Exact Gram part 2 Y_P^T K^-1 Y_P for every patch (list of dense)."""
        nn = self.nnele
        rows = self.fdof
        mask = rows >= 0
        JC = self.Jc[:, C]                                     # (nnele, m)
        Ys = []
        for KU in KUs:
            vals = (E1[:, None] * KU)[mask]
            G = sp.csr_matrix((vals, (rows[mask], np.nonzero(mask)[0])),
                              shape=(self.nfree, nn))
            Ys.append(sp.csc_matrix(G @ JC))
        out = [None] * len(patches)
        # batch patches so that at most ~512 right-hand sides are solved at once
        batch, cols = [], 0
        max_rhs = 512

        def flush(batch):
            if not batch:
                return
            idx = np.concatenate([patches[b] for b in batch])
            Yd = np.hstack([Y[:, idx].toarray() for Y in Ys])
            Z = multi_rhs_solve(self.solver, self.asm, data, Yd, self.log)
            nl = len(Ys)
            nb = idx.size
            off = 0
            for b in batch:
                k = patches[b].size
                Qb = np.zeros((k, k))
                for l in range(nl):
                    Yb = Yd[:, l * nb + off:l * nb + off + k]
                    Zb = Z[:, l * nb + off:l * nb + off + k]
                    Qb += 2.0 * (Yb.T @ Zb)
                out[b] = 0.5 * (Qb + Qb.T)
                off += k
        for b in range(len(patches)):
            k = patches[b].size * len(Ys)
            if cols + k > max_rhs and batch:
                flush(batch)
                batch, cols = [], 0
            batch.append(b)
            cols += k
        flush(batch)
        return out

    # ------------------------------------------------------------------
    def update(self, loop, vx, vxPhys, dc, dv, U, data, cval=None, stop_event=None,
               oc_fallback=None):
        """Return the new binary design vxnew (float array over the active elements)."""
        t0 = time.perf_counter()
        o = self.opts
        nn = self.nnele
        if not self.warm_done:
            # init="oc": OC warm-up iterations, then threshold to the target
            if loop <= int(o.n_warm):
                return oc_fallback()
            self.warm_done = True
            order = np.argsort(-vxPhys, kind="stable")
            xk = np.zeros(nn)
            xk[self.musd_pos] = 1.0
            vol_left = self.target - dv[self.musd_pos].sum()
            cum = np.cumsum(np.where(self.is_musd[order], 0.0, dv[order]))
            take = order[(cum <= vol_left + 1e-9) & ~self.is_musd[order]]
            xk[take] = 1.0
            self.V_k = self.target
            self.at_target = True
            self.iters_at_target = 1
            # return the thresholded design itself: the next FE solve measures
            # it, then the QUBO iterations start from it
            self.last = {"n_free": 0, "n_blocks": 0, "n_solves": 0, "backend": "threshold",
                         "hessian": self.hessian, "wall_time": time.perf_counter() - t0,
                         "hessian_time": 0.0, "solver_time": 0.0, "qpu_time": None,
                         "energy": None, "lambda": None, "volume_target": self.V_k / nn,
                         "n_flips": int(np.sum(np.abs(xk - np.asarray(vx)) > 0.5)),
                         "approx_ratio": None, "p_opt": None, "exact_match": None,
                         "max_block": 0, "repair_removed": 0, "repair_added": 0,
                         "iter": int(loop)}
            self.history.append(self.last)
            self._last_kind = "threshold"
            return xk
        xk = np.where(np.asarray(vx) > 0.5, 1.0, 0.0)
        xk[self.musd_pos] = 1.0
        diag = {} if o.diagnostics else None
        if diag is not None:
            diag["input"] = self.components(xk)[1]
        # volume schedule
        vcur = float(dv @ xk)
        if self.V_k is None:
            self.V_k = vcur
        self._Vk_before = self.V_k
        if o.init == "solid":
            self.V_k = max(self.V_k * (1.0 - float(o.er)), self.target)
        else:
            self.V_k = self.target
        self.at_target = self.V_k <= self.target * (1 + 1e-12)
        if self.at_target:
            self.iters_at_target += 1
        # ---- element terms and linear coefficients
        th = time.perf_counter()
        E, E1, E2, ces, KUs = self._element_terms(vxPhys, U)
        ce_sum = sum(ces)
        if self.interp == "secant":
            # modulus model E = Emin + S rho (secant between void and solid)
            E1m = np.full(nn, self.S)
        elif self.interp == "beso":
            # w = E'/p: the exact secant slope at solid elements (removal),
            # ~0 at voids (additions driven by the filtered neighbour
            # sensitivities, as in BESO); first-order term = dc / p
            E1m = E1 / (self.penal if self.simp else 1.0)
        else:
            E1m = E1
        if self.interp == "simp":
            g_raw = np.asarray(dc, dtype=np.float64)
        else:
            g_raw = self.H @ ((-E1m * ce_sum) / self.Hs)
        if o.history_average and self._g_prev is not None:
            g_raw = 0.5 * (g_raw + self._g_prev)     # BESO sensitivity history averaging
        self._g_prev = np.array(g_raw, copy=True)
        s = float(np.max(np.abs(g_raw)))
        if not s > 0:
            s = 1.0
        g = g_raw / s
        ratio = np.abs(g) / dv
        # ---- load protection (elements covering the loaded nodes stay solid)
        prot = self._protected(xk, ratio)
        n_prot_added = int(np.sum(prot & (xk < 0.5)))
        xk[prot] = 1.0
        if diag is not None:
            lab_p, diag["after_protect"] = self.components(xk)
            diag["protected_floating"] = int(np.sum(prot & (lab_p > 0)))
        # ---- free set
        free = ~self.is_musd & ~prot
        if o.free_set == "grey":
            grey = (vxPhys > 0.02) & (vxPhys < 0.98) & free
        else:
            # boundary band of the binary design: elements with a face
            # neighbour of the opposite value
            p1, p2 = self.pairs[:, 0], self.pairs[:, 1]
            diff = xk[p1] != xk[p2]
            grey = np.zeros(nn, dtype=bool)
            grey[p1[diff]] = True
            grey[p2[diff]] = True
            grey &= free
        nf = int(math.ceil(float(o.frontier_fraction) * self.V_k / max(dv.mean(), 1e-12)))
        sol = np.flatnonzero((xk > 0.5) & free)
        voi = np.flatnonzero((xk < 0.5) & free)
        rem = sol[np.argsort(ratio[sol], kind="stable")[:nf]]
        add = voi[np.argsort(-ratio[voi], kind="stable")[:nf]]
        inC = grey.copy()
        inC[rem] = True
        inC[add] = True
        C = np.flatnonzero(inC)
        m = C.size
        a = xk[C]
        v = dv[C]
        V_fixed = float(dv[~inC] @ xk[~inC])
        # ---- Hessian (design space, scaled by 1/s)
        hess = self.hessian
        blocks = None
        if self.block_size is not None and m > int(self.block_size):
            kb = int(self.block_size)
            if o.blocks == "rank":
                order = np.argsort(ratio[C], kind="stable")
            else:
                order = np.argsort(self.mkey[C], kind="stable")
            blocks = [order[i:i + kb] for i in range(0, m, kb)]
        if hess == "block" and m * len(ces) > int(o.hessian_max_rhs):
            if not self._hessian_warned:
                self.log(f"QUBO: {m} free elements x {len(ces)} load case(s) exceed "
                         f"hessian_max_rhs={o.hessian_max_rhs}; using hessian='diag' for "
                         "such iterations")
                self._hessian_warned = True
            hess = "diag"
        JC = self.Jc[:, C]
        if hess == "none" or m == 0:
            Qc = sp.csr_matrix((m, m))
        else:
            if self.interp == "simp":
                D = -E2 * ce_sum          # SIMP curvature (design-literal Taylor model)
            elif hess in ("diag", "scalar"):
                # element-local (Loewner) bound of the Gram term:
                #   2 w^2 (KE u)^T K^-1 (KE u) <= 2 w^2 c_e / E_e
                D = 2.0 * E1m ** 2 * ce_sum / (E if self.interp == "beso" else self.E0)
            else:
                D = np.zeros(nn)
            Qc = sp.csr_matrix(JC.T @ sp.diags(D / s) @ JC)
            if hess == "scalar":
                # separable control: keep only the diagonal of
                # Q_c = J_C^T diag(D/s) J_C (all couplings zeroed)
                Qc = sp.csr_matrix(sp.diags(Qc.diagonal()))
            if hess == "block":
                patches = self._patches(C, blocks)
                Gb = self._gram_blocks(C, patches, E1m, KUs, data)
                r_, c_, v_ = [], [], []
                for P, Qb in zip(patches, Gb):
                    rr, cc = np.meshgrid(P, P, indexing="ij")
                    r_.append(rr.ravel())
                    c_.append(cc.ravel())
                    v_.append(Qb.ravel() / s)
                Qc = Qc + sp.csr_matrix((np.concatenate(v_), (np.concatenate(r_),
                                                               np.concatenate(c_))),
                                        shape=(m, m))
        t_hess = time.perf_counter() - th
        # ---- standard form  y^T Q y + h.y + const
        Qc = sp.csr_matrix(Qc)
        if float(o.hessian_scale) != 1.0:
            # v2 control: scale the curvature model (block + diag parts); the
            # perimeter term gamma below is not scaled
            Qc = Qc * float(o.hessian_scale)
        dQ = Qc.diagonal()
        Qa = Qc @ a
        Q = 0.5 * (Qc - sp.diags(dQ))
        h = g[C] + 0.5 * dQ - Qa
        const = 0.5 * float(a @ Qa) + float(g[~inC] @ xk[~inC])
        # gamma in units of the mean |g| of the solid elements (max|dc| is
        # dominated by the load-point singularity)
        solid = xk > 0.5
        gscale = float(np.mean(np.abs(g[solid]))) if solid.any() else 1.0
        gam = float(o.gamma) * gscale
        if gam > 0 and self.pairs.size:
            p1, p2 = self.pairs[:, 0], self.pairs[:, 1]
            both = inC[p1] & inC[p2]
            one1 = inC[p1] & ~inC[p2]
            one2 = inC[p2] & ~inC[p1]
            cidx = np.full(nn, -1, dtype=np.int64)
            cidx[C] = np.arange(m)
            b1, b2 = cidx[p1[both]], cidx[p2[both]]
            P = sp.csr_matrix((np.full(b1.size, -gam), (b1, b2)), shape=(m, m))
            Q = Q + P + P.T
            np.add.at(h, b1, gam)
            np.add.at(h, b2, gam)
            np.add.at(h, cidx[p1[one1]], gam * (1.0 - 2.0 * xk[p2[one1]]))
            np.add.at(h, cidx[p2[one2]], gam * (1.0 - 2.0 * xk[p1[one2]]))
            const += gam * float(np.sum(xk[p2[one1]]) + np.sum(xk[p1[one2]]))
        mu = float(o.move_penalty)
        if mu > 0:
            h = h + mu * (1.0 - 2.0 * a)
            const += mu * float(a.sum())
        Q = sp.csr_matrix(Q)
        Q.eliminate_zeros()
        # ---- volume multiplier for the fixed-lambda (penalty) mode
        lam0 = None
        if o.volume == "penalty":
            if o.lambda_source == "oc":
                lam0 = oc_multiplier(vxPhys, dc, dv, self.target) / s
            else:
                from .blockqubo import separable_knapsack
                _, lam0 = separable_knapsack(g[C], v, max(self.V_k - V_fixed, 0.0))
        bopts = {}
        if self.backend in ("qaoa", "qiskit_aer", "ibm"):
            bopts = {"p": int(o.qaoa_p), "shots": int(o.qaoa_shots), "init": o.qaoa_init}
            if o.qaoa_maxiter:
                bopts["maxiter"] = int(o.qaoa_maxiter)
        if self.backend == "dwave_hybrid" and o.time_limit:
            bopts["time_limit"] = float(o.time_limit)
        if self.backend == "sa":
            bopts["sweeps"] = 100
        if self.backend == "qaoa":
            bopts["polish"] = bool(o.qaoa_polish)
        bopts.update(o.backend_options or {})
        y, info = solve_volume_model(
            Q, h, const, v, V_fixed, self.V_k, a, backend=self.backend, blocks=blocks,
            volume=o.volume, lam0=lam0, lambda_q=o.lambda_q, sweeps=o.sweeps,
            num_reads=o.num_reads, bisection_steps=o.bisection_steps,
            backend_opts=bopts, verify_exact=o.verify_exact, stop_event=stop_event,
            qaoa_params=self.qaoa_params, rng=self.rng, log=self.log)
        if QUBOUpdater.capture is not None:
            QUBOUpdater.capture({"Q": Q, "h": h, "const": const, "v": v, "V_fixed": V_fixed,
                                 "V_target": self.V_k, "a": a, "y": y, "lambda": info["lambda"],
                                 "mkey": self.mkey[C], "iter": int(loop)})
        if diag is not None:
            xr = xk.copy()
            xr[C] = y
            xr[self.musd_pos] = 1.0
            xr[prot] = 1.0
            diag["after_solve"] = self.components(xr)[1]
        # ---- adaptive move limit: keep at most cap flips (+ the removals the
        # volume step needs), chosen greedily on the Lagrangian of the model
        n_prop = int(np.sum(y != a))
        cap = None
        trunc = None
        if self.move is not None and m:
            budget = self.V_k - V_fixed
            solid_v = v[a > 0.5]
            vbar = float(solid_v.mean()) if solid_v.size else float(v.mean())
            n_need = int(math.ceil(max(float(v @ a) - budget, 0.0) / max(vbar, 1e-12)))
            cap = max(int(math.ceil(self.move * m)), n_need + 2)
            if n_prop > cap:
                y, t_rm, t_add = self._truncate(Q, h, v, a, y, float(info["lambda"] or 0.0),
                                                cap, budget)
                trunc = {"proposed": n_prop, "kept": int(np.sum(y != a)),
                         "repair_removed": int(t_rm), "repair_added": int(t_add)}
        xnew = xk.copy()
        xnew[C] = y
        xnew[self.musd_pos] = 1.0
        xnew[prot] = 1.0
        conn = None
        if o.connectivity:
            req = self.is_musd | prot | self.on_load
            vmax = max(float(dv @ xnew), float(self.V_k))
            xnew, conn = self.connect(xnew, xk, req, ratio)
            if conn["added"]:
                self._reb_tried = np.zeros(nn, dtype=bool)
                self._reb_required = req
                xnew, conn["rebalanced"] = self._rebalance(
                    xnew, ~self.is_musd & ~prot, ratio, dv, vmax)
            conn["floating_after"] = self.n_ungrounded(xnew)
        if diag is not None:
            lab_n, diag["output"] = self.components(xnew)
            fl = lab_n > 0
            diag["floating_sources"] = {
                "protected": int(np.sum(fl & prot)), "keep": int(np.sum(fl & self.is_musd)),
                "added_this_iter": int(np.sum(fl & (xk < 0.5))),
                "kept_from_input": int(np.sum(fl & (xk > 0.5) & ~prot & ~self.is_musd)),
                "outside_free_set": int(np.sum(fl & ~inC))}
        st = info["stats"]
        self.n_qubo_solves += st["n_solves"]
        stats = {"n_free": int(m), "n_blocks": int(st["n_blocks"]),
                 "n_solves": int(st["n_solves"]), "backend": info["backend"],
                 "hessian": hess, "wall_time": time.perf_counter() - t0,
                 "hessian_time": t_hess, "solver_time": float(st["solver_time"]),
                 "qpu_time": st["qpu_time"], "energy": info["energy"],
                 "lambda": info["lambda"], "volume_target": self.V_k / nn,
                 "n_flips": int(np.sum(np.abs(xnew - xk))),
                 "approx_ratio": st["approx_ratio"], "p_opt": st["p_opt"],
                 "exact_match": st["exact_match"], "max_block": int(st["max_block"]),
                 "repair_removed": info["repair_removed"],
                 "repair_added": info["repair_added"], "iter": int(loop),
                 "move_limit": self.move, "flip_cap": cap, "n_proposed_flips": n_prop,
                 "truncated": trunc, "guard": self.pending_guard,
                 "n_protected": int(prot.sum()), "n_protected_added": n_prot_added,
                 "exact_match_raw": st.get("exact_match_raw"),
                 "best_shot_ratio": st.get("best_shot_ratio")}
        if conn is not None:
            stats["connectivity_repair"] = conn
        if diag is not None:
            stats["connectivity"] = diag
        self.pending_guard = None
        self._last_kind = "qubo"
        self._last_at_target = bool(self.at_target)
        self.last = stats
        self.history.append(stats)
        return xnew
