"""Common crisp evaluation of a final design (QUBO study protocol,
docs/NOTES_quantum.md §7).

Every optimizer's final *pre-smoothing* filtered element field (``full_pre``
in :func:`freeto.core.run_freeto`: the filtered design with keep elements set
to 1, i.e. exactly what smoothedge3D receives) is turned into the fine-grid
node field ``xg`` exactly as smoothedge3D does (node averaging ``Hn``, 4x
linear upsampling), then projected to a crisp 0/1 fine-grid field with a
threshold chosen by bisection so that the element densities -- averaged over
the (ngrid+1)^3 windows as smoothedge3D does, void floor 0.001 -- have the
*same* volume fraction (over the active elements) for every method.  This is
the beta -> infinity limit of FreeTO's own volume-preserving Heaviside
projection, evaluated at a common volume, so it does not depend on the
optimizer's own final beta (OC 0.75-2, MMA ~30, QUBO 8) and does not cut
sub-element members like element-level thresholding does.

:func:`refined_voxel_compliance` (v2, 2026-10-04) is an independent check of
that element-window proxy: the same fine-grid field is binarised on a grid
refined f times per element (solid = 1, void = Emin, no SIMP exponent) at the
same volume fraction and solved by FE on the refined mesh.
"""
from __future__ import annotations

import time

import numpy as np

from .smoothedge import upsample_linear, window_reduce

__all__ = ["crisp_projection", "fine_field", "refined_voxel_compliance"]

VOID = 0.001


def fine_field(full, Hn, Hns, nelx, nely, nelz, ngrid=4):
    """Fine-grid node field xg of smoothedge3D (MATLAB layout)."""
    xn = (Hn @ np.asarray(full, dtype=float)) / Hns
    xn = xn.reshape((nely + 1, nelx + 1, nelz + 1), order="F")
    return upsample_linear(xn, ngrid)


def _elements(mask, ngrid, ele):
    xg = np.where(mask, 1.0, VOID)
    s = window_reduce(xg, ngrid, "sum")
    return (s / ((ngrid + 1) ** 3)).ravel(order="F")[ele]


def crisp_projection(full, Hn, Hns, nelx, nely, nelz, ele, target, ngrid=4, steps=60):
    """Crisp element densities of the design at volume fraction ``target``.

    Returns (rho (nnele,), threshold, achieved volume fraction).  The volume
    is a step function of the threshold; the bisection returns the largest
    volume <= target + half a fine cell (within ~1e-4 of the target at the
    study's mesh sizes)."""
    xg = fine_field(full, Hn, Hns, nelx, nely, nelz, ngrid)
    ele = np.asarray(ele)
    nn = ele.size
    lo, hi = float(xg.min()) - 1e-12, float(xg.max())
    best = None
    for _ in range(int(steps)):
        t = 0.5 * (lo + hi)
        rho = _elements(xg > t, ngrid, ele)
        v = float(rho.sum() / nn)
        if best is None or abs(v - target) < abs(best[2] - target):
            best = (rho, t, v)
        if v > target:
            lo = t
        else:
            hi = t
        if hi - lo < 1e-12:
            break
    return best


# ---------------------------------------------------------------------------
# refined binary voxel evaluation (v2)
# ---------------------------------------------------------------------------
def _refined_cells(nelx, nely, nelz, ele, f):
    """Grid indices (r, c, p) of the f^3 refined voxels of every active coarse
    element (rows: coarse elements in the order of ``ele``; columns: the f^3
    sub-voxels) and their element numbers on the refined grid (layout of
    :func:`freeto.mesh.element_dofs`: r + ny*c + ny*nx*p)."""
    ele = np.asarray(ele, dtype=np.int64)
    r = ele % nely
    c = (ele // nely) % nelx
    p = ele // (nely * nelx)
    rr, cc, pp = np.meshgrid(np.arange(f), np.arange(f), np.arange(f), indexing="ij")
    fr = (f * r)[:, None] + rr.ravel()[None, :]
    fc = (f * c)[:, None] + cc.ravel()[None, :]
    fp = (f * p)[:, None] + pp.ravel()[None, :]
    ny, nx = f * nely, f * nelx
    fele = (fr + ny * fc + ny * nx * fp).ravel()
    return fr.ravel(), fc.ravel(), fp.ravel(), fele


def _refined_node(n, nelx, nely, f):
    """Refined-grid node coincident with coarse node ``n``."""
    n = np.asarray(n, dtype=np.int64)
    r = n % (nely + 1)
    c = (n // (nely + 1)) % (nelx + 1)
    p = n // ((nely + 1) * (nelx + 1))
    ny, nx = f * nely, f * nelx
    return f * r + (ny + 1) * (f * c) + (ny + 1) * (nx + 1) * (f * p)


def _fine_coords(nelx, nely, nelz, f):
    """(R, C, P) refined-grid coordinates of every refined node (node order of
    :func:`freeto.mesh.element_dofs` on the refined grid)."""
    ny, nx, nz = f * nely, f * nelx, f * nelz
    n = np.arange((ny + 1) * (nx + 1) * (nz + 1), dtype=np.int64)
    return n % (ny + 1), (n // (ny + 1)) % (nx + 1), n // ((ny + 1) * (nx + 1))


def _interp_indicator(ind, nelx, nely, nelz, f):
    """Refined nodes whose trilinear interpolation of the coarse node indicator
    ``ind`` (bool, coarse node order) is 1, i.e. all corners of the smallest
    coarse vertex / edge / face / cell containing the node are flagged."""
    I = np.asarray(ind, dtype=bool).reshape((nely + 1, nelx + 1, nelz + 1), order="F")
    R, C, P = _fine_coords(nelx, nely, nelz, f)
    out = np.ones(R.size, dtype=bool)
    for a in (R // f, -(-R // f)):
        for b in (C // f, -(-C // f)):
            for c in (P // f, -(-P // f)):
                out &= I[a, b, c]
    return out


def _map_bcs(F, fixeddof, nelx, nely, nelz, f, act_nodes, bc_map, carry=None):
    """Coarse loads / fixed DOFs -> refined grid.

    ``"coincident"``: every coarse nodal load and fixed DOF is applied at the
    coincident refined node only (refined nodes between two fixed coarse nodes
    stay free; distributed loads become loads on every f-th node).
    ``"interp"``: a refined node is fixed in DOF d iff all coarse nodes of the
    smallest coarse vertex/edge/face/cell containing it are fixed in d (the
    support region refined with the mesh); each load case's region (coarse
    nodes with a non-zero load) is refined the same way and every coarse nodal
    load is spread over the refined region nodes with the trilinear weights of
    that coarse node, normalised so that it is conserved exactly.  A uniform
    distributed coarse load stays uniform (q/f^2 per refined face node).
    ``carry`` (bool per refined node, optional): refined nodes that touch at
    least one solid voxel; when given, the weights of a coarse nodal load are
    restricted to those nodes (the load acts on the material present in its
    region, as it does on the coarse grid, where every load node touches an
    element of density >= 0.001).  A coarse node whose region holds no such
    node keeps the unrestricted weights, so a load in void stays in void."""
    ny, nx, nz = f * nely, f * nelx, f * nelz
    nnf = (ny + 1) * (nx + 1) * (nz + 1)
    nnc = (nely + 1) * (nelx + 1) * (nelz + 1)
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    fixed = np.asarray(fixeddof, dtype=np.int64)
    Ff = np.zeros((3 * nnf, F.shape[1]))
    if bc_map == "coincident":
        nzr = np.flatnonzero(np.any(F != 0, axis=1))
        Ff[3 * _refined_node(nzr // 3, nelx, nely, f) + nzr % 3] = F[nzr]
        return Ff, 3 * _refined_node(fixed // 3, nelx, nely, f) + fixed % 3
    if bc_map != "interp":
        raise ValueError("bc_map must be 'interp' or 'coincident'")
    act = np.zeros(nnf, dtype=bool)
    act[act_nodes] = True
    fx = []
    for d in range(3):
        ind = np.zeros(nnc, dtype=bool)
        ind[fixed[fixed % 3 == d] // 3] = True
        fx.append(3 * np.flatnonzero(_interp_indicator(ind, nelx, nely, nelz, f) & act) + d)
    fixed_f = np.unique(np.concatenate(fx))
    R, C, P = _fine_coords(nelx, nely, nelz, f)
    Fc = F.reshape(nnc, 3, -1)
    offs = np.arange(-(f - 1), f)
    w1 = 1.0 - np.abs(offs) / f
    for k in range(F.shape[1]):
        reg = np.any(Fc[:, :, k] != 0, axis=1)
        if not reg.any():
            continue
        inr = _interp_indicator(reg, nelx, nely, nelz, f) & act
        j = np.flatnonzero(reg)
        r0 = f * (j % (nely + 1))
        c0 = f * ((j // (nely + 1)) % (nelx + 1))
        p0 = f * (j // ((nely + 1) * (nelx + 1)))
        idx, wts = [], []
        for a, wa in zip(offs, w1):
            for b, wb in zip(offs, w1):
                for c, wc in zip(offs, w1):
                    rr, cc, pp = r0 + a, c0 + b, p0 + c
                    ok = ((rr >= 0) & (rr <= ny) & (cc >= 0) & (cc <= nx)
                          & (pp >= 0) & (pp <= nz))
                    n = np.where(ok, rr + (ny + 1) * cc + (ny + 1) * (nx + 1) * pp, 0)
                    ok &= inr[n]
                    idx.append(n)
                    wts.append(np.where(ok, wa * wb * wc, 0.0))
        idx = np.stack(idx, axis=1)                     # (n_region, (2f-1)^3)
        wts = np.stack(wts, axis=1)
        if carry is not None:
            wc_ = np.where(carry[idx], wts, 0.0)
            has = wc_.sum(axis=1) > 0
            wts = np.where(has[:, None], wc_, wts)
        wts /= wts.sum(axis=1, keepdims=True)    # sum >= 1 (coincident node is in the region)
        for d in range(3):
            np.add.at(Ff[:, k], 3 * idx.ravel() + d, (wts * Fc[j, d, k][:, None]).ravel())
    return Ff, fixed_f


def refined_voxel_compliance(full_pre, Hn, Hns, nelx, nely, nelz, ele, target, F,
                             fixeddof, KE, E0, Emin, ngrid=4, f=2, solver="auto",
                             steps=60, return_solid=False, bc_map="interp",
                             load_on_solid=True):
    """Compliance of the binary voxel design on a grid refined ``f`` times.

    * ``xg = fine_field(full_pre, ...)``: the node field on FreeTO's ``ngrid``x
      fine grid, exactly as :func:`crisp_projection` uses it.
    * Every active coarse element is split into f x f x f voxels.  A voxel is
      solid iff the mean of ``xg`` at its 8 corners (corner spacing ngrid/f
      fine points; f = 2: fine points 0, 2, 4 of the element) exceeds t; t is
      bisected (``steps``) so that the solid fraction over the active region is
      as close as possible to ``target`` (achieved value returned).
    * Binary moduli: solid ``E0``, void ``Emin`` (no SIMP exponent).
    * Loads ``F`` (ndof x nload of the coarse grid) and supports ``fixeddof``
      are mapped to the refined grid by :func:`_map_bcs`: ``bc_map="interp"``
      (default) refines the support and load regions with the mesh (a refined
      node is fixed iff all coarse nodes of the coarse entity containing it
      are fixed; each nodal load is spread over the refined load region and
      conserved), ``"coincident"`` applies them only at the refined nodes
      coincident with coarse nodes (point supports / point loads on the
      refined mesh: the compliance then grows without bound under refinement,
      +82 % at f = 2 and +69 % more at f = 4 for a solid 6x3x3 cantilever,
      against +6 % and +2 % with "interp"; f = 1: identical).  Refined element
      stiffness ``KE / f`` (the unit-modulus 8-node brick stiffness scales with
      the edge length).
    * One FE solve (``freeto.fe.Assembler`` + ``make_solver(solver)``, the
      residual-checked :func:`freeto.quantum.update.robust_solve`) plus up to
      3 iterative-refinement steps with the same factorisation.

    Returns dict(compliance, volfrac, threshold, n_voxels, ndof_free,
    residual, time) (+ ``solid`` bool array per refined voxel, ordered as the
    active coarse elements x f^3, when ``return_solid``)."""
    from .fe import Assembler, make_solver
    from .mesh import element_dofs
    from .quantum.update import robust_solve, multi_rhs_solve
    t0 = time.perf_counter()
    f = int(f)
    ngrid = int(ngrid)
    if f < 1 or ngrid % f:
        raise ValueError(f"refinement f = {f} must be >= 1 and divide ngrid = {ngrid}")
    s = ngrid // f
    xg = fine_field(full_pre, Hn, Hns, nelx, nely, nelz, ngrid)
    fr, fc, fp, fele = _refined_cells(nelx, nely, nelz, ele, f)
    cv = np.zeros(fr.size)
    for dr in (0, 1):
        for dc in (0, 1):
            for dp in (0, 1):
                cv += xg[s * (fr + dr), s * (fc + dc), s * (fp + dp)]
    cv /= 8.0
    # threshold bisection at the target solid fraction (as crisp_projection)
    lo, hi = float(cv.min()) - 1e-12, float(cv.max())
    best = None
    for _ in range(int(steps)):
        t = 0.5 * (lo + hi)
        solid = cv > t
        v = float(solid.mean())
        if best is None or abs(v - target) < abs(best[1] - target):
            best = (t, v, solid)
        if v > target:
            lo = t
        else:
            hi = t
        if hi - lo < 1e-12:
            break
    thr, vf, solid = best
    # refined FE model
    nx, ny, nz = f * nelx, f * nely, f * nelz
    ndof_f = 3 * (nx + 1) * (ny + 1) * (nz + 1)
    edof = element_dofs(nx, ny, nz, fele)
    act = np.unique(edof)
    carry = None
    if load_on_solid and bc_map == "interp":
        # refined nodes attached to at least one solid voxel
        carry = np.zeros((nx + 1) * (ny + 1) * (nz + 1), dtype=bool)
        carry[np.unique(edof[solid][:, ::3] // 3)] = True
    Ff, fixed_f = _map_bcs(F, fixeddof, nelx, nely, nelz, f, act[::3] // 3, bc_map,
                           carry=carry)
    free = np.setdiff1d(act, fixed_f)
    asm = Assembler(edof, np.asarray(KE, dtype=float) / f, free, ndof_f)
    an = asm.act[::3] // 3
    coords = np.stack([(an // (ny + 1)) % (nx + 1), -(an % (ny + 1)),
                       an // ((ny + 1) * (nx + 1))], axis=1).astype(float)
    sol = make_solver(solver, asm, node_coords=coords, log=None)
    Ee = np.where(solid, float(E0), float(Emin))
    data = asm.data(Ee)
    Fr = np.ascontiguousarray(Ff[free])
    U = np.asarray(robust_solve(sol, asm, data, Fr, None)).reshape(free.size, -1)
    K = asm.free_view().full(data)
    nf = float(np.linalg.norm(Fr))
    R = Fr - K @ U
    resid = float(np.linalg.norm(R) / nf)
    for _ in range(3):          # iterative refinement with the stored factorisation
        if resid < 1e-10:
            break
        U = U + np.asarray(multi_rhs_solve(sol, asm, data, R)).reshape(U.shape)
        R = Fr - K @ U
        resid = float(np.linalg.norm(R) / nf)
    sol.close()
    out = {"compliance": float(np.sum(Fr * U)), "volfrac": float(vf),
           "threshold": float(thr), "n_voxels": int(fele.size),
           "ndof_free": int(free.size), "residual": resid,
           "time": time.perf_counter() - t0}
    if return_solid:
        out["solid"] = solid
    return out
