"""Point-in-closed-triangulation tests on structured grids.

Replaces ``intriangulation.m`` (J. Korsawe, based on ``voxelise`` by Adam A).
The original tests every point separately by casting a ray parallel to the
z axis through it, collecting the (rounded, de-duplicated) crossing
coordinates with the facets and classifying the point by parity.  Points whose
ray crosses an odd number of times are re-tested with rays parallel to x and
then y; points undecided after the three directions return -1.

Here the test points always form a structured grid, so all points on one grid
line share one ray: crossings are computed once per ray (vectorised over
(ray, candidate-facet) pairs) and all points on that ray are classified at
once.  The facet-crossing predicates, the plane intersection formula, the
mesh-limit filter, the ``round(z*1e10)/1e10`` de-duplication and the parity
rule are replicated operation-by-operation, so the result is identical to
``intriangulation`` evaluated on the grid points (including its handling of
rays that hit edges and vertices).
"""
from __future__ import annotations

import numpy as np

__all__ = ["inside_grid", "inside_points", "matlab_round"]

_UNDET = 2


def matlab_round(x):
    """MATLAB ``round`` (half away from zero)."""
    a = np.abs(x)
    r = np.floor(a)
    r = r + ((a - r) >= 0.5)
    return np.copysign(r, x)


def _crossings(tri, tx, ty):
    """Replicates the voxelise facet test + plane intersection for pairs.

    tri : (P, 3coords, 3verts) facet coordinates (already permuted so that the
          ray runs along coordinate index 2), tx, ty : (P,) ray position.
    Returns (hit mask (P,), crossing coordinate (P,)).
    """
    X1, X2, X3 = tri[:, 0, 0], tri[:, 0, 1], tri[:, 0, 2]
    Y1, Y2, Y3 = tri[:, 1, 0], tri[:, 1, 1], tri[:, 1, 2]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # edge 2-3 vs vertex 1
        Yp = Y2 - ((Y2 - Y3) * (X2 - X1) / (X2 - X3))
        YR = Y2 - ((Y2 - Y3) * (X2 - tx) / (X2 - X3))
        ok = (((Yp > Y1) & (YR > ty)) | ((Yp < Y1) & (YR < ty))
              | ((Y2 - Y3) * (X2 - tx) == 0))
        # edge 3-1 vs vertex 2
        Yp = Y3 - ((Y3 - Y1) * (X3 - X2) / (X3 - X1))
        YR = Y3 - ((Y3 - Y1) * (X3 - tx) / (X3 - X1))
        ok &= (((Yp > Y2) & (YR > ty)) | ((Yp < Y2) & (YR < ty))
               | ((Y3 - Y1) * (X3 - tx) == 0))
        # edge 1-2 vs vertex 3
        Yp = Y1 - ((Y1 - Y2) * (X1 - X3) / (X1 - X2))
        YR = Y1 - ((Y1 - Y2) * (X1 - tx) / (X1 - X2))
        ok &= (((Yp > Y3) & (YR > ty)) | ((Yp < Y3) & (YR < ty))
               | ((Y1 - Y2) * (X1 - tx) == 0))
        t = tri[ok]
        tx = tx[ok]
        ty = ty[ok]
        x1, x2, x3 = t[:, 0, 0], t[:, 0, 1], t[:, 0, 2]
        y1, y2, y3 = t[:, 1, 0], t[:, 1, 1], t[:, 1, 2]
        z1, z2, z3 = t[:, 2, 0], t[:, 2, 1], t[:, 2, 2]
        A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
        B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
        C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        D = (-x1 * (y2 * z3 - y3 * z2) - x2 * (y3 * z1 - y1 * z3)
             - x3 * (y1 * z2 - y2 * z1))
        C = np.where(np.abs(C) < 1e-14, 0.0, C)
        zc = (-D - A * tx - B * ty) / C
    return ok, zc


def _cast_rays(tri, u, v, w, max_pairs=2_000_000):
    """Classify grid points (u_i, v_j, w_k) with rays parallel to w.

    tri : (m, 3, 3) facets as [facet, coord, vertex] with coords ordered
          (ray-plane X, ray-plane Y, ray direction Z).
    Returns int8 array (nu, nv, nw): 0 outside, 1 inside, 2 undetermined.
    """
    nu, nv, nw = len(u), len(v), len(w)
    out = np.zeros((nu, nv, nw), dtype=np.int8)
    m = tri.shape[0]
    if m == 0 or nu == 0 or nv == 0 or nw == 0:
        return out
    zmin = tri[:, 2, :].min()
    zmax = tri[:, 2, :].max()
    bmin = tri.min(axis=2)
    bmax = tri.max(axis=2)
    # rays strictly inside the facet bounding box (voxelise uses
    # (t-min)*(max-t) > 0)
    i0 = np.searchsorted(u, bmin[:, 0], side="right")
    i1 = np.searchsorted(u, bmax[:, 0], side="left")
    j0 = np.searchsorted(v, bmin[:, 1], side="right")
    j1 = np.searchsorted(v, bmax[:, 1], side="left")
    ni = np.maximum(i1 - i0, 0)
    nj = np.maximum(j1 - j0, 0)
    cnt = ni.astype(np.int64) * nj
    sel = np.nonzero(cnt)[0]
    rays_all = []
    z_all = []
    if sel.size:
        csum = np.cumsum(cnt[sel])
        start = 0
        while start < sel.size:
            base = csum[start - 1] if start else 0
            stop = int(np.searchsorted(csum, base + max_pairs, side="right"))
            stop = max(stop, start + 1)
            ts = sel[start:stop]
            c = cnt[ts]
            tot = int(c.sum())
            tid = np.repeat(ts, c)
            offs = np.repeat(np.cumsum(c) - c, c)
            loc = np.arange(tot, dtype=np.int64) - offs
            njt = nj[tid]
            ii = i0[tid] + loc // njt
            jj = j0[tid] + loc % njt
            tx = u[ii]
            ty = v[jj]
            ok, zc = _crossings(tri[tid], tx, ty)
            ii = ii[ok]
            jj = jj[ok]
            keep = (zc >= zmin - 1e-12) & (zc <= zmax + 1e-12)
            zc = matlab_round(zc[keep] * 1e10) / 1e10
            rays_all.append((ii[keep] * nv + jj[keep]).astype(np.int64))
            z_all.append(zc)
            start = stop
    if not rays_all:
        return out
    ray = np.concatenate(rays_all)
    zc = np.concatenate(z_all)
    if ray.size == 0:
        return out
    order = np.lexsort((zc, ray))
    ray = ray[order]
    zc = zc[order]
    first = np.ones(ray.size, dtype=bool)
    first[1:] = (ray[1:] != ray[:-1]) | (zc[1:] != zc[:-1])
    ray = ray[first]
    zc = zc[first]
    nray = nu * nv
    ncross = np.bincount(ray, minlength=nray)
    odd = (ncross % 2) == 1
    flat = out.reshape(nray, nw)
    flat[odd] = _UNDET
    even_sel = ~odd[ray]
    ray_e = ray[even_sel]
    zc_e = zc[even_sel]
    if ray_e.size:
        # compress to rays that actually have crossings
        rays_u, rinv = np.unique(ray_e, return_inverse=True)
        nr = rays_u.size
        pr = np.searchsorted(w, zc_e, side="right")  # first w > zc
        pl = np.searchsorted(w, zc_e, side="left")   # first w >= zc
        n_lt = np.bincount(rinv * (nw + 1) + pr, minlength=nr * (nw + 1))
        n_le = np.bincount(rinv * (nw + 1) + pl, minlength=nr * (nw + 1))
        n_lt = np.cumsum(n_lt.reshape(nr, nw + 1)[:, :nw], axis=1)
        n_le = np.cumsum(n_le.reshape(nr, nw + 1)[:, :nw], axis=1)
        ins = ((n_lt % 2) == 1) & (n_lt == n_le)
        flat[rays_u] = ins.astype(np.int8)
    return out


def _orient_canon(ux, uy, vx, vy, px, py):
    """orient2d(u, v, p) computed from canonically ordered endpoints."""
    return (vx - ux) * (py - uy) - (vy - uy) * (px - ux)


def _robust_hits(tri, tx, ty):
    """Watertight point-in-projected-triangle test with symbolic
    perturbation p -> p + eps*(1, delta): every edge is evaluated from its
    lexicographically ordered endpoints (so the two triangles sharing an edge
    see bit-identical values) and exact zeros are resolved by the
    perturbation, so a ray through an edge or vertex is counted exactly once
    for a closed surface.  Returns (hit mask, crossing coordinate)."""
    X = tri[:, 0, :]
    Y = tri[:, 1, :]
    sgn = []
    for i in range(3):
        j = (i + 1) % 3
        ax, ay, bx, by = X[:, i], Y[:, i], X[:, j], Y[:, j]
        swap = (bx < ax) | ((bx == ax) & (by < ay))
        ux = np.where(swap, bx, ax)
        uy = np.where(swap, by, ay)
        vx = np.where(swap, ax, bx)
        vy = np.where(swap, ay, by)
        e = _orient_canon(ux, uy, vx, vy, tx, ty)
        pert = np.where(vy != uy, -np.sign(vy - uy), np.sign(vx - ux))
        s = np.where(e != 0, np.sign(e), pert)
        sgn.append(np.where(swap, -s, s))
    area = ((X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0])
            - (Y[:, 1] - Y[:, 0]) * (X[:, 2] - X[:, 0]))
    so = np.sign(area)
    ok = (so != 0) & (sgn[0] == so) & (sgn[1] == so) & (sgn[2] == so)
    t = tri[ok]
    tx = tx[ok]
    ty = ty[ok]
    x1, x2, x3 = t[:, 0, 0], t[:, 0, 1], t[:, 0, 2]
    y1, y2, y3 = t[:, 1, 0], t[:, 1, 1], t[:, 1, 2]
    z1, z2, z3 = t[:, 2, 0], t[:, 2, 1], t[:, 2, 2]
    # barycentric interpolation of z (well defined since area != 0)
    d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    l1 = ((y2 - y3) * (tx - x3) + (x3 - x2) * (ty - y3)) / d
    l2 = ((y3 - y1) * (tx - x3) + (x1 - x3) * (ty - y3)) / d
    zc = l1 * z1 + l2 * z2 + (1 - l1 - l2) * z3
    return ok, zc


def _cast_rays_robust(tri, u, v, w, max_pairs=2_000_000):
    """Ray parity per grid line with the watertight edge rule.
    Returns int8 (nu, nv, nw): 0 outside, 1 inside, 2 ray with odd number of
    crossings (open / non-manifold surface)."""
    nu, nv, nw = len(u), len(v), len(w)
    out = np.zeros((nu, nv, nw), dtype=np.int8)
    if tri.shape[0] == 0 or nu == 0 or nv == 0 or nw == 0:
        return out
    bmin = tri.min(axis=2)
    bmax = tri.max(axis=2)
    # half-open candidate intervals consistent with the perturbation
    i0 = np.searchsorted(u, bmin[:, 0], side="left")
    i1 = np.searchsorted(u, bmax[:, 0], side="left")
    j0 = np.searchsorted(v, bmin[:, 1], side="left")
    j1 = np.searchsorted(v, bmax[:, 1], side="left")
    ni = np.maximum(i1 - i0, 0)
    nj = np.maximum(j1 - j0, 0)
    cnt = ni.astype(np.int64) * nj
    sel = np.nonzero(cnt)[0]
    rays_all, z_all = [], []
    if sel.size:
        csum = np.cumsum(cnt[sel])
        start = 0
        while start < sel.size:
            base = csum[start - 1] if start else 0
            stop = int(np.searchsorted(csum, base + max_pairs, side="right"))
            stop = max(stop, start + 1)
            ts = sel[start:stop]
            c = cnt[ts]
            tot = int(c.sum())
            tid = np.repeat(ts, c)
            offs = np.repeat(np.cumsum(c) - c, c)
            loc = np.arange(tot, dtype=np.int64) - offs
            njt = nj[tid]
            ii = i0[tid] + loc // njt
            jj = j0[tid] + loc % njt
            with np.errstate(divide="ignore", invalid="ignore"):
                ok, zc = _robust_hits(tri[tid], u[ii], v[jj])
            rays_all.append((ii[ok] * nv + jj[ok]).astype(np.int64))
            z_all.append(zc)
            start = stop
    if not rays_all:
        return out
    ray = np.concatenate(rays_all)
    zc = np.concatenate(z_all)
    if ray.size == 0:
        return out
    nray = nu * nv
    ncross = np.bincount(ray, minlength=nray)
    flat = out.reshape(nray, nw)
    rays_u, rinv = np.unique(ray, return_inverse=True)
    nr = rays_u.size
    pr = np.searchsorted(w, zc, side="right")
    pl = np.searchsorted(w, zc, side="left")
    n_lt = np.bincount(rinv * (nw + 1) + pr, minlength=nr * (nw + 1))
    n_le = np.bincount(rinv * (nw + 1) + pl, minlength=nr * (nw + 1))
    n_lt = np.cumsum(n_lt.reshape(nr, nw + 1)[:, :nw], axis=1)
    n_le = np.cumsum(n_le.reshape(nr, nw + 1)[:, :nw], axis=1)
    ins = (((n_lt % 2) == 1) & (n_lt == n_le)).astype(np.int8)
    ins[(ncross[rays_u] % 2) == 1] = _UNDET
    flat[rays_u] = ins
    return out


def _facets(vertices, faces):
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    # [facet, coord, vertex]  (meshXYZ in voxelise)
    return np.ascontiguousarray(V[F].transpose(0, 2, 1))


def inside_grid(vertices, faces, gx, gy, gz, mode="matlab", undecided=1):
    """intriangulation() for all points of the grid ``gx x gy x gz``.

    ``mode="matlab"`` (default) reproduces intriangulation.m exactly,
    including its known blind spot: a ray passing exactly through a
    non-axis-parallel edge shared by two facets is not counted by either
    facet.  ``mode="robust"`` uses a watertight edge rule (each edge/vertex
    hit counted exactly once) along all three axes with a majority vote; it
    agrees with "matlab" except in such degenerate configurations.

    Points left undecided by the z, x and y tests (odd crossing counts along
    all three axes; only possible for open / non-manifold surfaces) get
    ``undecided`` in "matlab" mode.  The default 1 is bit-faithful to
    MATLAB/Octave: intriangulation stores -1 into a *logical* array, which
    coerces it to true, so every caller sees "inside".  Pass ``undecided=-1``
    to see them.  The "robust" mode never returns undecided points (majority
    vote over the axes with even crossing counts; none -> outside).

    Parameters
    ----------
    vertices, faces : closed triangulation (0-based faces)
    gx, gy, gz : strictly increasing 1-D coordinate arrays

    Returns
    -------
    int8 array of shape (len(gx), len(gy), len(gz)) with 1 (inside),
    0 (outside) and ``undecided``.
    """
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    gz = np.asarray(gz, dtype=np.float64)
    tri = _facets(vertices, faces)
    if mode == "robust":
        votes = np.zeros((len(gx), len(gy), len(gz)), dtype=np.int8)
        decided = np.zeros_like(votes)
        for perm, grids, back in (([0, 1, 2], (gx, gy, gz), (0, 1, 2)),
                                  ([1, 2, 0], (gy, gz, gx), (2, 0, 1)),
                                  ([2, 0, 1], (gz, gx, gy), (1, 2, 0))):
            r = _cast_rays_robust(np.ascontiguousarray(tri[:, perm, :]),
                                  *grids).transpose(back)
            votes += (r == 1)
            decided += (r != _UNDET)
        return np.where(2 * votes > decided, 1, 0).astype(np.int8)
    if mode != "matlab":
        raise ValueError("mode must be 'matlab' or 'robust'")
    res = _cast_rays(tri, gx, gy, gz)
    und = res == _UNDET
    if und.any():
        rx = _cast_rays(np.ascontiguousarray(tri[:, [1, 2, 0], :]), gy, gz,
                        gx).transpose(2, 0, 1)
        res = np.where(und, rx, res)
        und = res == _UNDET
        if und.any():
            ry = _cast_rays(np.ascontiguousarray(tri[:, [2, 0, 1], :]), gz,
                            gx, gy).transpose(1, 2, 0)
            res = np.where(und, ry, res)
            res[res == _UNDET] = undecided
    return res.astype(np.int8)


def _cast_points(tri, px, py, pz, chunk=4096):
    """Per-point version of the ray test (rays parallel to coordinate 2)."""
    n = px.size
    out = np.zeros(n, dtype=np.int8)
    if tri.shape[0] == 0:
        return out
    zmin = tri[:, 2, :].min()
    zmax = tri[:, 2, :].max()
    bmin = tri.min(axis=2)
    bmax = tri.max(axis=2)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        tx = px[s:e, None]
        ty = py[s:e, None]
        cand = (((ty - bmin[None, :, 1]) * (bmax[None, :, 1] - ty) > 0)
                & ((tx - bmin[None, :, 0]) * (bmax[None, :, 0] - tx) > 0))
        pi, fi = np.nonzero(cand)
        if pi.size == 0:
            continue
        ok, zc = _crossings(tri[fi], px[s:e][pi], py[s:e][pi])
        pi = pi[ok]
        keep = (zc >= zmin - 1e-12) & (zc <= zmax + 1e-12)
        pi = pi[keep]
        zc = matlab_round(zc[keep] * 1e10) / 1e10
        order = np.lexsort((zc, pi))
        pi = pi[order]
        zc = zc[order]
        first = np.ones(pi.size, dtype=bool)
        first[1:] = (pi[1:] != pi[:-1]) | (zc[1:] != zc[:-1])
        pi = pi[first]
        zc = zc[first]
        m = e - s
        nc = np.bincount(pi, minlength=m)
        tz = pz[s:e][pi]
        lt = np.bincount(pi, weights=(zc < tz), minlength=m)
        eq = np.bincount(pi, weights=(zc == tz), minlength=m)
        res = np.where((lt % 2 == 1) & (eq == 0), 1, 0).astype(np.int8)
        res[nc % 2 == 1] = _UNDET
        out[s:e] = res
    return out


def inside_points(vertices, faces, points, undecided=1):
    """intriangulation() for arbitrary points (O(npoints * nfacets)); see
    :func:`inside_grid` for ``undecided``."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    tri = _facets(vertices, faces)
    res = _cast_points(tri, P[:, 0], P[:, 1], P[:, 2])
    cl = np.nonzero(res == _UNDET)[0]
    res[cl] = 0
    if cl.size:
        r2 = _cast_points(np.ascontiguousarray(tri[:, [1, 2, 0], :]),
                          P[cl, 1], P[cl, 2], P[cl, 0])
        res[cl[r2 == 1]] = 1
        cl = cl[r2 == _UNDET]
        if cl.size:
            r3 = _cast_points(np.ascontiguousarray(tri[:, [2, 0, 1], :]),
                              P[cl, 2], P[cl, 0], P[cl, 1])
            res[cl[r3 == 1]] = 1
            cl = cl[r3 == _UNDET]
            res[cl] = undecided
    return res
