"""Grid construction and domain preparation (geomeshini, domainstokeep,
domainprep, forcevec, supportDOFs).

Conventions
-----------
Internally all element / node arrays use MATLAB's layout: a 3-D array of
shape (nely, nelx, nelz) (nodes: (nely+1, nelx+1, nelz+1)) stored and
linearly indexed in Fortran (column-major) order, where the first (row) axis
runs from the *largest* y down to the smallest (geomeshini flips the rows).
All indices here are 0-based (MATLAB index - 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import os

import numpy as np

from .inside import inside_grid
from .stl_io import read_stl

__all__ = ["matlab_linspace", "matlab_colon", "Grid", "build_grid",
           "Domain", "prepare_domain", "force_vectors", "support_dofs",
           "element_dofs", "xyz_to_matlab", "matlab_to_xyz"]

_EPS = np.finfo(float).eps


# ----------------------------------------------------------------------------
# MATLAB / Octave linspace and colon
# ----------------------------------------------------------------------------
def matlab_linspace(d1, d2, n, compat="matlab"):
    n = int(math.floor(n))
    if n < 1:
        return np.zeros(0)
    if n == 1:
        return np.array([float(d2)])
    d1 = float(d1)
    d2 = float(d2)
    if compat == "octave":
        # Octave >= 7: constructed symmetrically from both ends
        y = np.empty(n)
        if d1 == d2:
            y[:] = d2
            return y
        y[0] = d1
        y[-1] = d2
        delta = (d2 - d1) / (n - 1)
        n2 = n // 2
        i = np.arange(1, n2, dtype=float)
        y[1:n2] = d1 + i * delta
        y[n - 1 - np.arange(1, n2)] = d2 - i * delta
        if n % 2 == 1:
            y[n2] = 0.0 if d1 == -d2 else (d1 + d2) / 2
        return y
    n1 = n - 1
    y = d1 + (np.arange(n1 + 1, dtype=float) * (d2 - d1)) / n1
    if d1 == d2:
        y[:] = d1
    else:
        y[0] = d1
        y[-1] = d2
    return y


def _tfloor(x, ct):
    """Octave's tolerant floor (Hagerty's FL5)."""
    q = 1.0
    if x < 0:
        q = 1.0 - ct
    rmax = q / (2.0 - ct)
    t1 = 1.0 + math.floor(x)
    t1 = (ct / q) * (-t1 if t1 < 0 else t1)
    t1 = rmax if rmax < t1 else t1
    t1 = ct if ct > t1 else t1
    t1 = math.floor(x + t1)
    if x <= 0 or (t1 - x) < rmax:
        return t1
    return t1 - 1


def _teq(u, v, ct=3.0 * _EPS):
    tu = abs(u)
    tv = abs(v)
    return abs(u - v) < ((tu if tu > tv else tv) * ct)


def matlab_colon(a, d, b, compat="matlab"):
    """``a:d:b`` with MATLAB's (default) or Octave's element construction."""
    a = float(a)
    d = float(d)
    b = float(b)
    if not (math.isfinite(a) and math.isfinite(d) and math.isfinite(b)):
        raise ValueError("non-finite colon range")
    if d == 0 or (a < b and d < 0) or (b < a and d > 0):
        return np.zeros(0)
    if compat == "octave":
        ct = 3.0 * _EPS
        tmp = _tfloor((b - a + d) / d, ct)
        n = int(tmp) if tmp > 0 else 0
        if not _teq(a + (n - 1) * d, b):
            if _teq(a + (n - 2) * d, b):
                n -= 1
            elif _teq(a + n * d, b):
                n += 1
        out = a + np.arange(n, dtype=float) * d
        if n > 1:
            fin = a + (n - 1) * d
            if (d > 0 and fin >= b) or (d < 0 and fin <= b):
                fin = b
            out[-1] = fin
        return out
    # MATLAB (colonop.m, as published by MathWorks)
    tol = 2.0 * _EPS * max(abs(a), abs(b))
    sig = math.copysign(1.0, d)
    if a == math.floor(a) and d == 1:
        n = math.floor(b) - a
    elif a == math.floor(a) and d == math.floor(d):
        q = math.floor(a / d)
        r = a - q * d
        n = math.floor((b - r) / d) - math.floor((a - r) / d)
    else:
        q = (b - a) / d          # >= 0 here
        n = math.floor(q)
        if q - n >= 0.5:         # MATLAB round (half away from zero)
            n += 1
        if sig * (a + n * d - b) > tol:
            n -= 1
    n = int(n)
    last = a + n * d
    if sig * (last - b) > -tol:
        last = b
    out = np.zeros(n + 1)
    k = np.arange(0, n // 2 + 1, dtype=float)
    out[k.astype(int)] = a + k * d
    out[(n - k).astype(int)] = last - k * d
    if n % 2 == 0:
        out[n // 2] = (a + last) / 2
    return out


# ----------------------------------------------------------------------------
# layout helpers
# ----------------------------------------------------------------------------
def xyz_to_matlab(arr_xyz):
    """(nx, ny, nz) array with ascending y -> MATLAB (ny, nx, nz), rows = y
    descending."""
    return np.flip(np.transpose(arr_xyz, (1, 0, 2)), axis=0)


def matlab_to_xyz(arr_m):
    """Inverse of :func:`xyz_to_matlab` (returns a view)."""
    return np.transpose(np.flip(arr_m, axis=0), (1, 0, 2))


def _find_matlab(arr_xyz):
    """``find(out(:) > 0)`` after geomeshini's reshape+flip, 0-based."""
    m = xyz_to_matlab(arr_xyz)
    return np.flatnonzero(m.ravel(order="F") > 0)


# ----------------------------------------------------------------------------
# geomeshini: structured grid
# ----------------------------------------------------------------------------
@dataclass
class Grid:
    x: np.ndarray          # node coordinates (ascending)
    y: np.ndarray
    z: np.ndarray
    xc: np.ndarray         # element-centre coordinates
    yc: np.ndarray
    zc: np.ndarray
    ssz: float             # element size (mm)
    aa: float              # ssz / 2000  (KE scale factor aa/0.5 = ssz/1000 m)
    axis: int              # 0,1,2 : axis given MeshControl points
    bbox_min: np.ndarray
    bbox_max: np.ndarray

    @property
    def nelx(self):
        return len(self.x) - 1

    @property
    def nely(self):
        return len(self.y) - 1

    @property
    def nelz(self):
        return len(self.z) - 1

    @property
    def del_xyz(self):
        return self.bbox_max - self.bbox_min


def build_grid(stp1, mesh_control, compat="matlab"):
    """Grid of geomeshini.m, including its axis selection quirk: the axis that
    receives ``MeshControl`` points is the one holding the largest *absolute
    coordinate* of the domain bounding box (not the largest extent)."""
    stp1 = np.asarray(stp1, dtype=np.float64)
    mn = stp1.min(axis=0)
    mx = stp1.max(axis=0)
    stt = np.abs(np.array([mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]]))
    stff = int(np.argmax(stt))  # first index of the maximum
    ax = stff // 2
    g = [None, None, None]
    old = matlab_linspace(mn[ax], mx[ax], mesh_control, compat)
    ssz_old = (old[1] - old[0]) * 1e-2
    g[ax] = matlab_linspace(mn[ax] + ssz_old, mx[ax] - ssz_old, mesh_control,
                            compat)
    ssz = g[ax][1] - g[ax][0]
    aa = ssz / 2000
    for o in range(3):
        if o == ax:
            continue
        o_old = matlab_colon(mn[o], ssz, mx[o], compat)
        diff = mx[o] - o_old[-1]
        g[o] = matlab_colon(mn[o] + diff / 2, ssz, mx[o], compat)
    cen = []
    for o in range(3):
        c = g[o] + ssz / 2
        cen.append(c[:-1])
    return Grid(g[0], g[1], g[2], cen[0], cen[1], cen[2], float(ssz),
                float(aa), ax, mn, mx)


# ----------------------------------------------------------------------------
# geomeshini + domainstokeep
# ----------------------------------------------------------------------------
@dataclass
class Domain:
    grid: Grid
    nelx: int
    nely: int
    nelz: int
    nele: int
    ndof: int
    oute: np.ndarray            # (nely,nelx,nelz) 0/1 active elements
    outeM: np.ndarray           # (nely,nelx,nelz) 0/1 elements kept solid
    sup_all: np.ndarray         # node indices (0-based, MATLAB node order)
    sup_x: np.ndarray
    sup_y: np.ndarray
    sup_z: np.ndarray
    Fn: list                    # list of node-index arrays per force region
    timings: dict = field(default_factory=dict)
    # per region role ("fixed", "xfixed", ..., "force1", ..., "keepdom"):
    # {"nodes": grid nodes inside, "elements": active elements whose centre is
    # inside (kept regions only, else None), "elements_outside": inactive
    # elements whose centre is inside (kept regions only), "kept": bool}.
    # Diagnostics only (setup validation, freeto.audit); numerics unchanged.
    regions: dict = field(default_factory=dict)


def _load(path, cache):
    if path is None:
        return None
    key = str(path)
    if key not in cache:
        cache[key] = read_stl(path)
    return cache[key]


def prepare_domain(mesh_control, domain, fixed=None, xfixed=None, yfixed=None,
                   zfixed=None, forces=(), keepdom=None, keep_bc=True,
                   keep_bcx=False, keep_bcy=False, keep_bcz=False,
                   compat="matlab", inside_mode="robust", log=None):
    """geomeshini.m + domainstokeep.m.

    ``inside_mode``: "matlab" = exact intriangulation.m semantics, "robust" =
    watertight edge rule with 3-axis majority vote (see inside.inside_grid).
    """
    import time
    t0 = time.perf_counter()
    cache = {}
    Vd, Fd = _load(domain, cache)
    grid = build_grid(Vd, mesh_control, compat)
    x, y, z = grid.x, grid.y, grid.z
    nelx, nely, nelz = grid.nelx, grid.nely, grid.nelz
    if min(nelx, nely, nelz) < 1:
        raise ValueError("the grid has no elements along at least one axis "
                         "(domain too thin for this mesh_control)")
    nele = nelx * nely * nelz
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    tim = {"read_grid": time.perf_counter() - t0}

    inside_cache = {}

    def node_test(path):
        key = ("n", str(path))
        if key not in inside_cache:
            V, F = _load(path, cache)
            inside_cache[key] = inside_grid(V, F, x, y, z, mode=inside_mode)
        return inside_cache[key]

    def elem_test(path):
        key = ("e", str(path))
        if key not in inside_cache:
            V, F = _load(path, cache)
            inside_cache[key] = inside_grid(V, F, grid.xc, grid.yc, grid.zc,
                                              mode=inside_mode)
        return inside_cache[key]

    t1 = time.perf_counter()
    empty = np.zeros(0, dtype=np.int64)
    sup_all = _find_matlab(node_test(fixed)) if fixed else empty
    sup_x = _find_matlab(node_test(xfixed)) if xfixed else empty
    sup_y = _find_matlab(node_test(yfixed)) if yfixed else empty
    sup_z = _find_matlab(node_test(zfixed)) if zfixed else empty
    Fn = [_find_matlab(node_test(f)) for f in forces]
    tim["inside_nodes"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    oute_xyz = elem_test(domain).astype(np.float64)
    oute_xyz[oute_xyz < 0] = 0   # no-op unless inside_grid(undecided=-1)
    oute = np.ascontiguousarray(xyz_to_matlab(oute_xyz))
    # domainstokeep: list of region meshes in MATLAB's concatenation order
    if keep_bc:
        regions = [fixed]
        regions += [xfixed if keep_bcx else None]
        regions += [yfixed if keep_bcy else None]
        regions += [zfixed if keep_bcz else None]
        regions += list(forces)
        regions += [keepdom]
    else:
        regions = [keepdom]
    regions = [r for r in regions if r]
    outeM = np.zeros((nelx, nely, nelz))
    for r in regions:
        outeM = outeM + elem_test(r)
        outeM[outeM < 0] = 0
        outeM[outeM > 1] = 1
    outeM = np.ascontiguousarray(xyz_to_matlab(outeM))
    tim["inside_elements"] = time.perf_counter() - t1
    kept = {str(r) for r in regions}
    act_xyz = oute_xyz > 0
    reg = {}
    roles = [("fixed", fixed, sup_all), ("xfixed", xfixed, sup_x), ("yfixed", yfixed, sup_y),
             ("zfixed", zfixed, sup_z)]
    roles += [(f"force{i + 1}", f, Fn[i]) for i, f in enumerate(forces)]
    roles += [("keepdom", keepdom, None)]
    for role, path, nodes in roles:
        if not path:
            continue
        is_kept = (str(path) in kept and (role in ("fixed", "keepdom") or role.startswith("force")
                                          or (role == "xfixed" and keep_bcx)
                                          or (role == "yfixed" and keep_bcy)
                                          or (role == "zfixed" and keep_bcz)))
        if role != "keepdom" and not keep_bc:
            is_kept = False
        d = {"file": os.path.basename(str(path)), "nodes": None if nodes is None else int(len(nodes)),
             "elements": None, "elements_outside": None, "kept": bool(is_kept)}
        if is_kept:
            e = elem_test(path) > 0
            d["elements"] = int(np.count_nonzero(e & act_xyz))
            d["elements_outside"] = int(np.count_nonzero(e & ~act_xyz))
        reg[role] = d
    return Domain(grid, nelx, nely, nelz, nele, ndof, oute, outeM, sup_all,
                  sup_x, sup_y, sup_z, Fn, tim, reg)


# ----------------------------------------------------------------------------
# domainprep
# ----------------------------------------------------------------------------
def element_dofs(nelx, nely, nelz, elements=None):
    """edofMat (0-based) for the given element indices (default: all)."""
    nyp = nely + 1
    nxy = (nelx + 1) * (nely + 1)
    if elements is None:
        elements = np.arange(nelx * nely * nelz)
    e = np.asarray(elements, dtype=np.int64)
    r = e % nely
    c = (e // nely) % nelx
    p = e // (nely * nelx)
    n0 = r + nyp * c + nxy * p
    noff = np.array([1, 1 + nyp, nyp, 0, nxy + 1, nxy + 1 + nyp, nxy + nyp,
                     nxy], dtype=np.int64)
    nodes = n0[:, None] + noff[None, :]
    edof = (3 * nodes[:, :, None] + np.arange(3)[None, None, :]).reshape(-1, 24)
    return edof


def element_nodes(nelx, nely, nelz, elements):
    nyp = nely + 1
    nxy = (nelx + 1) * (nely + 1)
    e = np.asarray(elements, dtype=np.int64)
    r = e % nely
    c = (e // nely) % nelx
    p = e // (nely * nelx)
    n0 = r + nyp * c + nxy * p
    noff = np.array([1, 1 + nyp, nyp, 0, nxy + 1, nxy + 1 + nyp, nxy + nyp,
                     nxy], dtype=np.int64)
    return n0[:, None] + noff[None, :]


# ----------------------------------------------------------------------------
# forcevec / supportDOFs
# ----------------------------------------------------------------------------
def _as_list(v):
    if v is None:
        return [0.0]
    if np.isscalar(v):
        return [float(v)]
    return [float(a) for a in np.asarray(v, dtype=float).ravel()]


def force_vectors(Fn, ndof, fmagx, fmagy, fmagz, loadtype="distributed"):
    """forcevec.m.  Returns dense F (ndof, nloads) with all-zero columns
    removed, and nf (number of load cases before removal)."""
    fx, fy, fz = _as_list(fmagx), _as_list(fmagy), _as_list(fmagz)
    nf = max(len(fx), len(fy), len(fz))
    Fx = np.zeros((ndof, nf))
    Fy = np.zeros((ndof, nf))
    Fz = np.zeros((ndof, nf))
    for i in range(nf):
        frn = np.asarray(Fn[i], dtype=np.int64)
        if loadtype == "point":
            frn = np.sort(frn)
            k = int(math.floor(len(frn) / 2 + 0.5))  # MATLAB round, 1-based
            frn = frn[k - 1:k]
        cnt = len(frn)
        with np.errstate(divide="ignore", invalid="ignore"):
            if len(fx) > 1:
                Fx[3 * frn, i] = fx[i] / cnt if cnt else 0.0
            else:
                Fx[3 * frn, 0] = fx[0] / cnt if cnt else 0.0
            if len(fy) > 1:
                Fy[3 * frn + 1, i] = fy[i] / cnt if cnt else 0.0
            else:
                Fy[3 * frn + 1, 0] = fy[0] / cnt if cnt else 0.0
            if len(fz) > 1:
                Fz[3 * frn + 2, i] = fz[i] / cnt if cnt else 0.0
            else:
                Fz[3 * frn + 2, 0] = fz[0] / cnt if cnt else 0.0
    F = Fx + Fy + Fz
    keep = ~np.all(F == 0, axis=0)
    return F[:, keep], nf


def support_dofs(sup_all, sup_x, sup_y, sup_z):
    """supportDOFs.m (0-based, sorted unique)."""
    sa = np.asarray(sup_all, dtype=np.int64)
    parts = [np.stack([3 * sa, 3 * sa + 1, 3 * sa + 2], axis=1).ravel(),
             3 * np.asarray(sup_x, dtype=np.int64),
             3 * np.asarray(sup_y, dtype=np.int64) + 1,
             3 * np.asarray(sup_z, dtype=np.int64) + 2]
    return np.unique(np.concatenate(parts))
