"""Programmatic, watertight STL generation for the CONTINUUM example problems.

FreeTO's "inside" test (``freeto.inside.inside_grid``) needs a *closed*
surface with consistent outward-facing normals: every edge of the mesh must
be shared by exactly two triangles, and the signed volume must be positive
(``freeto.core._open_edges`` / the orientation check in
:func:`freeto.postprocess.surface_from_field` follow the same convention).
Everything in this module builds meshes that satisfy that by construction —
no CSG library is used, faces are only ever appended in matching pairs, so
"watertight" is an invariant of the construction rather than something that
needs to be checked after the fact (:func:`is_watertight` is provided anyway,
to verify in tests).

Units are **mm** throughout, matching the rest of FreeTO-Python.

Two families of shapes:

* :func:`box` — an axis-aligned box, the basic building block.
* :func:`l_shape` — a box with one axis-aligned corner notch removed (an "L"
  or "Γ" cross-section extruded along the third axis), built directly as two
  boxes with the shared internal face omitted from both sides (not a CSG
  union: the two "half" solids are built so their exposed faces already tile
  the outer boundary with no overlap and no gap).
* :func:`box_with_hole` — a box with a rectangular through-hole, built as a
  "picture frame": the frame's top/bottom caps are 4 non-overlapping
  rectangles and the hole gets its own (inward-facing) side wall.

Region helpers for supports and loads:

* :func:`face_slab` — a thin box hugging one face of a domain box, thick
  enough to survive FreeTO's 1% grid shrink (see module docstring below) and
  overlapping the domain so it actually captures grid nodes.
* :func:`patch` — a small box around a point, for tip/point loads.
* :func:`cylinder` — an approximated (n-gon prism) watertight cylinder, for
  bolt-hole-like regions.

Recommended workflow: build a domain box (or L-shape / box-with-hole) with
:func:`box` / :func:`l_shape` / :func:`box_with_hole`, then carve fixed/force
regions out of it with :func:`face_slab` / :func:`patch`, and write each with
:func:`write_mesh`.

Note on the 1% grid shrink
---------------------------
``freeto.mesh.build_grid`` places grid points for the axis that receives
``MeshControl`` points strictly *inside* the domain's bounding box: it first
computes a coarse ``linspace(min, max, MeshControl)``, takes 1% of that
spacing (``ssz_old``), and then rebuilds the axis as
``linspace(min + ssz_old, max - ssz_old, MeshControl)``. So the outermost
layer of grid nodes on that axis sits about 1% of the domain's extent
*inside* the true bounding-box face, never exactly on it. A support/load
region whose slab stops exactly at the nominal domain face is still fine
(the shrunk nodes are strictly inside it) — the real hazard is a *flat*,
zero-thickness region STL placed exactly on the face plane, which both has
no volume for the inside test and can numerically miss the shrunk node layer.
:func:`face_slab` therefore always gives the slab a real thickness and
extends it a little *outward* past the nominal domain face (``margin``,
default a fraction of ``thickness``) as well as the requested ``thickness``
inward, so it comfortably encloses the boundary node layer regardless of
which axis MeshControl lands on.
"""
from __future__ import annotations

import math
import os

import numpy as np

from .stl_io import write_stl

__all__ = ["box", "l_shape", "box_with_hole", "face_slab", "patch",
           "cylinder", "write_mesh", "is_watertight", "combine",
           "translate", "bbox_of"]

_AXES = {"x": 0, "y": 1, "z": 2}


# ----------------------------------------------------------------------------
# low-level: a single closed box
# ----------------------------------------------------------------------------
def box(lo, hi):
    """Closed, outward-oriented triangulated axis-aligned box.

    Parameters
    ----------
    lo, hi : (3,) array-like
        Opposite corners (``lo`` need not be the numerically smaller corner
        on every axis; it is sorted internally).

    Returns
    -------
    vertices (8, 3) float64, faces (12, 3) int64
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],   # 0-3: z0 face
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],   # 4-7: z1 face
    ], dtype=np.float64)
    F = np.array([
        [0, 2, 1], [0, 3, 2],   # z0 (outward = -z)
        [4, 5, 6], [4, 6, 7],   # z1 (outward = +z)
        [0, 1, 5], [0, 5, 4],   # y0 (outward = -y)
        [1, 2, 6], [1, 6, 5],   # x1 (outward = +x)
        [2, 3, 7], [2, 7, 6],   # y1 (outward = +y)
        [3, 0, 4], [3, 4, 7],   # x0 (outward = -x)
    ], dtype=np.int64)
    return V, F


def translate(mesh, delta):
    V, F = mesh
    return V + np.asarray(delta, dtype=np.float64), F


def bbox_of(mesh):
    V, _ = mesh
    return V.min(axis=0), V.max(axis=0)


def _weld(V, F):
    """Merge bit-identical coincident vertices and remap faces. Used to fuse
    the touching faces between adjacent grid cells in :func:`l_shape` /
    :func:`box_with_hole` (each cell is built as an independent ``box()``, so
    the shared boundary has duplicate vertices at identical coordinates until
    welded) and, harmlessly, whenever :func:`combine` is used on meshes that
    happen to share vertices exactly."""
    if V.shape[0] == 0:
        return V, F
    uv, inv = np.unique(V, axis=0, return_inverse=True)
    inv = inv.ravel()
    return uv, inv[F]


def combine(*meshes):
    """Concatenate several closed meshes into one STL, welding any bit-
    identical coincident vertices across pieces (so touching pieces built as
    independent boxes — see :func:`l_shape` / :func:`box_with_hole` — fuse
    into a single watertight shell rather than leaving duplicated vertices
    along the shared faces). Also fine for genuinely disjoint shells (the
    weld is then a no-op): FreeTO's inside test accepts several disjoint
    watertight shells in one file, but prefer a single fused shell for
    anything that must read as one solid."""
    Vs, Fs = [], []
    off = 0
    for V, F in meshes:
        V = np.asarray(V, dtype=np.float64)
        F = np.asarray(F, dtype=np.int64)
        Vs.append(V)
        Fs.append(F + off)
        off += V.shape[0]
    return _weld(np.concatenate(Vs, axis=0), np.concatenate(Fs, axis=0))


def write_mesh(path, mesh, header="FreeTO-Python generated geometry"):
    V, F = mesh
    write_stl(path, V, F, header=header)
    return str(path)


def is_watertight(mesh, check_orientation=True):
    """True if every edge is shared by exactly two triangles and (optionally)
    the mesh encloses positive signed volume with outward-pointing normals
    (matches the convention in :func:`freeto.core._open_edges` /
    :func:`freeto.postprocess.surface_from_field`)."""
    V, F = mesh
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if F.size == 0:
        return False
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]),
                axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    if np.any(cnt != 2):
        return False
    if check_orientation:
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        vol6 = np.einsum("ij,ij->", v0, np.cross(v1 - v0, v2 - v0))
        if vol6 <= 0:
            return False
    return True


# ----------------------------------------------------------------------------
# L-shape: an outer box with one axis-aligned corner notch removed
# ----------------------------------------------------------------------------
def l_shape(lo, hi, notch_lo, notch_hi):
    """L-shaped (or "Γ"-shaped) prism: the box ``[lo, hi]`` minus a corner
    notch ``[notch_lo, notch_hi]``.

    The notch must touch exactly two of the outer box's faces on two
    different axes (i.e. it removes a rectangular corner column that runs
    the full extent of the box on the third axis) — this is what keeps the
    remaining solid a single L cross-section extruded along that third axis,
    rather than a hole or a slot. Concretely, for each axis the notch's
    ``[lo, hi]`` interval must either equal the outer box's own interval on
    that axis (the "full-extent" axis) or touch one of the outer box's two
    ends on the other axes (a "cut" axis) — exactly one axis is full-extent.

    Built directly as two boxes (the two arms of the L) with the internal
    face where they meet omitted from both, so the result is a single
    watertight shell with no internal (invisible) geometry.

    Returns
    -------
    vertices (n, 3) float64, faces (m, 3) int64
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    nlo = np.asarray(notch_lo, dtype=np.float64)
    nhi = np.asarray(notch_hi, dtype=np.float64)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    nlo, nhi = np.minimum(nlo, nhi), np.maximum(nlo, nhi)

    tol = 1e-9 * max(1.0, float(np.max(hi - lo)))
    full_axis = None
    for ax in range(3):
        if abs(nlo[ax] - lo[ax]) < tol and abs(nhi[ax] - hi[ax]) < tol:
            if full_axis is not None:
                raise ValueError("notch spans the full extent on more than "
                                 "one axis; not a corner notch")
            full_axis = ax
    if full_axis is None:
        raise ValueError("notch does not span the full extent on any axis "
                         "(it must run the whole prism depth); pass "
                         "notch_lo/notch_hi matching lo/hi on the depth axis")
    a, b = [ax for ax in range(3) if ax != full_axis]
    # which side of the outer box does the notch touch on axes a, b?
    touch_a_min = abs(nlo[a] - lo[a]) < tol
    touch_a_max = abs(nhi[a] - hi[a]) < tol
    touch_b_min = abs(nlo[b] - lo[b]) < tol
    touch_b_max = abs(nhi[b] - hi[b]) < tol
    if touch_a_min == touch_a_max or touch_b_min == touch_b_max:
        raise ValueError("notch must touch exactly one end of the outer box "
                         "on each of the two non-depth axes (a corner cut)")
    cut_a = nlo[a] if touch_a_max else nhi[a]   # split coordinate on axis a
    cut_b = nlo[b] if touch_b_max else nhi[b]   # split coordinate on axis b

    # 2x2 grid of cells on the (a, b) plane, full `full_axis` depth; the one
    # cell matching the notch's corner is skipped. Every internal face
    # between the 3 remaining cells is shared in full by exactly one present
    # neighbour (the grid lines line up exactly, unlike a naive 2-box split
    # which would drop a whole face where only part of it is internal), so
    # "drop the face touching a present neighbour" is exact.
    bins_a = [(lo[a], cut_a), (cut_a, hi[a])]
    bins_b = [(lo[b], cut_b), (cut_b, hi[b])]
    notch_ia = 1 if touch_a_max else 0
    notch_ib = 1 if touch_b_max else 0

    def _present(ia, ib):
        return 0 <= ia <= 1 and 0 <= ib <= 1 and not (ia == notch_ia and ib == notch_ib)

    def _cell_minus(ia, ib):
        blo, bhi = lo.copy(), hi.copy()
        blo[a], bhi[a] = bins_a[ia]
        blo[b], bhi[b] = bins_b[ib]
        V, F = box(blo, bhi)
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        keep = np.ones(F.shape[0], dtype=bool)
        neighbours = [(a, 1 - ia, ib, bins_a[ia][1 - ia] if ia == 0 else bins_a[ia][0],
                      +1 if ia == 0 else -1),
                     (b, ia, 1 - ib, bins_b[ib][1 - ib] if ib == 0 else bins_b[ib][0],
                      +1 if ib == 0 else -1)]
        for axis_i, nia, nib, plane, sign in neighbours:
            if not _present(nia, nib):
                continue
            coord = v0[:, axis_i]
            on_plane = np.abs(coord - plane) < tol
            face_sign = n[:, axis_i]
            is_face = on_plane & ((face_sign > 0) if sign > 0 else (face_sign < 0))
            keep &= ~is_face
        return V, F[keep]

    cells = [_cell_minus(ia, ib) for ia in range(2) for ib in range(2)
            if _present(ia, ib)]
    return combine(*cells)


# ----------------------------------------------------------------------------
# box with a rectangular through-hole ("picture frame" prism)
# ----------------------------------------------------------------------------
def box_with_hole(lo, hi, hole_lo, hole_hi, axis="z"):
    """Box with a rectangular through-hole along ``axis`` (default z).

    ``hole_lo``/``hole_hi`` are given in the box's own (x, y, z) mm
    coordinates and must span the full box extent on ``axis`` (the hole goes
    all the way through) and lie strictly inside the box on the other two
    axes. Built directly as a "picture frame": the two caps perpendicular to
    ``axis`` are 4 non-overlapping rectangles (no polygon-with-hole
    triangulation needed since both loops are axis-aligned rectangles), the
    outer perimeter gets 4 outward side walls and the hole gets 4
    inward-facing side walls.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    hlo = np.asarray(hole_lo, dtype=np.float64)
    hhi = np.asarray(hole_hi, dtype=np.float64)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    hlo, hhi = np.minimum(hlo, hhi), np.maximum(hlo, hhi)
    ax = _AXES[axis] if isinstance(axis, str) else int(axis)
    a, b = [i for i in range(3) if i != ax]
    tol = 1e-9 * max(1.0, float(np.max(hi - lo)))
    if not (abs(hlo[ax] - lo[ax]) < tol and abs(hhi[ax] - hi[ax]) < tol):
        raise ValueError("the hole must span the full box extent on `axis`")
    if not (lo[a] < hlo[a] < hhi[a] < hi[a] and lo[b] < hlo[b] < hhi[b] < hi[b]):
        raise ValueError("the hole must lie strictly inside the box on the "
                         "other two axes")

    # 3x3 grid of cells on the (a, b) plane (bin 1 = the hole's own range on
    # that axis), full `ax` depth; the centre cell (hole) is skipped. Every
    # internal face is shared in full by exactly one present neighbour cell
    # (all grid lines line up exactly), so a simple "drop the face touching
    # a present neighbour" rule leaves each remaining face — outer wall,
    # hole wall, or cap — appearing exactly once.
    bins_a = [(lo[a], hlo[a]), (hlo[a], hhi[a]), (hhi[a], hi[a])]
    bins_b = [(lo[b], hlo[b]), (hlo[b], hhi[b]), (hhi[b], hi[b])]

    def _present(ia, ib):
        return 0 <= ia <= 2 and 0 <= ib <= 2 and not (ia == 1 and ib == 1)

    def _cell_minus(ia, ib):
        blo, bhi = lo.copy(), hi.copy()
        blo[a], bhi[a] = bins_a[ia]
        blo[b], bhi[b] = bins_b[ib]
        V, F = box(blo, bhi)
        v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        keep = np.ones(F.shape[0], dtype=bool)
        neighbours = [(a, ia - 1, ib, bins_a[ia][0], -1),
                     (a, ia + 1, ib, bins_a[ia][1], +1),
                     (b, ia, ib - 1, bins_b[ib][0], -1),
                     (b, ia, ib + 1, bins_b[ib][1], +1)]
        for axis_i, nia, nib, plane, sign in neighbours:
            if not _present(nia, nib):
                continue    # boundary (outer wall) or hole wall: keep
            coord = v0[:, axis_i]
            on_plane = np.abs(coord - plane) < tol
            face_sign = n[:, axis_i]
            is_face = on_plane & ((face_sign > 0) if sign > 0 else (face_sign < 0))
            keep &= ~is_face
        return V, F[keep]

    cells = [_cell_minus(ia, ib) for ia in range(3) for ib in range(3)
            if _present(ia, ib)]
    return combine(*cells)


# ----------------------------------------------------------------------------
# region helpers: face slabs, patches, cylinders
# ----------------------------------------------------------------------------
def face_slab(domain_box, face, thickness, margin=None, extent=None):
    """Thin box hugging one face of ``domain_box`` (a ``(lo, hi)`` pair),
    for support / load regions.

    Parameters
    ----------
    domain_box : (lo, hi)
        The design domain's own bounding box (mm).
    face : str
        One of ``"x-", "x+", "y-", "y+", "z-", "z+"`` — which domain face the
        slab hugs.
    thickness : float
        How far the slab extends *inward* from the nominal face (mm); should
        be at least ~1.5 element sizes at the mesh you plan to run so the
        slab reliably contains a full layer of grid nodes.
    margin : float, optional
        How far the slab extends *outward*, past the nominal domain face
        (mm) — guards against FreeTO's 1% grid-point shrink on the
        MeshControl axis (see module docstring). Defaults to
        ``max(0.2 * thickness, 0.05 * (hi-lo along the face axis))``.
    extent : float, optional
        Half-width (mm) to shrink the slab's in-plane footprint by on the
        two axes parallel to the face (use this for a smaller patch such as
        a line/point load near mid-face instead of the whole face); if given
        as a single float, the slab is centred on the domain's face and
        shrunk symmetrically by that amount on both in-plane axes. Leave
        ``None`` to cover the whole face.

    Returns
    -------
    vertices, faces
    """
    lo, hi = (np.asarray(domain_box[0], dtype=np.float64),
              np.asarray(domain_box[1], dtype=np.float64))
    axis = _AXES[face[0]]
    side = face[1]
    if side not in ("-", "+"):
        raise ValueError("face must be like 'x-', 'x+', 'y-', 'y+', 'z-', 'z+'")
    span = float(hi[axis] - lo[axis])
    if margin is None:
        margin = max(0.2 * float(thickness), 0.05 * span)
    slo, shi = lo.copy(), hi.copy()
    if side == "-":
        slo[axis] = lo[axis] - margin
        shi[axis] = lo[axis] + float(thickness)
    else:
        slo[axis] = hi[axis] - float(thickness)
        shi[axis] = hi[axis] + margin
    if extent is not None:
        for ax in range(3):
            if ax == axis:
                continue
            c = 0.5 * (lo[ax] + hi[ax])
            e = float(extent)
            slo[ax] = max(slo[ax], c - e)
            shi[ax] = min(shi[ax], c + e)
    return box(slo, shi)


def patch(center, size):
    """Small box of full side length(s) ``size`` centred at ``center`` — for
    a compact tip/point load or a small support pad. ``size`` is a scalar
    (cube) or a (3,) sequence (box)."""
    center = np.asarray(center, dtype=np.float64)
    size = np.broadcast_to(np.asarray(size, dtype=np.float64), (3,))
    half = size / 2.0
    return box(center - half, center + half)


def cylinder(center, axis, radius, height, n=24):
    """Watertight n-gon-prism approximation of a cylinder (for bolt-hole /
    round support or load regions).

    Parameters
    ----------
    center : (3,) — the cylinder's centroid (mid-height, mid-radius)
    axis : "x" | "y" | "z" — the cylinder's axis
    radius, height : float (mm)
    n : number of sides of the polygon approximation (>= 8 recommended)
    """
    center = np.asarray(center, dtype=np.float64)
    ax = _AXES[axis] if isinstance(axis, str) else int(axis)
    u, v = [i for i in range(3) if i != ax]
    theta = np.linspace(0.0, 2 * np.pi, int(n), endpoint=False)
    ring = np.zeros((n, 3))
    ring[:, u] = radius * np.cos(theta)
    ring[:, v] = radius * np.sin(theta)
    lo_ring = ring.copy()
    lo_ring[:, ax] = -height / 2.0
    hi_ring = ring.copy()
    hi_ring[:, ax] = height / 2.0
    V = np.concatenate([lo_ring, hi_ring], axis=0) + center
    n0 = np.arange(n)
    n1 = (n0 + 1) % n
    side = np.concatenate([
        np.stack([n0, n1, n1 + n], axis=1),
        np.stack([n0, n1 + n, n0 + n], axis=1),
    ], axis=0)
    # bottom cap (outward = -axis): fan from vertex 0, wound so the normal
    # points in -axis; top cap (outward = +axis): fan wound the other way.
    fan_lo = np.stack([np.zeros(n - 2, dtype=np.int64),
                       np.arange(2, n), np.arange(1, n - 1)], axis=1)
    fan_hi = (np.stack([np.zeros(n - 2, dtype=np.int64),
                        np.arange(1, n - 1), np.arange(2, n)], axis=1) + n)
    F = np.concatenate([side, fan_lo, fan_hi], axis=0)
    # orient consistently with the axis sign convention used by box(): a
    # right-handed ring (theta increasing) with u,v in axis order (u,v) such
    # that (u,v,ax) is a right-handed permutation of (x,y,z) gives outward
    # normals already; for the one permutation that is left-handed (when ax
    # is the middle axis, u=x,v=z) flip the side/caps winding.
    perm_sign = {(1, 2, 0): 1, (2, 0, 1): 1, (0, 1, 2): 1,
                (2, 1, 0): -1, (1, 0, 2): -1, (0, 2, 1): -1}[(u, v, ax)]
    if perm_sign < 0:
        F = F[:, ::-1]
    return V, F
