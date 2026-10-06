"""Post-processing: symmetry mirroring and closed-surface extraction.

The level-set field is carried as a :class:`FieldSnapshot` whose ``top``
array is in (x, y, z) axis order with ascending coordinates, so that index
(i, j, k) sits at ``origin + (i, j, k) * spacing`` (mm, the STL units).
"""
from __future__ import annotations

import numpy as np

__all__ = ["FieldSnapshot", "apply_symmetry", "surface_from_field",
           "smooth3", "SYM_PLANES"]

SYM_PLANES = ("x-y", "y-z", "z-x")


class FieldSnapshot:
    """Level-set snapshot: ``top`` = xg - ls (solid where > 0)."""

    __slots__ = ("top", "origin", "spacing", "ls")

    def __init__(self, top, origin, spacing, ls=0.0):
        top = np.asarray(top)
        if top.flags.writeable:
            try:
                top.flags.writeable = False
            except ValueError:
                pass
        self.top = top
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
        self.ls = float(ls)

    @property
    def shape(self):
        return self.top.shape

    @property
    def extent_max(self):
        return self.origin + (np.array(self.top.shape) - 1) * self.spacing

    def __repr__(self):
        return (f"FieldSnapshot(shape={self.top.shape}, origin={self.origin}, "
                f"spacing={self.spacing})")


def apply_symmetry(field, plane, direction="right"):
    """Mirror a field like symmetry.m.

    Array semantics follow symmetry.m exactly (the boundary slice is
    duplicated, so the mirror plane lies half a fine-grid spacing beyond the
    last grid plane).  In physical (x,y,z) terms:

    * ``"x-y"``: mirror along z;  right -> copy at +z,  left -> copy at -z
    * ``"y-z"``: mirror along x;  right -> copy at +x,  left -> copy at -x
    * ``"z-x"``: mirror along y;  right -> copy at -y,  left -> copy at +y
      (symmetry.m works on the row-flipped MATLAB array, hence the reversal)

    The original half keeps its physical coordinates; the origin is shifted
    when the copy is placed on the negative side.
    """
    plane = str(plane).lower()
    direction = str(direction).lower()
    top = np.asarray(field.top)
    origin = field.origin.copy()
    if plane == "x-y":
        ax, neg = 2, direction == "left"
    elif plane == "y-z":
        ax, neg = 0, direction == "left"
    elif plane == "z-x":
        ax, neg = 1, direction == "right"   # 'right' -> copy at -y
    else:
        raise ValueError(f"unknown symmetry plane {plane!r}")
    mir = np.flip(top, axis=ax)
    if neg:
        new = np.concatenate([mir, top], axis=ax)
        origin[ax] -= top.shape[ax] * field.spacing[ax]
    else:
        new = np.concatenate([top, mir], axis=ax)
    return FieldSnapshot(new, origin, field.spacing, field.ls)


def smooth3(v):
    """MATLAB smooth3(v) (3x3x3 box, replicate padding)."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(np.asarray(v, dtype=np.float64), size=3,
                          mode="nearest")


def surface_from_field(field, smooth=True, level=0.0):
    """Closed triangle surface of {top > level} in physical coordinates.

    Equivalent to MATLAB's ``isosurface(smooth3(phi),0)`` + ``isocaps``: the
    (optionally smoothed) field is padded with a large negative value so the
    marching-cubes surface is closed; the vertices generated on edges towards
    the padding lie within 1e-3 voxel of the grid boundary, giving flat caps
    as isocaps does.  The mesh is watertight (every edge shared by exactly
    two faces) and wound counter-clockwise seen from outside (outward
    normals).

    Returns vertices (n,3) float32 and faces (m,3) int32.
    """
    from skimage.measure import marching_cubes
    phi = np.asarray(field.top, dtype=np.float64)
    if smooth:
        phi = smooth3(phi)
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    if phi.size == 0 or not np.any(phi > level):
        return empty
    big = max(1e-12, float(np.abs(phi).max())) * 1e3
    pad = np.pad(phi - level, 1, mode="constant", constant_values=-big)
    try:
        verts, faces, _, _ = marching_cubes(pad, level=0.0,
                                            allow_degenerate=True)
    except (ValueError, RuntimeError):
        return empty
    # vertices on edges towards the padding sit <= 1e-3 voxel outside the
    # boundary plane: the caps are flat and practically on the grid boundary
    # (like isocaps) while every triangle keeps a non-zero area.
    verts = field.origin + (verts - 1.0) * field.spacing
    faces = faces.astype(np.int64)
    # weld coincident vertices and drop faces that became degenerate
    uv, inv = np.unique(verts, axis=0, return_inverse=True)
    inv = inv.ravel()
    verts = uv
    faces = inv[faces]
    good = ((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
            & (faces[:, 0] != faces[:, 2]))
    faces = faces[good]
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    # orientation: make the enclosed signed volume positive
    if faces.shape[0]:
        vol6 = np.einsum("ij,ij->", verts[faces[:, 0]], cr)
        if vol6 < 0:
            faces = faces[:, ::-1]
    # compact vertices
    used = np.unique(faces)
    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return (verts[used].astype(np.float32),
            remap[faces].astype(np.int32))
