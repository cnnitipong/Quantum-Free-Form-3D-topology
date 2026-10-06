"""STL reading / writing.

``read_stl`` mirrors MATLAB's ``stlread`` as used by FreeTO: binary or ASCII
files, single-precision coordinates promoted to float64, and duplicate
vertices merged so that ``faces`` index a set of unique points (the triangle
vertex order inside each face is preserved, which is what the ray-casting
predicates depend on).
"""
from __future__ import annotations

import os
import re
import struct

import numpy as np

__all__ = ["read_stl", "write_stl", "face_normals"]

_FLOAT_RE = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"
_VERTEX_RE = re.compile(
    rb"vertex\s+(" + _FLOAT_RE.encode() + rb")\s+(" + _FLOAT_RE.encode()
    + rb")\s+(" + _FLOAT_RE.encode() + rb")")


def _read_raw(path):
    """Return the raw (ntri, 3, 3) float64 triangle array of an STL file."""
    with open(path, "rb") as fh:
        data = fh.read()
    n = len(data)
    if n >= 84:
        ntri = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * ntri == n:
            rec = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)),
                            ("attr", "<u2")])
            arr = np.frombuffer(data, dtype=rec, count=ntri, offset=84)
            return arr["v"].astype(np.float64)
    # ASCII
    toks = _VERTEX_RE.findall(data)
    if not toks:
        if data.lstrip()[:5].lower() == b"solid":
            return np.zeros((0, 3, 3))
        raise ValueError(f"{path}: not a valid binary or ASCII STL file")
    vals = np.array([[float(a.replace(b"d", b"e").replace(b"D", b"e"))
                      for a in t] for t in toks], dtype=np.float64)
    if vals.shape[0] % 3:
        raise ValueError(f"{path}: ASCII STL vertex count is not a multiple of 3")
    return vals.reshape(-1, 3, 3)


def read_stl(path):
    """Read an STL file.

    Returns
    -------
    vertices : (n, 3) float64 array of unique points
    faces : (m, 3) int64 array of 0-based vertex indices
    """
    path = os.fspath(path)
    tri = _read_raw(path)
    m = tri.shape[0]
    if m == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    V = tri.reshape(-1, 3)
    uniq, inv = np.unique(V, axis=0, return_inverse=True)
    faces = inv.reshape(m, 3).astype(np.int64)
    return np.ascontiguousarray(uniq), faces


def face_normals(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if f.size == 0:
        return np.zeros((0, 3))
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return n / ln


def write_stl(path, vertices, faces, header="FreeTO-Python binary STL"):
    """Write a binary STL file (float32 coordinates, unit face normals)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    m = f.shape[0]
    rec = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)),
                    ("attr", "<u2")])
    arr = np.zeros(m, dtype=rec)
    if m:
        arr["normal"] = face_normals(v, f)
        arr["v"] = v[f]
    hdr = header.encode("ascii", "replace")[:80].ljust(80, b" ")
    with open(os.fspath(path), "wb") as fh:
        fh.write(hdr)
        fh.write(struct.pack("<I", m))
        fh.write(arr.tobytes())
