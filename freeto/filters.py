"""Density filters HHs3D and HnHns3D, vectorised with index offsets."""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

__all__ = ["HHs3D", "HnHns3D"]


def HHs3D(nelx, nely, nelz, rmin, ele, nele):
    """Element filter restricted to the active elements ``ele``.

    Returns (H (nnele x nnele CSR), Hs (nnele,)) with H = H_full(ele, ele)
    and Hs = row sums of the restricted matrix (as in HHs3D.m).
    """
    c = int(math.ceil(rmin)) - 1
    ele = np.asarray(ele, dtype=np.int64)
    nnele = ele.size
    # position of each element in the active list (-1 = passive)
    pos = np.full(nele, -1, dtype=np.int64)
    pos[ele] = np.arange(nnele)
    j1 = ele % nely
    i1 = (ele // nely) % nelx
    k1 = ele // (nely * nelx)
    rows, cols, vals = [], [], []
    for dk in range(-c, c + 1):
        for di in range(-c, c + 1):
            for dj in range(-c, c + 1):
                w = max(0.0, rmin - math.sqrt(di * di + dj * dj + dk * dk))
                if w == 0.0:
                    continue   # sparse() drops zeros
                j2 = j1 + dj
                i2 = i1 + di
                k2 = k1 + dk
                ok = ((j2 >= 0) & (j2 < nely) & (i2 >= 0) & (i2 < nelx)
                      & (k2 >= 0) & (k2 < nelz))
                e2 = k2 * (nelx * nely) + i2 * nely + j2
                idx = np.nonzero(ok)[0]
                p2 = pos[e2[idx]]
                sel = p2 >= 0
                rows.append(idx[sel])
                cols.append(p2[sel])
                vals.append(np.full(sel.sum(), w))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(nnele, nnele))
    H.sum_duplicates()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def HnHns3D(nelx, nely, nelz, rnmin):
    """Element -> node averaging filter with HnHns3D.m's loop bounds
    (element offsets -ceil(rnmin) .. ceil(rnmin)-1 around each node, distance
    measured from the node to the element centre).

    Returns (Hn ((nelx+1)(nely+1)(nelz+1) x nele CSR), Hns (nnodes,)).
    """
    c = int(math.ceil(rnmin))
    nny, nnx, nnz = nely + 1, nelx + 1, nelz + 1
    nn = nny * nnx * nnz
    n = np.arange(nn, dtype=np.int64)
    jn = n % nny
    inn = (n // nny) % nnx
    kn = n // (nny * nnx)
    rows, cols, vals = [], [], []
    for dk in range(-c, c):
        for di in range(-c, c):
            for dj in range(-c, c):
                # distance node (1-based n1) -> centre of element n1+d (1-based)
                # elex = in2 + 0.5  =>  in1 - elex = -d - 0.5
                w = max(0.0, rnmin - math.sqrt((di + 0.5) ** 2 + (dj + 0.5) ** 2
                                               + (dk + 0.5) ** 2))
                if w == 0.0:
                    continue
                j2 = jn + dj
                i2 = inn + di
                k2 = kn + dk
                ok = ((j2 >= 0) & (j2 < nely) & (i2 >= 0) & (i2 < nelx)
                      & (k2 >= 0) & (k2 < nelz))
                idx = np.nonzero(ok)[0]
                rows.append(idx)
                cols.append(k2[idx] * (nelx * nely) + i2[idx] * nely + j2[idx])
                vals.append(np.full(idx.size, w))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    Hn = sp.csr_matrix((vals, (rows, cols)), shape=(nn, nelx * nely * nelz))
    Hn.sum_duplicates()
    Hns = np.asarray(Hn.sum(axis=1)).ravel()
    return Hn, Hns
