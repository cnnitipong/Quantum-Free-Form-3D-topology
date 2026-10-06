"""QUBO containers and conversions.

Convention (docs/QUANTUM_DESIGN.md §C): minimise

    E(x) = x^T Q x + h . x + const,     x in {0,1}^n,

with ``Q`` symmetric and zero-diagonal (the diagonal is folded into ``h``
because x_i^2 = x_i).  ``Q`` may be a dense ndarray or a scipy sparse matrix.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["normalize_qubo", "energy", "to_ising", "from_ising", "to_bqm_dict",
           "to_bqm", "is_sparse", "dense", "qubo_max_abs"]


def is_sparse(Q):
    return sp.issparse(Q)


def dense(Q):
    return Q.toarray() if sp.issparse(Q) else np.asarray(Q, dtype=np.float64)


def normalize_qubo(Q, h=None, const=0.0):
    """Symmetrise ``Q`` and fold its diagonal into ``h``.

    Returns (Q_sym_zero_diag, h (n,), const) — sparse stays sparse (CSR)."""
    if sp.issparse(Q):
        Q = sp.csr_matrix(Q, dtype=np.float64)
        n = Q.shape[0]
        d = Q.diagonal().copy()
        Q = sp.csr_matrix((Q + Q.T) * 0.5 - sp.diags(d))
        Q.eliminate_zeros()
    else:
        Q = np.array(Q, dtype=np.float64, copy=True)
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Q must be a square matrix")
        n = Q.shape[0]
        d = np.diag(Q).copy()
        Q = 0.5 * (Q + Q.T)
        np.fill_diagonal(Q, 0.0)
    h = np.zeros(n) if h is None else np.asarray(h, dtype=np.float64).reshape(-1).copy()
    if h.size != n:
        raise ValueError(f"h has {h.size} entries, Q is {n}x{n}")
    return Q, h + d, float(const)


def energy(Q, h, const, X):
    """Energies of samples ``X`` ((R, n) or (n,)) — vectorised."""
    X = np.asarray(X, dtype=np.float64)
    one = X.ndim == 1
    X2 = X[None, :] if one else X
    if sp.issparse(Q):
        QX = (Q @ X2.T).T
    else:
        QX = X2 @ Q
    E = np.einsum("ij,ij->i", QX, X2) + X2 @ h + const
    return float(E[0]) if one else E


def to_ising(Q, h, const=0.0):
    """x = (1 - z)/2  ->  E = z^T J z + b . z + c  (J symmetric, zero diag).

    x^T Q x = 1/4 (1-z)^T Q (1-z) = 1/4 [sum Q - 2 (Q 1).z + z^T Q z]."""
    Qd = dense(Q)
    row = Qd.sum(axis=1)
    J = 0.25 * Qd
    b = -0.5 * row - 0.5 * h
    c = 0.25 * Qd.sum() + 0.5 * h.sum() + const
    return J, b, float(c)


def from_ising(J, b, c=0.0):
    """Inverse of :func:`to_ising` (z = 1 - 2x)."""
    J = dense(J)
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)
    Q = 4.0 * J
    h = -4.0 * J.sum(axis=1) - 2.0 * b
    const = J.sum() + b.sum() + c
    return Q, h, float(const)


def to_bqm_dict(Q, h, const=0.0):
    """(linear, quadratic, offset) dicts for dimod ({(i, j): 2 Q_ij}, i < j)."""
    linear = {i: float(v) for i, v in enumerate(np.asarray(h))}
    quad = {}
    if sp.issparse(Q):
        C = sp.triu(Q, k=1).tocoo()
        for i, j, v in zip(C.row, C.col, C.data):
            if v != 0:
                quad[(int(i), int(j))] = 2.0 * float(v)
    else:
        iu, ju = np.nonzero(np.triu(Q, k=1))
        for i, j in zip(iu, ju):
            quad[(int(i), int(j))] = 2.0 * float(Q[i, j])
    return linear, quad, float(const)


def to_bqm(Q, h, const=0.0):
    """dimod.BinaryQuadraticModel (requires dimod)."""
    import dimod
    lin, quad, off = to_bqm_dict(Q, h, const)
    return dimod.BinaryQuadraticModel(lin, quad, off, dimod.BINARY)


def qubo_max_abs(Q, h):
    qa = abs(Q).max() if sp.issparse(Q) else (np.abs(Q).max() if Q.size else 0.0)
    return float(max(qa, np.abs(h).max() if h.size else 0.0))
