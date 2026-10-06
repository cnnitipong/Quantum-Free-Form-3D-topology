"""Exact brute-force QUBO solver (n <= 24) and full energy spectrum.

The low ``L = min(n, 16)`` variables are enumerated as a (2^L, L) 0/1 matrix;
the high variables are looped over (<= 256 iterations for n = 24).
Bitstring index convention (used by the QAOA simulator): little-endian,
``index = sum_i x_i 2^i``.
"""
from __future__ import annotations

import numpy as np

from .qubo import dense, normalize_qubo

__all__ = ["spectrum", "exact_solve", "EXACT_MAX_N", "bits_of"]

EXACT_MAX_N = 24
_LOW = 16


def bits_of(idx, n):
    """(len(idx), n) float 0/1 matrix, little-endian."""
    idx = np.asarray(idx, dtype=np.int64)
    return ((idx[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.float64)


def spectrum(Q, h, const=0.0):
    """Energies of all 2^n bitstrings (little-endian index), float64."""
    Q, h, const = normalize_qubo(Q, h, const)
    Qd = dense(Q)
    n = h.size
    if n > EXACT_MAX_N:
        raise ValueError(f"exact spectrum limited to n <= {EXACT_MAX_N} (got {n})")
    if n == 0:
        return np.array([const])
    L = min(n, _LOW)
    XL = bits_of(np.arange(2 ** L), L)
    QLL = Qd[:L, :L]
    E_low = np.einsum("ij,ij->i", XL @ QLL, XL) + XL @ h[:L]
    nh = n - L
    if nh == 0:
        return E_low + const
    QLH = Qd[:L, L:]
    QHH = Qd[L:, L:]
    XH = bits_of(np.arange(2 ** nh), nh)
    out = np.empty((2 ** nh, 2 ** L))
    for k in range(2 ** nh):
        xh = XH[k]
        lin = 2.0 * (QLH @ xh)
        eh = xh @ QHH @ xh + h[L:] @ xh + const
        np.add(E_low, XL @ lin, out=out[k])
        out[k] += eh
    return out.reshape(-1)


def exact_solve(Q, h, const=0.0, return_spectrum=False):
    """Return (x_best (n,), E_best, info[, spectrum])."""
    E = spectrum(Q, h, const)
    n = int(np.log2(E.size)) if E.size > 1 else 0
    i = int(np.argmin(E))
    x = bits_of([i], n)[0] if n else np.zeros(0)
    Emin = float(E[i])
    info = {"n_optimal": int(np.count_nonzero(E <= Emin + 1e-9 * max(1.0, abs(Emin)))),
            "E_max": float(E.max())}
    if return_spectrum:
        return x, Emin, info, E
    return x, Emin, info
