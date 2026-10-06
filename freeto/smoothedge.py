"""smoothedge3D: node densities -> fine grid (factor ngrid) -> Heaviside
projection with volume-preserving threshold -> element densities."""
from __future__ import annotations

import numpy as np

__all__ = ["upsample_linear", "smoothedge3D", "window_reduce"]


def _up1(a, axis, f):
    """Exact linear interpolation of integer-spaced samples at spacing 1/f
    along ``axis`` (interp3 'linear' on the node grid)."""
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    out = np.empty(((n - 1) * f + 1,) + a.shape[1:], dtype=a.dtype)
    lo = a[:-1]
    d = a[1:] - lo
    for r in range(f):
        t = r / f
        out[r:(n - 1) * f:f] = lo + t * d if r else lo
    out[-1] = a[-1]
    return np.moveaxis(out, 0, axis)


def upsample_linear(xn, f=4):
    """interp3(nodex,nodey,nodez,xn,fnx,fny,fnz,'linear') for the regular
    query grid 0:1/f:nel (separable exact linear interpolation)."""
    out = xn
    for ax in range(3):
        out = _up1(out, ax, f)
    return np.ascontiguousarray(out)


def window_reduce(X, f, op):
    """For each coarse cell (i,j,k) reduce X over the (f+1)^3 window
    X[f*i:f*i+f+1, f*j:..., f*k:...] (inclusive of both ends) with op in
    {'sum','min','max'}; separable along the three axes."""
    out = X
    for ax in range(3):
        a = np.moveaxis(out, ax, 0)
        n = (a.shape[0] - 1) // f
        body = a[:n * f].reshape((n, f) + a.shape[1:])
        endp = a[f::f]
        if op == "sum":
            r = body.sum(axis=1) + endp
        elif op == "min":
            r = np.minimum(body.min(axis=1), endp)
        else:
            r = np.maximum(body.max(axis=1), endp)
        out = np.moveaxis(r, 0, ax)
    return out


def smoothedge3D(vxPhys, Hn, Hns, nelx, nely, nelz, nele, nnele, beta,
                 ngrid=4):
    """Port of smoothedge3D.m.

    vxPhys : (nele,) full element densities (Fortran order of (nely,nelx,nelz))
    Returns (vxPhys_new (nele,), xg (fine grid, MATLAB layout), ls, top, tol).
    """
    xn = (Hn @ vxPhys) / Hns
    xn = xn.reshape((nely + 1, nelx + 1, nelz + 1), order="F")
    xg = upsample_linear(xn, ngrid)
    npts = ((ngrid * nelx + 1) * (ngrid * nely + 1) * (ngrid * nelz + 1))
    target = vxPhys.sum() / nele
    # Points with xg == 0 give exactly 0.001 (tanh is odd) and points with
    # xg == 1 give exactly 1 (x/x); only the remaining points need tanh.
    flat = xg.ravel()
    m0 = flat == 0.0
    m1 = flat == 1.0
    mid = ~(m0 | m1)
    xm = flat[mid]
    n0 = int(np.count_nonzero(m0))
    n1 = int(np.count_nonzero(m1))
    buf = np.empty_like(xm)
    l1, l2 = 0.0, 1.0
    ls = 0.5
    while (l2 - l1) > 1.0e-5:
        ls = (l1 + l2) / 2.0
        tb = np.tanh(beta * ls)
        den = tb + np.tanh(beta * (1 - ls))
        np.subtract(xm, ls, out=buf)
        buf *= beta
        np.tanh(buf, out=buf)
        buf += tb
        buf /= den
        np.maximum(buf, 0.001, out=buf)
        total = buf.sum() + n0 * 0.001 + n1 * 1.0
        if total / npts - target > 0:
            l1 = ls
        else:
            l2 = ls
    xgnew = np.full(flat.shape, 0.001)
    xgnew[m1] = 1.0
    xgnew[mid] = buf
    xgnew = xgnew.reshape(xg.shape)
    # element conversion: sums / min / max over (ngrid+1)^3 windows
    s = window_reduce(xgnew, ngrid, "sum")
    mn = window_reduce(xgnew, ngrid, "min")
    mx = window_reduce(xgnew, ngrid, "max")
    terr = int(np.count_nonzero((mn > 0.001) & (mx < 1)))
    vx_new = (s / ((ngrid + 1) ** 3)).ravel(order="F")
    tol = terr / nnele
    top = xg - ls
    return vx_new, xg, ls, top, tol
