"""Design update schemes.

* :func:`oc_update` -- the optimality-criteria update of SIMP.m / SEMDOT.m
  (bisection on the Lagrange multiplier, move 0.1, based on ``vxPhys``).
* :class:`MMA` -- an independent implementation of Svanberg's Method of Moving
  Asymptotes (K. Svanberg, "The method of moving asymptotes - a new method
  for structural optimization", IJNME 24, 1987; asymptote update and
  subproblem as in "MMA and GCMMA - two methods for nonlinear optimization",
  2007), with the subproblem solved by a primal-dual interior-point Newton
  method on the KKT conditions.  Written from the published equations.
"""
from __future__ import annotations

import numpy as np

__all__ = ["oc_update", "MMA"]


def oc_update(vxPhys, dc, dv, vol, nnele, move=0.1):
    """OC update exactly as in SIMP.m / SEMDOT.m (lines 79-84)."""
    l1, l2 = 0.0, 1e9
    target = vol * nnele
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = -dc / dv
    lo = vxPhys - move
    hi = vxPhys + move
    vxnew = vxPhys
    while l1 + l2 > 0 and (l2 - l1) / (l1 + l2) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        with np.errstate(invalid="ignore"):
            cand = vxPhys * np.sqrt(ratio / lmid)
        vxnew = np.maximum(0.0, np.maximum(lo, np.minimum(1.0, np.minimum(hi, cand))))
        if vxnew.sum() > target:
            l1 = lmid
        else:
            l2 = lmid
    return vxnew


class MMA:
    """Method of Moving Asymptotes for

        min f0(x) + a0 z + sum(c_i y_i + 0.5 d_i y_i^2)
        s.t. f_i(x) - a_i z - y_i <= 0,  xmin <= x <= xmax,  y, z >= 0.

    Usage mirrors the classic ``mmasub`` call::

        mma = MMA(n, m, xmin, xmax)
        xnew = mma.update(it, xval, f0val, df0dx, fval, dfdx)
    """

    def __init__(self, n, m=1, xmin=0.0, xmax=1.0, a0=1.0, a=None, c=None,
                 d=None, move=0.5, asyinit=0.5, asyincr=1.2, asydecr=0.7,
                 albefa=0.1, raa0=1e-5, epsimin=1e-7):
        self.n = n
        self.m = m
        self.xmin = np.broadcast_to(np.asarray(xmin, float), (n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(xmax, float), (n,)).copy()
        self.a0 = float(a0)
        self.a = np.zeros(m) if a is None else np.asarray(a, float).reshape(m)
        self.c = (1e4 * np.ones(m)) if c is None else np.asarray(c, float).reshape(m)
        self.d = np.zeros(m) if d is None else np.asarray(d, float).reshape(m)
        self.move = move
        self.asyinit = asyinit
        self.asyincr = asyincr
        self.asydecr = asydecr
        self.albefa = albefa
        self.raa0 = raa0
        self.epsimin = epsimin
        self.low = np.ones(n)
        self.upp = np.ones(n)
        self.xold1 = None
        self.xold2 = None

    def update(self, it, xval, f0val, df0dx, fval, dfdx):
        """One MMA iteration (``it`` is 1-based as in mmasub)."""
        n, m = self.n, self.m
        xval = np.asarray(xval, float).ravel()
        df0dx = np.asarray(df0dx, float).ravel()
        fval = np.asarray(fval, float).reshape(m)
        dfdx = np.asarray(dfdx, float).reshape(m, n)
        if self.xold1 is None:
            self.xold1 = xval.copy()
            self.xold2 = xval.copy()
        xmin, xmax = self.xmin, self.xmax
        xmami = xmax - xmin
        # asymptotes
        if it <= 2:
            low = xval - self.asyinit * xmami
            upp = xval + self.asyinit * xmami
        else:
            zzz = (xval - self.xold1) * (self.xold1 - self.xold2)
            factor = np.ones(n)
            factor[zzz > 0] = self.asyincr
            factor[zzz < 0] = self.asydecr
            low = xval - factor * (self.xold1 - self.low)
            upp = xval + factor * (self.upp - self.xold1)
            low = np.clip(low, xval - 10 * xmami, xval - 0.01 * xmami)
            upp = np.clip(upp, xval + 0.01 * xmami, xval + 10 * xmami)
        # move limits
        alfa = np.maximum(np.maximum(low + self.albefa * (xval - low),
                                     xval - self.move * xmami), xmin)
        beta = np.minimum(np.minimum(upp - self.albefa * (upp - xval),
                                     xval + self.move * xmami), xmax)
        # approximation coefficients
        xmamiinv = 1.0 / np.maximum(xmami, 1e-5)
        ux1 = upp - xval
        xl1 = xval - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        p0 = np.maximum(df0dx, 0.0)
        q0 = np.maximum(-df0dx, 0.0)
        pq0 = 0.001 * (p0 + q0) + self.raa0 * xmamiinv
        p0 = (p0 + pq0) * ux2
        q0 = (q0 + pq0) * xl2
        P = np.maximum(dfdx, 0.0)
        Q = np.maximum(-dfdx, 0.0)
        PQ = 0.001 * (P + Q) + self.raa0 * xmamiinv[None, :]
        P = (P + PQ) * ux2[None, :]
        Q = (Q + PQ) * xl2[None, :]
        b = P @ (1.0 / ux1) + Q @ (1.0 / xl1) - fval
        xnew = self._subsolve(low, upp, alfa, beta, p0, q0, P, Q, b)
        self.xold2 = self.xold1
        self.xold1 = xval.copy()
        self.low = low
        self.upp = upp
        return xnew

    # ------------------------------------------------------------------
    def _subsolve(self, low, upp, alfa, beta, p0, q0, P, Q, b):
        """Primal-dual interior point method for the MMA subproblem."""
        m = self.m
        a0, a, c, d = self.a0, self.a, self.c, self.d
        epsi = 1.0
        x = 0.5 * (alfa + beta)
        y = np.ones(m)
        z = 1.0
        lam = np.ones(m)
        xsi = np.maximum(1.0 / (x - alfa), 1.0)
        eta = np.maximum(1.0 / (beta - x), 1.0)
        mu = np.maximum(np.ones(m), 0.5 * c)
        zet = 1.0
        s = np.ones(m)

        def residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi):
            ux1 = upp - x
            xl1 = x - low
            plam = p0 + P.T @ lam
            qlam = q0 + Q.T @ lam
            gvec = P @ (1.0 / ux1) + Q @ (1.0 / xl1)
            dpsidx = plam / (ux1 * ux1) - qlam / (xl1 * xl1)
            r = [dpsidx - xsi + eta,
                 c + d * y - mu - lam,
                 np.atleast_1d(a0 - zet - a @ lam),
                 gvec - a * z - y + s - b,
                 xsi * (x - alfa) - epsi,
                 eta * (beta - x) - epsi,
                 mu * y - epsi,
                 np.atleast_1d(zet * z - epsi),
                 lam * s - epsi]
            r = np.concatenate(r)
            return np.sqrt(r @ r), np.abs(r).max()

        while epsi > self.epsimin:
            resnorm, resmax = residual(x, y, z, lam, xsi, eta, mu, zet, s,
                                       epsi)
            ittt = 0
            while resmax > 0.9 * epsi and ittt < 200:
                ittt += 1
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                ux3 = ux1 * ux2
                xl3 = xl1 * xl2
                uxinv1 = 1.0 / ux1
                xlinv1 = 1.0 / xl1
                uxinv2 = 1.0 / ux2
                xlinv2 = 1.0 / xl2
                plam = p0 + P.T @ lam
                qlam = q0 + Q.T @ lam
                gvec = P @ uxinv1 + Q @ xlinv1
                GG = P * uxinv2[None, :] - Q * xlinv2[None, :]
                dpsidx = plam / ux2 - qlam / xl2
                delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x)
                dely = c + d * y - lam - epsi / y
                delz = a0 - a @ lam - epsi / z
                dellam = gvec - a * z - y - b + epsi / lam
                diagx = 2.0 * (plam / ux3 + qlam / xl3) + xsi / (x - alfa) \
                    + eta / (beta - x)
                diagxinv = 1.0 / diagx
                diagy = d + mu / y
                diagyinv = 1.0 / diagy
                diaglam = s / lam
                diaglamyi = diaglam + diagyinv
                # reduced system in (dlam, dz)   (m < n case)
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                Alam = np.diag(diaglamyi) + (GG * diagxinv[None, :]) @ GG.T
                AA = np.zeros((m + 1, m + 1))
                AA[:m, :m] = Alam
                AA[:m, m] = a
                AA[m, :m] = a
                AA[m, m] = -zet / z
                bb = np.concatenate([blam, [delz]])
                sol = np.linalg.solve(AA, bb)
                dlam = sol[:m]
                dz = sol[m]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
                dy = -dely / diagy + dlam / diagy
                dxsi = -xsi + epsi / (x - alfa) - (xsi * dx) / (x - alfa)
                deta = -eta + epsi / (beta - x) + (eta * dx) / (beta - x)
                dmu = -mu + epsi / y - (mu * dy) / y
                dzet = -zet + epsi / z - zet * dz / z
                ds = -s + epsi / lam - (s * dlam) / lam
                xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
                dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet],
                                      ds])
                stmxx = np.max(-1.01 * dxx / xx)
                stmalfa = np.max(-1.01 * dx / (x - alfa))
                stmbeta = np.max(1.01 * dx / (beta - x))
                steg = 1.0 / max(stmalfa, stmbeta, stmxx, 1.0)
                old = (x, y, z, lam, xsi, eta, mu, zet, s)
                itto = 0
                resinew = 2 * resnorm
                while resinew > resnorm and itto < 50:
                    itto += 1
                    x = old[0] + steg * dx
                    y = old[1] + steg * dy
                    z = old[2] + steg * dz
                    lam = old[3] + steg * dlam
                    xsi = old[4] + steg * dxsi
                    eta = old[5] + steg * deta
                    mu = old[6] + steg * dmu
                    zet = old[7] + steg * dzet
                    s = old[8] + steg * ds
                    resinew, resmax = residual(x, y, z, lam, xsi, eta, mu,
                                               zet, s, epsi)
                    steg = steg / 2
                resnorm = resinew
            epsi = 0.1 * epsi
        return x
