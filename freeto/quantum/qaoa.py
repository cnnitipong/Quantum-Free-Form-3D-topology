"""Own numpy state-vector QAOA simulator (docs/QUANTUM_DESIGN.md §C.1).

* The cost Hamiltonian is diagonal: the energies of all 2^n bitstrings are
  computed once (:func:`freeto.quantum.exact.spectrum`, little-endian qubit
  order, bit i of the basis index = x_i) and scaled to [0, 1].
* Cost layer: psi <- exp(-i gamma E~) psi (element-wise).
  Mixer: exp(-i beta sum_j X_j), applied qubit by qubit in place on
  ``psi.reshape(2^(n-q-1), 2, 2^q)`` views (no copies of the full vector).
* Parameters (gamma_1..p, beta_1..p); initialisation ``linear_ramp``
  (TQA-like: gamma_k = (k/p) D, beta_k = (1 - k/p) D, D = 0.75, with the
  k -> k - 1/2 midpoint rule) or ``interp`` (optimise p = 1, extend by linear
  interpolation of the angle sequence, Zhou et al. 2020).
* Classical optimiser: scipy COBYLA (``maxiter = 100 p``) on <E~>.
* Output: ``shots`` samples from |psi|^2, the best sample (raw -- the
  ``qaoa`` backend optionally polishes it by greedy descent and reports both),
  ``approx_ratio = (E_max - <E>) / (E_max - E_min)``, ``p_opt`` = total
  probability of the optimal bitstring(s), ``p_opt_shots = 1 - (1 -
  p_opt)^shots``, ``best_shot_ratio`` = (E_max - E_best_shot)/(E_max - E_min),
  and the p = 0 baseline of the uniform superposition (``uniform_ratio``,
  ``uniform_p_opt`` = fraction of optimal bitstrings).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .exact import bits_of, spectrum

__all__ = ["QAOA_MAX_N", "qaoa_state", "apply_mixer", "qaoa_expectation",
           "initial_angles", "interp_angles", "run_qaoa"]

QAOA_MAX_N = 20


def apply_mixer(psi, beta, n):
    """In place: psi <- prod_q exp(-i beta X_q) psi."""
    c = math.cos(beta)
    s = -1j * math.sin(beta)
    for q in range(n):
        v = psi.reshape(-1, 2, 2 ** q)
        a = v[:, 0, :]
        b = v[:, 1, :]
        t = a.copy()
        a *= c
        a += s * b
        b *= c
        b += s * t
    return psi


def qaoa_state(Es, params, n, p):
    """|psi(gamma, beta)> for scaled energies ``Es`` (2^n,)."""
    gam = params[:p]
    bet = params[p:2 * p]
    psi = np.full(2 ** n, 2.0 ** (-n / 2.0), dtype=np.complex128)
    for k in range(p):
        psi *= np.exp(-1j * gam[k] * Es)
        apply_mixer(psi, bet[k], n)
    return psi


def qaoa_expectation(Es, params, n, p):
    psi = qaoa_state(Es, params, n, p)
    pr = psi.real * psi.real + psi.imag * psi.imag
    return float(pr @ Es)


def initial_angles(p, delta=0.75):
    k = np.arange(1, p + 1) - 0.5
    gam = k / p * delta
    bet = (1.0 - k / p) * delta
    return np.concatenate([gam, bet])


def interp_angles(params, p):
    """INTERP: angles for depth p+1 from optimal depth-p angles."""
    gam, bet = params[:p], params[p:2 * p]

    def ext(a):
        out = np.zeros(p + 1)
        for i in range(p + 1):          # i = 0..p  (1-based i+1)
            left = a[i - 1] if i - 1 >= 0 else 0.0
            right = a[i] if i < p else 0.0
            out[i] = (i / p) * left + ((p - i) / p) * right
        return out
    return np.concatenate([ext(gam), ext(bet)])


def _optimise(Es, n, p, x0, maxiter, method):
    from scipy.optimize import minimize
    nfev = [0]

    def f(x):
        nfev[0] += 1
        return qaoa_expectation(Es, x, n, p)
    if method.upper() == "COBYLA":
        res = minimize(f, x0, method="COBYLA",
                       options={"maxiter": int(maxiter), "rhobeg": 0.2})
    else:
        res = minimize(f, x0, method=method, options={"maxiter": int(maxiter)})
    return np.asarray(res.x, dtype=float), float(res.fun), nfev[0]


def run_qaoa(Q, h, const=0.0, p=3, shots=1000, seed=None, init="linear_ramp",
             initial_params=None, maxiter=None, optimizer="COBYLA", E_all=None,
             optimise=True):
    """Simulate QAOA on the QUBO (Q, h, const); returns a dict with
    ``samples, energies, x, energy, info``.  ``E_all`` may pass a precomputed
    spectrum."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = int(np.asarray(h).size)
    if n > QAOA_MAX_N:
        raise ValueError(f"QAOA state-vector simulation is limited to n <= "
                         f"{QAOA_MAX_N} qubits (got {n}); use block_size <= {QAOA_MAX_N}")
    E = spectrum(Q, h, const) if E_all is None else np.asarray(E_all, float)
    Emin, Emax = float(E.min()), float(E.max())
    span = Emax - Emin
    p = int(p)
    maxiter = int(maxiter) if maxiter else 100 * p
    if span <= 0:
        # constant energy: every bitstring optimal
        X = bits_of(rng.integers(0, 2 ** n, size=shots), n)
        return dict(samples=X, energies=np.full(shots, Emin), x=X[0], energy=Emin,
                    info={"p": p, "approx_ratio": 1.0, "p_opt": 1.0, "p_opt_shots": 1.0,
                          "nfev": 0, "angles": [], "qaoa_time": time.perf_counter() - t0})
    Es = (E - Emin) / span
    nfev = 0
    if initial_params is not None and len(initial_params) == 2 * p:
        x0 = np.asarray(initial_params, dtype=float)
    elif init == "interp" and p > 1:
        x = initial_angles(1)
        if optimise:
            x, _, nf = _optimise(Es, n, 1, x, 100, optimizer)
            nfev += nf
        for q in range(1, p):
            x = interp_angles(x, q)
            if optimise and q + 1 < p:
                x, _, nf = _optimise(Es, n, q + 1, x, 100 * (q + 1), optimizer)
                nfev += nf
        x0 = x
    else:
        x0 = initial_angles(p)
    if optimise:
        params, fun, nf = _optimise(Es, n, p, x0, maxiter, optimizer)
        nfev += nf
    else:
        params = x0
    psi = qaoa_state(Es, params, n, p)
    prob = psi.real ** 2 + psi.imag ** 2
    prob /= prob.sum()
    expE = float(prob @ E)
    tol = 1e-9 * max(1.0, abs(Emin), abs(Emax))
    opt_mask = E <= Emin + tol
    p_opt = float(prob[opt_mask].sum())
    cdf = np.cumsum(prob)
    idx = np.searchsorted(cdf, rng.random(int(shots)) * cdf[-1])
    idx = np.minimum(idx, E.size - 1)
    X = bits_of(idx, n)
    En = E[idx]
    b = int(np.argmin(En))
    info = {"p": p, "approx_ratio": (Emax - expE) / span, "p_opt": p_opt,
            "p_opt_shots": 1.0 - (1.0 - p_opt) ** int(shots), "expected_energy": expE,
            "E_min": Emin, "E_max": Emax, "nfev": nfev, "angles": params.tolist(),
            "shots": int(shots), "best_shot_optimal": bool(En[b] <= Emin + tol),
            "best_shot_energy": float(En[b]),
            "best_shot_ratio": float((Emax - En[b]) / span),
            # p = 0 baseline: the uniform superposition (random guessing)
            "uniform_ratio": float((Emax - float(E.mean())) / span),
            "uniform_p_opt": float(np.count_nonzero(opt_mask)) / E.size,
            "qaoa_time": time.perf_counter() - t0}
    return dict(samples=X, energies=En, x=X[b], energy=float(En[b]), info=info)
