"""QUBO solver backend registry, :func:`solve_qubo`, and the optional D-Wave /
Qiskit wrappers (all imports lazy; tokens from the environment).

Timing convention (docs/QUANTUM_DESIGN.md §C.2): ``timing["wall"]`` is the
Python call to return (includes network, embedding, queueing),
``timing["solver"]`` the time inside the sampler as far as it reports it --
for the QAOA backends (``qaoa``, ``qiskit_aer``, ``ibm``) this is the whole
variational loop *including* the classical angle optimisation on the
state-vector simulator (for ``qaoa`` ~100 % of it is classical simulation),
excluding the optional greedy polish (``info["polish_time"]``); all local
times are CPU seconds on this machine,
``timing["qpu_access"]`` / ``["qpu_sampling"]`` the QPU times reported by the
cloud service (None for local backends), ``["hybrid_run_time"]`` the Leap
hybrid run time.
"""
from __future__ import annotations

import importlib.util
import os
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .qubo import dense, energy as qubo_energy, normalize_qubo, to_bqm_dict

__all__ = ["QUBOResult", "QuantumBackendUnavailable", "solve_qubo",
           "available_backends", "check_backend", "BACKENDS", "backend_capacity",
           "default_block_size", "is_batch_backend"]


class QuantumBackendUnavailable(RuntimeError):
    """An optional backend was requested but its SDK / token is missing."""


@dataclass
class QUBOResult:
    x: np.ndarray
    energy: float
    samples: np.ndarray
    energies: np.ndarray
    timing: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:  # noqa: BLE001 -- ImportError, ValueError, broken packages
        return False


# name -> (description, kind, max_n, required modules, token env var)
BACKENDS = {
    "exact": ("Exact brute force (all 2^n states, n <= 24)", "classical", 24, (), None),
    "sa": ("Simulated annealing (own vectorised numpy)", "classical", None, (), None),
    "tabu": ("Tabu search, parallel restarts (own numpy)", "classical", None, (), None),
    "greedy": ("Steepest-descent local search (own numpy)", "classical", None, (), None),
    "qaoa": ("QAOA, own numpy state-vector simulator (n <= 20)", "simulator", 20, (), None),
    "dwave_sa": ("D-Wave dwave-samplers SimulatedAnnealingSampler (C++, local)",
                 "classical", None, ("dimod", "dwave.samplers"), None),
    "dwave_tabu": ("D-Wave dwave-samplers TabuSampler (C++, local)", "classical", None,
                   ("dimod", "dwave.samplers"), None),
    "dwave_qpu": ("D-Wave QPU (DWaveSampler + EmbeddingComposite, Leap cloud)", "qpu",
                  150, ("dimod", "dwave.system"), "DWAVE_API_TOKEN"),
    "dwave_hybrid": ("D-Wave Leap hybrid BQM solver (cloud)", "hybrid", 1_000_000,
                     ("dimod", "dwave.system"), "DWAVE_API_TOKEN"),
    "qiskit_aer": ("QAOA circuit on qiskit-aer AerSimulator (shot noise; angles from "
                   "the own simulator)", "simulator", 20, ("qiskit", "qiskit_aer"), None),
    "ibm": ("QAOA circuit on IBM Quantum hardware (qiskit-ibm-runtime SamplerV2)",
            "qpu", 20, ("qiskit", "qiskit_ibm_runtime"), "QISKIT_IBM_TOKEN"),
}

_BLOCK = {"qaoa": 14, "qiskit_aer": 14, "ibm": 14, "exact": 20, "dwave_qpu": 80}
_BATCH = {"sa", "exact", "greedy", "tabu"}


def backend_capacity(name):
    return BACKENDS[name][2] if name in BACKENDS else None


def default_block_size(name):
    return _BLOCK.get(name)


def is_batch_backend(name):
    """Backends that can solve several linear terms h (K, n) in one call."""
    return name in _BATCH


def _status(name):
    desc, kind, max_n, mods, tok = BACKENDS[name]
    missing = [m for m in mods if not _has(m)]
    reason = ""
    if missing:
        reason = (f"not installed: {', '.join(missing)} "
                  f"(pip install -r requirements-quantum.txt)")
    token_set = None
    if tok:
        token_set = bool(os.environ.get(tok))
        if not token_set and not reason:
            reason = f"environment variable {tok} is not set"
        elif not token_set:
            reason += f"; environment variable {tok} is not set"
    return {"name": name, "available": reason == "", "description": desc,
            "needs_token": tok is not None, "token_env": tok, "token_set": token_set,
            "kind": kind, "max_n": max_n, "reason": reason}


def available_backends():
    """Status of every backend (never raises)."""
    out = []
    for name in BACKENDS:
        try:
            out.append(_status(name))
        except Exception as e:  # noqa: BLE001
            out.append({"name": name, "available": False, "description": BACKENDS[name][0],
                        "needs_token": BACKENDS[name][4] is not None,
                        "token_env": BACKENDS[name][4], "token_set": None,
                        "kind": BACKENDS[name][1], "max_n": BACKENDS[name][2],
                        "reason": f"{type(e).__name__}: {e}"})
    return out


def check_backend(name):
    """Raise ValueError (unknown) / QuantumBackendUnavailable (missing SDK or token)."""
    name = str(name).lower()
    if name == "auto":
        return True
    if name not in BACKENDS:
        raise ValueError(f"unknown QUBO backend {name!r} (known: auto, {', '.join(BACKENDS)})")
    st = _status(name)
    if not st["available"]:
        raise QuantumBackendUnavailable(f"QUBO backend '{name}' is unavailable: {st['reason']}")
    return True


# ---------------------------------------------------------------------------
def solve_qubo(Q, h, const=0.0, *, backend="auto", num_reads=None, seed=None,
               initial_state=None, time_limit=None, normalized=False, **opts):
    """Minimise x^T Q x + h.x + const over x in {0,1}^n (see module docs)."""
    t0 = time.perf_counter()
    if not normalized:
        Q, h, const = normalize_qubo(Q, h, const)
    h = np.asarray(h, dtype=np.float64)
    n = h.size
    name = str(backend).lower()
    if name == "auto":
        name = "exact" if n <= 20 else "sa"
    if name not in BACKENDS:
        raise ValueError(f"unknown QUBO backend {name!r}")
    if n == 0:
        return QUBOResult(np.zeros(0), float(const), np.zeros((1, 0)), np.array([const]),
                          {"wall": 0.0, "solver": 0.0, "qpu_access": None,
                           "qpu_sampling": None, "hybrid_run_time": None},
                          {"backend": name, "n": 0})
    fn = _DISPATCH[name]
    x, E, S, Es, timing, info = fn(Q, h, const, num_reads=num_reads, seed=seed,
                                   initial_state=initial_state, time_limit=time_limit,
                                   **opts)
    timing = {"wall": time.perf_counter() - t0, "solver": timing.get("solver"),
              "qpu_access": timing.get("qpu_access"),
              "qpu_sampling": timing.get("qpu_sampling"),
              "hybrid_run_time": timing.get("hybrid_run_time")}
    if timing["solver"] is None:
        timing["solver"] = timing["wall"]
    info = dict(info)
    info.setdefault("backend", name)
    info["n"] = n
    return QUBOResult(np.asarray(x, dtype=np.float64), float(E), S, Es, timing, info)


def _best(S, E):
    i = int(np.argmin(E))
    return S[i], float(E[i])


def _exact(Q, h, const, **kw):
    from .exact import exact_solve, EXACT_MAX_N
    n = h.size
    if n > EXACT_MAX_N:
        raise ValueError(f"backend 'exact' is limited to n <= {EXACT_MAX_N} (got {n}); "
                         "use block_size <= 24")
    t = time.perf_counter()
    x, E, info = exact_solve(Q, h, const)
    return x, E, x[None, :], np.array([E]), {"solver": time.perf_counter() - t}, info


def _sa(Q, h, const, num_reads=None, seed=None, initial_state=None, sweeps=None, **kw):
    from .anneal import simulated_annealing
    t = time.perf_counter()
    r = simulated_annealing(Q, h, const, num_reads=num_reads or 32, sweeps=sweeps,
                            seed=seed, initial_state=initial_state)
    x, E = _best(r["samples"], r["energies"])
    return x, E, r["samples"], r["energies"], {"solver": time.perf_counter() - t}, r["info"]


def _tabu(Q, h, const, num_reads=None, seed=None, initial_state=None, steps=None,
          tenure=None, **kw):
    from .anneal import tabu_search
    t = time.perf_counter()
    r = tabu_search(Q, h, const, num_reads=num_reads or 8, steps=steps, tenure=tenure,
                    seed=seed, initial_state=initial_state)
    x, E = _best(r["samples"], r["energies"])
    return x, E, r["samples"], r["energies"], {"solver": time.perf_counter() - t}, r["info"]


def _greedy(Q, h, const, num_reads=None, seed=None, initial_state=None, **kw):
    from .anneal import greedy_descent
    t = time.perf_counter()
    r = greedy_descent(Q, h, const, initial_state=initial_state,
                       num_reads=num_reads or (1 if initial_state is not None else 16),
                       seed=seed)
    x, E = _best(r["samples"], r["energies"])
    return x, E, r["samples"], r["energies"], {"solver": time.perf_counter() - t}, r["info"]


def _qaoa(Q, h, const, num_reads=None, seed=None, initial_state=None, p=3, shots=None,
          init="linear_ramp", initial_params=None, maxiter=None, optimizer="COBYLA",
          polish=True, **kw):
    """Own state-vector QAOA.  ``polish=True`` (default) replaces the best shot
    by a greedy steepest descent started from it (classical post-processing);
    ``polish=False`` returns the raw best shot.  Both are always reported:
    ``info["best_shot_energy"]`` / ``["best_shot_x"]`` (raw) and
    ``info["polished_energy"]``, ``info["polished"]`` (True if the descent
    improved the shot).  Timing: ``solver`` = the whole variational loop
    (spectrum, classical COBYLA angle optimisation on the simulator, sampling);
    the greedy polish is excluded and reported as ``info["polish_time"]``."""
    from .qaoa import run_qaoa
    from .anneal import greedy_descent
    t = time.perf_counter()
    shots = shots or num_reads or 1000
    r = run_qaoa(Q, h, const, p=p, shots=shots, seed=seed, init=init,
                 initial_params=initial_params, maxiter=maxiter, optimizer=optimizer)
    info = dict(r["info"])
    x, E = r["x"], r["energy"]
    info["best_shot_energy"] = float(E)
    info["best_shot_x"] = np.asarray(x, dtype=float).tolist()
    ts = time.perf_counter() - t
    tp = time.perf_counter()
    g = greedy_descent(Q, h, const, initial_state=x, num_reads=1)
    Ep = float(g["energies"][0])
    improved = Ep < E - 1e-12 * max(1.0, abs(E))
    info["polished_energy"] = Ep if improved else float(E)
    info["polished"] = bool(improved)
    info["polish_time"] = time.perf_counter() - tp
    info["polish_applied"] = bool(polish)
    if polish and improved:
        x, E = g["samples"][0], Ep
    return x, E, r["samples"], r["energies"], {"solver": ts}, info


# ---- D-Wave ------------------------------------------------------------------
def _sampleset_to_arrays(ss, n):
    import dimod  # noqa: F401
    rec = ss.record
    var_order = list(ss.variables)
    S = np.zeros((rec.sample.shape[0], n))
    cols = np.array([int(v) for v in var_order])
    S[:, cols] = rec.sample
    reps = np.asarray(rec.num_occurrences)
    S = np.repeat(S, reps, axis=0)
    E = np.repeat(np.asarray(rec.energy, dtype=float), reps)
    return S, E


def _dwave_local(Q, h, const, kind, num_reads=None, seed=None, **kw):
    try:
        import dimod
        from dwave.samplers import SimulatedAnnealingSampler, TabuSampler
    except Exception as e:  # noqa: BLE001 -- missing or broken optional SDK
        raise QuantumBackendUnavailable(
            "dimod / dwave-samplers are not installed or fail to import "
            f"(pip install dimod dwave-samplers): {type(e).__name__}: {e}") from e
    lin, quad, off = to_bqm_dict(Q, h, const)
    bqm = dimod.BinaryQuadraticModel(lin, quad, off, dimod.BINARY)
    t = time.perf_counter()
    if kind == "sa":
        ss = SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads or 32, seed=seed)
    else:
        kws = {"num_reads": num_reads or 8}
        if seed is not None:
            kws["seed"] = int(seed)
        ss = TabuSampler().sample(bqm, **kws)
    ts = time.perf_counter() - t
    S, E = _sampleset_to_arrays(ss, h.size)
    x, Eb = _best(S, E)
    return x, Eb, S, E, {"solver": ts}, {"backend": f"dwave_{kind}"}


def _dwave_token():
    tok = os.environ.get("DWAVE_API_TOKEN")
    if not tok:
        raise QuantumBackendUnavailable(
            "DWAVE_API_TOKEN is not set: create a Leap account (cloud.dwavesys.com/leap), "
            "then set the environment variable DWAVE_API_TOKEN to your API token.")
    return tok


def _dwave_qpu(Q, h, const, num_reads=None, seed=None, annealing_time=20.0,
               chain_strength=None, **kw):
    try:
        import dimod
        from dwave.system import DWaveSampler, EmbeddingComposite
    except Exception as e:  # noqa: BLE001 -- missing or broken optional SDK
        raise QuantumBackendUnavailable(
            "dimod / dwave-system are not installed or fail to import "
            f"(pip install dwave-ocean-sdk): {type(e).__name__}: {e}") from e
    tok = _dwave_token()
    lin, quad, off = to_bqm_dict(Q, h, const)
    bqm = dimod.BinaryQuadraticModel(lin, quad, off, dimod.BINARY)
    sampler = EmbeddingComposite(DWaveSampler(token=tok))
    kws = {"num_reads": int(num_reads or 500), "annealing_time": float(annealing_time),
           "return_embedding": True}
    if chain_strength is not None:
        kws["chain_strength"] = chain_strength
    t = time.perf_counter()
    ss = sampler.sample(bqm, **kws)
    ss.resolve()
    ts = time.perf_counter() - t
    S, E = _sampleset_to_arrays(ss, h.size)
    tim = ss.info.get("timing", {}) or {}
    info = {"backend": "dwave_qpu", "timing_raw": {k: float(v) for k, v in tim.items()
                                                   if isinstance(v, (int, float))}}
    emb = ss.info.get("embedding_context", {}).get("embedding")
    if emb:
        lens = [len(c) for c in emb.values()]
        info.update(n_physical=int(sum(lens)), max_chain=int(max(lens)),
                    mean_chain=float(np.mean(lens)))
    if "chain_break_fraction" in ss.record.dtype.names:
        info["chain_break_fraction"] = float(np.mean(ss.record.chain_break_fraction))
    x, Eb = _best(S, E)
    return x, Eb, S, E, {"solver": ts,
                         "qpu_access": tim.get("qpu_access_time", 0) * 1e-6 if tim else None,
                         "qpu_sampling": tim.get("qpu_sampling_time", 0) * 1e-6 if tim else None
                         }, info


def _dwave_hybrid(Q, h, const, time_limit=None, **kw):
    try:
        import dimod
        from dwave.system import LeapHybridSampler
    except Exception as e:  # noqa: BLE001 -- missing or broken optional SDK
        raise QuantumBackendUnavailable(
            "dimod / dwave-system are not installed or fail to import "
            f"(pip install dwave-ocean-sdk): {type(e).__name__}: {e}") from e
    tok = _dwave_token()
    lin, quad, off = to_bqm_dict(Q, h, const)
    bqm = dimod.BinaryQuadraticModel(lin, quad, off, dimod.BINARY)
    sampler = LeapHybridSampler(token=tok)
    kws = {}
    if time_limit is not None:
        kws["time_limit"] = max(3.0, float(time_limit))
    t = time.perf_counter()
    ss = sampler.sample(bqm, **kws)
    ss.resolve()
    ts = time.perf_counter() - t
    S, E = _sampleset_to_arrays(ss, h.size)
    inf = ss.info or {}
    x, Eb = _best(S, E)
    return x, Eb, S, E, {"solver": ts,
                         "qpu_access": inf.get("qpu_access_time", 0) * 1e-6,
                         "hybrid_run_time": inf.get("run_time", 0) * 1e-6}, \
        {"backend": "dwave_hybrid", "problem_id": inf.get("problem_id")}


# ---- Qiskit -----------------------------------------------------------------
def qaoa_circuit(Q, h, const, params, p, measure=True):
    """QAOA circuit for the QUBO with the same scaled-energy convention as the
    own simulator (angles are interchangeable).  Qubit i <-> x_i."""
    from qiskit import QuantumCircuit
    from .exact import spectrum
    from .qubo import to_ising
    E = spectrum(Q, h, const)
    Emin, Emax = float(E.min()), float(E.max())
    span = (Emax - Emin) or 1.0
    J, b, _ = to_ising(Q, h, const)
    J, b = J / span, b / span
    n = h.size
    qc = QuantumCircuit(n)
    qc.h(range(n))
    gam, bet = params[:p], params[p:2 * p]
    for k in range(p):
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0:
                    # exp(-i gamma 2 J_ij Z_i Z_j) = RZZ(4 gamma J_ij)
                    qc.rzz(4.0 * gam[k] * J[i, j], i, j)
        for i in range(n):
            if b[i] != 0:
                qc.rz(2.0 * gam[k] * b[i], i)
        for i in range(n):
            qc.rx(2.0 * bet[k], i)
    if measure:
        qc.measure_all()
    return qc


def _counts_to_samples(counts, n):
    xs, reps = [], []
    for bitstr, c in counts.items():
        s = bitstr.replace(" ", "")
        xs.append([int(s[-1 - i]) for i in range(n)])     # little-endian
        reps.append(int(c))
    X = np.repeat(np.array(xs, dtype=float), reps, axis=0)
    return X


def _qiskit(Q, h, const, which, num_reads=None, seed=None, p=3, shots=None,
            init="linear_ramp", initial_params=None, maxiter=None, **kw):
    from .qaoa import run_qaoa
    try:
        import qiskit  # noqa: F401
    except Exception as e:  # noqa: BLE001 -- missing or broken optional SDK
        raise QuantumBackendUnavailable("qiskit is not installed (pip install qiskit)") from e
    shots = int(shots or num_reads or 1000)
    # classical angle optimisation with the own simulator (n <= 20); counted
    # in timing["solver"] like for the 'qaoa' backend (whole variational loop)
    t = time.perf_counter()
    sim = run_qaoa(Q, h, const, p=p, shots=16, seed=seed, init=init,
                   initial_params=initial_params, maxiter=maxiter)
    params = np.asarray(sim["info"]["angles"])
    qc = qaoa_circuit(Q, h, const, params, p)
    info = {"backend": which, "p": p, "angles": params.tolist(),
            "sim_approx_ratio": sim["info"]["approx_ratio"], "sim_p_opt": sim["info"]["p_opt"],
            "angle_opt_time": time.perf_counter() - t}
    qpu = None
    if which == "qiskit_aer":
        try:
            from qiskit_aer import AerSimulator
        except Exception as e:  # noqa: BLE001 -- missing or broken optional SDK
            raise QuantumBackendUnavailable("qiskit-aer is not installed") from e
        from qiskit import transpile
        be = AerSimulator(seed_simulator=seed)
        job = be.run(transpile(qc, be), shots=shots)
        counts = job.result().get_counts()
    else:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        except Exception as e:  # noqa: BLE001 -- missing or broken optional SDK
            raise QuantumBackendUnavailable("qiskit-ibm-runtime is not installed") from e
        tok = os.environ.get("QISKIT_IBM_TOKEN")
        if not tok:
            raise QuantumBackendUnavailable(
                "QISKIT_IBM_TOKEN is not set (IBM Quantum API key; optionally "
                "QISKIT_IBM_INSTANCE and QISKIT_IBM_BACKEND)")
        kws = {"channel": os.environ.get("QISKIT_IBM_CHANNEL", "ibm_quantum_platform"),
               "token": tok}
        if os.environ.get("QISKIT_IBM_INSTANCE"):
            kws["instance"] = os.environ["QISKIT_IBM_INSTANCE"]
        service = QiskitRuntimeService(**kws)
        bname = os.environ.get("QISKIT_IBM_BACKEND")
        be = service.backend(bname) if bname else service.least_busy(
            operational=True, simulator=False, min_num_qubits=h.size)
        pm = generate_preset_pass_manager(optimization_level=1, backend=be)
        isa = pm.run(qc)
        job = SamplerV2(mode=be).run([isa], shots=shots)
        res = job.result()
        counts = res[0].data.meas.get_counts()
        try:
            qpu = float(job.metrics()["usage"]["quantum_seconds"])
        except Exception:  # noqa: BLE001
            qpu = None
        info["ibm_backend"] = be.name
        info["job_id"] = job.job_id()
    ts = time.perf_counter() - t
    X = _counts_to_samples(counts, h.size)
    E = qubo_energy(Q, h, const, X)
    x, Eb = _best(X, E)
    E_all_min = sim["info"]["E_min"]
    info["best_shot_optimal"] = bool(Eb <= E_all_min + 1e-9 * max(1, abs(E_all_min)))
    info["p_opt_empirical"] = float(np.mean(E <= E_all_min + 1e-9 * max(1, abs(E_all_min))))
    span = sim["info"]["E_max"] - E_all_min
    info["approx_ratio"] = float((sim["info"]["E_max"] - E.mean()) / span) if span > 0 else 1.0
    return x, Eb, X, E, {"solver": ts, "qpu_access": qpu}, info


_DISPATCH = {
    "exact": _exact, "sa": _sa, "tabu": _tabu, "greedy": _greedy, "qaoa": _qaoa,
    "dwave_sa": lambda Q, h, c, **k: _dwave_local(Q, h, c, "sa", **k),
    "dwave_tabu": lambda Q, h, c, **k: _dwave_local(Q, h, c, "tabu", **k),
    "dwave_qpu": _dwave_qpu, "dwave_hybrid": _dwave_hybrid,
    "qiskit_aer": lambda Q, h, c, **k: _qiskit(Q, h, c, "qiskit_aer", **k),
    "ibm": lambda Q, h, c, **k: _qiskit(Q, h, c, "ibm", **k),
}
