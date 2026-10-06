"""QUBO / quantum design updates for FreeTO-Python (docs/QUANTUM_API.md).

Everything that runs locally is a classical simulation of a quantum
algorithm (state-vector QAOA) or a quantum-inspired heuristic (simulated
annealing, tabu); D-Wave and IBM backends are optional extras
(``requirements-quantum.txt``) that need an account token.
"""
from .options import QUBOOptions
from .backends import (QUBOResult, QuantumBackendUnavailable, solve_qubo,
                       available_backends, check_backend, BACKENDS)
from .qubo import normalize_qubo, energy, to_ising, from_ising, to_bqm

__all__ = ["QUBOOptions", "QUBOResult", "QuantumBackendUnavailable", "solve_qubo",
           "available_backends", "check_backend", "BACKENDS", "normalize_qubo",
           "energy", "to_ising", "from_ising", "to_bqm"]
