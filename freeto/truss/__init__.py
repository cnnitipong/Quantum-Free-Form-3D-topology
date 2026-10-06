"""Truss ground-structure topology optimisation with exact enumeration,
continuous OC, sorting/BESO and iterative QUBO updates (docs/QUANTUM_API.md §3)."""
from .ground import TrussProblem, GroundStructure, candidate_bars, grid_nodes
from .fe import TrussFE
from .optimize import (TrussResult, enumerate_exact, oc_continuous, round_sorted,
                       qubo_rounding, qubo_iterative, solve_truss, TRUSS_METHODS)
from .benchmarks import BENCHMARKS, ALIASES, get_benchmark, list_benchmarks

__all__ = ["TrussProblem", "GroundStructure", "candidate_bars", "grid_nodes", "TrussFE",
           "TrussResult", "enumerate_exact", "oc_continuous", "round_sorted",
           "qubo_rounding", "qubo_iterative", "solve_truss", "TRUSS_METHODS",
           "BENCHMARKS", "ALIASES", "get_benchmark", "list_benchmarks"]
