"""Truss benchmark problems T1-T5 (docs/QUANTUM_DESIGN.md §B).

Pinned values (``c_exact``) were computed once with
:func:`freeto.truss.optimize.enumerate_exact` and are regression-tested in
tests/test_truss.py; ``c_best_known`` for the two large problems (no exact
reference) is the best value found by the study's multi-start runs.
"""
from __future__ import annotations

import numpy as np

from .ground import TrussProblem

__all__ = ["BENCHMARKS", "get_benchmark", "list_benchmarks", "ALIASES"]


def ten_bar():
    """Classic ten-bar truss (Venkayya): 360 in bays, E = 1e4 ksi,
    A_full = 10 in^2, 100 kip down at nodes 2 and 4, nodes 5 and 6 pinned.
    Units: in, kip, ksi -> compliance in kip*in."""
    nodes = np.array([[720, 360], [720, 0], [360, 360], [360, 0], [0, 360], [0, 0]], float)
    bars = np.array([(5, 3), (3, 1), (6, 4), (4, 2), (3, 4), (1, 2), (5, 4), (6, 3),
                     (3, 2), (4, 1)]) - 1
    supports = [(4, 0), (4, 1), (5, 0), (5, 1)]
    loads = [(1, 0.0, -100.0), (3, 0.0, -100.0)]
    return TrussProblem(nodes, bars, supports, loads, E=1e4, A_full=10.0,
                        vmax_fraction=0.65, id="ten_bar", alias="T1",
                        title="Ten-bar truss (classic)",
                        description="6 nodes, 10 bars, two 100-kip loads, 65 % volume "
                                    "(the lightest stable binary sub-truss needs 58.6 %; "
                                    "at 50 % none exists); 1,024 designs, exact optimum "
                                    "by enumeration.",
                        c_exact=C_EXACT["ten_bar"])


def _grid_cantilever(nx, ny, vfrac, id_, alias, title, drop_fixed=True, lmax=None,
                     loads=None):
    p = TrussProblem.grid2d(nx, ny, 1.0, 1.0, lmax=lmax, drop_fixed=drop_fixed,
                            loads=[], vmax_fraction=vfrac, id=id_, alias=alias, title=title)
    if loads is None:
        tip = int(np.argmax(p.nodes[:, 0] + 0.01 * (p.nodes[:, 1] == 0)))
        loads = [(tip, 0.0, -1.0)]
    p.loads = [[tuple(r) for r in loads]]
    return p


def gs_3x2():
    p = _grid_cantilever(3, 2, 0.5, "gs_3x2", "T2s", "3x2 ground structure (12 bars)")
    p.description = ("3x2 node grid, 1 m spacing, 12 bars after overlap filtering and "
                     "dropping the bar between the two fixed nodes, left column fixed, unit "
                     "load down at the bottom-right node, 50 % volume; the whole QUBO fits "
                     "the QAOA simulator (12 qubits).")
    p.c_exact = C_EXACT["gs_3x2"]
    return p


def gs_4x2():
    p = _grid_cantilever(4, 2, 0.4, "gs_4x2", "T2", "4x2 ground structure (21 bars)")
    p.description = ("4x2 node grid, 21 bars after overlap filtering and dropping the bar "
                     "between the two fixed nodes, left column fixed, unit load down at the "
                     "bottom-right node, 40 % volume; 2.1 M designs (batched exact "
                     "enumeration).")
    p.c_exact = C_EXACT["gs_4x2"]
    return p


def tower3d():
    base = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
    mid = [(x, y, 1.0) for x, y, _ in base]
    nodes = np.array(base + mid + [(0.5, 0.5, 2.0)], float)
    supports = [(n, k) for n in range(4) for k in range(3)]
    loads = [(8, 1.0, 0.0, -1.0)]
    p = TrussProblem.ground_structure(nodes, supports, loads, lmax=1.5, drop_fixed=True,
                                      vmax_fraction=0.55, id="tower3d", alias="T3",
                                      title="3-D tower (22 bars)")
    p.description = ("Square base (4 fixed nodes), 4 nodes at z = 1, apex at z = 2, "
                     "lmax = 1.5 -> 22 bars; apex load Fx = 1, Fz = -1; 55 % volume (the "
                     "lightest kinematically stable sub-truss needs 53.0 %; at the former "
                     "45 % every candidate is a mechanism).")
    p.c_exact = C_EXACT["tower3d"]
    return p


def gs_9x3():
    p = _grid_cantilever(9, 3, 0.35, "gs_9x3", "T4", "9x3 ground structure (large 2-D)",
                         lmax=2.3, drop_fixed=True, loads=[])
    nodes = p.nodes
    n1 = int(np.flatnonzero((nodes[:, 0] == 4) & (nodes[:, 1] == 0))[0])
    n2 = int(np.flatnonzero((nodes[:, 0] == 8) & (nodes[:, 1] == 0))[0])
    p.loads = [[(n1, 0.0, -1.0), (n2, 0.0, -1.0)]]
    p.description = (f"9x3 node grid, lmax = 2.3 -> {p.n_bars} bars (no bars between "
                     "the fixed nodes), left column fixed, unit loads down at bottom-chord "
                     "nodes x = 4 and x = 8, 35 % volume (at 25 % no method found a stable "
                     "binary design: the continuous optimum spreads the volume over thin "
                     "bars and its stable full-area subset needed 33.6 % of the former "
                     "118-bar total length, about 34 % of the present one); no exact "
                     "reference (best known from the study).")
    p.c_best_known = C_BEST_KNOWN.get("gs_9x3")
    return p


def column3d():
    p = TrussProblem.grid3d(2, 2, 4, lmax=float(np.sqrt(3.0)) * (1 + 1e-9), drop_fixed=True,
                            loads=[], vmax_fraction=0.30, id="column3d", alias="T5",
                            title="2x2x4 3-D column (66 bars)")
    top = np.flatnonzero(p.nodes[:, 2] == p.nodes[:, 2].max())
    p.loads = [[(int(n), 0.25, 0.0, -0.25) for n in top]]
    p.description = (f"2x2x4 node column, lmax = sqrt(3) -> {p.n_bars} bars (no base-base "
                     "bars), base fixed, lateral + vertical load (1, 0, -1) shared by the "
                     "4 top nodes, 30 % volume; no exact reference.")
    p.c_best_known = C_BEST_KNOWN.get("column3d")
    return p


# pinned by enumerate_exact with the kinematic stability test (see
# tests/test_truss.py).  Before the fix-round (M4) the "optima" of gs_3x2
# (9.0897), gs_4x2 (23.159) and tower3d (8.0967 at 45 % volume) were
# kinematic mechanisms whose zero-energy mode is orthogonal to the load.
# Re-pinned 2026-10-04 (v2) after dropping the bar between the two fixed nodes
# of the 2-D grids (drop_fixed=True everywhere): it carries no force but its
# length entered the volume budget vmax_fraction * sum(L).  gs_3x2: 13 -> 12
# bars, budget 8.5645 -> 8.0645, optimum 9.96376 (volume 8.243, now
# infeasible) -> 11.6569; gs_4x2: 22 -> 21 bars, same optimal bar set and
# compliance (indices shifted by one).  ten_bar and tower3d are unchanged.
C_EXACT = {"ten_bar": 629.4701293633094, "gs_3x2": 11.656854216862358,
           "gs_4x2": 23.89152970038057, "tower3d": 7.846666208853116}
C_BEST_KNOWN = {}

BENCHMARKS = {"ten_bar": ten_bar, "gs_4x2": gs_4x2, "gs_3x2": gs_3x2, "tower3d": tower3d,
              "gs_9x3": gs_9x3, "column3d": column3d}
ALIASES = {"T1": "ten_bar", "T2": "gs_4x2", "T2s": "gs_3x2", "T3": "tower3d",
           "T4": "gs_9x3", "T5": "column3d"}


def get_benchmark(name):
    key = ALIASES.get(name, name)
    if key not in BENCHMARKS:
        raise ValueError(f"unknown truss benchmark {name!r} "
                         f"(known: {', '.join(BENCHMARKS)}; aliases {', '.join(ALIASES)})")
    return BENCHMARKS[key]()


def list_benchmarks():
    return [get_benchmark(k).summary() for k in BENCHMARKS]
