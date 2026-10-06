"""Truss ground structures (2-D / 3-D): nodes, candidate bars with overlap
filtering, supports and loads (docs/QUANTUM_DESIGN.md §B)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = ["TrussProblem", "GroundStructure", "candidate_bars", "grid_nodes"]


def grid_nodes(shape, spacing=None):
    """Regular grid of nodes; index order = itertools.product over the axes
    (last axis fastest), e.g. 2-D (nx, ny): node = i*ny + j at (i dx, j dy)."""
    shape = tuple(int(s) for s in shape)
    d = len(shape)
    spacing = np.ones(d) if spacing is None else np.broadcast_to(
        np.asarray(spacing, float), (d,))
    idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"),
                   axis=-1).reshape(-1, d)
    return idx * spacing


def candidate_bars(nodes, lmax=None, fixed_nodes=None, drop_fixed=True, tol=1e-9):
    """All node pairs (a < b, lexicographic order) that pass the overlap
    filter (no other node strictly inside the bar), the length limit, and
    (``drop_fixed``) are not between two fully fixed nodes."""
    nodes = np.asarray(nodes, dtype=float)
    N = nodes.shape[0]
    a, b = np.triu_indices(N, k=1)
    order = np.lexsort((b, a))
    a, b = a[order], b[order]
    d = nodes[b] - nodes[a]
    L2 = np.einsum("ij,ij->i", d, d)
    keep = L2 > 0
    if lmax is not None:
        keep &= L2 <= (float(lmax) * (1 + 1e-12)) ** 2
    # overlap: node c strictly between a and b, collinear
    P = nodes[None, :, :] - nodes[a][:, None, :]            # (npair, N, dim)
    t = np.einsum("pnd,pd->pn", P, d) / L2[:, None]
    proj = P - t[:, :, None] * d[:, None, :]
    dist2 = np.einsum("pnd,pnd->pn", proj, proj)
    scale = np.maximum(L2, 1e-300)[:, None]
    inside = (dist2 <= tol * scale) & (t > tol) & (t < 1 - tol)
    keep &= ~inside.any(axis=1)
    if drop_fixed and fixed_nodes is not None:
        fx = np.zeros(N, dtype=bool)
        fx[np.asarray(list(fixed_nodes), dtype=int)] = True
        keep &= ~(fx[a] & fx[b])
    return np.stack([a[keep], b[keep]], axis=1)


@dataclass
class TrussProblem:
    """Ground-structure truss problem.

    ``supports``: list of (node, dof) with dof in 0..dim-1; ``loads``: list of
    (node, fx, fy[, fz]) for a single load case, or a list of such lists for
    several load cases.  Areas are relative (x in [0, 1], A = A_full x); the
    volume budget is ``vmax_fraction * sum(L)`` (in units of A_full)."""
    nodes: np.ndarray
    bars: np.ndarray
    supports: list
    loads: list
    E: float = 1.0
    A_full: float = 1.0
    vmax_fraction: float = 0.5
    id: str = "custom"
    title: str = ""
    description: str = ""
    alias: str = ""
    c_exact: Optional[float] = None
    exact_design: Optional[list] = None
    c_best_known: Optional[float] = None
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=float)
        self.bars = np.asarray(self.bars, dtype=np.int64).reshape(-1, 2)
        self.supports = [(int(n), int(k)) for n, k in self.supports]
        ld = self.loads
        if len(ld) and not isinstance(ld[0][0], (list, tuple, np.ndarray)):
            ld = [ld]
        self.loads = [[tuple(float(v) if i else int(v) for i, v in enumerate(row))
                       for row in case] for case in ld]

    # ------------------------------------------------------------------
    @property
    def dim(self):
        return self.nodes.shape[1]

    @property
    def n_nodes(self):
        return self.nodes.shape[0]

    @property
    def n_bars(self):
        return self.bars.shape[0]

    @property
    def ndof(self):
        return self.n_nodes * self.dim

    @property
    def lengths(self):
        d = self.nodes[self.bars[:, 1]] - self.nodes[self.bars[:, 0]]
        return np.sqrt(np.einsum("ij,ij->i", d, d))

    @property
    def vmax(self):
        return float(self.vmax_fraction * self.lengths.sum())

    @property
    def fixed_dofs(self):
        return np.unique(np.array([n * self.dim + k for n, k in self.supports], dtype=np.int64))

    @property
    def free_dofs(self):
        return np.setdiff1d(np.arange(self.ndof), self.fixed_dofs)

    @property
    def load_matrix(self):
        F = np.zeros((self.ndof, len(self.loads)))
        for c, case in enumerate(self.loads):
            for row in case:
                n = row[0]
                for k, val in enumerate(row[1:1 + self.dim]):
                    F[n * self.dim + k, c] += val
        return F

    @property
    def loaded_nodes(self):
        F = self.load_matrix.reshape(self.n_nodes, self.dim, -1)
        return np.flatnonzero(np.abs(F).sum(axis=(1, 2)) > 0)

    @property
    def support_nodes(self):
        return np.unique([n for n, _ in self.supports])

    def to_dict(self):
        loads = [[list(r) for r in case] for case in self.loads]
        return {"id": self.id, "alias": self.alias, "title": self.title,
                "description": self.description, "dim": self.dim,
                "nodes": self.nodes.tolist(), "bars": self.bars.tolist(),
                "supports": [list(s) for s in self.supports],
                "loads": loads[0] if len(loads) == 1 else loads,
                "load_cases": loads,
                "lengths": self.lengths.tolist(), "E": self.E, "A_full": self.A_full,
                "vmax": self.vmax, "vmax_fraction": self.vmax_fraction,
                "c_exact": self.c_exact, "c_best_known": self.c_best_known,
                "n_nodes": self.n_nodes, "n_bars": self.n_bars}

    def summary(self):
        return {"id": self.id, "alias": self.alias, "title": self.title,
                "description": self.description, "dim": self.dim,
                "n_nodes": self.n_nodes, "n_bars": self.n_bars,
                "vmax_fraction": self.vmax_fraction,
                "exact_available": self.n_bars <= 22, "c_exact": self.c_exact,
                "c_best_known": self.c_best_known}

    # ------------------------------------------------------------------
    @classmethod
    def ground_structure(cls, nodes, supports, loads, lmax=None, drop_fixed=True, **kw):
        nodes = np.asarray(nodes, dtype=float)
        dim = nodes.shape[1]
        cnt = {}
        for n, _ in supports:
            cnt[n] = cnt.get(n, 0) + 1
        fully = [n for n, c in cnt.items() if c >= dim]
        bars = candidate_bars(nodes, lmax=lmax, fixed_nodes=fully, drop_fixed=drop_fixed)
        return cls(nodes=nodes, bars=bars, supports=supports, loads=loads, **kw)

    @classmethod
    def grid2d(cls, nx, ny, dx=1.0, dy=1.0, lmax=None, supports=None, loads=None, **kw):
        nodes = grid_nodes((nx, ny), (dx, dy))
        if supports is None:        # left column fully fixed
            supports = [(n, k) for n in range(len(nodes)) if nodes[n, 0] == 0 for k in (0, 1)]
        return cls.ground_structure(nodes, supports, loads or [], lmax=lmax, **kw)

    @classmethod
    def grid3d(cls, nx, ny, nz, dx=1.0, dy=1.0, dz=1.0, lmax=None, supports=None,
               loads=None, **kw):
        nodes = grid_nodes((nx, ny, nz), (dx, dy, dz))
        if supports is None:        # base (z = 0) fully fixed
            supports = [(n, k) for n in range(len(nodes)) if nodes[n, 2] == 0
                        for k in (0, 1, 2)]
        return cls.ground_structure(nodes, supports, loads or [], lmax=lmax, **kw)

    @classmethod
    def from_lists(cls, nodes, bars, supports, loads, **kw):
        return cls(nodes=nodes, bars=bars, supports=supports, loads=loads, **kw)


GroundStructure = TrussProblem
