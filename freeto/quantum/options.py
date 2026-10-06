"""QUBOOptions: configuration of the QUBO design update (``optimizer="QUBO"``).

Kept dependency-free (imported by :mod:`freeto.core`)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional

__all__ = ["QUBOOptions", "HESSIAN_MODES", "VOLUME_MODES", "BLOCK_MODES",
           "INIT_MODES", "QAOA_INITS"]

HESSIAN_MODES = ("auto", "block", "exact-block", "diag", "scalar", "none")
VOLUME_MODES = ("bisection", "penalty")
BLOCK_MODES = ("morton", "rank")
INIT_MODES = ("solid", "oc")
QAOA_INITS = ("linear_ramp", "interp")


@dataclass
class QUBOOptions:
    """Options of the QUBO design update (see docs/QUANTUM_API.md)."""
    backend: str = "auto"
    hessian: str = "auto"
    volume: str = "bisection"
    lambda_q: Optional[float] = None
    gamma: float = 0.0
    move_penalty: float = 0.0
    frontier_fraction: float = 0.25
    block_size: Optional[int] = None
    blocks: str = "morton"
    sweeps: int = 2
    init: str = "solid"
    er: float = 0.05
    n_warm: int = 10
    patience: int = 8
    num_reads: Optional[int] = None
    seed: Optional[int] = None
    qaoa_p: int = 3
    qaoa_shots: int = 1000
    qaoa_init: str = "linear_ramp"
    qaoa_maxiter: Optional[int] = None
    time_limit: Optional[float] = None
    verify_exact: bool = False
    hessian_block_size: int = 64
    hessian_max_rhs: int = 6000
    hessian_scale: float = 1.0
    bisection_steps: int = 14
    lambda_source: str = "sort"
    free_set: str = "band"
    history_average: bool = True
    interp: str = "beso"
    qaoa_polish: bool = True
    move_limit: Optional[float] = 0.25
    move_limit_min: float = 0.02
    protect_loads: bool = True
    guard: bool = True
    guard_tol: float = 0.5
    guard_tol_target: float = 0.25
    max_rejects: int = 4
    connectivity: bool = True
    diagnostics: bool = False
    backend_options: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_any(cls, v):
        if v is None:
            return cls()
        if isinstance(v, cls):
            return cls(**asdict(v))
        if isinstance(v, dict):
            return cls.from_dict(v)
        raise ValueError(f"qubo options must be a QUBOOptions or dict, got {type(v).__name__}")

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        bad = sorted(set(d) - names)
        if bad:
            raise ValueError(f"unknown QUBO option(s): {', '.join(bad)} "
                             f"(allowed: {', '.join(sorted(names))})")
        return cls(**{k: v for k, v in d.items()})

    def to_dict(self):
        return asdict(self)

    def normalized(self):
        o = QUBOOptions.from_any(self)
        o.backend = str(o.backend).lower()
        o.hessian = str(o.hessian).lower()
        if o.hessian == "exact-block":
            o.hessian = "block"
        o.volume = str(o.volume).lower()
        o.blocks = str(o.blocks).lower()
        o.init = str(o.init).lower()
        o.qaoa_init = str(o.qaoa_init).lower()
        o.lambda_source = str(o.lambda_source).lower()
        o.free_set = str(o.free_set).lower()
        o.interp = str(o.interp).lower()
        return o

    def validate(self, check_backend=True):
        """Raise ValueError with a readable message for invalid options."""
        o = self.normalized()
        if o.hessian not in HESSIAN_MODES:
            raise ValueError(f"qubo.hessian must be one of {', '.join(HESSIAN_MODES)}")
        if o.volume not in VOLUME_MODES:
            raise ValueError(f"qubo.volume must be one of {', '.join(VOLUME_MODES)}")
        if o.blocks not in BLOCK_MODES:
            raise ValueError(f"qubo.blocks must be one of {', '.join(BLOCK_MODES)}")
        if o.init not in INIT_MODES:
            raise ValueError(f"qubo.init must be one of {', '.join(INIT_MODES)}")
        if o.qaoa_init not in QAOA_INITS:
            raise ValueError(f"qubo.qaoa_init must be one of {', '.join(QAOA_INITS)}")
        if o.interp not in ("beso", "secant", "simp"):
            raise ValueError("qubo.interp must be 'beso', 'secant' or 'simp'")
        if o.free_set not in ("band", "grey"):
            raise ValueError("qubo.free_set must be 'band' or 'grey'")
        if o.lambda_source not in ("sort", "oc"):
            raise ValueError("qubo.lambda_source must be 'sort' or 'oc'")
        if not (0.0 < float(o.frontier_fraction) <= 1.0):
            raise ValueError("qubo.frontier_fraction must be in (0, 1]")
        if not (0.0 < float(o.er) < 1.0):
            raise ValueError("qubo.er must be in (0, 1)")
        if o.block_size is not None and int(o.block_size) < 1:
            raise ValueError("qubo.block_size must be >= 1 (or None)")
        if int(o.sweeps) < 1:
            raise ValueError("qubo.sweeps must be >= 1")
        if int(o.qaoa_p) < 1:
            raise ValueError("qubo.qaoa_p must be >= 1")
        if int(o.qaoa_shots) < 1:
            raise ValueError("qubo.qaoa_shots must be >= 1")
        if float(o.gamma) < 0 or float(o.move_penalty) < 0:
            raise ValueError("qubo.gamma and qubo.move_penalty must be >= 0")
        if o.lambda_q is not None and not float(o.lambda_q) > 0:
            raise ValueError("qubo.lambda_q must be positive")
        if int(o.patience) < 1:
            raise ValueError("qubo.patience must be >= 1")
        if not (float(o.hessian_scale) >= 0.0 and float(o.hessian_scale) < float("inf")):
            raise ValueError("qubo.hessian_scale must be a finite number >= 0")
        if int(o.hessian_block_size) < 1:
            raise ValueError("qubo.hessian_block_size must be >= 1")
        if o.move_limit is not None and not (0.0 < float(o.move_limit) <= 1.0):
            raise ValueError("qubo.move_limit must be in (0, 1] (or None to disable)")
        if not (0.0 < float(o.move_limit_min) <= 1.0):
            raise ValueError("qubo.move_limit_min must be in (0, 1]")
        if float(o.guard_tol) < 0 or float(o.guard_tol_target) < 0:
            raise ValueError("qubo.guard_tol and qubo.guard_tol_target must be >= 0")
        if int(o.max_rejects) < 0:
            raise ValueError("qubo.max_rejects must be >= 0")
        if o.block_size is not None:
            # a block larger than what the backend can solve (exact 24, QAOA 20
            # qubits, ...) would otherwise only fail inside the first update
            from .backends import BACKENDS
            if o.backend in BACKENDS and BACKENDS[o.backend][2] is not None \
                    and int(o.block_size) > BACKENDS[o.backend][2]:
                raise ValueError(f"qubo.block_size = {int(o.block_size)} exceeds what backend "
                                 f"'{o.backend}' can solve (n <= {BACKENDS[o.backend][2]})")
        if check_backend:
            from .backends import check_backend as _cb
            _cb(o.backend)
        return True
