"""Paper settings as web-app field values (docs/PAPER_SETTINGS.md).

The Setup panel's fields are named like the JSON fields of POST /api/jobs
(``mesh_control``, ``qubo_backend`` ...).  :func:`web_preset` turns the
settings of one manuscript run (:mod:`freeto.paper`, read from the study
suite "quick2") into those field values, so loading a paper example, the
"Paper settings" button and the page's initial state all put exactly the
study's values into the form.  Examples outside the paper get the same run
protocol (QUBO-SA with the block Hessian, seed 0, iteration cap 300,
refined binary voxel evaluation f = 2, MMA with the filtered volume
constraint) on their own geometry, mesh and volume fraction.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # the real core only (the stub core has no paper module)
    from freeto import paper as _paper
    from freeto.quantum.options import QUBOOptions as _QUBOOptions
    PAPER_USABLE = True
except Exception:  # noqa: BLE001
    _paper = None
    _QUBOOptions = None
    PAPER_USABLE = False

#: FreeTOConfig fields set from the form (besides the files / loads / symmetry)
CONFIG_FIELDS = ("mesh_control", "volfrac", "youngs_modulus", "poisson_ratio", "method",
                 "optimizer", "penal", "rmin", "loadtype", "keep_bc", "keep_bcx", "keep_bcy",
                 "keep_bcz", "max_iter", "solver", "eval_binary", "eval_crisp", "eval_beta",
                 "eval_refined", "mma_constraint", "mma_feasible_stop", "init_perturb",
                 "init_seed", "audit")
DEFAULT_EXAMPLE = "cantilever_beam"


def qubo_option_names() -> List[str]:
    if _QUBOOptions is None:
        return []
    from dataclasses import fields
    return [f.name for f in fields(_QUBOOptions)]


def is_paper_example(name: Optional[str]) -> bool:
    return bool(PAPER_USABLE and name in _paper.PAPER_EXAMPLES)


def _protocol(problem_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Paper protocol (QUBO-SA block, seed 0, cap 300, evaluations) on an
    arbitrary example's own problem settings."""
    s = _paper.paper_settings(_paper.PAPER_EXAMPLES[0], _paper.DEFAULT_METHOD)
    for k in ("mesh_control", "volfrac", "youngs_modulus", "poisson_ratio", "method",
              "penal", "rmin", "loadtype", "keep_bc", "keep_bcx", "keep_bcy", "keep_bcz"):
        if k in problem_kwargs:
            s[k] = problem_kwargs[k]
    from freeto.core import FreeTOConfig
    dflt = FreeTOConfig()
    for k in ("youngs_modulus", "poisson_ratio", "method", "penal", "rmin", "loadtype",
              "keep_bc", "keep_bcx", "keep_bcy", "keep_bcz"):
        if k not in problem_kwargs:
            s[k] = getattr(dflt, k)
    return s


def _fields(s: Dict[str, Any], qubo: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: s.get(k) for k in CONFIG_FIELDS}
    out["method"] = str(out["method"]).upper()
    out["optimizer"] = str(out["optimizer"]).upper()
    for k, v in qubo.items():
        out[f"qubo_{k}"] = v
    return out


def web_preset(example: str, method: Optional[str] = None, seed: int = 0,
               problem_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Form values of the paper run ``method`` (default QUBO-SA block) on a
    paper example, or of the paper protocol on another example (its
    ``config_kwargs`` in ``problem_kwargs``).  The QUBO fields always hold the
    resolved QUBOOptions; for OC / MMA runs they are those of QUBO-SA (block)
    (not sent by the form unless the optimizer is QUBO)."""
    if not PAPER_USABLE:
        raise RuntimeError("freeto.paper is not available")
    method = method or _paper.DEFAULT_METHOD
    if is_paper_example(example):
        s = _paper.paper_settings(example, method, seed)
        q = s["qubo"] or _paper.paper_settings(example, _paper.DEFAULT_METHOD, seed)["qubo"]
        out = _fields(s, q)
        out.update(paper_example=example, paper_method=method, paper_seed=int(seed))
        return out
    s = _protocol(dict(problem_kwargs or {}))
    out = _fields(s, s["qubo"])
    out.update(paper_example=None, paper_method=None, paper_seed=int(seed))
    return out


def paper_info(example: str, problem_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """What the UI needs for one example: whether it is a paper example, its
    methods with their form presets, the default method and the published
    reference values."""
    if not PAPER_USABLE:
        return {"available": False, "is_paper": False}
    if not is_paper_example(example):
        return {"available": True, "is_paper": False, "default_method": None,
                "methods": [], "presets": {},
                "protocol_preset": web_preset(example, problem_kwargs=problem_kwargs)}
    methods = _paper.paper_methods(example)
    ref = {m: _paper.paper_record(example, m, 0) for m in methods}
    return {"available": True, "is_paper": True, "default_method": _paper.DEFAULT_METHOD,
            "methods": methods,
            "presets": {m: web_preset(example, m) for m in methods},
            "mma_reference_refined": _paper.mma_reference(example),
            "records_seed0": ref}


def paper_comparison(example: Optional[str], cfg) -> Optional[Dict[str, Any]]:
    """Paper context of a submitted config: matched paper method/seed (if every
    setting equals that run), the published record of that run and the MMA
    reference for the refined gap (only when the problem and the refined
    evaluation f = 2 are the paper's)."""
    if not PAPER_USABLE or not is_paper_example(example):
        return None
    diff = _paper.same_problem(cfg, example)
    out: Dict[str, Any] = {"example": example, "same_problem": not diff,
                           "problem_differences": diff}
    refined_f = getattr(cfg, "eval_refined", None)
    out["gap_reference"] = (_paper.mma_reference(example)
                            if (not diff and refined_f == 2) else None)
    m = _paper.match_paper_method(cfg, example)
    if m is not None:
        out.update(method=m[0], seed=m[1], record=_paper.paper_record(example, m[0], m[1]))
    else:
        out.update(method=None, seed=None, record=None)
    return out
