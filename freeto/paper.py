"""Settings and reference values of the manuscript's continuum runs.

The manuscript's continuum results (Table 1/2, Fig. 5) are the S3/S4 runs of
the FreeTO-Python study suite ``"quick2"`` (:func:`freeto.study.suite_spec`).
This module reads those run definitions from the suite itself, so the
settings it returns are the study's by construction, and reproduces the
keyword set that :func:`freeto.study._run_continuum` passes to
:func:`freeto.examples.example_config` (``tests/test_paper_settings.py``
checks both against the study code path).  It is read-only with respect to
the study: nothing here is used by ``freeto.study``.

The reference values (``paper_reference.json``, written by
``scripts/export_paper_reference.py`` from ``results/quick2/results.json``)
are the published records: iterations, native compliance and volume, crisp
and refined binary voxel compliance and volume, and the MMA reference run's
refined compliance of every example.  See docs/PAPER_SETTINGS.md.
"""
from __future__ import annotations

import copy
import json
import os
from functools import lru_cache

__all__ = ["PAPER_EXAMPLES", "DEFAULT_METHOD", "PAPER_SEED", "STUDY_EVAL_KW",
           "paper_methods", "paper_run", "paper_config_kwargs", "paper_config",
           "paper_settings", "paper_reference", "paper_record", "mma_reference",
           "match_paper_method", "same_problem"]

#: the five continuum examples of the manuscript (Table 1), in its order
PAPER_EXAMPLES = ("cantilever_beam", "mbb_beam", "bridge_deck", "l_bracket", "GE_bracket")
#: the method the web app selects by default (the paper's QUBO-SA, block Hessian)
DEFAULT_METHOD = "QUBO-sa (block)"
PAPER_SEED = 0
#: evaluation keywords _run_continuum adds to every continuum run (common-beta
#: re-smoothing at beta 8 and the crisp element proxy at V*)
STUDY_EVAL_KW = {"eval_beta": 8.0, "eval_crisp": True}
#: methods offered per example, in display order (only those the study ran on
#: that example are listed by paper_methods)
METHOD_ORDER = ("QUBO-sa (block)", "MMA", "BESO-sort", "QUBO-sa (diag)",
                "QUBO-qaoa kb8 p1 penalty (+greedy)", "BESO-sort (move 0.04)",
                "QUBO-sa (scalar)", "QUBO-sa (block, Qx3)", "OC (FreeTO default, flagged)")
_STUDIES = ("S3", "S4")
_REF_FILE = os.path.join(os.path.dirname(__file__), "paper_reference.json")


@lru_cache(maxsize=1)
def _runs():
    from .study import suite_spec
    out = {}
    for r in suite_spec("quick2")["runs"]:
        if r.get("kind") == "continuum" and r.get("study") in _STUDIES \
                and r.get("problem") in PAPER_EXAMPLES and r.get("label") in METHOD_ORDER:
            out.setdefault((r["problem"], r["label"]), r)
    return out


def paper_methods(problem):
    """Labels of the paper methods run on ``problem`` (display order)."""
    runs = _runs()
    return [m for m in METHOD_ORDER if (problem, m) in runs]


def paper_run(problem, method=DEFAULT_METHOD):
    """The quick2 run dict of (problem, method) (a deep copy)."""
    try:
        return copy.deepcopy(_runs()[(problem, method)])
    except KeyError:
        raise KeyError(f"no paper run {method!r} for {problem!r} "
                       f"(paper examples: {', '.join(PAPER_EXAMPLES)})") from None


def paper_config_kwargs(problem, method=DEFAULT_METHOD, seed=PAPER_SEED):
    """Keyword overrides of example_config for this run, exactly as
    freeto.study._run_continuum builds them."""
    run = paper_run(problem, method)
    opt = run.get("optimizer", "QUBO")
    kw = {"mesh_control": int(run["mesh_control"]), "optimizer": opt,
          "max_iter": int(run.get("max_iter", 100)), **STUDY_EVAL_KW}
    ro = dict(run.get("run_options") or {})
    if float(ro.get("init_perturb") or 0.0) > 0 and "init_seed" not in ro:
        ro["init_seed"] = int(seed)
    kw.update(ro)
    if run.get("volfrac") is not None:
        kw["volfrac"] = float(run["volfrac"])
    if opt == "QUBO":
        o = dict(run.get("options") or {})
        o.setdefault("seed", seed)
        kw["qubo"] = o
    if run.get("solver"):
        kw["solver"] = run["solver"]
    return kw


def paper_config(problem, method=DEFAULT_METHOD, seed=PAPER_SEED, stl_dir=None):
    """FreeTOConfig of the paper run (absolute STL paths)."""
    from .examples import example_config
    return example_config(problem, stl_dir=stl_dir, **paper_config_kwargs(problem, method, seed))


def paper_settings(problem, method=DEFAULT_METHOD, seed=PAPER_SEED):
    """Every FreeTOConfig value of the paper run except the STL paths, with the
    QUBO options fully resolved (QUBOOptions defaults filled in; None for
    OC/MMA)."""
    from dataclasses import fields
    from .quantum.options import QUBOOptions
    cfg = paper_config(problem, method, seed)
    out = {}
    for f in fields(cfg):
        if f.name in ("domain", "forces", "fixed", "xfixed", "yfixed", "zfixed", "keepdom"):
            continue
        out[f.name] = copy.deepcopy(getattr(cfg, f.name))
    out["qubo"] = QUBOOptions.from_any(cfg.qubo).to_dict() if cfg.qubo is not None else None
    out["symmetry"] = [list(s) for s in (out.get("symmetry") or [])]
    for k in ("fmagx", "fmagy", "fmagz"):
        out[k] = [float(v) for v in out[k]]
    return out


# ---------------------------------------------------------------------------
# reference values (published records)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def paper_reference():
    """Contents of paper_reference.json ({} when the file is missing)."""
    try:
        with open(_REF_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except OSError:
        return {}


def paper_record(problem, method, seed=PAPER_SEED):
    """Published record (dict) of one run, or None."""
    return (paper_reference().get("records", {}).get(problem, {}).get(method, {})
            .get(str(int(seed))))


def mma_reference(problem):
    """Refined compliance of the paper's MMA reference run of ``problem`` (the
    denominator of every refined gap), or None."""
    return paper_reference().get("mma_reference_refined", {}).get(problem)


# ---------------------------------------------------------------------------
# comparison of an arbitrary FreeTOConfig with the paper runs
# ---------------------------------------------------------------------------
_PROBLEM_FIELDS = ("mesh_control", "volfrac", "keep_bc", "keep_bcx", "keep_bcy", "keep_bcz",
                   "youngs_modulus", "poisson_ratio", "method", "penal", "rmin", "loadtype",
                   "symmetry", "compat", "inside_mode")
_PATH_FIELDS = ("domain", "fixed", "xfixed", "yfixed", "zfixed", "keepdom")


def _norm_path(p):
    return os.path.normcase(os.path.realpath(p)) if p else None


def _bcast(v, n):
    v = [float(x) for x in (v if isinstance(v, (list, tuple)) else [v])]
    return v * n if len(v) == 1 and n > 1 else v


def _diff(a, b, names):
    an, bn = a.normalized(), b.normalized()
    bad = []
    for k in names:
        if getattr(an, k) != getattr(bn, k):
            bad.append(k)
    return bad


def same_problem(cfg, problem):
    """List of differences between ``cfg`` and the paper's ``problem``
    (geometry, BCs, loads, mesh, volume fraction, material, filter); [] when
    it is the paper's problem."""
    ref = paper_config(problem, "MMA")
    bad = _diff(cfg, ref, _PROBLEM_FIELDS)
    for k in _PATH_FIELDS:
        if _norm_path(getattr(cfg, k)) != _norm_path(getattr(ref, k)):
            bad.append(k)
    if [_norm_path(f) for f in cfg.forces] != [_norm_path(f) for f in ref.forces]:
        bad.append("forces")
    n = len(ref.forces)
    for k in ("fmagx", "fmagy", "fmagz"):
        if _bcast(getattr(cfg, k), n) != _bcast(getattr(ref, k), n):
            bad.append(k)
    return bad


def match_paper_method(cfg, problem):
    """(method label, seed) of the paper run whose complete settings equal
    ``cfg`` (problem, optimizer, iteration cap, evaluation options, QUBO
    options incl. the seed), or None."""
    from .quantum.options import QUBOOptions
    if problem not in PAPER_EXAMPLES or same_problem(cfg, problem):
        return None
    c = cfg.normalized()
    q = QUBOOptions.from_any(c.qubo).to_dict() if c.optimizer == "QUBO" else None
    seed = (q or {}).get("seed")
    if seed is None:
        seed = int(c.init_seed) if c.init_perturb else PAPER_SEED
    for m in paper_methods(problem):
        ref = paper_config(problem, m, seed=int(seed)).normalized()
        if ref.optimizer != c.optimizer:
            continue
        rest = ("max_iter", "solver", "tolx", "tol_thresh", "beta_init", "beta_step",
                "beta_max", "mma_move", "eval_beta", "eval_crisp", "eval_refined",
                "mma_constraint", "mma_feasible_stop", "init_perturb", "init_seed")
        if _diff(c, ref, rest):
            continue
        if q is not None and QUBOOptions.from_any(ref.qubo).to_dict() != q:
            continue
        return m, int(seed)
    return None
