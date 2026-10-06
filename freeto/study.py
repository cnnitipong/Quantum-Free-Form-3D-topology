"""Study runner for the QUBO / quantum design-update extension
(docs/QUANTUM_DESIGN.md §E, API in docs/QUANTUM_API.md §4, protocol in
docs/NOTES_quantum.md §7).

    python -m freeto.study --suite quick --out results/quick
    python -m freeto.study --suite quick2 --dry-run      # list the runs only
    python -m freeto.study --suite quick2 --jobs 2 --out results/quick2   # 2 processes
    python -m freeto.study --suite full  --out results/full
    python -m freeto.study --spec my_study.json --out results/mine

Writes ``results.json`` (all records), ``results.csv`` (scalar columns),
``summary.md`` (tables) and the figures F1-F7 (matplotlib, Agg backend).

What the study can and cannot show: everything runs on a classical
computer -- simulated annealing and tabu are classical heuristics and QAOA is
an exact state-vector *simulation* whose cost grows as 2^n.  The study
measures solution quality (gap to the exact optimum for small trusses,
compliance against MMA for continua under a common crisp evaluation at equal
volume) and algorithmic behaviour (QAOA approximation ratio vs depth p and
size n against the uniform-superposition and greedy baselines, sensitivity to
penalty weights).  It cannot demonstrate any quantum speed-up, and continuum
designs have no exact optimum to compare with.

Continuum protocol (fix round, docs/VERIFICATION_quantum.md §3):

* every final design is evaluated on the **common crisp model**
  (:mod:`freeto.evaluate`): the final pre-smoothing filtered field is
  projected crisp on FreeTO's 4x fine grid with the threshold chosen so that
  the element densities have the run's target volume fraction exactly, then
  one FE solve with the same SIMP model -- the headline metric; the native
  FreeTO compliance / volume of each method are reported next to it;
* MMA with its native beta continuation is the primary baseline (gaps are
  relative to MMA's crisp compliance at the same volume); an MMA
  compliance-volume curve (volfrac 0.26-0.34) gives the MMA value at the
  native volume of every other method;
* stochastic methods run >= 5 seeds (mean +- sd); OC is reported but
  flagged (with the FreeTO defaults it stops grey at beta ~ 0.7);
* BESO sorting (same pipeline, no QUBO) and QAOA without the greedy polish
  are controls.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import threading
import time
import traceback

import numpy as np

__all__ = ["run_study", "reprocess", "resume_records", "suite_spec", "SUITES", "main"]

DISCLAIMER = (
    "**What this study shows and does not show.** All runs are classical: "
    "SA/tabu/greedy are classical heuristics, QAOA is an exact state-vector "
    "simulation (cost 2^n) whose angles are optimised classically, no cloud QPU was "
    "used unless a D-Wave/IBM backend appears in the tables. All times are CPU "
    "seconds on this machine. The numbers measure solution quality (gap to the exact "
    "optimum for trusses with <= 22 bars; compliance against MMA at equal volume "
    "under a common crisp evaluation for continua) and algorithmic behaviour (QAOA "
    "approximation ratio vs p and n against the uniform-superposition and greedy "
    "baselines, penalty sensitivity). They cannot show a quantum speed-up, and "
    "continuum designs have no exact reference.")

PROTOCOL = (
    "**Continuum evaluation protocol.** *crisp c* = compliance of the common crisp "
    "design: each method's final pre-smoothing filtered field is interpolated to "
    "FreeTO's 4x fine grid (as smoothedge3D does) and projected to 0/1 with the "
    "threshold chosen by bisection so that the element densities (fine-grid window "
    "averages, void 0.001) have exactly the target volume fraction V* (same V* for "
    "every method), then one FE solve with the same SIMP model (p = 3, Emin = 1e-3). "
    "This is the headline metric. *native c @ V* = FreeTO's own reported compliance "
    "and volume fraction of the method (OC/MMA: compliance of the last iterate at "
    "its own Heaviside beta; QUBO/BESO: one FE solve on the returned design, which "
    "is the best design found at the target volume). *gap* = crisp c / crisp c of MMA "
    "(native beta continuation, same problem, mesh and V*) - 1; 'n/a' marks diverged "
    "runs (crisp c > 10x MMA or singular). *MMA @ native V* = MMA's native compliance "
    "interpolated (log-linear) on the MMA compliance-volume curve at the method's mean "
    "native volume. Stochastic methods: mean +- sd over the listed seeds; MMA, OC "
    "and BESO-sort are deterministic (1 run). OC uses the FreeTO defaults (beta <= 2, "
    "stopping rule change <= 3e-3) and stops grey: it is reported, not used as a "
    "baseline. beta_final = Heaviside beta of the projection that produced the "
    "returned design. For OC/MMA the native volume is that of the iterate whose "
    "compliance is reported (hist volfrac[-2]), not of FreeTO's twice-smoothed returned "
    "field (recorded as volume_fraction_returned; its own compliance is "
    "compliance_returned_design, 2-15x higher). **Accuracy of the crisp proxy:** the "
    "crisp element field keeps window-averaged (grey) boundary elements that SIMP "
    "penalises; a refined-voxel FE of the same crisp fine fields (independent check, "
    "docs/VERIFICATION_quantum_round2.md §2: cantilever MC 36 at 2x, MMA 0.0946, QUBO-sa "
    "diag 0.0953 (+0.8 %), BESO-sort 0.1020 (+7.9 %)) keeps the ranking but shows that the "
    "proxy flatters the binary methods by about 1-4 points; QUBO-vs-MMA differences of "
    "a few % are therefore not differences. Single-seed numbers depend on the thread "
    "count (chaotic SA trajectories: cantilever MC 25 seed 0 crisp 0.150 with 1 thread, "
    "0.175 with 2); the study pins the thread count (results.json machine.threads). "
    "**Connectivity repair and physics check (corrections of 2026-10-02).** QUBO/BESO runs use "
    "the connectivity repair of the binary update (QUBOOptions.connectivity = True, the "
    "default: ungrounded components without keep/load elements are removed, the others "
    "reconnected along the cheapest void path, the added volume rebalanced; "
    "docs/PHYSICS_AUDIT.md §5). Every continuum run is audited (freeto.audit, "
    "docs/AUDIT_API.md) on its crisp design at V*: face-connected components of the crisp "
    "fine-grid solid, floating share outside the supported main component, share of |F| on "
    "solid / on the main body, rigid-body restraint of the main body, captured keep/support/load "
    "regions and the volume target. *audit* = runs passing all checks / runs; a group with a "
    "run failing a physics check (everything but the volume bookkeeping) is labelled "
    "**physically invalid**, one failing only the volume target 'volume off target'. The "
    "examples were corrected the same day: bridge_deck's kept deck and support pads now hold "
    "element centres at every mesh (volfrac 0.2 -> 0.35, study mesh MC 31 -> 41), "
    "l_bracket's support no longer keeps elements outside the domain, mbb_beam's and "
    "multi_load_beam's load/support regions hold element centres at coarse meshes. "
    "**v2 (suite quick2, 2026-10-04).** *refined c* = compliance of the binary voxel design "
    "on a grid refined 2x per element (freeto.evaluate.refined_voxel_compliance): voxel solid "
    "iff the mean of the same fine-grid field at its 8 corners exceeds a threshold bisected "
    "to the same V*, solid E0 / void the crisp model's void modulus, no SIMP exponent, "
    "supports and load regions refined with the mesh; *gap vs MMA (refined)* uses the MMA "
    "baseline's refined c. In quick2 MMA's volume constraint is evaluated on the filtered "
    "(pre-projection) field and MMA only stops on the tolerances when |fval| <= 1e-3; every "
    "continuum run has max_iter = 300. *MMA (init s)* = MMA from x0 = V + U(-0.05, 0.05) "
    "(seed s): the baseline's spread.")

# -- palette (reference categorical order, light surface) -------------------
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#4a3aa7", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

QUICK_MAIN = (("cantilever_beam", 36), ("mbb_beam", 46))
# bridge_deck at MC 41 (2026-10-02): with the corrected example the kept deck row is
# captured at every mesh, but at MC 31 (4 element rows) it is 25 % of the domain and
# leaves < 9 % of the 35 % budget for the load path; at MC 41 it is 17 % (+ 3 % pads)
QUICK_ROBUST = (("bridge_deck", 41), ("l_bracket", 30), ("GE_bracket", 24))
MMA_CURVE = (0.26, 0.28, 0.32, 0.34)
PROXY_BIAS = 0.04
#: a truss run counts as an exact hit if gap <= HIT_TOL (v2: was 1e-9; designs
#: differing only in zero-force bars differ by ~1e-9 through the 1e-9 void area)
HIT_TOL = 1e-6       # refined-voxel check: the crisp proxy flatters binary designs by 1-4 pts
DIVERGED = 10.0


# ---------------------------------------------------------------------------
# suites
# ---------------------------------------------------------------------------
def _truss_runs(study, problems, seeds, backends, qaoa_ps, qaoa_problems, qaoa_seeds,
                block_exact=20):
    runs = []
    for pb in problems:
        runs.append({"study": study, "kind": "truss", "problem": pb, "method": "exact",
                     "label": "exact", "seeds": [0]})
    for pb in problems:
        for meth, lab in (("sort", "sort/BESO"), ("oc_round", "OC+round"),
                          ("oc_qubo", "OC+QUBO-round")):
            runs.append({"study": study, "kind": "truss", "problem": pb, "method": meth,
                         "backend": "auto", "label": lab, "seeds": [0]})
        for be in backends:
            opts = {}
            if be == "exact":
                opts["block_size"] = block_exact
            runs.append({"study": study, "kind": "truss", "problem": pb, "method": "qubo",
                         "backend": be, "label": f"QUBO-{be}", "options": opts,
                         "seeds": [0] if be == "exact" else list(seeds)})
    for pb in qaoa_problems:
        for p in qaoa_ps:
            for pol, tag in ((False, "raw"), (True, "+greedy")):
                runs.append({"study": study, "kind": "truss", "problem": pb, "method": "qubo",
                             "backend": "qaoa", "label": f"QUBO-qaoa p={p} ({tag})",
                             "options": {"qaoa_p": p, "volume": "penalty", "block_size": 14,
                                         "qaoa_polish": pol},
                             "seeds": list(qaoa_seeds)})
    return runs


def _cont(study, problem, mc, label, optimizer="QUBO", options=None, seeds=(0,),
          max_iter=100, volfrac=None, role="method"):
    r = {"study": study, "kind": "continuum", "problem": problem, "mesh_control": mc,
         "optimizer": optimizer, "label": label, "max_iter": max_iter,
         "seeds": list(seeds), "role": role}
    if options is not None:
        r["options"] = dict(options)
    if volfrac is not None:
        r["volfrac"] = float(volfrac)
    return r


def _continuum_block(study, problem, mc, seeds, methods, curve=MMA_CURVE, max_iter=100,
                     oc=True):
    """MMA (primary baseline) + its compliance-volume curve, OC (flagged),
    BESO-sort and the QUBO methods of ``methods`` [(label, options)]."""
    runs = [_cont(study, problem, mc, "MMA", "MMA", max_iter=max_iter, role="baseline")]
    for vf in curve:
        runs.append(_cont(study, problem, mc, f"MMA (V={vf:.2f})", "MMA", max_iter=max_iter,
                          volfrac=vf, role="curve"))
    if oc:
        runs.append(_cont(study, problem, mc, "OC (FreeTO default, flagged)", "OC",
                          max_iter=max_iter, role="flagged"))
    runs.append(_cont(study, problem, mc, "BESO-sort", "QUBO", {"hessian": "none"},
                      max_iter=max_iter, role="control"))
    for lab, opts in methods:
        runs.append(_cont(study, problem, mc, lab, "QUBO", opts, seeds=seeds,
                          max_iter=max_iter))
    return runs


def _qaoa_controls(study, problem, mc, seeds, max_iter=30):
    """Same block-penalty model (10-variable Morton blocks, one sweep) solved by
    SA, by raw QAOA (p = 1) and by QAOA + greedy polish."""
    base = {"hessian": "block", "block_size": 10, "volume": "penalty", "sweeps": 1,
            "verify_exact": True}
    return [_cont(study, problem, mc, "MMA", "MMA", max_iter=100, role="baseline"),
            _cont(study, problem, mc, "QUBO-sa blocks10 penalty", "QUBO",
                  dict(base, backend="sa"), seeds=seeds, max_iter=max_iter),
            _cont(study, problem, mc, "QUBO-qaoa p=1 (raw)", "QUBO",
                  dict(base, backend="qaoa", qaoa_p=1, qaoa_polish=False), seeds=seeds,
                  max_iter=max_iter, role="control"),
            _cont(study, problem, mc, "QUBO-qaoa p=1 (+greedy)", "QUBO",
                  dict(base, backend="qaoa", qaoa_p=1, qaoa_polish=True), seeds=seeds,
                  max_iter=max_iter)]


def suite_spec(name):
    """Built-in study matrices: 'smoke' (about a minute), 'quick' (about an
    hour on a 2-core laptop: design-size continuum meshes, 5 seeds), 'full'
    (several hours)."""
    name = str(name).lower()
    if name == "smoke":
        runs = _truss_runs("S1", ["gs_3x2"], [0], ["exact", "sa"], [1], ["gs_3x2"], [0])
        runs += [_cont("S3", "cantilever_beam", 20, "MMA", "MMA", max_iter=30,
                       role="baseline"),
                 _cont("S3", "cantilever_beam", 20, "MMA (V=0.26)", "MMA", max_iter=30,
                       volfrac=0.26, role="curve"),
                 _cont("S3", "cantilever_beam", 20, "OC (FreeTO default, flagged)", "OC",
                       max_iter=30, role="flagged"),
                 _cont("S3", "cantilever_beam", 20, "QUBO-sa (diag)", "QUBO",
                       {"backend": "sa", "hessian": "diag"}, seeds=[0], max_iter=30)]
        runs += [{"study": "F2", "kind": "qaoa_scan", "problem": "gs_3x2", "p_values": [1, 2],
                  "n_max": 10, "seeds": [0]},
                 {"study": "F5", "kind": "truss", "problem": "gs_3x2", "method": "qubo",
                  "backend": "sa", "label": "penalty x1", "scan": "lambda_q",
                  "scan_value": 1.0, "lambda_q_mult": 1.0, "options": {"volume": "penalty"},
                  "seeds": [0]}]
        return {"name": "smoke", "runs": runs}
    if name == "quick":
        seeds5 = [0, 1, 2, 3, 4]
        runs = _truss_runs("S1", ["ten_bar", "gs_3x2", "gs_4x2", "tower3d"], seeds5,
                           ["exact", "sa", "tabu", "greedy"], [1, 3],
                           ["ten_bar", "gs_3x2"], [0, 1, 2])
        for pb in ("gs_9x3", "column3d"):
            runs.append({"study": "S2", "kind": "truss", "problem": pb, "method": "sort",
                         "label": "sort/BESO", "seeds": [0]})
            runs.append({"study": "S2", "kind": "truss", "problem": pb,
                         "method": "oc_round", "label": "OC+round", "seeds": [0]})
            for be in ("sa", "tabu"):
                runs.append({"study": "S2", "kind": "truss", "problem": pb, "method": "qubo",
                             "backend": be, "label": f"QUBO-{be}", "seeds": [0, 1, 2]})
        meth = [("QUBO-sa (block)", {"backend": "sa", "hessian": "block"}),
                ("QUBO-sa (diag)", {"backend": "sa", "hessian": "diag"})]
        for pb, mc in QUICK_MAIN:
            runs += _continuum_block("S3", pb, mc, seeds5, meth)
        runs += _qaoa_controls("S3q", "cantilever_beam", 25, [0])
        for pb, mc in QUICK_ROBUST:
            runs += _continuum_block("S4", pb, mc, [0, 1, 2],
                                     [("QUBO-sa (block)", {"backend": "sa",
                                                           "hessian": "block"})],
                                     curve=(), oc=False)
            # block QAOA in penalty mode (the only QPU-sized formulation), started
            # from the solid design like the other QUBO runs; with init="oc" the
            # thresholded 6-iteration OC field is itself the catastrophic step
            # (c ~ 1e3 on bridge/l_bracket, docs/NOTES_quantum.md §6)
            runs.append(_cont("S4", pb, mc, "QUBO-qaoa kb8 p1 penalty (+greedy)", "QUBO",
                              {"backend": "qaoa", "hessian": "block", "block_size": 8,
                               "qaoa_p": 1, "qaoa_shots": 300, "volume": "penalty",
                               "sweeps": 1, "verify_exact": True}, seeds=[0], max_iter=45))
        runs += [{"study": "F2", "kind": "qaoa_scan", "problem": pb, "p_values": [1, 2, 3, 5],
                  "n_max": 14, "seeds": [0]} for pb in ("ten_bar", "gs_3x2", "tower3d")]
        runs += [{"study": "F2", "kind": "qaoa_scan", "problem": "cantilever_beam",
                  "mesh_control": 20, "p_values": [1, 2, 3, 5], "n_values": [8, 12],
                  "n_instances": 2, "seeds": [0]}]
        runs += _penalty_runs(["ten_bar", "gs_3x2"], [0.1, 0.3, 1.0, 3.0, 10.0], [0, 1])
        runs += _gamma_runs("cantilever_beam", 25, [0.0, 0.05], [0])
        # appended 2026-10-04 (run index 98, so the earlier indices are unchanged):
        # block QAOA on the MBB beam with the same settings as the S4 QAOA runs,
        # so that every continuum example has a QAOA column.
        runs.append(_cont("S3", "mbb_beam", 46, "QUBO-qaoa kb8 p1 penalty (+greedy)", "QUBO",
                          {"backend": "qaoa", "hessian": "block", "block_size": 8,
                           "qaoa_p": 1, "qaoa_shots": 300, "volume": "penalty",
                           "sweeps": 1, "verify_exact": True}, seeds=[0], max_iter=45))
        return {"name": "quick", "runs": runs}
    if name == "quick2":
        return _quick2(suite_spec("quick"))
    if name == "full":
        seeds = list(range(10))
        runs = _truss_runs("S1", ["ten_bar", "gs_3x2", "gs_4x2", "tower3d"], seeds,
                           ["exact", "sa", "tabu", "greedy"] + _opt_local(),
                           [1, 2, 3, 5], ["ten_bar", "gs_3x2", "tower3d", "gs_4x2"],
                           [0, 1, 2])
        for pb in ("gs_9x3", "column3d"):
            for meth_, lab in (("sort", "sort/BESO"), ("oc_round", "OC+round"),
                               ("oc_qubo", "OC+QUBO-round")):
                runs.append({"study": "S2", "kind": "truss", "problem": pb, "method": meth_,
                             "label": lab, "seeds": [0]})
            for be in ["sa", "tabu"] + _opt_local():
                runs.append({"study": "S2", "kind": "truss", "problem": pb, "method": "qubo",
                             "backend": be, "label": f"QUBO-{be}", "seeds": list(range(5))})
        meth = []
        for be in ("sa", "tabu"):
            for hs in ("diag", "block"):
                meth.append((f"QUBO-{be} ({hs})", {"backend": be, "hessian": hs}))
        for pb, mc in (("cantilever_beam", 40), ("mbb_beam", 50), ("bridge_deck", 41)):
            runs += _continuum_block("S3", pb, mc, list(range(5)), meth)
        runs += _qaoa_controls("S3q", "cantilever_beam", 25, [0, 1, 2])
        for pb, mc in QUICK_ROBUST + (("GE_bracket", 40),):
            runs += _continuum_block("S4", pb, mc, list(range(5)),
                                     [("QUBO-sa (block)", {"backend": "sa",
                                                           "hessian": "block"})],
                                     curve=(), oc=False)
        runs += [{"study": "F2", "kind": "qaoa_scan", "problem": pb, "p_values": [1, 2, 3, 5],
                  "n_max": 16, "seeds": [0, 1]} for pb in ("ten_bar", "gs_3x2", "tower3d",
                                                            "gs_4x2")]
        runs += [{"study": "F2", "kind": "qaoa_scan", "problem": "cantilever_beam",
                  "mesh_control": 25, "p_values": [1, 2, 3, 5], "n_values": [8, 12, 16],
                  "n_instances": 4, "seeds": [0]}]
        runs += _penalty_runs(["ten_bar", "gs_3x2", "gs_4x2"], [0.03, 0.1, 0.3, 1.0, 3.0,
                                                                 10.0, 30.0], list(range(5)))
        runs += _gamma_runs("cantilever_beam", 25, [0.0, 0.01, 0.02, 0.05, 0.1], [0, 1])
        runs += _hardware_runs()
        return {"name": "full", "runs": runs}
    raise ValueError(f"unknown suite {name!r} (smoke, quick, quick2, full)")


#: options of every continuum run of the suite "quick2" (passed to FreeTOConfig
#: through ``run["run_options"]``, docs/NOTES_quantum.md §9)
V2_RUN_OPTIONS = {"eval_refined": 2, "mma_constraint": "filtered", "mma_feasible_stop": True}
#: FreeTOConfig fields a study run may set through ``run_options``
RUN_OPTION_KEYS = ("eval_refined", "mma_constraint", "mma_feasible_stop", "init_perturb",
                   "init_seed")
MMA_INIT_PERTURB = 0.05


def _d5_controls(study, problem, mc, seeds, hessian_x3=False):
    """v2 controls of one continuum example: BESO sorting with the QUBO move
    limit 0.04 (1 run), the separable 'scalar' Hessian (same seeds as the
    block variant), optionally the block Hessian scaled by 3 (``hessian_x3``)
    and MMA from 5 perturbed initial designs (spread of the baseline;
    init_seed = seed, see _run_continuum)."""
    mma = _cont(study, problem, mc, "MMA (init s)", "MMA", seeds=list(range(5)),
                role="baseline_spread")
    mma["run_options"] = {"init_perturb": MMA_INIT_PERTURB}
    out = [_cont(study, problem, mc, "BESO-sort (move 0.04)", "QUBO",
                 {"hessian": "none", "move_limit": 0.04}, role="control"),
           _cont(study, problem, mc, "QUBO-sa (scalar)", "QUBO",
                 {"backend": "sa", "hessian": "scalar"}, seeds=seeds, role="control")]
    if hessian_x3:
        out.append(_cont(study, problem, mc, "QUBO-sa (block, Qx3)", "QUBO",
                         {"backend": "sa", "hessian": "block", "hessian_scale": 3.0},
                         seeds=seeds, role="control"))
    return out + [mma]


def _quick2(quick):
    """Suite 'quick2' (manuscript v2) derived from the 'quick' list: every
    continuum run at max_iter 300 with V2_RUN_OPTIONS; S3q controls with 5
    seeds; the mbb_beam QAOA run (index 98 of 'quick') moved next to the
    mbb_beam S3 block; D5 controls (_d5_controls) after every S3/S4 example
    block (+ "QUBO-sa (block, Qx3)" on the S3 examples); truss QUBO / sorting
    runs (S1, S2, F5 penalty) with ``kinematic_repair=True`` (not OC rounding or
    exact enumeration); qaoa_scan runs unchanged."""
    import copy
    runs = copy.deepcopy(quick["runs"])
    qaoa_lab = "QUBO-qaoa kb8 p1 penalty (+greedy)"
    i_mbb = [i for i, r in enumerate(runs) if r.get("kind") == "continuum"
             and r.get("study") == "S3" and r.get("problem") == "mbb_beam"
             and r.get("label") == qaoa_lab]
    assert len(i_mbb) == 1, "quick suite changed: mbb_beam QAOA run not found"
    mbb_qaoa = runs.pop(i_mbb[0])
    out = []
    n = len(runs)
    for i, r in enumerate(runs):
        out.append(r)
        if r.get("kind") != "continuum" or r.get("study") not in ("S3", "S4"):
            continue
        key = (r["study"], r["problem"], r["mesh_control"])
        nxt = runs[i + 1] if i + 1 < n else {}
        if (nxt.get("kind") == "continuum" and (nxt.get("study"), nxt.get("problem"),
                                                nxt.get("mesh_control")) == key):
            continue
        # last run of this example block
        if key == ("S3", "mbb_beam", mbb_qaoa["mesh_control"]):
            out.append(mbb_qaoa)
        blk = [b for b in runs if b.get("kind") == "continuum"
               and (b.get("study"), b.get("problem"), b.get("mesh_control")) == key
               and b.get("label") == "QUBO-sa (block)"]
        out += _d5_controls(*key, seeds=blk[0]["seeds"] if blk else [0],
                            hessian_x3=key[0] == "S3")
    for r in out:
        if r.get("kind") == "truss" and r.get("method") in ("qubo", "sort"):
            # truss kinematic repair (v2, P0b) for every QUBO / sorting run
            r["options"] = dict(r.get("options") or {}, kinematic_repair=True)
        if r.get("kind") != "continuum":
            continue
        # cap 300 for every continuum run (2026-10-04, Fable): the binary
        # updates stop by their own rules well before 100 iterations, whereas
        # MMA with its unbounded beta continuation reached a cap of 100 on 12
        # runs with the grey fraction still above 1e-3 (compliance already flat
        # within about 1 %); a cap that binds on the baseline only would bias
        # the comparison, so all runs share the higher cap
        r["max_iter"] = 300
        if r.get("study") == "S3q" and r.get("role") != "baseline":
            r["seeds"] = [0, 1, 2, 3, 4]
        r["run_options"] = dict(V2_RUN_OPTIONS, **(r.get("run_options") or {}))
    return {"name": "quick2", "runs": out}


def _opt_local():
    from .quantum.backends import available_backends
    ok = {b["name"] for b in available_backends() if b["available"]}
    return [b for b in ("dwave_sa", "dwave_tabu") if b in ok]


def _hardware_runs():
    """S5: only when tokens are present (never in CI)."""
    from .quantum.backends import available_backends
    ok = {b["name"] for b in available_backends() if b["available"]}
    runs = []
    if "dwave_qpu" in ok:
        for pb in ("gs_3x2", "tower3d"):
            runs.append({"study": "S5", "kind": "truss", "problem": pb, "method": "qubo",
                         "backend": "dwave_qpu", "label": "QUBO-dwave_qpu",
                         "options": {"volume": "penalty", "num_reads": 500}, "seeds": [0, 1, 2]})
    if "dwave_hybrid" in ok:
        runs.append({"study": "S5", "kind": "truss", "problem": "gs_9x3", "method": "qubo",
                     "backend": "dwave_hybrid", "label": "QUBO-dwave_hybrid",
                     "options": {"volume": "penalty"}, "seeds": [0]})
    if "ibm" in ok:
        runs.append({"study": "S5", "kind": "truss", "problem": "gs_3x2", "method": "qubo",
                     "backend": "ibm", "label": "QUBO-ibm p=1",
                     "options": {"volume": "penalty", "qaoa_p": 1}, "seeds": [0]})
    return runs


def _penalty_runs(problems, mults, seeds):
    runs = []
    for pb in problems:
        for m in mults:
            runs.append({"study": "F5", "kind": "truss", "problem": pb, "method": "qubo",
                         "backend": "sa", "label": f"penalty x{m:g}", "scan": "lambda_q",
                         "scan_value": m, "lambda_q_mult": m,
                         "options": {"volume": "penalty"},
                         "seeds": list(seeds)})
    return runs


def _gamma_runs(problem, mc, gammas, seeds):
    return [{"study": "F5", "kind": "continuum", "problem": problem, "mesh_control": mc,
             "optimizer": "QUBO", "label": f"gamma={g:g}", "scan": "gamma", "scan_value": g,
             "options": {"backend": "sa", "hessian": "diag", "gamma": g},
             "max_iter": 80, "seeds": list(seeds), "role": "scan"} for g in gammas]


SUITES = {"smoke": lambda: suite_spec("smoke"), "quick": lambda: suite_spec("quick"),
          "quick2": lambda: suite_spec("quick2"), "full": lambda: suite_spec("full")}


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------
def _nanmean(v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def _nansd(v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return float(np.std(v, ddof=1)) if len(v) > 1 else None


def _qstats(stats):
    if not stats:
        return {}
    sizes = [s.get("max_block", s.get("n_free", 0)) for s in stats]
    ar = [a for s in stats for a in (s.get("approx_ratios") or
                                     ([s["approx_ratio"]] if s.get("approx_ratio") is not None
                                      else []))]
    po = [a for s in stats for a in (s.get("p_opts") or
                                     ([s["p_opt"]] if s.get("p_opt") is not None else []))]
    em = [s["exact_match"] for s in stats if s.get("exact_match") is not None]
    emr = [s["exact_match_raw"] for s in stats if s.get("exact_match_raw") is not None]
    bsr = [s["best_shot_ratio"] for s in stats if s.get("best_shot_ratio") is not None]
    qpu = [s["qpu_time"] for s in stats if s.get("qpu_time") is not None]
    rep = [s.get("repair_added") for s in stats if s.get("repair_added") is not None]
    fl = [s.get("n_flips") for s in stats if s.get("n_flips") is not None]
    return {"mean_qubo_size": float(np.mean([s.get("n_free", 0) for s in stats])),
            "max_qubo_size": int(max(sizes)) if sizes else 0,
            "solver_time": float(sum(s.get("solver_time") or 0.0 for s in stats)),
            "qpu_access_time": float(sum(qpu)) if qpu else None,
            "approx_ratio": _nanmean(ar), "p_opt": _nanmean(po),
            "exact_match": _nanmean(em), "exact_match_raw": _nanmean(emr),
            "best_shot_ratio": _nanmean(bsr),
            "repair_added_mean": _nanmean(rep), "flips_mean": _nanmean(fl)}


def _run_truss(run, seed):
    from .truss import get_benchmark, solve_truss
    prob = get_benchmark(run["problem"])
    opts = dict(run.get("options") or {})
    if run.get("lambda_q_mult") is not None:
        # multiple of the default penalty weight 2 / v_max^2 (scaled units)
        opts["lambda_q"] = float(run["lambda_q_mult"]) * 2.0 / float(prob.lengths.max()) ** 2
    t0 = time.perf_counter()
    res = solve_truss(prob, method=run.get("method", "qubo"), backend=run.get("backend", "sa"),
                      seed=seed, **opts)
    wall = time.perf_counter() - t0
    gap = res.gap if (res.feasible and res.gap is not None and math.isfinite(res.gap)) \
        else None
    rec = {"compliance": res.compliance, "c_ref": res.c_exact, "gap": gap,
           "feasible": bool(res.feasible), "volume_fraction": res.volume_fraction,
           "iterations": res.iterations, "fe_solves": res.fe_solves,
           "qubo_solves": res.qubo_solves, "wall_time": wall,
           "history": list(res.history.get("compliance", [])),
           "n": prob.n_bars, "bars_on": np.flatnonzero(res.on).tolist(),
           "backend": res.backend}
    rec.update(_qstats(res.qubo_stats))
    ka = [s_["kin_repair_added"] for s_ in res.qubo_stats if "kin_repair_added" in s_]
    kr = [s_["kin_repair_removed"] for s_ in res.qubo_stats if "kin_repair_removed" in s_]
    if (run.get("options") or {}).get("kinematic_repair"):
        # kinematic repair (v2): bars added / removed per iteration (mean)
        rec["repair_added"] = float(np.mean(ka)) if ka else 0.0
        rec["repair_removed"] = float(np.mean(kr)) if kr else 0.0
    if res.timing.get("solver") is not None and "solver_time" not in rec:
        rec["solver_time"] = res.timing["solver"]
    if run.get("method") == "exact":
        rec["solver_time"] = wall
        rec["n_feasible_designs"] = res.info.get("n_feasible")
    return rec, None


def _run_continuum(run, seed, keep_surface=True, out_dir=None, run_id=None):
    """One continuum run.  ``run["run_options"]`` (RUN_OPTION_KEYS) go to
    FreeTOConfig; with ``init_perturb`` > 0 and no explicit ``init_seed`` the
    run's seed is the initial-design seed.  With ``out_dir`` the evaluated
    pre-smoothing field (+ binary design, histories) is saved to
    ``<out_dir>/fields/<run_id>.npz`` (record field ``fields_file``)."""
    from .core import run_freeto
    from .examples import example_config
    opt = run.get("optimizer", "QUBO")
    kw = {"mesh_control": int(run["mesh_control"]), "optimizer": opt,
          "max_iter": int(run.get("max_iter", 100)), "eval_beta": 8.0,
          "eval_crisp": True}
    ro = dict(run.get("run_options") or {})
    bad = sorted(set(ro) - set(RUN_OPTION_KEYS))
    if bad:
        raise ValueError(f"unknown run_options {bad} (allowed: {', '.join(RUN_OPTION_KEYS)})")
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
    cfg = example_config(run["problem"], **kw)
    t0 = time.perf_counter()
    res = run_freeto(cfg, log=None)
    wall = time.perf_counter() - t0
    ex = res.extra
    qh = ex.get("qubo_history", [])
    hv = [float(v) for v in res.history.get("volfrac", [])]
    # native (c, V) pair of the *same* design: for OC/MMA res.comp is the compliance
    # of the once-smoothed iterate entering the last iteration (volume hv[-2]);
    # res.finalvol is the volume of FreeTO's twice-smoothed returned field
    v_nat = hv[-2] if (opt != "QUBO" and len(hv) >= 2) else float(res.finalvol)
    rec = {"compliance": float(res.comp), "volume_fraction": float(v_nat),
           "volume_fraction_returned": float(res.finalvol), "history_volfrac": hv,
           "crisp_compliance": ex.get("crisp_compliance"),
           "crisp_volfrac": ex.get("crisp_volfrac"), "volfrac_target": ex.get("crisp_target"),
           "compliance_at_beta": ex.get("compliance_at_beta"),
           "volfrac_at_beta": ex.get("volfrac_at_beta"),
           "compliance_returned_design": ex.get("compliance_returned_design"),
           "refined_compliance": ex.get("refined_compliance"),
           "refined_volfrac": ex.get("refined_volfrac"),
           "refined_threshold": ex.get("refined_threshold"),
           "refined_f": ex.get("refined_f"), "refined_time": ex.get("refined_time"),
           "mma_constraint": ex.get("mma_constraint"),
           "mma_fval_final": ex.get("mma_fval_final"),
           "init_perturb": ro.get("init_perturb"), "init_seed": ro.get("init_seed"),
           "beta_final": ex.get("beta_final"), "returned_design": ex.get("returned_design"),
           "best_iteration": ex.get("best_iteration"),
           "n_rejects": int(sum(1 for q in qh if q.get("guard"))) if qh else None,
           "eval_errors": ex.get("eval_errors"), "role": run.get("role"),
           "iterations": res.iterations,
           "fe_solves": ex.get("fe_solves", res.iterations),
           "qubo_solves": ex.get("qubo_solves", 0), "wall_time": wall,
           "history": list(res.history["compliance"]), "nnele": res.elenum1,
           "mesh_control": run["mesh_control"],
           "backend": (qh[-1]["backend"] if qh else None),
           "update_time_per_iter": float(np.mean([q["wall_time"] for q in qh])) if qh else None,
           "hessian_time_per_iter": float(np.mean([q["hessian_time"] for q in qh])) if qh else None,
           "fe_time_per_iter": res.timings.get("fe_per_iter")}
    rec.update(_qstats(qh))
    if opt != "QUBO":
        rec["solver_time"] = None
    rec.update(_audit_fields(ex.get("audit"), ex.get("audit_error")))
    if out_dir and run_id:
        rec["fields_file"] = _save_fields(out_dir, run_id, res)
    surf = None
    if keep_surface:
        try:
            v, f = res.surface(smooth=True)
            surf = (np.asarray(v), np.asarray(f))
        except Exception:  # noqa: BLE001
            surf = None
    return rec, surf


def _save_fields(out_dir, run_id, res):
    """Write <out_dir>/fields/<run_id>.npz (full_pre float32, binary_design
    uint8 for QUBO/BESO runs, history arrays, grid size); returns the path
    relative to out_dir."""
    rel = os.path.join("fields", f"{run_id}.npz")
    os.makedirs(os.path.join(out_dir, "fields"), exist_ok=True)
    ex = res.extra
    fp = ex.get("full_pre")
    arrs = {"full_pre": np.asarray(fp if fp is not None else [], dtype=np.float32),
            "grid": np.array([res.setup_info.get("nelx"), res.setup_info.get("nely"),
                              res.setup_info.get("nelz")], dtype=np.int64)}
    if ex.get("binary_design") is not None:
        arrs["binary_design"] = (np.asarray(ex["binary_design"]) > 0.5).astype(np.uint8)
    for k, v in res.history.items():
        arrs[f"history_{k}"] = np.asarray(v, dtype=np.float64)
    np.savez_compressed(os.path.join(out_dir, rel), **arrs)
    return rel.replace(os.sep, "/")


AUDIT_FIELDS = ("audit_ok", "audit_physics_ok", "audit_failed", "n_components",
                "floating_frac", "loads_solid", "load_on_main", "supports_solid",
                "binary_n_components", "binary_floating_frac")


def _audit_fields(aud, err=None):
    """Study record fields of a freeto.audit report (docs/AUDIT_API.md)."""
    if not isinstance(aud, dict):
        return {"audit_ok": None, "audit_error": err} if err else {}
    ch = aud.get("checks", {})
    b = aud.get("binary") or {}
    return {"audit_ok": bool(aud.get("ok")),
            "audit_physics_ok": bool(aud.get("physics_ok", aud.get("ok"))),
            "audit_failed": ",".join(aud.get("failed", [])),
            "n_components": aud.get("n_components"), "floating_frac": aud.get("floating_frac"),
            "loads_solid": (ch.get("loads_solid") or {}).get("value"),
            "load_on_main": (ch.get("load_on_main") or {}).get("value"),
            "supports_solid": (ch.get("supports_solid") or {}).get("value"),
            "binary_n_components": b.get("n_components"),
            "binary_floating_frac": b.get("floating_frac")}


def _truss_subqubos(problem, n_max, seed):
    """Fixed-lambda QUBO of the first iteration (solid design, target Vmax)
    of a truss benchmark; problems larger than n_max give Morton blocks."""
    from .truss import get_benchmark, TrussFE
    from .truss.optimize import _model, _bar_blocks
    from .quantum.blockqubo import solve_volume_model
    prob = get_benchmark(problem)
    fe = TrussFE(prob)
    fem = TrussFE(prob, amin=1e-3)
    x = np.ones(prob.n_bars)
    _, g, U, _ = fem.grad(x)
    H = fem.hessian(x, U)
    s = float(np.max(np.abs(g)))
    Q, h, const = _model(g, H, x, s)
    Vk = max(0.85 * float(fe.L.sum()), prob.vmax)
    y, info = solve_volume_model(Q, h, const, fe.L, 0.0, Vk, x, backend="exact" if
                                 prob.n_bars <= 20 else "sa", seed=seed)
    lam = info["lambda"]
    hl = h + lam * fe.L
    out = []
    if prob.n_bars <= n_max:
        out.append((Q, hl, f"{problem} (n={prob.n_bars})"))
    else:
        for B in _bar_blocks(prob, n_max):
            if len(B) < 4:
                continue
            Bc = np.setdiff1d(np.arange(prob.n_bars), B)
            hB = hl[B] + 2.0 * Q[np.ix_(B, Bc)] @ x[Bc]
            out.append((Q[np.ix_(B, B)], hB, f"{problem} block (n={len(B)})"))
    return out


def _continuum_subqubos(problem, mc, n_values, n_inst, seed):
    from .core import run_freeto
    from .examples import example_config
    from .quantum.update import QUBOUpdater
    cap = []
    QUBOUpdater.capture = cap.append
    try:
        run_freeto(example_config(problem, mesh_control=mc, optimizer="QUBO", max_iter=6,
                                  qubo={"backend": "sa", "hessian": "block", "seed": seed}),
                   log=None)
    finally:
        QUBOUpdater.capture = None
    m = cap[-1]
    Q = m["Q"].tocsr()
    h = m["h"] + m["lambda"] * m["v"]
    a = m["a"]
    order = np.argsort(m["mkey"], kind="stable")
    rng = np.random.default_rng(seed)
    out = []
    for n in n_values:
        starts = rng.choice(max(1, order.size - n), size=min(n_inst, max(1, order.size - n)),
                            replace=False)
        for st in starts:
            B = order[st:st + n]
            Bc = np.setdiff1d(np.arange(order.size), B)
            QBB = Q[B][:, B].toarray()
            hB = h[B] + 2.0 * (Q[B][:, Bc] @ a[Bc])
            out.append((QBB, hB, f"{problem} MC{mc} block (n={n})"))
    return out


def _run_qaoa_scan(run, seed):
    from .quantum.qaoa import run_qaoa
    from .quantum.exact import spectrum
    from .quantum.anneal import greedy_descent
    if run.get("mesh_control"):
        inst = _continuum_subqubos(run["problem"], int(run["mesh_control"]),
                                   run.get("n_values", [8, 12]), int(run.get("n_instances", 2)),
                                   seed)
    else:
        inst = _truss_subqubos(run["problem"], int(run.get("n_max", 14)), seed)
    recs = []
    for k, (Q, h, name) in enumerate(inst):
        E = spectrum(Q, h, 0.0)
        Emin, Emax = float(E.min()), float(E.max())
        span = (Emax - Emin) or 1.0
        tol = 1e-9 * max(1.0, abs(Emin), abs(Emax))
        # classical greedy-only baseline: 16 steepest-descent restarts from random states
        gd = greedy_descent(Q, h, 0.0, num_reads=16, seed=seed)
        g_best = float(np.min(gd["energies"]))
        g_hit = float(np.mean(gd["energies"] <= Emin + tol))
        for p in run.get("p_values", [1, 2, 3]):
            t0 = time.perf_counter()
            r = run_qaoa(Q, h, 0.0, p=int(p), shots=int(run.get("shots", 1000)), seed=seed,
                         E_all=E)
            inf = r["info"]
            pol = greedy_descent(Q, h, 0.0, initial_state=r["x"], num_reads=1)
            Ep = min(float(pol["energies"][0]), float(r["energy"]))
            recs.append({"instance": name, "instance_index": k, "n": int(h.size), "p": int(p),
                         "approx_ratio": inf["approx_ratio"], "p_opt": inf["p_opt"],
                         "p_opt_shots": inf["p_opt_shots"], "nfev": inf["nfev"],
                         "best_shot_optimal": inf["best_shot_optimal"],
                         "best_shot_ratio": inf["best_shot_ratio"],
                         "polished_ratio": (Emax - Ep) / span,
                         "polished_optimal": bool(Ep <= Emin + tol),
                         "uniform_ratio": inf["uniform_ratio"],
                         "uniform_p_opt": inf["uniform_p_opt"],
                         "greedy16_ratio": (Emax - g_best) / span,
                         "greedy16_optimal": bool(g_best <= Emin + tol),
                         "greedy16_hit_fraction": g_hit,
                         "wall_time": time.perf_counter() - t0,
                         "solver_time": inf["qaoa_time"], "compliance": None,
                         "label": f"QAOA p={p}", "backend": "qaoa"})
    return recs


# ---------------------------------------------------------------------------
def _expand(spec):
    out = []
    for i, run in enumerate(spec.get("runs", [])):
        for seed in run.get("seeds", [0]):
            rid = "{}-{}-{}-{}-s{}".format(run.get("study", "S"), i, run.get("problem", ""),
                                           str(run.get("label", run.get("method", "")))
                                           .replace(" ", "_").replace("/", "-"), seed)
            out.append((rid, run, seed))
    return out


def _load_spec(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class _ThreadPin:
    """Pin BLAS/LAPACK/OpenMP/MKL (PARDISO) thread pools for reproducibility.

    The SA/QUBO trajectories are chaotic: the same seed gives different designs
    with 1 and 2 threads (cantilever MC 25 crisp 0.150 vs 0.175), so the study
    pins the thread count (default 1) with threadpoolctl when it is installed
    (effective even after numpy is imported) and the usual environment
    variables (effective for libraries loaded later)."""

    def __init__(self, n):
        self.n = None if n is None else int(n)
        self.ctx = None
        self.saved = {}
        self.info = {"requested": self.n, "method": None, "pools": []}

    def __enter__(self):
        if self.n is None:
            return self
        for k in _THREAD_ENV:
            self.saved[k] = os.environ.get(k)
            os.environ[k] = str(self.n)
        self.info["method"] = "env"
        try:
            from threadpoolctl import threadpool_limits, threadpool_info
            self.ctx = threadpool_limits(limits=self.n)
            self.ctx.__enter__()
            self.info["method"] = "threadpoolctl+env"
            self.info["pools"] = [{"api": i.get("internal_api"),
                                   "num_threads": i.get("num_threads")}
                                  for i in threadpool_info()]
        except Exception:  # noqa: BLE001 -- threadpoolctl optional
            self.ctx = None
        return self

    def __exit__(self, *a):
        if self.ctx is not None:
            self.ctx.__exit__(*a)
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def run_study(spec, callback=None, out_dir=None, stop_event=None, figures=True, log=print,
              threads=1, jobs=1, fresh_workers=False, resume_records=None):
    """Run a study matrix (dict, suite name, or path to a JSON spec).

    ``threads``: BLAS/OpenMP/MKL thread count used for the whole study (default
    1, for reproducibility -- results depend on it; None = leave unchanged).
    Recorded in ``results["machine"]["threads"]``.

    ``jobs`` > 1 runs the expanded runs in a pool of ``jobs`` worker processes
    (spawn context; every worker pins its thread pools to ``threads`` exactly as
    the serial path does and seeds every run from the spec, so the records equal
    those of the serial run except for timings); records are collected in the
    original run order.  ``results.json`` is (re)written every
    ``PARTIAL_EVERY`` finished runs (``"partial": true``) when ``out_dir`` is set.

    ``fresh_workers``: every run in a fresh worker process (pool with
    ``max_tasks_per_child=1``; also for ``jobs=1``), so memory cannot accumulate
    across runs (the CLI default).  ``resume_records``: {run_id: [records]} of
    finished runs to keep (see :func:`resume_records`); only the other runs are
    executed."""
    with _ThreadPin(threads) as pin:
        return _run_study(spec, callback, out_dir, stop_event, figures, log, pin.info,
                          threads=threads, jobs=jobs, fresh_workers=fresh_workers,
                          kept=resume_records)


def resume_records(out_dir, spec, log=print):
    """Records of ``out_dir/results.json`` (partial or complete) to keep when
    resuming: every run of the current expanded ``spec`` whose records are all
    error-free.  Returns {run_id: [records]} (derived fields stripped)."""
    log = log or (lambda s: None)
    pj = os.path.join(out_dir, "results.json")
    with open(pj, encoding="utf-8") as f:
        old = json.load(f)
    want = {rid for rid, _, _ in _expand(spec)}
    groups = {}
    for r in old.get("records", []):
        groups.setdefault(r.get("run_id"), []).append(r)
    kept = {}
    for rid, rs in groups.items():
        if rid in want and rs and not any(r.get("error") for r in rs):
            kept[rid] = [{k: v for k, v in r.items() if k not in _DERIVED} for r in rs]
    log(f"resume {out_dir}: {len(kept)} of {len(want)} runs kept, {len(want) - len(kept)} to run "
        f"({sum(1 for rid, rs in groups.items() if rid in want and any(r.get('error') for r in rs))}"
        f" with errors, {len(want - set(groups))} missing; stored file "
        f"{'partial' if old.get('partial') else 'complete'})")
    return kept


PARTIAL_EVERY = 5
_WORKER_PIN = None
_WORKER_EXACT_CACHE = {}


def _run_base(rid, run, seed):
    return {"run_id": rid, "study": run.get("study"), "kind": run.get("kind"),
            "problem": run.get("problem"), "label": run.get("label", run.get("method")),
            "method": run.get("method", run.get("optimizer")),
            "backend": run.get("backend"), "seed": seed, "scan": run.get("scan"),
            "scan_value": run.get("scan_value"), "error": None}


def _execute_run(rid, run, seed, out_dir, figures, exact_cache):
    """One expanded run -> (records, surface or None).  Never raises: a failing
    run gives one record with ``error`` (+ traceback)."""
    base = _run_base(rid, run, seed)
    t0 = time.perf_counter()
    new = []
    surf = None
    try:
        kind = run.get("kind")
        if kind == "truss":
            key = (run["problem"], run.get("method"))
            if run.get("method") == "exact" and key in exact_cache:
                rec = dict(exact_cache[key])
            else:
                rec, _ = _run_truss(run, seed)
                if run.get("method") == "exact":
                    exact_cache[key] = rec
            new.append({**base, **rec})
        elif kind == "continuum":
            rec, surf = _run_continuum(run, seed, keep_surface=figures, out_dir=out_dir,
                                       run_id=rid)
            new.append({**base, **rec})
        elif kind == "qaoa_scan":
            for rec in _run_qaoa_scan(run, seed):
                new.append({**base, **rec, "label": rec["label"]})
        else:
            raise ValueError(f"unknown run kind {kind!r}")
    except Exception as e:  # noqa: BLE001
        new = [{**base, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=4),
                "wall_time": time.perf_counter() - t0}]
        surf = None
    return new, surf


def _worker_init(threads):
    """Pool initializer: pin the worker's thread pools for its whole life."""
    global _WORKER_PIN
    _WORKER_PIN = _ThreadPin(threads)
    _WORKER_PIN.__enter__()


def _worker_run(idx, rid, run, seed, out_dir, figures):
    t0 = time.perf_counter()
    new, surf = _execute_run(rid, run, seed, out_dir, figures, _WORKER_EXACT_CACHE)
    return idx, new, surf, time.perf_counter() - t0


def _run_msg(last):
    msg = last.get("error") or ("c=%.5g" % last["compliance"]
                                if last.get("compliance") is not None else "")
    if last.get("crisp_compliance") is not None:
        msg += " crisp=%.5g" % last["crisp_compliance"]
    if last.get("refined_compliance") is not None:
        msg += " refined=%.5g" % last["refined_compliance"]
    if last.get("gap") is not None and np.isfinite(last["gap"]):
        msg += " gap=%.2f%%" % (100 * last["gap"])
    return msg


def _machine(thread_info, jobs=1):
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor(), "cpus": os.cpu_count(),
            "threads": thread_info, "jobs": int(jobs)}


def _write_partial(out_dir, spec, slots, n, thread_info, jobs, t_start):
    """results.json with the records finished so far (run order), so that a
    crash loses at most PARTIAL_EVERY runs; ``reprocess`` can read it."""
    recs = [r for sl in slots if sl is not None for r in sl]
    res = {"suite": spec.get("name", "custom"), "partial": True,
           "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "machine": _machine(thread_info, jobs), "elapsed": time.perf_counter() - t_start,
           "n_done": sum(1 for sl in slots if sl is not None), "n_runs": n,
           "records": recs, "summary": [], "figures": [], "files": {}, "spec": spec}
    os.makedirs(out_dir, exist_ok=True)
    pj = os.path.join(out_dir, "results.json")
    tmp = pj + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_clean(res), f, indent=1, default=_json_default)
    os.replace(tmp, pj)


def _run_study(spec, callback, out_dir, stop_event, figures, log, thread_info, threads=1,
               jobs=1, fresh_workers=False, kept=None):
    if isinstance(spec, str):
        spec = _load_spec(spec) if os.path.isfile(spec) else suite_spec(spec)
    log = log or (lambda s: None)
    cb = callback or (lambda ev: None)
    t_start = time.perf_counter()
    expanded = _expand(spec)
    n = len(expanded)
    jobs = max(1, int(jobs or 1))
    cb({"event": "start", "n_runs": n, "suite": spec.get("name", "custom")})
    slots = [None] * n             # records of run idx (filled in completion order)
    kept = kept or {}
    for i_, (rid_, _, _) in enumerate(expanded):
        if rid_ in kept:
            slots[i_] = kept[rid_]
    todo = [i_ for i_ in range(n) if slots[i_] is None]
    n_todo = len(todo)
    if kept:
        log(f"resuming: {n - n_todo} run(s) kept from the stored results, {n_todo} to run "
            "(F4 shows only the designs run now)")
    surfaces = {}
    n_done = 0

    def finish(idx, new, surf, dt):
        nonlocal n_done
        rid = expanded[idx][0]
        slots[idx] = new
        if surf is not None:
            surfaces[rid] = surf
        n_done += 1
        last = new[-1]
        log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{n_done}/{n_todo}] #{idx} {rid}: "
            f"{_run_msg(last)} ({dt:.1f} s; study {time.perf_counter() - t_start:.0f} s)")
        if out_dir and n_done % PARTIAL_EVERY == 0:
            try:
                _write_partial(out_dir, spec, slots, n, thread_info, jobs, t_start)
            except Exception as e:  # noqa: BLE001 -- never stop the study for this
                log(f"partial results.json not written: {type(e).__name__}: {e}")
        cb({"event": "run_end", "index": idx, "n_runs": n, "run_id": rid,
            "record": _slim(last)})

    def start(idx):
        rid, run, _ = expanded[idx]
        cb({"event": "run_start", "index": idx, "n_runs": n, "run_id": rid,
            "label": run.get("label"), "study": run.get("study")})

    if jobs == 1 and not fresh_workers:
        exact_cache = {}
        for idx in todo:
            rid, run, seed = expanded[idx]
            if stop_event is not None and stop_event.is_set():
                log("study stopped by request")
                break
            start(idx)
            t0 = time.perf_counter()
            new, surf = _execute_run(rid, run, seed, out_dir, figures, exact_cache)
            finish(idx, new, surf, time.perf_counter() - t0)
    else:
        _run_parallel(expanded, jobs, threads, out_dir, figures, stop_event, log, start,
                      finish, todo=todo, fresh=fresh_workers)
    records = [r for sl in slots if sl is not None for r in sl]
    _postprocess(records)
    summary = _summary(records)
    results = {"suite": spec.get("name", "custom"),
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "partial": False,
               "machine": dict(_machine(thread_info, jobs), fresh_workers=bool(fresh_workers)),
               "elapsed": time.perf_counter() - t_start, "records": records,
               "summary": summary, "figures": [], "files": {}, "spec": spec}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if figures:
            try:
                results["figures"] = make_figures(records, surfaces, out_dir, log=log)
                for f in results["figures"]:
                    cb({"event": "figure", "path": f})
            except Exception as e:  # noqa: BLE001
                log(f"figure generation failed: {type(e).__name__}: {e}")
                traceback.print_exc()
        files = _write_outputs(results, out_dir)
        results["files"] = files
    cb({"event": "done", "results": _slim(results)})
    return results


def _run_parallel(expanded, jobs, threads, out_dir, figures, stop_event, log, start, finish,
                  todo=None, fresh=False):
    """Process-pool execution of the expanded runs ``todo`` (default all; at most
    ``jobs`` in flight; ``fresh``: one run per worker process,
    max_tasks_per_child=1).  A worker exception or a crashed worker process is
    recorded as the run's ``error``; a broken pool is replaced and the study
    continues."""
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool
    ctx = mp.get_context("spawn")

    def new_pool():
        kw = {"max_tasks_per_child": 1} if fresh else {}
        return ProcessPoolExecutor(max_workers=jobs, mp_context=ctx,
                                   initializer=_worker_init, initargs=(threads,), **kw)
    pool = new_pool()
    todo = list(range(len(expanded))) if todo is None else list(todo)
    pending = {}
    t_sub = {}
    try:
        while todo or pending:
            while todo and len(pending) < jobs and not (stop_event is not None
                                                        and stop_event.is_set()):
                idx = todo.pop(0)
                rid, run, seed = expanded[idx]
                start(idx)
                t_sub[idx] = time.perf_counter()
                try:
                    fut = pool.submit(_worker_run, idx, rid, run, seed, out_dir, figures)
                except BrokenProcessPool:
                    pool = new_pool()
                    fut = pool.submit(_worker_run, idx, rid, run, seed, out_dir, figures)
                pending[fut] = idx
            if stop_event is not None and stop_event.is_set() and todo:
                log("study stopped by request (waiting for the running runs)")
                todo = []
            if not pending:
                break
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            broken = False
            for fut in done:
                idx = pending.pop(fut)
                rid, run, seed = expanded[idx]
                try:
                    i2, new, surf, dt = fut.result()
                except Exception as e:  # noqa: BLE001 -- worker crash / pickling error
                    broken = broken or isinstance(e, BrokenProcessPool)
                    new = [{**_run_base(rid, run, seed),
                            "error": f"{type(e).__name__}: {e}",
                            "wall_time": time.perf_counter() - t_sub[idx]}]
                    surf, dt = None, time.perf_counter() - t_sub[idx]
                finish(idx, new, surf, dt)
            if broken:
                # every in-flight run of a broken pool fails the same way
                for fut, idx in list(pending.items()):
                    rid, run, seed = expanded[idx]
                    pending.pop(fut)
                    finish(idx, [{**_run_base(rid, run, seed),
                                  "error": "BrokenProcessPool: worker process died",
                                  "wall_time": time.perf_counter() - t_sub[idx]}],
                           None, time.perf_counter() - t_sub[idx])
                pool.shutdown(wait=False, cancel_futures=True)
                pool = new_pool()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _slim(obj):
    return json.loads(json.dumps(_clean(obj), default=_json_default))


def _clean(o):
    """inf/nan -> None so that results.json is strict JSON."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    return o


def _json_default(o):
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _mma_curve(records, problem, mc):
    """MMA native (V, c) and crisp (V*, c) points of one problem/mesh."""
    pts = [r for r in records if r.get("kind") == "continuum" and not r.get("error")
           and r.get("problem") == problem and r.get("mesh_control") == mc
           and r.get("method") == "MMA" and r.get("study") in ("S3", "S3q", "S4")
           and r.get("role") != "baseline_spread"]
    nat = sorted((r["volume_fraction"], r["compliance"]) for r in pts
                 if _finite(r.get("compliance")) and _finite(r.get("volume_fraction")))
    cr = sorted((r["volfrac_target"], r["crisp_compliance"]) for r in pts
                if _finite(r.get("crisp_compliance")) and _finite(r.get("volfrac_target")))
    return nat, cr


def _interp_logc(curve, V):
    """log-linear interpolation of c(V) (None outside the curve's range)."""
    if len(curve) < 2 or not _finite(V):
        return None
    vs = np.array([p[0] for p in curve])
    cs = np.log(np.array([p[1] for p in curve]))
    if V < vs.min() - 1e-9 or V > vs.max() + 1e-9:
        return None
    return float(np.exp(np.interp(V, vs, cs)))


def _postprocess(records):
    """Truss: gap vs exact (<= 22 bars) or vs the best feasible run (large
    trusses); infeasible -> gap None ('n/a').  Continuum: gap of the crisp
    compliance vs MMA's crisp compliance (same problem, mesh, target volume);
    diverged (crisp > 10x MMA or singular) -> gap None.  gap_refined: the same
    for the refined-voxel compliance (eval_refined) vs the MMA baseline's
    refined compliance (None when either is missing / > 10x, or when the
    baseline's own refined c exceeds 10x its crisp c)."""
    best = {}
    for r in records:
        if r.get("error") or r.get("kind") != "truss":
            continue
        if r.get("feasible") and _finite(r.get("compliance")):
            k = r["problem"]
            best[k] = min(best.get(k, math.inf), r["compliance"])
    for r in records:
        if r.get("kind") == "truss" and not r.get("error") and r.get("c_ref") is None:
            ref = best.get(r["problem"])
            if ref:
                r["c_ref"] = ref
                r["gap"] = (r["compliance"] / ref - 1.0) if r.get("feasible") else None
    refs, refs_r = {}, {}
    for r in records:
        if (r.get("kind") == "continuum" and not r.get("error") and r.get("method") == "MMA"
                and r.get("role") == "baseline" and _finite(r.get("crisp_compliance"))):
            k = (r["problem"], r.get("mesh_control"), round(r["volfrac_target"], 6))
            refs[k] = r["crisp_compliance"]
            # a baseline whose refined design is broken (refined c > 10x its own crisp c,
            # e.g. a member lost at the 2x voxel scale) is no reference: gap_refined None
            if (_finite(r.get("refined_compliance"))
                    and r["refined_compliance"] <= DIVERGED * r["crisp_compliance"]):
                refs_r[k] = r["refined_compliance"]
    for r in records:
        if r.get("kind") != "continuum" or r.get("error"):
            continue
        k = (r["problem"], r.get("mesh_control"), round(r.get("volfrac_target") or 0, 6))
        # refined-voxel gap (suite quick2): same reference run, its refined compliance
        ref_r, cr_ = refs_r.get(k), r.get("refined_compliance")
        r["c_ref_refined"] = ref_r
        r["gap_refined"] = (cr_ / ref_r - 1.0) if (ref_r and _finite(cr_)
                                                   and cr_ <= DIVERGED * ref_r) else None
        ref = refs.get(k)
        c = r.get("crisp_compliance")
        if ref is None:
            r["gap"] = None
            r["feasible"] = _finite(c)
            continue
        r["c_ref"] = ref
        div = not _finite(c) or c > DIVERGED * ref
        r["diverged"] = bool(div)
        r["feasible"] = not div
        r["gap"] = None if div else c / ref - 1.0
        nat, _ = _mma_curve(records, r["problem"], r.get("mesh_control"))
        r["mma_native_at_V"] = _interp_logc(nat, r.get("volume_fraction"))


def _group(records, keys):
    g = {}
    for r in records:
        g.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    return g


def _summary(records):
    rows = []
    tr = [r for r in records if r.get("kind") == "truss"]
    for (study, problem, label), rs in sorted(
            _group(tr, ["study", "problem", "label"]).items(),
            key=lambda kv: tuple(str(x) for x in kv[0])):
        ok = [r for r in rs if not r.get("error")]
        fin = [r["gap"] for r in ok if r.get("feasible") and _finite(r.get("gap"))]
        rows.append({
            "study": study, "kind": "truss", "problem": problem, "label": label,
            "runs": len(rs), "errors": len(rs) - len(ok),
            "feasible": sum(1 for r in ok if r.get("feasible")),
            "gap_mean": float(np.mean(fin)) if fin else None,
            "gap_sd": _nansd(fin),
            "gap_min": float(np.min(fin)) if fin else None,
            "gap_max": float(np.max(fin)) if fin else None,
            "n_infeasible": sum(1 for r in ok if not r.get("feasible")),
            "n_exact_hits": sum(1 for g in fin if g <= HIT_TOL),
            "hit_tol": HIT_TOL,
            "repair_added_mean": _nanmean([r.get("repair_added") for r in ok]),
            "repair_removed_mean": _nanmean([r.get("repair_removed") for r in ok]),
            "compliance_mean": _nanmean([r.get("compliance") for r in ok
                                         if r.get("feasible")]),
            "wall_mean": _nanmean([r.get("wall_time") for r in ok]),
            "solver_mean": _nanmean([r.get("solver_time") for r in ok]),
            "qpu_mean": _nanmean([r.get("qpu_access_time") for r in ok]),
            "iterations_mean": _nanmean([r.get("iterations") for r in ok]),
            "approx_ratio_mean": _nanmean([r.get("approx_ratio") for r in ok]),
            "exact_match_mean": _nanmean([r.get("exact_match") for r in ok])})
    co = [r for r in records if r.get("kind") == "continuum"]
    order = {"baseline": 0, "baseline_spread": 1, "curve": 2, "flagged": 3, "control": 4,
             "method": 5, "scan": 6}
    spread = {}
    for (study, problem, mc), rs in _group([r for r in co if not r.get("error")
                                            and r.get("role") == "baseline_spread"],
                                           ["study", "problem", "mesh_control"]).items():
        cr_, rf_ = ([r.get("crisp_compliance") for r in rs],
                    [r.get("refined_compliance") for r in rs])
        spread[(study, problem, mc)] = {
            "n": len(rs), "crisp_mean": _nanmean(cr_), "crisp_sd": _nansd(cr_),
            "refined_mean": _nanmean(rf_), "refined_sd": _nansd(rf_)}
    groups = _group(co, ["study", "problem", "mesh_control", "label"])
    for (study, problem, mc, label), rs in sorted(
            groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2]),
                                            order.get(kv[1][0].get("role"), 9),
                                            str(kv[0][3]))):
        ok = [r for r in rs if not r.get("error")]
        good = [r for r in ok if not r.get("diverged")]
        cr = [r.get("crisp_compliance") for r in good]
        nat = [r.get("compliance") for r in good]
        Vn = [r.get("volume_fraction") for r in good]
        Vn_mean = _nanmean(Vn)
        nat_curve, _ = _mma_curve(records, problem, mc)
        mma_at = _interp_logc(nat_curve, Vn_mean)
        nat_mean = _nanmean(nat)
        rf = [r.get("refined_compliance") for r in good]
        sp = spread.get((study, problem, mc)) or {}
        rows.append({
            "study": study, "kind": "continuum", "problem": problem, "mesh_control": mc,
            "label": label, "role": (rs[0].get("role") if rs else None),
            "nnele": rs[0].get("nnele") if rs else None,
            "runs": len(rs), "errors": len(rs) - len(ok),
            "seeds": [r.get("seed") for r in rs],
            "n_diverged": sum(1 for r in ok if r.get("diverged")),
            "volfrac_target": _nanmean([r.get("volfrac_target") for r in ok]),
            "crisp_mean": _nanmean(cr), "crisp_sd": _nansd(cr),
            "crisp_min": (min(float(x) for x in cr if _finite(x))
                          if any(_finite(x) for x in cr) else None),
            "crisp_max": (max(float(x) for x in cr if _finite(x))
                          if any(_finite(x) for x in cr) else None),
            "flips_mean": _nanmean([r.get("flips_mean") for r in ok]),
            "repair_added_mean": _nanmean([r.get("repair_added_mean") for r in ok]),
            "crisp_V_mean": _nanmean([r.get("crisp_volfrac") for r in good]),
            "gap_mean": _nanmean([r.get("gap") for r in good]),
            "gap_sd": _nansd([r.get("gap") for r in good]),
            "refined_mean": _nanmean(rf), "refined_sd": _nansd(rf),
            "refined_V_mean": _nanmean([r.get("refined_volfrac") for r in good]),
            "gap_refined_mean": _nanmean([r.get("gap_refined") for r in good]),
            "gap_refined_sd": _nansd([r.get("gap_refined") for r in good]),
            "mma_spread_n": sp.get("n"),
            "mma_spread_crisp_mean": sp.get("crisp_mean"),
            "mma_spread_crisp_sd": sp.get("crisp_sd"),
            "mma_spread_refined_mean": sp.get("refined_mean"),
            "mma_spread_refined_sd": sp.get("refined_sd"),
            "native_mean": nat_mean, "native_sd": _nansd(nat),
            "native_V_mean": Vn_mean, "native_V_sd": _nansd(Vn),
            "mma_native_at_V": mma_at,
            "gap_native_eqV": (nat_mean / mma_at - 1.0) if (nat_mean and mma_at) else None,
            "beta_final": sorted({round(r["beta_final"], 2) for r in ok
                                  if _finite(r.get("beta_final"))}),
            "iterations_mean": _nanmean([r.get("iterations") for r in ok]),
            "wall_mean": _nanmean([r.get("wall_time") for r in ok]),
            "solver_mean": _nanmean([r.get("solver_time") for r in ok]),
            "rejects_mean": _nanmean([r.get("n_rejects") for r in ok]),
            "exact_match_mean": _nanmean([r.get("exact_match") for r in ok]),
            "exact_match_raw_mean": _nanmean([r.get("exact_match_raw") for r in ok]),
            "approx_ratio_mean": _nanmean([r.get("approx_ratio") for r in ok]),
            "audit_n": sum(1 for r in ok if r.get("audit_ok") is not None),
            "audit_pass": sum(1 for r in ok if r.get("audit_ok")),
            "audit_physics_fail": sum(1 for r in ok if r.get("audit_physics_ok") is False),
            "audit_volume_only_fail": sum(1 for r in ok if r.get("audit_ok") is False
                                          and r.get("audit_physics_ok")),
            "audit_failed": sorted({c for r in ok for c in (r.get("audit_failed") or "").split(",")
                                    if c}),
            "n_components_mean": _nanmean([r.get("n_components") for r in ok]),
            "n_components_max": (max(r["n_components"] for r in ok
                                     if r.get("n_components") is not None)
                                 if any(r.get("n_components") is not None for r in ok) else None),
            "floating_frac_mean": _nanmean([r.get("floating_frac") for r in ok]),
            "floating_frac_max": (max(r["floating_frac"] for r in ok
                                      if _finite(r.get("floating_frac")))
                                  if any(_finite(r.get("floating_frac")) for r in ok) else None),
            "loads_solid_min": (min(r["loads_solid"] for r in ok if _finite(r.get("loads_solid")))
                                if any(_finite(r.get("loads_solid")) for r in ok) else None),
            "scan": rs[0].get("scan") if rs else None})
    for (inst, p), rs in sorted(_group([r for r in records if r.get("kind") == "qaoa_scan"
                                        and not r.get("error")],
                                       ["instance", "p"]).items(),
                                key=lambda kv: (str(kv[0][0]), kv[0][1])):
        m = lambda k: _nanmean([float(r[k]) for r in rs if r.get(k) is not None])  # noqa
        rows.append({"study": "F2", "kind": "qaoa_scan", "problem": inst,
                     "label": f"QAOA p={p}", "p": p, "runs": len(rs), "errors": 0,
                     "approx_ratio_mean": m("approx_ratio"), "p_opt_mean": m("p_opt"),
                     "p_opt_shots_mean": m("p_opt_shots"),
                     "best_shot_ratio_mean": m("best_shot_ratio"),
                     "best_shot_optimal_mean": m("best_shot_optimal"),
                     "polished_ratio_mean": m("polished_ratio"),
                     "polished_optimal_mean": m("polished_optimal"),
                     "uniform_ratio_mean": m("uniform_ratio"),
                     "uniform_p_opt_mean": m("uniform_p_opt"),
                     "greedy16_optimal_mean": m("greedy16_optimal"),
                     "greedy16_hit_fraction_mean": m("greedy16_hit_fraction"),
                     "wall_mean": m("wall_time"), "n": rs[0]["n"]})
    return rows


CSV_COLS = ["run_id", "study", "kind", "problem", "label", "role", "method", "backend",
            "seed", "scan", "scan_value", "compliance", "volume_fraction",
            "crisp_compliance", "crisp_volfrac", "volfrac_target", "c_ref", "gap",
            "refined_compliance", "refined_volfrac", "refined_threshold", "c_ref_refined",
            "gap_refined", "diverged", "feasible", "mma_native_at_V", "compliance_at_beta",
            "volfrac_at_beta", "compliance_returned_design", "beta_final",
            "returned_design", "best_iteration", "n_rejects", "iterations", "fe_solves",
            "qubo_solves", "mean_qubo_size", "max_qubo_size", "flips_mean",
            "repair_added_mean", "wall_time", "solver_time", "qpu_access_time",
            "update_time_per_iter", "hessian_time_per_iter", "fe_time_per_iter",
            "approx_ratio", "p_opt", "p_opt_shots", "best_shot_ratio", "polished_ratio",
            "uniform_ratio", "greedy16_hit_fraction", "exact_match", "exact_match_raw", "n",
            "p", "instance", "mesh_control", "nnele", "audit_ok", "audit_physics_ok",
            "audit_failed", "n_components", "floating_frac", "loads_solid", "load_on_main",
            "repair_added", "repair_removed",
            "supports_solid", "binary_n_components", "binary_floating_frac",
            "mma_constraint", "mma_fval_final", "init_perturb", "init_seed", "fields_file",
            "error"]


def _fmt(v, pct=False, nd=4):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not math.isfinite(v):
            return "n/a" if pct else "∞"
        if pct:
            return f"{100 * v:+.1f} %"
        return f"{v:.{nd}g}"
    return str(v)


def _pm(m, sd, nd=4):
    if m is None:
        return "–"
    return _fmt(m, nd=nd) + (f" ± {sd:.2g}" if sd is not None else "")


def _write_outputs(results, out_dir):
    pj = os.path.join(out_dir, "results.json")
    with open(pj, "w", encoding="utf-8") as f:
        json.dump(_clean(results), f, indent=1, default=_json_default)
    pc = os.path.join(out_dir, "results.csv")
    with open(pc, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in results["records"]:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in CSV_COLS])
    pm = os.path.join(out_dir, "summary.md")
    with open(pm, "w", encoding="utf-8") as f:
        f.write(summary_markdown(results))
    return {"json": pj, "csv": pc, "summary_md": pm}


def _verdicts(rows):
    """One honest sentence per (problem, mesh, QUBO method): compare the
    seed-mean crisp compliance with MMA's at the same volume."""
    out = []
    base = {(r["study"], r["problem"], r["mesh_control"]): r for r in rows
            if r["kind"] == "continuum" and r.get("role") == "baseline"}
    beso = {(r["study"], r["problem"], r["mesh_control"]): r for r in rows
            if r["kind"] == "continuum" and r.get("label") == "BESO-sort"}
    for r in rows:
        if r["kind"] != "continuum" or r.get("role") not in ("method",):
            continue
        b = base.get((r["study"], r["problem"], r["mesh_control"]))
        if not b or b.get("crisp_mean") is None:
            continue
        nd, n = r["n_diverged"], r["runs"] - r["errors"]
        head = (f"* **{r['problem']} MC{r['mesh_control']} ({r.get('nnele')} el.), "
                f"{r['label']}** ({n} seed(s), V* = {_fmt(r['volfrac_target'], nd=3)}): ")
        if r.get("crisp_mean") is None:
            out.append(head + f"all runs diverged (crisp c > {DIVERGED:g}x MMA).")
            continue
        m, cm = r["crisp_mean"], b["crisp_mean"]
        lo, hi = r.get("crisp_min"), r.get("crisp_max")
        rel = m / cm - 1.0
        if r.get("crisp_sd") is None:
            word = "single run, no seed spread -- not evidence either way"
        elif lo <= cm <= hi:
            word = "MMA lies within the QUBO seed spread: indistinguishable"
        else:
            side = "below" if hi < cm else "above"
            r_lo, r_hi = lo / cm - 1.0, hi / cm - 1.0
            word = (f"MMA lies outside the QUBO seed spread (all seeds {side} MMA, "
                    f"{r_lo * 100:+.1f} … {r_hi * 100:+.1f} %)")
            if abs(rel) < PROXY_BIAS:
                word += ("; the difference is smaller than the crisp proxy's bias "
                         "(≈ 1–4 points in favour of binary designs), so "
                         "indistinguishable from MMA")
        s = (head + f"crisp c = {_pm(m, r.get('crisp_sd'))} vs MMA {_fmt(cm)} "
             f"({rel * 100:+.1f} %): {word}")
        if r.get("gap_refined_mean") is not None:
            s += (f"; refined-voxel FE (2x): gap {r['gap_refined_mean'] * 100:+.1f} %"
                  + (f" ± {100 * r['gap_refined_sd']:.1f}" if r.get("gap_refined_sd")
                     is not None else ""))
            if b.get("mma_spread_refined_sd") is not None and b.get("refined_mean"):
                s += (f" (MMA from perturbed starts: sd "
                      f"{100 * b['mma_spread_refined_sd'] / b['refined_mean']:.1f} %)")
        if nd:
            s += f"; {nd} of {n} runs diverged and are excluded from the mean"
        bs = beso.get((r["study"], r["problem"], r["mesh_control"]))
        if bs and bs.get("crisp_mean") is not None:
            s += (f"; BESO-sort control (no QUBO) {_fmt(bs['crisp_mean'])} "
                  f"({(bs['crisp_mean'] / cm - 1) * 100:+.1f} %)")
        tag = _validity(r)
        if tag:
            s += f"; **{tag}**"
        elif r.get("audit_n"):
            s += f"; physics check passed ({r['audit_pass']}/{r['audit_n']} runs)"
        if r.get("gap_native_eqV") is not None:
            s += (f". Native: {_pm(r['native_mean'], r.get('native_sd'))} @ V = "
                  f"{_fmt(r['native_V_mean'], nd=3)} vs MMA interpolated at that V "
                  f"{_fmt(r['mma_native_at_V'])} ({r['gap_native_eqV'] * 100:+.1f} %)")
        out.append(s + ".")
    return out


def _truss_design_notes(records):
    """Per exact-reference truss: which distinct feasible designs (bar sets) the
    methods reached -- equal compliance does not mean the same design."""
    out = []
    tr = [r for r in records if r.get("kind") == "truss" and r.get("study") == "S1"
          and not r.get("error") and r.get("method") != "exact"]
    for pb in dict.fromkeys(r["problem"] for r in tr):
        g = [r for r in tr if r["problem"] == pb]
        feas = [r for r in g if r.get("feasible")]
        des = {}
        for r in feas:
            k = (round(float(r["compliance"]), 6), tuple(r.get("bars_on") or []))
            des.setdefault(k, set()).add(r["label"])
        inf = {}
        for r in g:
            if not r.get("feasible"):
                inf.setdefault(r["label"], [0, 0])[0] += 1
            inf.setdefault(r["label"], [0, 0])[1] += 1
        bad = [f"{lab} {v[0]}/{v[1]}" for lab, v in inf.items() if v[0]]
        txt = "; ".join(f"c = {k[0]:.6g} bars {list(k[1])} ({', '.join(sorted(l))})"
                        for k, l in sorted(des.items()))
        out.append(f"* **{pb}**: {len(des)} distinct feasible design(s): {txt or 'none'}."
                   + (f" Infeasible runs: {', '.join(bad)}." if bad else ""))
    if out:
        out = ["Designs reached (S1; equal compliance does not imply the same bar set):",
               ""] + out
    return out


def _validity(r):
    """'physically invalid' / 'volume off target' tag of a summary row."""
    if r.get("audit_physics_fail"):
        return (f"physically invalid ({r['audit_physics_fail']}/{r['audit_n']} runs: "
                f"{', '.join(c for c in r.get('audit_failed', []) if c != 'volume_target')})")
    if r.get("audit_volume_only_fail"):
        return f"volume off target ({r['audit_volume_only_fail']}/{r['audit_n']} runs)"
    return ""


def _audit_cell(r):
    if not r.get("audit_n"):
        return "–"
    return f"{r['audit_pass']}/{r['audit_n']}"


def summary_markdown(results):
    rows = results["summary"]
    L = [f"# FreeTO QUBO study — suite `{results['suite']}`", "",
         DISCLAIMER, "",
         f"Created {results['created']} on {results['machine']['platform']} "
         f"({results['machine']['cpus']} CPUs), total {results['elapsed']:.0f} s.", ""]
    if results.get("note"):
        L += [f"*Note: {results['note']}*", ""]
    co = [r for r in rows if r["kind"] == "continuum"]
    if co:
        L += ["## Continuum: common crisp evaluation at equal volume (S3 headline, S3q "
              "QAOA controls, S4 robustness, F5 γ scan)", "", PROTOCOL, ""]
        vs = _verdicts(co)
        if vs:
            L += ["**Verdicts (generated from the table below; a method is only called "
                  "better than MMA if its seed-mean crisp compliance plus one sd is "
                  "below MMA's at the same volume):**", ""] + vs + [""]
        L += ["| study | problem (elements) | method | seeds | crisp c @ V* | gap vs MMA "
              "(crisp) | refined c (2x voxels) | gap vs MMA (refined) | native c | native V | "
              "MMA @ native V | audit | comps | float % | "
              "loads solid | β_final | iters | rejects | flips/it | repair+/it | wall s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
              "---|---|"]
        for r in co:
            seeds = (",".join(str(s) for s in r["seeds"]) if r.get("role") in
                     ("method", "baseline_spread")
                     or r.get("role") == "control" and r["runs"] > 1 else "det.")
            if r["n_diverged"]:
                seeds += f" ({r['n_diverged']} diverged)"
            gap = ("n/a (diverged)" if r.get("crisp_mean") is None and r["n_diverged"]
                   else (_fmt(r["gap_mean"], True) + (f" ± {100 * r['gap_sd']:.1f}"
                                                      if r.get("gap_sd") is not None
                                                      else "")))
            tag = _validity(r)
            lab = r["label"] + (f" — **{tag}**" if tag else "")
            ffm = r.get("floating_frac_max")
            gap_r = (_fmt(r.get("gap_refined_mean"), True)
                     + (f" ± {100 * r['gap_refined_sd']:.1f}"
                        if r.get("gap_refined_sd") is not None else ""))
            L.append(
                f"| {r['study']} | {r['problem']} MC{r['mesh_control']} ({r.get('nnele')}) | "
                f"{lab} | {seeds} | {_pm(r['crisp_mean'], r['crisp_sd'])} @ "
                f"{_fmt(r['volfrac_target'], nd=3)} | {gap} | "
                f"{_pm(r.get('refined_mean'), r.get('refined_sd'))} | {gap_r} | "
                f"{_pm(r['native_mean'], r['native_sd'])} | "
                f"{_pm(r['native_V_mean'], r['native_V_sd'], nd=3)} | "
                f"{_fmt(r['mma_native_at_V'])} | {_audit_cell(r)} | "
                f"{_fmt(r.get('n_components_max'))} | "
                f"{'–' if ffm is None else f'{100 * ffm:.1f}'} | "
                f"{_fmt(r.get('loads_solid_min'), nd=3)} | "
                f"{', '.join(_fmt(b, nd=3) for b in r['beta_final']) or '–'} | "
                f"{_fmt(r['iterations_mean'], nd=3)} | {_fmt(r['rejects_mean'], nd=2)} | "
                f"{_fmt(r.get('flips_mean'), nd=3)} | {_fmt(r.get('repair_added_mean'), nd=2)} | "
                f"{_fmt(r['wall_mean'], nd=3)} |")
        L += ["", "audit = runs passing every check of freeto.audit / audited runs; comps = "
              "largest number of face-connected crisp components over the seeds; float % = "
              "largest crisp volume share outside the supported main component; loads solid = "
              "smallest share of |F| on nodes next to crisp material.",
              "", "flips/it = mean Hamming distance between consecutive binary designs; "
              "repair+/it = mean number of elements per iteration added by the greedy "
              "volume repair (fills to V_k even when the model energy rises), i.e. the part "
              "of each step that the QUBO solution itself did not choose.", ""]
        sp = [r for r in co if r.get("role") == "baseline_spread"]
        if sp:
            base = {(r["study"], r["problem"], r["mesh_control"]): r for r in co
                    if r.get("role") == "baseline"}
            L += ["MMA spread (MMA from 5 perturbed initial designs, x0 = V + U(−0.05, "
                  "0.05), same options; role baseline_spread): the seed-to-seed variation of "
                  "the deterministic baseline, to compare with the QUBO seed spread and gaps.",
                  "", "| study | problem | MMA crisp c | perturbed MMA crisp c (mean ± sd) | "
                  "sd / mean | MMA refined c | perturbed MMA refined c (mean ± sd) | "
                  "sd / mean |", "|---|---|---|---|---|---|---|---|"]
            for r in sp:
                b = base.get((r["study"], r["problem"], r["mesh_control"])) or {}
                cv = (r["crisp_sd"] / r["crisp_mean"]) if (r.get("crisp_sd") is not None
                                                           and r.get("crisp_mean")) else None
                rv = (r["refined_sd"] / r["refined_mean"]) if (
                    r.get("refined_sd") is not None and r.get("refined_mean")) else None
                L.append(f"| {r['study']} | {r['problem']} MC{r['mesh_control']} | "
                         f"{_fmt(b.get('crisp_mean'))} | {_pm(r['crisp_mean'], r['crisp_sd'])} | "
                         f"{'–' if cv is None else f'{100 * cv:.1f} %'} | "
                         f"{_fmt(b.get('refined_mean'))} | "
                         f"{_pm(r.get('refined_mean'), r.get('refined_sd'))} | "
                         f"{'–' if rv is None else f'{100 * rv:.1f} %'} |")
            L.append("")
        qc = [r for r in co if r["study"] == "S3q" and r.get("exact_match_mean") is not None]
        if qc:
            L += ["QAOA controls (S3q, same block-penalty model, 10-variable blocks, "
                  "verify_exact): fraction of blocks where the solver output is the exact "
                  "block optimum.", "",
                  "| method | exact-match (returned) | exact-match (raw best shot) | "
                  "approx ratio |", "|---|---|---|---|"]
            for r in qc:
                L.append(f"| {r['label']} | {_fmt(r['exact_match_mean'], nd=3)} | "
                         f"{_fmt(r['exact_match_raw_mean'], nd=3)} | "
                         f"{_fmt(r['approx_ratio_mean'], nd=3)} |")
            L.append("")
    tr = [r for r in rows if r["kind"] == "truss"]
    if tr:
        L += ["## Trusses (S1 exactness, S2 scale, F5 penalty)", "",
              "gap = c / c_exact − 1 (≤ 22 bars; exact = enumeration with the kinematic "
              "stability test) or vs the best feasible run of the study (S2); feasible = "
              "kinematically stable and volume ≤ V_max; 'n/a' = no feasible run; 'hits "
              f"(gap ≤ {HIT_TOL:g})' = feasible runs within {HIT_TOL:g} of the reference "
              "(designs that differ only in zero-force bars differ by ~1e-9 through the "
              "1e-9 void area). QUBO-qaoa '(raw)' uses the best measured shot of "
              "each block, '(+greedy)' polishes it by greedy descent (classical); "
              "QUBO-greedy is the greedy-only baseline in the same loop. repair +/− = "
              "bars added / removed per iteration by the kinematic repair (suite quick2; "
              "'–' = repair off).", "",
              f"| study | problem | method | runs | feasible | hits (gap ≤ {HIT_TOL:g}) | "
              "gap mean ± sd | gap min | gap max | repair +/− per it. | wall s | solver s |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in tr:
            gm = (_fmt(r["gap_mean"], True) + (f" ± {100 * r['gap_sd']:.1f}"
                                               if r.get("gap_sd") is not None else "")
                  if r["gap_mean"] is not None else "n/a")
            if r["n_infeasible"]:
                gm += f" ({r['n_infeasible']} infeasible)"
            L.append(f"| {r['study']} | {r['problem']} | {r['label']} | {r['runs']} | "
                     f"{r['feasible']} | {r['n_exact_hits']} | {gm} | "
                     f"{_fmt(r['gap_min'], True)} | {_fmt(r['gap_max'], True)} | "
                     + ("–" if r.get("repair_added_mean") is None else
                        f"{r['repair_added_mean']:.2f} / {r['repair_removed_mean']:.2f}")
                     + f" | {_fmt(r['wall_mean'], nd=3)} | {_fmt(r['solver_mean'], nd=3)} |")
        L.append("")
        L += _truss_design_notes(results["records"]) + [""]
    qa = [r for r in rows if r["kind"] == "qaoa_scan"]
    if qa:
        L += ["## QAOA state-vector simulation (F2) with baselines", "",
              "approx ratio = (E_max − ⟨E⟩)/(E_max − E_min) of the final state; p = 0 "
              "baseline = uniform superposition (same formula, ⟨E⟩ = mean of the spectrum); "
              "P(opt) = probability of the optimal bitstring(s), uniform P(opt) = fraction "
              "of optimal bitstrings; best-shot ratio = (E_max − E_best of 1000 shots)/span "
              "(raw), polished = after greedy descent from the best shot; greedy-16 = 16 "
              "greedy-descent restarts from random states without QAOA (fraction of "
              "restarts that end at the optimum). COBYLA with maxiter = 100 p; solver "
              "time = whole variational loop on this CPU.", "",
              "| instance | n | p | approx ratio (uniform) | P(opt) (uniform) | best shot "
              "opt. raw | after polish | greedy-16 hit frac. | wall s |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in qa:
            L.append(f"| {r['problem']} | {r['n']} | {r['p']} | "
                     f"{_fmt(r['approx_ratio_mean'], nd=3)} "
                     f"({_fmt(r['uniform_ratio_mean'], nd=3)}) | "
                     f"{_fmt(r['p_opt_mean'], nd=3)} ({_fmt(r['uniform_p_opt_mean'], nd=2)}) | "
                     f"{_fmt(r['best_shot_optimal_mean'], nd=2)} | "
                     f"{_fmt(r['polished_optimal_mean'], nd=2)} | "
                     f"{_fmt(r['greedy16_hit_fraction_mean'], nd=2)} | "
                     f"{_fmt(r['wall_mean'], nd=3)} |")
        L.append("")
    errs = [r for r in results["records"] if r.get("error")]
    if errs:
        L += ["## Errors", ""] + [f"* `{r['run_id']}`: {r['error']}" for r in errs] + [""]
    ev = [r for r in results["records"] if r.get("eval_errors")]
    if ev:
        L += ["## Failed post-run evaluations (run kept, metric = ∞)", ""] + \
            [f"* `{r['run_id']}`: {r['eval_errors']}" for r in ev] + [""]
    if results.get("figures"):
        L += ["## Figures", ""] + [f"![{os.path.basename(p)}]({os.path.basename(p)})"
                                   for p in results["figures"]] + [""]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2.0})
    return plt


def make_figures(records, surfaces, out_dir, log=print):
    os.makedirs(out_dir, exist_ok=True)
    plt = _mpl()
    ok = [r for r in records if not r.get("error")]
    figs = []
    for fn in (_fig_truss_gap, _fig_qaoa, _fig_history, _fig_topologies, _fig_penalty,
               _fig_timing, _fig_equal_volume):
        try:
            p = fn(plt, ok, surfaces, out_dir)
            if p:
                figs.append(p)
        except Exception as e:  # noqa: BLE001
            log(f"figure {fn.__name__} failed: {type(e).__name__}: {e}")
        plt.close("all")
    return figs


def _label_colors(labels):
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(labels)}


def _fig_truss_gap(plt, recs, surfaces, out_dir):
    rs = [r for r in recs if r["kind"] == "truss" and r["study"] in ("S1", "S2")
          and r.get("method") != "exact"]
    if not rs:
        return None
    probs = list(dict.fromkeys(r["problem"] for r in rs))
    labels = list(dict.fromkeys(r["label"] for r in rs))
    cap = 2.0
    fig, axes = plt.subplots(len(probs), 1, figsize=(10, 2.5 * len(probs) + 0.8),
                             squeeze=False)
    for ax, pb in zip(axes[:, 0], probs):
        prs = [r for r in rs if r["problem"] == pb]
        labs = [l for l in labels if any(r["label"] == l for r in prs)]
        for i, lab in enumerate(labs):
            g = [r for r in prs if r["label"] == lab]
            col = PALETTE[0]
            if lab.startswith("QUBO-qaoa"):
                col = PALETTE[1] if "raw" in lab else PALETTE[3]
            elif lab.startswith(("sort", "OC")):
                col = PALETTE[6]
            elif lab == "QUBO-greedy":
                col = PALETTE[2]
            fin = [min(r["gap"], cap) for r in g if r.get("feasible") and _finite(r.get("gap"))]
            if fin:          # bar = mean over the feasible seeds only
                ax.bar(i, float(np.mean(fin)) * 100, width=0.62, color=col, alpha=0.35,
                       edgecolor="none")
            jit = (np.arange(len(g)) - (len(g) - 1) / 2) * 0.06
            for j, r in enumerate(g):
                if r.get("feasible") and _finite(r.get("gap")):
                    ax.scatter(i + jit[j], min(r["gap"], cap) * 100, s=16, color=col,
                               zorder=3, edgecolor=SURFACE, linewidth=0.8)
                else:
                    ax.scatter(i + jit[j], cap * 100, s=28, marker="x", color=INK, zorder=3,
                               linewidth=1.2)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=7)
        ax.grid(axis="x", visible=False)
        exact = pb in ("ten_bar", "gs_3x2", "gs_4x2", "tower3d")
        ax.set_ylabel("gap [%]")
        ax.set_ylim(-5, cap * 100 + 15)
        ax.set_title(f"{pb}: gap to {'exact optimum (kinematically stable)' if exact else 'best feasible run'}"
                     f"  (dots = seeds, × = infeasible / mechanism, capped at {cap * 100:.0f} %)",
                     loc="left", fontsize=9)
    fig.suptitle("F1  Truss ground structures — binary design quality per problem", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    p = os.path.join(out_dir, "F1_truss_gap.png")
    fig.savefig(p, dpi=130)
    return p


def _fig_qaoa(plt, recs, surfaces, out_dir):
    rs = [r for r in recs if r["kind"] == "qaoa_scan"]
    if not rs:
        return None
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.0))
    groups = _group(rs, ["instance"])
    insts = sorted(groups, key=lambda k: (groups[k][0]["n"], str(k[0])))
    names = []
    for i, key in enumerate(insts):
        g = groups[key]
        ps = sorted(set(r["p"] for r in g))
        ar = [np.mean([r["approx_ratio"] for r in g if r["p"] == p]) for p in ps]
        po = [np.mean([r["p_opt"] for r in g if r["p"] == p]) for p in ps]
        u_ar = float(np.mean([r["uniform_ratio"] for r in g]))
        u_po = float(np.mean([r["uniform_p_opt"] for r in g]))
        col = PALETTE[i % len(PALETTE)]
        mk = "o" if "block" not in key[0] else "s"
        a1.plot([0] + ps, [u_ar] + ar, color=col, marker=mk, markersize=5, label=key[0])
        a2.plot([0] + ps, [u_po] + po, color=col, marker=mk, markersize=5, label=key[0])
        names.append(key[0])
        raw = float(np.mean([r["best_shot_optimal"] for r in g]))
        pol = float(np.mean([r["polished_optimal"] for r in g]))
        gr = float(np.mean([r["greedy16_hit_fraction"] for r in g]))
        w = 0.26
        a3.bar(i - w, raw, w, color=PALETTE[1], label="QAOA best shot (raw)" if i == 0 else None)
        a3.bar(i, pol, w, color=PALETTE[3], label="QAOA + greedy polish" if i == 0 else None)
        a3.bar(i + w, gr, w, color=PALETTE[2],
               label="greedy only (fraction of 16 restarts)" if i == 0 else None)
    allp = sorted({0} | {r["p"] for r in rs})
    for a in (a1, a2):
        a.set_xlabel("QAOA depth p (p = 0: uniform superposition baseline)")
        a.set_xticks(allp)
    a1.set_ylabel("approximation ratio of ⟨E⟩")
    a1.set_title("Approximation ratio vs depth", loc="left")
    a2.set_ylabel("P(optimal bitstring)")
    a2.set_yscale("log")
    a2.set_title("Ground-state probability vs depth", loc="left")
    a1.legend(fontsize=6, loc="lower right")
    a3.set_xticks(range(len(names)))
    a3.set_xticklabels([n.replace(" block", "\nblock") for n in names], fontsize=6,
                       rotation=25, ha="right")
    a3.set_ylabel("fraction optimal (mean over p, instances)")
    a3.set_ylim(0, 1.05)
    a3.set_title("Exact optimum found: raw vs polished vs greedy", loc="left")
    a3.legend(fontsize=6, loc="lower left")
    fig.suptitle("F2  QAOA (numpy state-vector simulation, COBYLA 100·p evaluations, 1000 "
                 "shots) against the uniform and greedy baselines", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "F2_qaoa.png")
    fig.savefig(p, dpi=130)
    return p


def _main_problems(recs):
    return list(dict.fromkeys((r["problem"], r["mesh_control"], r["study"]) for r in recs
                              if r["kind"] == "continuum" and r["study"] in ("S3", "S3q", "S4")))


def _fig_history(plt, recs, surfaces, out_dir):
    rs = [r for r in recs if r["kind"] == "continuum" and r["study"] in ("S3", "S3q", "S4")
          and r.get("history") and r.get("role") not in ("curve", "baseline_spread")]
    if not rs:
        return None
    keys = _main_problems(rs)
    ncol = min(3, len(keys))
    nrow = int(math.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.6 * nrow), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        g = [r for r in rs if (r["problem"], r["mesh_control"], r["study"]) == key]
        labs = list(dict.fromkeys(r["label"] for r in g))
        cols = _label_colors(labs)
        for lab in labs:
            r0 = sorted([r for r in g if r["label"] == lab], key=lambda r: r["seed"])[0]
            h = np.asarray(r0["history"], dtype=float)
            ax.plot(np.arange(1, h.size + 1), h, color=cols[lab], label=lab, linewidth=1.4)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("compliance of the iterate [N·mm]")
        ax.set_title(f"{key[2]} {key[0]} MC{key[1]} ({g[0].get('nnele')} el.)", loc="left",
                     fontsize=9)
        ax.legend(fontsize=6)
    for ax in axes.ravel()[len(keys):]:
        ax.set_axis_off()
    fig.suptitle("F3  Continuum compliance histories per problem (lowest seed of each method; "
                 "QUBO/BESO start solid and remove 5 %/iteration; QUBO histories show "
                 "accepted designs only — guard rejections are restored)", x=0.01, ha="left",
                 fontweight="bold", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(out_dir, "F3_continuum_history.png")
    fig.savefig(p, dpi=120)
    return p


def _render(ax, v, f, color=PALETTE[0]):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as mcolors
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    light = np.array([0.4, 0.8, 0.45])
    light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(n @ light, 0, 1)
    base = np.array(mcolors.to_rgb(color))
    fc = np.clip(base[None, :] * shade[:, None] + (1 - shade[:, None]) * 0.12, 0, 1)
    pc = Poly3DCollection(tri, facecolors=fc, edgecolors="none", linewidths=0)
    ax.add_collection3d(pc)
    lo, hi = v.min(axis=0), v.max(axis=0)
    ext = np.maximum(hi - lo, 1e-9)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(tuple(ext / ext.max()), zoom=1.0)
    ax.view_init(elev=24, azim=-62, vertical_axis="y")
    ax.set_axis_off()


def _fig_topologies(plt, recs, surfaces, out_dir):
    rs = [r for r in recs if r["kind"] == "continuum" and r["run_id"] in surfaces
          and r["study"] in ("S3", "S4") and r.get("role") not in ("curve", "baseline_spread")]
    if not rs:
        return None
    seen, sel = set(), []
    for r in sorted(rs, key=lambda r: r["seed"]):
        k = (r["problem"], r["mesh_control"], r["label"])
        if k not in seen:
            seen.add(k)
            sel.append(r)
    order = {k: i for i, k in enumerate(_main_problems(rs))}
    sel.sort(key=lambda r: order.get((r["problem"], r["mesh_control"], r["study"]), 99))
    sel = sel[:25]
    ncol = min(5, len(sel))
    nrow = int(math.ceil(len(sel) / ncol))
    fig = plt.figure(figsize=(3.3 * ncol, 3.0 * nrow + 0.5))
    for i, r in enumerate(sel):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        v, f = surfaces[r["run_id"]]
        if f.shape[0] > 60000:
            f = f[:: int(math.ceil(f.shape[0] / 60000))]
        if f.shape[0]:
            _render(ax, v, f)
        ax.set_title(f"{r['problem']} MC{r['mesh_control']}\n{r['label']} (seed {r['seed']})\n"
                     f"crisp c = {_fmt(r.get('crisp_compliance'))}; "
                     f"β_final = {_fmt(r.get('beta_final'), nd=3)}"
                     + ("" if r.get("audit_ok") is None else
                        ("; audit pass" if r.get("audit_ok") else
                         "; audit FAIL: " + (r.get("audit_failed") or ""))), fontsize=7,
                     color=(INK if r.get("audit_physics_ok") is not False else "#b00020"))
    fig.suptitle("F4  Final topologies (FreeTO surfaces of the returned designs at each "
                 "method's own β; lowest seed; the crisp c of the title is the common "
                 "evaluation at V*; OC with the FreeTO defaults stops grey (β_final ≤ 2), so "
                 "its iso-surface is not a converged topology)",
                 x=0.01, ha="left", fontweight="bold", fontsize=8)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0 - 0.95 / (3.0 * nrow + 0.5),
                        wspace=0.02, hspace=0.35)
    p = os.path.join(out_dir, "F4_topologies.png")
    fig.savefig(p, dpi=110)
    return p


def _fig_penalty(plt, recs, surfaces, out_dir):
    tr = [r for r in recs if r.get("scan") == "lambda_q"]
    ga = [r for r in recs if r.get("scan") == "gamma"]
    if not tr and not ga:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    cap = 2.0
    for i, (pb,), in enumerate([(k[0],) for k in _group(tr, ["problem"])]):
        g = [r for r in tr if r["problem"] == pb]
        xs = sorted(set(r["scan_value"] for r in g))
        col = PALETTE[i % len(PALETTE)]
        means = []
        for x in xs:
            gg = [min(r["gap"], cap) if (r.get("feasible") and _finite(r.get("gap"))) else cap
                  for r in g if r["scan_value"] == x]
            ax.scatter([x] * len(gg), np.array(gg) * 100, s=14, color=col, alpha=0.6,
                       edgecolor="none")
            means.append(np.mean(gg) * 100 if gg else np.nan)
        ax.plot(xs, means, color=col, marker="o", markersize=5, label=pb)
    ax.set_xscale("log")
    ax.set_xlabel("λ_q / default (default = 2 max|g| / L_max²)")
    ax.set_ylabel("gap to exact [%] (infeasible plotted at 200)")
    ax.set_title("Truss, volume='penalty' (QUBO-sa)", loc="left")
    if tr:
        ax.legend(fontsize=7)
    ax = axes[1]
    if ga:
        g = sorted(ga, key=lambda r: r["scan_value"])
        xs = sorted(set(r["scan_value"] for r in g))
        for j, key in enumerate(("crisp_compliance", "compliance")):
            ys = [_nanmean([r.get(key) for r in g if r["scan_value"] == x]) for x in xs]
            ax.plot(xs, ys, color=PALETTE[j], marker="o", markersize=5,
                    label={"crisp_compliance": "crisp c at V*",
                           "compliance": "native c"}[key])
        ax.set_yscale("log")
        ax.set_xlabel("perimeter weight γ (× mean solid |g|)")
        ax.set_ylabel("compliance [N·mm]")
        ax.set_title(f"Continuum {g[0]['problem']} MC{g[0]['mesh_control']} (QUBO-sa diag, "
                     f"seed {g[0]['seed']})", loc="left", fontsize=9)
        ax.legend(fontsize=7)
    else:
        ax.set_axis_off()
    fig.suptitle("F5  Penalty-weight sensitivity", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "F5_penalty.png")
    fig.savefig(p, dpi=130)
    return p


def _fig_timing(plt, recs, surfaces, out_dir):
    rs = [r for r in recs if r.get("kind") in ("truss", "continuum") and r.get("wall_time")
          and r.get("role") != "curve"]
    if not rs:
        return None
    keys = sorted(_group(rs, ["kind", "label"]),
                  key=lambda k: (k[0], str(k[1])))
    fig, ax = plt.subplots(figsize=(12, 4.2))
    for i, key in enumerate(keys):
        g = [r for r in rs if (r["kind"], r["label"]) == key]
        jit = (np.arange(len(g)) - (len(g) - 1) / 2) * 0.05
        col = PALETTE[0] if key[0] == "truss" else PALETTE[1]
        ax.scatter(i + jit, [r["wall_time"] for r in g], s=12, color=col, alpha=0.8,
                   edgecolor="none", label=(f"{key[0]} wall time per run" if i == 0 or
                                            key[0] != keys[i - 1][0] else None))
        st = [r.get("solver_time") for r in g if r.get("solver_time")]
        if st:
            ax.scatter(i + jit[:len(st)], st, s=12, marker="_", color=INK,
                       label="time inside the QUBO solver" if i == 0 else None)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k[1] for k in keys], fontsize=6, rotation=55, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("seconds per run (CPU, this machine)")
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=7, loc="upper left")
    ax.set_title("F6  Wall time per run (dots = runs of all problems/sizes; QPU access time "
                 "is n/a: no QPU was used)", loc="left")
    fig.tight_layout()
    p = os.path.join(out_dir, "F6_timing.png")
    fig.savefig(p, dpi=130)
    return p


def _fig_equal_volume(plt, recs, surfaces, out_dir):
    keys = [k for k in _main_problems(recs) if k[2] == "S3"]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(6.0 * len(keys), 4.2), squeeze=False)
    for ax, key in zip(axes[0], keys):
        g = [r for r in recs if r["kind"] == "continuum"
             and (r["problem"], r["mesh_control"], r["study"]) == key]
        nat, cr = _mma_curve(recs, key[0], key[1])
        if nat:
            ax.plot([p[0] for p in nat], [p[1] for p in nat], color=INK2, marker="o",
                    markersize=4, label="MMA native c(V) curve")
        if cr:
            ax.plot([p[0] for p in cr], [p[1] for p in cr], color=INK, marker="D",
                    markersize=4, linestyle="--", label="MMA crisp c(V*) curve")
        labs = [l for l in dict.fromkeys(r["label"] for r in g)
                if not l.startswith("MMA") and not l.startswith("OC")]
        cols = _label_colors(labs)
        for lab in labs:
            gg = [r for r in g if r["label"] == lab and not r.get("diverged")]
            if not gg:
                continue
            Vn = [r["volume_fraction"] for r in gg]
            cn = [r["compliance"] for r in gg]
            ax.scatter(Vn, cn, s=14, color=cols[lab], alpha=0.55, edgecolor="none")
            cc = [r["crisp_compliance"] for r in gg if _finite(r.get("crisp_compliance"))]
            Vc = [r["crisp_volfrac"] for r in gg if _finite(r.get("crisp_compliance"))]
            if cc:
                ax.errorbar(np.mean(Vc), np.mean(cc),
                            yerr=(np.std(cc, ddof=1) if len(cc) > 1 else None),
                            color=cols[lab], marker="D", markersize=6, capsize=3,
                            label=f"{lab}: crisp (mean ± sd, n={len(cc)}); dots = native")
        ax.set_yscale("log")
        ax.set_xlabel("volume fraction")
        ax.set_ylabel("compliance [N·mm]")
        ax.set_title(f"{key[0]} MC{key[1]} ({g[0].get('nnele')} el.)", loc="left")
        ax.legend(fontsize=6)
    fig.suptitle("F7  Equal-volume comparison: crisp evaluation at V* (diamonds) and native "
                 "(c, V) points against the MMA compliance-volume curves", x=0.01, ha="left",
                 fontweight="bold", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(out_dir, "F7_equal_volume.png")
    fig.savefig(p, dpi=130)
    return p


# ---------------------------------------------------------------------------
_DERIVED = ("gap", "c_ref", "diverged", "mma_native_at_V", "gap_refined", "c_ref_refined")


def reprocess(out_dir, log=print, threads=1, rerun_baselines=True):
    """Rebuild summary.md, results.json/csv and the figures of a finished study
    from its stored records, without re-running it.

    Records written before the native-volume fix (OC/MMA without
    ``volume_fraction_returned``) need the volume history of their last
    iterations; with ``rerun_baselines`` these cheap deterministic OC/MMA runs
    (seconds each) are re-run once to recover it (their compliance is checked
    against the stored value).  QUBO/truss/QAOA runs are never re-run.  F4
    needs the surfaces, which are not stored: an existing F4 file is kept."""
    log = log or (lambda s: None)
    pj = os.path.join(out_dir, "results.json")
    with open(pj, encoding="utf-8") as f:
        res = json.load(f)
    recs = res["records"]
    spec = res.get("spec") or {"runs": []}
    runs = {rid: (run, seed) for rid, run, seed in _expand(spec)}
    n_rerun = 0
    with _ThreadPin(threads):
        for r in recs:
            if (r.get("kind") == "continuum" and not r.get("error")
                    and r.get("method") in ("OC", "MMA")
                    and "volume_fraction_returned" not in r):
                if not rerun_baselines or r["run_id"] not in runs:
                    log(f"{r['run_id']}: no volume history (kept as is)")
                    continue
                run, seed = runs[r["run_id"]]
                new, _ = _run_continuum(run, seed, keep_surface=False)
                n_rerun += 1
                if not math.isclose(new["compliance"], r["compliance"], rel_tol=1e-6):
                    log(f"WARNING {r['run_id']}: re-run compliance {new['compliance']:.6g} "
                        f"!= stored {r['compliance']:.6g}")
                for k in ("volume_fraction", "volume_fraction_returned", "history_volfrac"):
                    r[k] = new[k]
                log(f"{r['run_id']}: native V {r['volume_fraction']:.4f} (returned field "
                    f"{r['volume_fraction_returned']:.4f})")
    for r in recs:
        for k in _DERIVED:
            r.pop(k, None)
    _postprocess(recs)
    res["summary"] = _summary(recs)
    note = (f"reprocessed {time.strftime('%Y-%m-%d %H:%M')} from stored records "
            f"(python -m freeto.study --reprocess); {n_rerun} OC/MMA baseline run(s) re-run "
            "for their volume history (native V = volume of the reported iterate)")
    res["note"] = (res.get("note") + " " if res.get("note") else "") + note + "."
    ok = [r for r in recs if not r.get("error")]
    figs = make_figures(ok, {}, out_dir, log=log)
    f4 = os.path.join(out_dir, "F4_topologies.png")
    if os.path.isfile(f4):
        figs.insert(min(3, len(figs)), f4)
    res["figures"] = figs
    res["files"] = _write_outputs(res, out_dir)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m freeto.study",
                                 description="QUBO / quantum design-update study runner "
                                             "(docs/QUANTUM_DESIGN.md §E)")
    ap.add_argument("--suite", default=None, choices=["smoke", "quick", "quick2", "full"],
                    help="built-in suite (default quick; with --resume: the stored spec)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the runs of the suite / spec (after --only) and exit")
    ap.add_argument("--spec", help="JSON study spec (overrides --suite)")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--only", help="comma-separated study ids to keep (e.g. S1,F2)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--threads", type=int, default=1,
                    help="BLAS/OpenMP/MKL threads (default 1: results depend on it)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes running the study's runs in parallel (default 1; "
                         "each worker uses --threads threads; records are identical to the "
                         "serial run except for timings)")
    ap.add_argument("--fresh-workers", action=argparse.BooleanOptionalAction, default=True,
                    help="run every run in a fresh worker process (default; also with "
                         "--jobs 1), so memory cannot grow across runs; --no-fresh-workers: "
                         "--jobs 1 runs in this process, --jobs N reuses the workers")
    ap.add_argument("--resume", metavar="DIR",
                    help="continue the study in DIR: keep the error-free runs of DIR/results.json "
                         "(partial or complete), run the missing / failed ones, write the "
                         "complete results to DIR (spec: --spec/--suite, else the stored one)")
    ap.add_argument("--reprocess", metavar="DIR",
                    help="rebuild summary/figures of a finished study in DIR from its "
                         "results.json (no new runs except cheap OC/MMA volume histories)")
    a = ap.parse_args(argv)
    logf = None if a.quiet else (lambda s: print(s, flush=True))
    if a.reprocess:
        res = reprocess(a.reprocess, log=logf, threads=a.threads)
        print(f"rewrote {res['files'].get('summary_md')} and {len(res['figures'])} figure(s)")
        return 0
    kept = None
    if a.spec:
        spec = _load_spec(a.spec)
    elif a.suite or not a.resume:
        spec = suite_spec(a.suite or "quick")
    else:
        with open(os.path.join(a.resume, "results.json"), encoding="utf-8") as f:
            spec = json.load(f).get("spec")
        if not spec:
            ap.error(f"{a.resume}/results.json has no stored spec; pass --suite or --spec")
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        spec = dict(spec, runs=[r for r in spec["runs"] if r.get("study") in keep])
    if a.resume and not a.dry_run:
        kept = resume_records(a.resume, spec, log=lambda s: print(s, flush=True))
        a.out = a.resume
    if a.dry_run:
        jobs = _expand(spec)
        for i, (rid, run, seed) in enumerate(jobs):
            extra = ""
            if run.get("kind") == "truss":
                extra = (f" method={run.get('method')} backend={run.get('backend')}"
                         f" options={json.dumps(run.get('options') or {}, sort_keys=True)}")
            if run.get("kind") == "continuum":
                extra = (f" opt={run.get('optimizer')} max_iter={run.get('max_iter')}"
                         f" options={json.dumps(run.get('options') or {}, sort_keys=True)}"
                         f" run_options={json.dumps(run.get('run_options') or {}, sort_keys=True)}")
            print(f"{i:4d} {rid} [{run.get('kind')}]{extra}")
        n_c = sum(1 for _, r, _ in jobs if r.get("kind") == "continuum")
        print(f"{len(jobs)} runs ({len(spec.get('runs', []))} run entries; {n_c} continuum, "
              f"{len(jobs) - n_c} truss / QAOA scan) in suite {spec.get('name', 'custom')}")
        return 0
    stop = threading.Event()
    try:
        res = run_study(spec, out_dir=a.out, figures=not a.no_figures, stop_event=stop,
                        log=logf, threads=a.threads, jobs=a.jobs,
                        fresh_workers=a.fresh_workers, resume_records=kept)
    except KeyboardInterrupt:
        stop.set()
        return 130
    print(f"wrote {res['files'].get('json')}, {res['files'].get('csv')}, "
          f"{res['files'].get('summary_md')} and {len(res['figures'])} figure(s) "
          f"in {res['elapsed']:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
