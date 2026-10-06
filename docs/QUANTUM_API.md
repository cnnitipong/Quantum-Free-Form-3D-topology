# QUANTUM_API — public interface of the QUBO / quantum extension

Status: **stable contract for the web UI** (written before the implementation;
the implementation follows it — additions only, no renames).  Design rationale:
`docs/QUANTUM_DESIGN.md`; engineering notes / deviations: `docs/NOTES_quantum.md`.

All quantum SDKs are optional.  Nothing below requires `dwave-*` or `qiskit*`
to be installed; cloud backends are never selected unless named explicitly.

**Changes in the fix round (2026-10, docs/VERIFICATION_quantum.md) — additions only,
no renames or removals:**
* `FreeTOConfig.eval_crisp` (new; common crisp evaluation → `extra["crisp_compliance"]`,
  `["crisp_volfrac"]`, `["crisp_target"]`, `["crisp_threshold"]`,
  `["compliance_returned_design"]`); `extra["beta_final"]` (all optimizers);
  `extra["eval_errors"]` (failed post-run evaluations no longer raise: metric = `inf`);
  QUBO mode: `extra["returned_design"]` (`"best"` | `"last"`).
* `QUBOOptions`: `qaoa_polish`, `move_limit`, `move_limit_min`, `protect_loads`, `guard`,
  `guard_tol`, `guard_tol_target`, `max_rejects` (defaults change the QUBO update's
  behaviour: load protection, adaptive move limit and accept-if-improves guard are ON).
* per-iteration `qubo` dict: `move_limit`, `flip_cap`, `n_proposed_flips`, `truncated`,
  `guard`, `n_protected`, `n_protected_added`, `exact_match_raw`, `best_shot_ratio`.
* `QUBOOptions.connectivity` (default True, changes the QUBO update: floating solids are
  removed / reconnected each iteration) and `QUBOOptions.diagnostics` (default False);
  per-iteration `qubo["connectivity_repair"]` / `qubo["connectivity"]`;
  `QUBOUpdater.components(x)` (face-connected components of a binary design).
* `qaoa` backend: `polish` option (default True); `info` always has `best_shot_energy`,
  `best_shot_x`, `best_shot_ratio`, `polished_energy`, `polished`, `polish_applied`,
  `polish_time`, `uniform_ratio`, `uniform_p_opt`.  Timing: `timing["solver"]` of all
  QAOA backends = whole variational loop incl. the classical angle optimisation
  (qiskit backends now include it too), polish excluded.
* `dwave_*` backends list `dimod` as a required module; broken optional SDKs are reported
  as unavailable instead of raising.
* Truss: kinematic stability test (`TrussFE.kinematic`, `stable_mask`, `prune`);
  `feasible` now means kinematically stable; pinned optima changed (T2s 9.9638, T2 23.8915,
  T3 7.8467 at the new 55 % volume); `solve_truss` options `prune` (True),
  `stability_load` (0), `qaoa_polish` (True); `nbits` < 1 raises `ValueError`.
* Study: records gain `crisp_compliance`, `crisp_volfrac`, `volfrac_target`, `beta_final`,
  `diverged`, `mma_native_at_V`, `role`, …; `gap` is `None` (not ∞) for infeasible /
  diverged runs, continuum `gap` is vs MMA's crisp compliance; new figure
  `F7_equal_volume.png`; `results.json` is strict JSON (no Infinity/NaN).
* Round 2: continuum `volume_fraction` of OC/MMA records = volume of the reported iterate
  (`hist["volfrac"][-2]`), new `volume_fraction_returned`, `history_volfrac`;
  `run_study(..., threads=1)` pins BLAS/OpenMP/MKL threads (recorded in
  `results["machine"]["threads"]`); `freeto.study.reprocess(out_dir)` / CLI
  `--reprocess DIR` and `--threads N`.

**v2 changes (2026-10-04, manuscript v2; docs/NOTES_quantum.md §9) — additions only; the
defaults of `run_freeto` are unchanged (compare_ref 6/6) except the truss benchmarks:**
* `FreeTOConfig` (all opt-in): `eval_refined: int | None = None` (refined binary voxel
  evaluation with f voxels per element edge, f ∈ {1, 2, 4} →
  `extra["refined_compliance"]`, `["refined_volfrac"]`, `["refined_threshold"]`,
  `["refined_time"]`, `["refined_f"]`, `["refined_target"]`, `["refined_residual"]`,
  `["refined_ndof_free"]`; failures → `inf` + `extra["eval_errors"]["eval_refined"]`);
  `mma_constraint: "projected"` (default, FreeTO) | `"filtered"` (MMA volume constraint on
  the pre-projection filtered field); `mma_feasible_stop: bool = False` (MMA stops on the
  change / topology tolerances only when |fval| ≤ 1e-3; `extra["mma_fval_final"]`);
  `init_perturb: float = 0.0`, `init_seed: int = 0` (OC/MMA start from
  volfrac + U(−a, a) per non-keep element, clipped to [0.001, 1]).  MMA runs report
  `extra["mma_constraint"]`.  Every run returns `extra["full_pre"]` (float32 copy of the
  evaluated pre-smoothing field, length nele).
* `freeto.evaluate.refined_voxel_compliance(full_pre, Hn, Hns, nelx, nely, nelz, ele,
  target, F, fixeddof, KE, E0, Emin, ngrid=4, f=2, solver="auto", steps=60,
  return_solid=False, bc_map="interp")` → dict `compliance, volfrac, threshold, n_voxels,
  ndof_free, residual, time` (+ `solid`).
* `QUBOOptions.hessian = "scalar"`: "diag" with all off-diagonal couplings of Q_c zeroed.
* Truss: every ground structure drops bars between two fully fixed nodes
  (`drop_fixed=True`): gs_3x2 13 → **12** bars, gs_4x2 22 → **21**, gs_9x3 118 → **116**
  (ten_bar, tower3d, column3d unchanged); re-pinned `C_EXACT`: gs_3x2 **11.6568542**
  (was 9.9637618), gs_4x2 23.8915297 (unchanged value, bar indices shifted by one).
* Study: suite `"quick2"`; run key `run_options` (FreeTOConfig fields `eval_refined`,
  `mma_constraint`, `mma_feasible_stop`, `init_perturb`, `init_seed`; with
  `init_perturb` > 0 and no `init_seed` the run seed is used); role `"baseline_spread"`;
  records gain `refined_compliance`, `refined_volfrac`, `refined_threshold`, `refined_f`,
  `refined_time`, `c_ref_refined`, `gap_refined`, `mma_constraint`, `mma_fval_final`,
  `init_perturb`, `init_seed`, `fields_file`; summary rows gain `refined_mean/sd`,
  `gap_refined_mean/sd`, `refined_V_mean`, `mma_spread_*`; with `out_dir` every continuum
  run writes `fields/<run_id>.npz`; CLI `--suite quick2`, `--dry-run`.
* P0b (2026-10-04): truss `solve_truss(..., kinematic_repair=False)` (QUBO / sort only;
  `freeto.truss.optimize.kinematic_repair(fe, x, V_target) -> (x, n_added, n_removed)`;
  per-iteration `qubo_stats` keys `kin_repair_added`, `kin_repair_removed`; study truss
  records `repair_added`, `repair_removed`, quick2 turns it on for every truss QUBO / sort
  run); `QUBOOptions.hessian_scale: float = 1.0` (scales Q_c, not the γ term; quick2
  control "QUBO-sa (block, Qx3)"); truss exact hits use `freeto.study.HIT_TOL = 1e-6`
  (summary rows `n_exact_hits`, `hit_tol`); `run_study(..., jobs=1)` / CLI `--jobs N`
  (process pool, records identical to the serial run except timings,
  `results["machine"]["jobs"]`), `results.json` rewritten every 5 finished runs with
  `"partial": true` until the study ends.
* `run_study(..., fresh_workers=False, resume_records=None)`, `freeto.study.resume_records(
  out_dir, spec) -> {run_id: [records]}`; CLI `--fresh-workers` (default) /
  `--no-fresh-workers` (one fresh worker process per run, `max_tasks_per_child=1`, also for
  `--jobs 1`) and `--resume DIR` (keep error-free runs, run missing / failed ones, write the
  complete results to DIR); final `results.json` has `"partial": false`.

---------------------------------------------------------------------------
## 1. Continuum optimizer `optimizer="QUBO"`

```python
from freeto import FreeTOConfig, run_freeto
from freeto.quantum import QUBOOptions

cfg = FreeTOConfig(..., optimizer="QUBO",
                   qubo=QUBOOptions(backend="sa", hessian="auto"))   # or a plain dict
res = run_freeto(cfg, callback=cb, stop_event=ev, log=print)
```

`FreeTOConfig` gains two fields (defaults keep every existing behaviour):

| field | type | default | meaning |
|---|---|---|---|
| `qubo` | `QUBOOptions \| dict \| None` | `None` | options for `optimizer="QUBO"` (None → all defaults); ignored for OC/MMA |
| `eval_binary` | `bool` | `False` | after the run, also compute the compliance of the *binary-thresholded* final design (keep the `volfrac·nnele` densest elements, void density 0.001, no smoothing) → `res.extra["binary_compliance"]`, `["binary_volfrac"]` |
| `eval_beta` | `float \| None` | `None` | after the run, re-smooth the final filtered design at Heaviside β = `eval_beta` and solve → `res.extra["compliance_at_beta"]`, `["volfrac_at_beta"]` |
| `eval_crisp` | `float \| True \| None` | `None` | common crisp evaluation (`freeto/evaluate.py`): the final pre-smoothing filtered field is interpolated to the 4× fine grid and projected to 0/1 with the threshold set (bisection) so that the element densities have volume fraction `eval_crisp` (True = `volfrac`); one FE solve → `res.extra["crisp_compliance"]`, `["crisp_volfrac"]`, `["crisp_target"]`, `["crisp_threshold"]`, plus `["compliance_returned_design"]` (FE solve on `res.eleden`) |
| `audit` | `bool` | `True` | physics / connectivity check of the returned design (`freeto.audit`, docs/AUDIT_API.md) → `res.extra["audit"]`; QUBO runs also get `info["audit_live"]` = {n_components, floating_frac, ungrounded_frac} of the binary design in every iteration callback; a failing audit never raises |

Post-run evaluations never discard a finished run: a solver failure is logged, the metric
is `inf` and the message is stored in `res.extra["eval_errors"]`.  `res.extra["beta_final"]`
= Heaviside β of the projection that produced the returned design (all optimizers).
`res.comp` semantics: OC/MMA — compliance computed in the last iteration (of the design
entering it; FreeTO/MATLAB convention); QUBO — one extra FE solve on the returned design.

`optimizer` accepts `"OC" | "MMA" | "QUBO"` (case-insensitive).
In QUBO mode the defaults of `tolx`, `tol_thresh`, `beta_init`, `beta_step`,
`beta_max` become `1e-3, 1e-3, 0.5, 0.5, 8.0` (explicit values still win).

### `QUBOOptions` (dataclass, `freeto.quantum.QUBOOptions`; every field optional)

| field | default | allowed values / meaning |
|---|---|---|
| `backend` | `"auto"` | any name from `available_backends()` (`"auto"` = `exact` for n ≤ 20 else `sa`; never a cloud backend) |
| `hessian` | `"auto"` | `"auto"` (→ `"block"` with a direct solver, `"diag"` with AMG), `"block"` (alias `"exact-block"`: exact Gram Hessian inside Morton patches, one multi-RHS solve with the reused factorisation), `"diag"` (element-local bound, no extra solve), `"scalar"` (v2: `"diag"` with every off-diagonal coupling of Q_c = J_Cᵀ diag(D/s) J_C zeroed — separable control), `"none"` (first order; with `gamma=0` this is the BESO/sorting control, solved exactly by sorting) |
| `interp` | `"beso"` | modulus model of the QUBO: `"beso"` (weights E′/p: exact secant expansion at solid elements, filter-driven additions — default), `"secant"` (E = Emin + ρ(E0−Emin) everywhere), `"simp"` (design-literal Taylor model in the SIMP density incl. the −E″c curvature; unstable for 0/1 moves, kept for the record) |
| `volume` | `"bisection"` | `"bisection"` (λ bisection, exact volume) or `"penalty"` (λ fixed + quadratic penalty λ_q(v·x−V)², for expensive backends) |
| `lambda_q` | `None` | penalty weight in units of max\|dc\| / v_max² (None → 2.0) |
| `gamma` | `0.0` | perimeter / anti-checkerboard weight in units of the mean \|g\| of the solid elements; 0 disables |
| `move_penalty` | `0.0` | Hamming move penalty μ |
| `frontier_fraction` | `0.25` | free-set size parameter f |
| `block_size` | `None` | None → auto by backend (qaoa/qiskit_aer/ibm 14, exact 20, dwave_qpu 80, others: one block) |
| `blocks` | `"morton"` | `"morton"` or `"rank"` |
| `sweeps` | `2` | block Gauss–Seidel sweeps |
| `init` | `"solid"` | `"solid"` (BESO-like volume schedule) or `"oc"` (n_warm OC iterations, then the thresholded design is returned once — its compliance is the next iteration's — and QUBO iterations follow) |
| `er` | `0.05` | evolutionary volume ratio per iteration (init="solid") |
| `n_warm` | `10` | OC warm-up iterations (init="oc") |
| `patience` | `8` | stop after this many iterations without improvement once the volume target is reached |
| `num_reads` | `None` | reads / replicas for sa, tabu, dwave (None → backend default) |
| `seed` | `None` | RNG seed (stochastic backends) |
| `qaoa_p` | `3` | QAOA depth |
| `qaoa_shots` | `1000` | samples drawn from the final state |
| `qaoa_init` | `"linear_ramp"` | `"linear_ramp"` or `"interp"` |
| `time_limit` | `None` | seconds, for `dwave_hybrid` (min 3 s) |
| `verify_exact` | `False` | re-solve every block with n ≤ 20 by brute force and record whether the backend found the exact block optimum |
| `hessian_scale` | `1.0` | v2: multiplies the curvature model Q_c (block and diag parts) before the standard form; the perimeter term γ is not scaled |
| `hessian_block_size` | `64` | patch size for the exact block Hessian when the backend takes the whole free set |
| `hessian_max_rhs` | `6000` | above this many right-hand sides (free elements × load cases) an iteration falls back to `"diag"` |
| `free_set` | `"band"` | free set = boundary band of the binary design (+ frontier) — `"grey"` = the smooth-edge grey band of the design doc |
| `history_average` | `True` | BESO sensitivity history averaging g_k ← (g_k + g_{k−1})/2 |
| `bisection_steps` | `14` | sequential λ-bisection steps (backends that cannot batch λ values) |
| `lambda_source` | `"sort"` | fixed λ for `volume="penalty"`: `"sort"` (knapsack threshold of the first-order model) or `"oc"` (OC multiplier) |
| `qaoa_maxiter` | `None` | COBYLA evaluations (None → 100·p) |
| `qaoa_polish` | `True` | `qaoa` backend: replace each block's best shot by a greedy descent from it (classical); False = raw QAOA samples. Both energies are always reported |
| `protect_loads` | `True` | keep a greedy set cover of the loaded nodes solid (every loaded node keeps ≥ 1 solid element; FreeTO's keep_bc misses thin load regions at coarse meshes) |
| `move_limit` | `0.25` | adaptive move limit: at most ⌈move_limit·|C_k|⌉ flips per iteration (+ the removals the volume step needs), chosen greedily on the model's Lagrangian; halved on every guard rejection, ×1.25 after acceptance; None disables |
| `move_limit_min` | `0.02` | lower bound of the adaptive move limit |
| `guard` | `True` | accept-if-improves guard: a design whose FE compliance exceeds the previous accepted one by more than `guard_tol` (volume still decreasing) / `guard_tol_target` (at the target volume) is rejected, the previous design restored and the update redone with the smaller move limit |
| `guard_tol` | `0.5` | relative compliance rise tolerated while the volume target decreases |
| `guard_tol_target` | `0.25` | relative compliance rise tolerated at the target volume |
| `max_rejects` | `4` | consecutive rejections after which the design is accepted anyway |
| `connectivity` | `True` | connectivity repair of every binary step (physics audit 2026-10, docs/NOTES_quantum.md §3.18): solid components that touch no support element are removed if they hold no keep / protected / loaded element, otherwise reconnected to the nearest grounded component along the cheapest void path (Dijkstra on the face graph); the added volume is then removed again from boundary elements with the smallest \|g\|/v whose removal keeps every required element grounded. Per-iteration record: `qubo["connectivity_repair"]` = {`floating_before`, `components_before`, `removed`, `added`, `reconnected`, `rebalanced`}. False = the previous behaviour |
| `diagnostics` | `False` | record per-iteration connectivity statistics of the binary design (input, after load protection, after the QUBO solve, output; floating elements by source) in `qubo["connectivity"]`; no effect on the result |
| `backend_options` | `{}` | passed through to the backend (e.g. `{"sweeps": 200}` for sa, `{"annealing_time": 50}` for dwave_qpu) |

`QUBOOptions.from_dict(d)` / `.to_dict()` convert to/from JSON; unknown keys
raise `ValueError`.  `cfg.validate()` raises `FreeTOError` for bad enum values,
unknown backends, or a cloud backend whose SDK is not installed / whose token
environment variable is not set.

### Callback additions (QUBO mode only)

* `setup` dict gains `"qubo": {...}` = the resolved options
  (`backend`, `hessian`, `volume`, `block_size`, `init`, ...).
* every `iter` dict gains `"qubo"`:

```python
{"n_free": int,            # free-set size |C_k|
 "n_blocks": int,          # number of blocks in the Gauss–Seidel sweep
 "n_solves": int,          # QUBO solver calls this iteration (incl. λ bisection)
 "backend": str,           # backend actually used ("separable" when the QUBO is diagonal,
                           #  "threshold" for the init="oc" switch iteration)
 "hessian": str,           # hessian mode actually used
 "wall_time": float,       # s, whole design update (incl. Hessian)
 "hessian_time": float,    # s, extra solves for the block Hessian
 "solver_time": float,     # s, time inside the QUBO solver(s)
 "qpu_time": float | None, # s, QPU access time (D-Wave / IBM), None for local backends
 "energy": float,          # final model energy (scaled units)
 "lambda": float,          # volume multiplier used (scaled units)
 "volume_target": float,   # V_k as a fraction of the active elements
 "n_flips": int,           # Hamming distance to the previous binary design
 "approx_ratio": float | None,   # mean QAOA approximation ratio over blocks (QAOA backends)
 "p_opt": float | None,          # mean QAOA P(optimum) over blocks
 "exact_match": float | None,    # fraction of blocks where the backend hit the exact optimum (verify_exact)
 "max_block": int}               # largest block solved
```

### Result additions

`FreeTOResult.extra: dict` (always present, `{}` for OC/MMA unless
`eval_binary`).  In QUBO mode:
`extra["qubo_history"]` (list of the per-iteration `qubo` dicts above),
`extra["qubo_options"]` (resolved options), `extra["final_compliance"]`
(= `res.comp`, from one extra FE solve on the returned design),
`extra["best_iteration"]`, `extra["last_compliance"]`, `extra["binary_design"]`
(0/1 array over the active elements), `extra["fe_solves"]`, `extra["qubo_solves"]`.
`res.comp` corresponds to `res.eleden` in QUBO mode (extra final solve); the returned
design is the best design found at the target volume (`extra["returned_design"]` =
`"best"`) unless the last one is better (`"last"`).

---------------------------------------------------------------------------
## 2. QUBO solvers / backends — `freeto.quantum`

```python
from freeto.quantum import (solve_qubo, available_backends, QUBOResult,
                            QuantumBackendUnavailable, QUBOOptions)

available_backends() -> list[dict]      # never raises, never imports cloud SDKs eagerly beyond find_spec
# each: {"name": "sa", "available": True, "description": "...", "needs_token": False,
#        "token_env": None, "token_set": None, "kind": "classical"|"simulator"|"qpu"|"hybrid",
#        "max_n": int | None, "reason": ""}      # reason = why unavailable ("" when available)
```

Backend names: `exact`, `sa`, `tabu`, `greedy`, `qaoa` (always available);
`dwave_sa`, `dwave_tabu` (need `dwave-samplers`), `dwave_qpu`, `dwave_hybrid`
(need `dwave-system` + `DWAVE_API_TOKEN`), `qiskit_aer` (need `qiskit` +
`qiskit-aer`), `ibm` (need `qiskit-ibm-runtime` + `QISKIT_IBM_TOKEN`; optional
`QISKIT_IBM_INSTANCE`, `QISKIT_IBM_BACKEND`).

```python
res = solve_qubo(Q, h, const=0.0, backend="auto", num_reads=None, seed=None,
                 initial_state=None, time_limit=None, **opts) -> QUBOResult
```
minimises `E(x) = xᵀQx + h·x + const`, x ∈ {0,1}ⁿ; `Q` dense or scipy
sparse, any square matrix (symmetrised, diagonal folded into h).

```python
@dataclass
class QUBOResult:
    x: np.ndarray          # best sample, float 0/1, shape (n,)
    energy: float
    samples: np.ndarray    # (R, n)
    energies: np.ndarray   # (R,)
    timing: dict           # {"wall", "solver", "qpu_access", "qpu_sampling", "hybrid_run_time"} (s or None)
    info: dict             # backend specific: "backend", "n", QAOA: "p", "approx_ratio", "p_opt",
                           # "p_opt_shots", "nfev", "angles"; D-Wave: "chain_break_fraction", ...
```

---------------------------------------------------------------------------
## 3. Truss ground structures — `freeto.truss`

```python
from freeto.truss import (TrussProblem, list_benchmarks, get_benchmark,
                          solve_truss, TrussResult, TRUSS_METHODS)

list_benchmarks() -> list[dict]
# {"id": "gs_3x2", "alias": "T2s", "title": str, "description": str, "dim": 2|3,
#  "n_nodes": int, "n_bars": int, "vmax_fraction": float,
#  "exact_available": bool, "c_exact": float | None}
# ids: ten_bar (T1), gs_4x2 (T2), gs_3x2 (T2s), tower3d (T3), gs_9x3 (T4), column3d (T5)
# bars (v2, no bar between two fully fixed nodes): 10, 21, 12, 22, 116, 66

prob = get_benchmark("gs_3x2")            # TrussProblem; aliases "T1".."T5", "T2s" accepted
prob.to_dict() -> {"id", "dim", "nodes": [[x,y(,z)]...], "bars": [[a,b]...],
                   "supports": [[node, dof]...], "loads": [[node, fx, fy(, fz)]...],
                   "lengths": [...], "vmax": float, "vmax_fraction": float}

res = solve_truss(prob, method="qubo", backend="sa", seed=0,
                  callback=None, stop_event=None, **options) -> TrussResult
```

`TRUSS_METHODS = ["exact", "qubo", "sort", "oc", "oc_round", "oc_qubo"]`:
`exact` = enumeration (≤ 22 bars), `qubo` = iterative QUBO update (options as
`QUBOOptions` fields that apply: `hessian` ("exact"|"none"), `volume`,
`lambda_q`, `block_size`, `sweeps`, `er`, `num_reads`, `qaoa_p`, `qaoa_shots`,
`nbits`, `max_iter`, `patience`, `er` (default 0.15), `amin_model` (1e-3),
`trust_region` (True: re-solve with a Hamming move penalty / halved volume step
when the proposed design is a mechanism), `prune` (True: drop dangling zero-force
bars), `kinematic_repair` (False; v2: greedy add-back of bars until kinematically
stable and connected, then volume-restoring removals, NOTES §9.6), `stability_load` (0: weight of nominal random stability load cases added to the
QUBO model only), `qaoa_polish` (True), `verify_exact`, `backend_options`),
`sort` = BESO / first-order sorting control, `oc` =
continuous OC (areas in [A_min, 1]), `oc_round` = OC + keep-largest rounding,
`oc_qubo` = OC + one-shot quadratic QUBO rounding.

`callback(info)` per iteration (qubo/sort/oc): `{"stage": "iter", "iter",
"compliance", "volume", "volume_fraction", "on": [0/1 per bar] or
"areas": [...], "qubo": {same keys as the continuum qubo dict}}`.

`TrussResult` attributes and `res.to_dict()` (JSON-safe) keys:
`problem` (id), `method`, `backend`, `seed`, `nodes` (N×d), `bars` (M×2),
`on` (M bool), `areas` (M, relative 0..1 — equals `on` for binary methods),
`compliance`, `volume`, `vmax`, `volume_fraction`, `feasible` (stable and
volume ≤ vmax), `stable` (finite residual-checked solve, c ≤ 1e3 c_full, loaded nodes
connected to a support, and *kinematically stable*: the unit-area stiffness of the present
bars on the free DOFs of their nodes is positive definite), `c_exact` (None if unknown), `gap` (c/c_exact − 1 or
None), `iterations`, `fe_solves`, `qubo_solves`, `history` ({"compliance":
[...], "volume": [...], "n_changed": [...]}), `timing` ({"wall", "solver",
"qpu_access"}), `qubo_stats` (list of per-iteration dicts), `info` (method
specific, e.g. `exact`: `n_feasible`, `n_evaluated`, `top` list).

---------------------------------------------------------------------------
## 4. Study runner — `freeto.study`

```python
from freeto.study import run_study, suite_spec, SUITES
spec = suite_spec("quick")                # also "quick2", "full", "smoke"; or a user dict (below)
results = run_study(spec, callback=None, out_dir="results/quick",
                    stop_event=None, figures=True)
```

CLI: `python -m freeto.study --suite quick|quick2|full|smoke --out results/quick
[--spec my.json] [--no-figures] [--only S1,S3] [--threads 1] [--jobs 1]
[--fresh-workers | --no-fresh-workers] [--resume DIR] [--dry-run]`
(`--dry-run` lists the runs and exits);
`python -m freeto.study --reprocess results/quick` rebuilds summary/figures from the stored
`results.json` without re-running (see NOTES §7.7).  `run_study(..., threads=1)` pins the
thread pools (default 1; None leaves them).

Spec (JSON-compatible):
```python
{"name": "quick",
 "runs": [ {"study": "S1", "kind": "truss", "problem": "gs_3x2", "method": "qubo",
            "backend": "sa", "seeds": [0, 1, 2], "options": {...}, "label": "QUBO-sa"},
           {"study": "S3", "kind": "continuum", "problem": "cantilever_beam",
            "mesh_control": 25, "optimizer": "QUBO", "options": {...},
            "max_iter": 60, "seeds": [0], "label": "QUBO-sa",
            "run_options": {"eval_refined": 2, "mma_constraint": "filtered",
                            "mma_feasible_stop": True}},          # run_options: v2
           {"study": "F2", "kind": "qaoa_scan", "problem": "gs_3x2",
            "p_values": [1, 2, 3], "seeds": [0]} ]}
```

Progress events passed to `callback(ev)`:
`{"event": "start", "n_runs", "suite"}`,
`{"event": "run_start", "index", "n_runs", "run_id", "label", "study"}`,
`{"event": "run_end", "index", "n_runs", "run_id", "record"}` (record = one
row of results, see below; `"error"` key set if the run failed — the study
continues), `{"event": "figure", "path"}`, `{"event": "done", "results"}`.
`stop_event` stops after the current run.

`results` dict (also written to `out_dir/results.json`):
`{"suite", "created", "machine": {...}, "elapsed", "records": [...],
"summary": [...], "figures": [paths], "files": {"json", "csv", "summary_md"}}`.
Each record: `run_id, study, kind, problem, label, method, backend, seed,
compliance, c_ref, gap, feasible, volume_fraction, iterations, fe_solves,
qubo_solves, mean_qubo_size, max_qubo_size, wall_time, solver_time,
qpu_access_time, approx_ratio, p_opt, exact_match, history (list), error` (+ for
continuum: `crisp_compliance, crisp_volfrac, volfrac_target, compliance_at_beta,
beta_final, returned_design, n_rejects, diverged, mma_native_at_V, role, mesh_control,
nnele`, v2: `refined_compliance, refined_volfrac, refined_threshold, c_ref_refined,
gap_refined, mma_constraint, mma_fval_final, init_perturb, init_seed, fields_file`; for qaoa_scan rows: `p, n, best_shot_ratio, polished_ratio, uniform_ratio,
uniform_p_opt, greedy16_hit_fraction`).  `gap` is `None` for infeasible (truss) or
diverged (continuum: crisp c > 10× MMA) runs; continuum gaps are relative to MMA's crisp
compliance at the same target volume.  `results.csv` has the scalar columns; `summary.md` is the
markdown summary table; figures `F1_truss_gap.png`, `F2_qaoa.png`,
`F3_continuum_history.png`, `F4_topologies.png`, `F5_penalty.png`,
`F6_timing.png`, `F7_equal_volume.png`.
