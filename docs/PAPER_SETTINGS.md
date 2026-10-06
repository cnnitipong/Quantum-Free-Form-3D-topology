# Paper settings (manuscript v2, suite `quick2`)

The continuum results of the manuscript (Table 1, Table 2, Fig. 5) are the
S3 and S4 runs of the study suite `quick2` of `freeto/study.py`
(`suite_spec("quick2")`: `_quick2`, `V2_RUN_OPTIONS`, `QUICK_MAIN`,
`QUICK_ROBUST`), run with one pinned BLAS/OpenMP thread per process through
`freeto.study._run_continuum`, which calls
`freeto.examples.example_config(problem, **overrides)` and `run_freeto`.
The values below are read from that code and from the stored spec and
records of `results/quick2/results.json` (created 2026-10-04).
`freeto/paper.py` returns them programmatically (`paper_settings(problem,
method, seed)`, `paper_config(...)`), and `tests/test_webapp_paper.py`
checks that the module, the web app's presets and a web job equal the
study's configuration field by field.

## Examples (Table 1)

Geometry, supports and loads are `freeto/examples.py` `EXAMPLES[problem]["config_kwargs"]`
(STL files in `examples/STLs/generated/` and `examples/STLs/`); the study
overrides only the mesh control and the run options.

| Example | key | MeshControl | V* | Loads (N) | Supports | Symmetry / keep | Active el. |
|---|---|---|---|---|---|---|---|
| Cantilever | `cantilever_beam` | 36 | 0.30 | Fy = -1000 (tip patch) | x = 0 face clamped | - | 1925 |
| MBB beam (half) | `mbb_beam` | 46 | 0.30 | Fy = -1000 (top of symmetry plane) | roller (y), symmetry plane (x), z restraint | mirrored y-z (output only) | 2205 |
| Bridge deck | `bridge_deck` | 41 | 0.35 | Fy = -1000 over the deck slab | two end pads clamped | deck slab kept solid | 2400 |
| L-bracket | `l_bracket` | 30 | 0.30 | Fy = -1000 (arm tip) | top face clamped | - | 1920 |
| GE bracket | `GE_bracket` | 24 | 0.30 | case 1 Fz = 1500, case 2 Fy = -2000 | four bolt holes clamped | - | 776 |

Shared by every run: E0 = 210 GPa (`youngs_modulus=210e9`), nu = 0.3,
SIMP (`method="SIMP"`), penalty q = 3, filter radius rmin = 1.5 elements,
load type `distributed`, `keep_bc=True`, solver `auto` (PARDISO in the
paper's runs), `inside_mode="robust"`, `compat="matlab"`.

## Methods

Every S3/S4 continuum run has the iteration cap 300 and the run options
`V2_RUN_OPTIONS = {eval_refined: 2, mma_constraint: "filtered",
mma_feasible_stop: True}`; `_run_continuum` adds `eval_beta=8.0` and
`eval_crisp=True` (crisp element proxy at V*). QUBO runs get `qubo.seed =
seed`. All QUBO options not listed are the `QUBOOptions` defaults:
volume `bisection` (multisection), frontier fraction 0.25, blocks `morton`,
sweeps 2, init `solid`, evolutionary ratio R_E = `er` = 0.05, patience 8,
`history_average` on, interpolation `beso`, move limit 0.25 (min 0.02,
x0.5 after a rejection, x1.25 after an accepted step), guard 0.5 / 0.25 with 4
rejects, load protection and connectivity repair on, gamma = 0,
move penalty 0, Hessian patches of 64 (`hessian_block_size`), SA 32 replicas
(`num_reads` unset).

| Method (run label) | Optimizer | QUBO backend | Hessian | Block size | Other options | Examples | Seeds |
|---|---|---|---|---|---|---|---|
| QUBO-sa (block) **(web default)** | QUBO | `sa` | `block` | whole free set (`None`) | - | all five | 0-4 beams, 0-2 others |
| QUBO-sa (diag) | QUBO | `sa` | `diag` | whole free set | - | cantilever, MBB | 0-4 |
| MMA (reference) | MMA | - | - | - | move 0.5, `mma_constraint=filtered`, `mma_feasible_stop` | all five | 0 |
| BESO-sort | QUBO | `auto` | `none` | - | - | all five | 0 |
| QUBO-qaoa kb8 p1 penalty (+greedy) | QUBO | `qaoa` | `block` | 8 | `qaoa_p=1`, `qaoa_shots=300`, `volume=penalty`, `sweeps=1`, `verify_exact=True`, polish on | MBB, bridge, L, GE | 0 |
| BESO-sort (move 0.04) | QUBO | `auto` | `none` | - | `move_limit=0.04` | all five | 0 |
| QUBO-sa (scalar) | QUBO | `sa` | `scalar` | whole free set | - | all five | as block |
| QUBO-sa (block, Qx3) | QUBO | `sa` | `block` | whole free set | `hessian_scale=3` | cantilever, MBB | 0-4 |
| OC (FreeTO default, flagged) | OC | - | - | - | - | cantilever, MBB | 0 |

Not offered in the web app's "Paper run" list: the MMA compliance-volume
curve (MMA at V* = 0.26, 0.28, 0.32, 0.34 on the beams), "MMA (init s)"
(`init_perturb=0.05`, `init_seed` = seed 0-4; the API accepts both fields)
and the S3q cantilever runs at MC 25 (QAOA blocks of 10).

### Projection (beta) schedule and stopping

| Optimizer | beta | tolerances / stop |
|---|---|---|
| QUBO (all binary runs, incl. BESO-sort) | 0.5, +0.5 per iteration, up to 8 | at V*: Hamming change < 1e-3, or no improvement for 8 iterations (patience), or 300 iterations; returns the best design at V* |
| MMA | 0.5, +0.5 per iteration, no cap | change or grey fraction < 1e-3 while the volume constraint holds within 1e-3 (`mma_feasible_stop`), or 300 iterations; last iterate |
| OC | 0.1, +0.05 per iteration, up to 2 | change < 3e-3, or 300 iterations |

### Evaluation (every run)

* native: the optimizer's own compliance and volume (QUBO: the returned
  design, one extra solve; OC/MMA: the iterate entering the last iteration
  and its volume, `history_volfrac[-2]`);
* crisp element proxy at V* (`eval_crisp=True`);
* common-beta compliance at beta 8 (`eval_beta=8`);
* refined binary voxel evaluation, f = 2 (`eval_refined=2`, loads on solid
  voxels): the headline metric; gap = c_ref / c_ref(MMA, seed 0) - 1;
* physics audit on.

Refined compliance of the MMA reference run (the gap denominator):
cantilever 0.092518, MBB 0.288049, bridge deck 0.0028827, L-bracket 0.179240,
GE bracket 0.072716 (`freeto/paper_reference.json`, `mma_reference_refined`).

## Per-run settings (all paper runs offered by the web app)

One row per example and method, generated from `freeto.paper.paper_settings` (= the study's configuration). Seeds are those of the published records; the web app's default is seed 0. E0 = 210 GPa and nu = 0.3 throughout.

| Example | Mesh control | V* | Method (run label) | Optimizer | QUBO backend | Hessian | Block size | Move limit | R_E | Beta schedule | r_min | Penalty q | Iter. cap | Evaluation | Seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cantilever | 36 | 0.30 | QUBO-sa (block) | QUBO | sa | block | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2,3,4 |
| Cantilever | 36 | 0.30 | MMA | MMA | - | - | - | 0.5 (MMA) | - | 0.5 +0.5/it, no cap | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2; MMA filtered vol., feasible stop | 0 |
| Cantilever | 36 | 0.30 | BESO-sort | QUBO | auto | none | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| Cantilever | 36 | 0.30 | QUBO-sa (diag) | QUBO | sa | diag | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2,3,4 |
| Cantilever | 36 | 0.30 | BESO-sort (move 0.04) | QUBO | auto | none | free set | 0.04 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| Cantilever | 36 | 0.30 | QUBO-sa (scalar) | QUBO | sa | scalar | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2,3,4 |
| Cantilever | 36 | 0.30 | QUBO-sa (block, Qx3) | QUBO | sa | block x3 | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2,3,4 |
| Cantilever | 36 | 0.30 | OC (FreeTO default, flagged) | OC | - | - | - | 0.1 (OC) | - | 0.1 +0.05/it, max 2 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| MBB (half) | 46 | 0.30 | QUBO-sa (block) | QUBO | sa | block | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2,3,4 |
| MBB (half) | 46 | 0.30 | MMA | MMA | - | - | - | 0.5 (MMA) | - | 0.5 +0.5/it, no cap | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2; MMA filtered vol., feasible stop | 0 |
| MBB (half) | 46 | 0.30 | BESO-sort | QUBO | auto | none | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| MBB (half) | 46 | 0.30 | QUBO-sa (diag) | QUBO | sa | diag | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2,3,4 |
| MBB (half) | 46 | 0.30 | QUBO-qaoa kb8 p1 penalty (+greedy) | QUBO | qaoa (penalty vol., p=1, 300 shots, 1 sweep) | block | 8 | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| MBB (half) | 46 | 0.30 | BESO-sort (move 0.04) | QUBO | auto | none | free set | 0.04 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| MBB (half) | 46 | 0.30 | QUBO-sa (scalar) | QUBO | sa | scalar | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2,3,4 |
| MBB (half) | 46 | 0.30 | QUBO-sa (block, Qx3) | QUBO | sa | block x3 | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2,3,4 |
| MBB (half) | 46 | 0.30 | OC (FreeTO default, flagged) | OC | - | - | - | 0.1 (OC) | - | 0.1 +0.05/it, max 2 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| Bridge deck | 41 | 0.35 | QUBO-sa (block) | QUBO | sa | block | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2 |
| Bridge deck | 41 | 0.35 | MMA | MMA | - | - | - | 0.5 (MMA) | - | 0.5 +0.5/it, no cap | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2; MMA filtered vol., feasible stop | 0 |
| Bridge deck | 41 | 0.35 | BESO-sort | QUBO | auto | none | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| Bridge deck | 41 | 0.35 | QUBO-qaoa kb8 p1 penalty (+greedy) | QUBO | qaoa (penalty vol., p=1, 300 shots, 1 sweep) | block | 8 | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| Bridge deck | 41 | 0.35 | BESO-sort (move 0.04) | QUBO | auto | none | free set | 0.04 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| Bridge deck | 41 | 0.35 | QUBO-sa (scalar) | QUBO | sa | scalar | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2 |
| L-bracket | 30 | 0.30 | QUBO-sa (block) | QUBO | sa | block | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2 |
| L-bracket | 30 | 0.30 | MMA | MMA | - | - | - | 0.5 (MMA) | - | 0.5 +0.5/it, no cap | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2; MMA filtered vol., feasible stop | 0 |
| L-bracket | 30 | 0.30 | BESO-sort | QUBO | auto | none | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| L-bracket | 30 | 0.30 | QUBO-qaoa kb8 p1 penalty (+greedy) | QUBO | qaoa (penalty vol., p=1, 300 shots, 1 sweep) | block | 8 | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| L-bracket | 30 | 0.30 | BESO-sort (move 0.04) | QUBO | auto | none | free set | 0.04 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| L-bracket | 30 | 0.30 | QUBO-sa (scalar) | QUBO | sa | scalar | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2 |
| GE bracket | 24 | 0.30 | QUBO-sa (block) | QUBO | sa | block | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2 |
| GE bracket | 24 | 0.30 | MMA | MMA | - | - | - | 0.5 (MMA) | - | 0.5 +0.5/it, no cap | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2; MMA filtered vol., feasible stop | 0 |
| GE bracket | 24 | 0.30 | BESO-sort | QUBO | auto | none | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| GE bracket | 24 | 0.30 | QUBO-qaoa kb8 p1 penalty (+greedy) | QUBO | qaoa (penalty vol., p=1, 300 shots, 1 sweep) | block | 8 | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| GE bracket | 24 | 0.30 | BESO-sort (move 0.04) | QUBO | auto | none | free set | 0.04 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0 |
| GE bracket | 24 | 0.30 | QUBO-sa (scalar) | QUBO | sa | scalar | free set | 0.25 (min 0.02) | 0.05 | 0.5 +0.5/it, max 8 | 1.5 | 3 | 300 | native; crisp proxy; beta 8; refined f=2 | 0,1,2 |

## Reproducing a run

* CLI / Python: `freeto.paper.paper_config("cantilever_beam", "QUBO-sa (block)", seed=0)`
  gives the study's `FreeTOConfig`; run it with one BLAS thread
  (`OMP_NUM_THREADS=1`). Repeated runs are bit-identical to the records.
* Web app: pick a paper example and the paper run, Load (or "Paper
  settings"), Run. The job runs with one pinned thread (`FREETO_THREADS`,
  default 1, like the study) and the result card shows native, crisp and
  refined values, the refined gap to the paper's MMA reference and the
  published values of the matching paper run.

The trajectory of a binary run depends on the floating-point summation
order: a different thread count, BLAS library or sparse solver (e.g.
SuperLU instead of PARDISO when `pypardiso` is missing) gives a different,
equally valid design, so bit-identical reproduction needs the same software
stack as the study (Python 3.11, NumPy 2.4.4, SciPy 1.17.1, PARDISO).
