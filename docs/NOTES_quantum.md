# QUBO / quantum extension — engineering notes

Scope: `freeto/quantum/`, `freeto/truss/`, `freeto/study.py`, the QUBO hooks in
`freeto/core.py` (+ `FreeTOConfig.qubo / eval_binary / eval_beta / eval_crisp`,
`FreeTOResult.extra`), `freeto/evaluate.py`,
`--optimizer QUBO` / `--qubo-*` flags in `freeto/cli.py`, `tests/test_quantum.py`,
`tests/test_truss.py`, `requirements-quantum.txt`.  Design: `docs/QUANTUM_DESIGN.md`;
public API: `docs/QUANTUM_API.md`.  Results of the quick study: `results/quick/summary.md`.

**Fix round (2026-10).**  After the independent review (`docs/VERIFICATION_quantum.md`)
the following changed: kinematic truss stability + re-pinned optima (§3.14), continuum
stabilisation (load protection, adaptive move limit, accept-if-improves guard; §3.15),
explicit QAOA polish and baselines (§3.16), fail-safe post-run evaluations and UTF-8 output
(§3.17), corrected reasons in §3.1/§3.2, and a new continuum protocol (common crisp
evaluation at equal volume, MMA baseline + c–V curve, 5 seeds; §7).  §6 replaces all earlier
findings.

**Honest framing (repeated from the design).** Everything that runs locally is classical:
SA / tabu / greedy are classical heuristics, QAOA is an exact state-vector *simulation*
(cost ∝ 2^n).  The study measures solution quality and algorithmic behaviour; it cannot show
a quantum speed-up, and continuum designs have no exact reference.

## 1. Layout

| module | content |
|---|---|
| `quantum/options.py` | `QUBOOptions` dataclass (+ `from_dict`, `validate`) — dependency-free, imported by `core` |
| `quantum/qubo.py` | normal form x^T Q x + h·x + c (Q symmetric, zero diagonal), energy, Ising / dimod-BQM conversion |
| `quantum/exact.py` | brute force ≤ 24 variables: low 16 bits as a (65536, 16) matrix, high bits looped; full spectrum (little-endian index) |
| `quantum/anneal.py` | vectorised SA (dense: per-variable Metropolis over R replicas; sparse: greedy graph colouring → chromatic parallel Metropolis, one numpy step per colour), tabu (R restarts vectorised, aspiration, stall restarts), greedy steepest descent; SA and tabu accept one linear term per replica group (h of shape (K, n)) |
| `quantum/qaoa.py` | own state-vector QAOA: spectrum-based diagonal cost layer, in-place axis-wise mixer, linear-ramp / INTERP initialisation, COBYLA, shots, approx ratio, P(opt), P(opt in shots) |
| `quantum/backends.py` | registry, `solve_qubo`, `available_backends`, lazy D-Wave (dwave-samplers SA/tabu, DWaveSampler+EmbeddingComposite, LeapHybridSampler) and Qiskit (AerSimulator, IBM SamplerV2) wrappers, timing extraction |
| `quantum/blockqubo.py` | volume-constrained model minimisation shared by continuum and truss: block Gauss–Seidel, λ bisection / vectorised multisection, penalty mode, greedy repair, statistics |
| `quantum/update.py` | `QUBOUpdater` (continuum update), `multi_rhs_solve` (reuses the FE factorisation), `robust_solve` (iterative refinement for 0/1 designs), Morton keys |
| `truss/ground.py, fe.py, optimize.py, benchmarks.py` | ground structures with overlap filter, batched dense FE, exact enumeration, OC + rounding, iterative QUBO, benchmarks T1–T5 |
| `evaluate.py` | common crisp evaluation (fine-grid projection at a common volume) used by `eval_crisp` and the study |
| `study.py` | suites smoke / quick / full, runners, post-processing (crisp gaps vs MMA, MMA c–V curve interpolation), CSV/JSON/markdown, figures F1–F7 |

## 2. Core integration (no change to OC/MMA numerics)

`run_freeto` gets `qubo_mode = optimizer == "QUBO"`; the `QUBOUpdater` is built after the
solver and called exactly where `oc_update` is.  Everything else of the loop (filtering,
`MusD`, smooth-edge, change, logging, callback) is unchanged.  QUBO-only differences:
defaults tolx = tol_thresh = 1e-3, β 0.5 → +0.5 → 8 (design A.5); the loop ends on
`updater.should_stop` (volume target reached and change ≤ tolx, or `patience` iterations
without improvement) instead of the Topo measure, without MATLAB's second smoothing; the
best design at the target volume is remembered and one extra FE solve is done on the
returned design (`res.comp` matches `res.eleden`); FE solves go through `robust_solve`.
The OC/MMA code paths are untouched (the new `while` condition reduces to the old one for
them; `full_pre = full` and `beta_used = beta` are references only).  QUBO mode only: the
accept-if-improves guard (§3.15) can restore the previous design and redo its FE solve
before the update (`_qubo_fe`, same formulas as the loop body).  The post-run evaluations
(`eval_beta`, `eval_binary`, `eval_crisp`, QUBO final solve) are fail-safe (§3.17).  `fe.SuperLUSolver` now keeps `last_lu`
(one attribute, no numerical change) so the block Hessian can reuse the factorisation;
CHOLMOD (`_factor`) and PARDISO (phase 33) are reused without changes.  Verified:
`tests/compare_ref.py` 524/524 checks pass, all pre-existing tests pass.

## 3. Deviations from QUANTUM_DESIGN.md (with reasons)

1. **Modulus-space model instead of the SIMP-density Taylor model (`interp="beso"`).**
   The design expands c in the SIMP density ρ (Hessian 2E′E′(ku)ᵀK⁻¹(ku) − δE″c).
   *Correct reason for the deviation (fix round, VERIFICATION M8):* with the density
   filter a single design flip changes each element density only by J_ej ≲ 0.33, and in
   that regime the design-literal ρ-space model (`interp="simp"`) is the **accurate**
   local model — finite differences on a filtered 192-element problem give a median
   relative error of 5 % (single removals), 8 % (additions), 12 % (adjacent pairs),
   whereas the `beso` weights underestimate every Δc by ≈ 1/p (model ÷ truth ≈ 0.3).
   The `simp` model nevertheless **diverges in the loop** (4 of 6 seeds, c = 10–600):
   the design that is evaluated is the *Heaviside-projected* field (β up to 8), not Jx,
   and a boundary flip changes the projected density of the neighbouring elements by ≈ 1
   — exactly the regime where a Taylor model in ρ fails (and where a secant model in the
   modulus is right; `secant` in turn fails for additions because its Neumann series
   diverges in void).  The `beso` interpolation (w = E′/p = ρ^{p−1}(E0 − Emin), first-
   order term = the BESO sensitivity, Hessian = the PSD Gram term 2w_e w_f (KE u_e)ᵀK⁻¹
   (KE u_f), w ≈ 0 at voids so additions are driven by the filtered neighbour
   sensitivities) is biased low by ≈ 1/p but its *ranking* of moves is good and the loop
   is stable; that is why it is the default.  It is **not** "the exact secant expansion"
   for partial (filtered) density changes — it is exact only for a full 0↔1 change of an
   unfiltered solid element.  (The previous text's "1 − 3 + 3 = 1" argument applies only
   to an unfiltered 0→1 change and was the wrong reason.)
2. **`hessian="diag"`** is the element-local (Loewner) bound of the Gram term,
   D_e = 2w_e²c_e/E_e (≥ 0, exact for a statically determinate chain), mapped through the
   filter (sparse, ~27–125 non-zeros per row).  No extra solve.  `hessian="block"`
   (default with a direct solver; `"exact-block"` accepted) = exact Gram term inside Morton
   patches (= the solver blocks, or `hessian_block_size` = 64 for whole-set backends);
   **there are no cross-patch couplings** (the code sets the diagonal bound D = 0 in block
   mode unless `interp="simp"`, to avoid double counting the patch-diagonal part — the
   previous note "cross-patch couplings from the diag part" was wrong).  All patches'
   right-hand sides are batched into ≤ 512-column multi-RHS solves with the existing
   factorisation; AMG → automatic `diag`.
3. **Free set = boundary band of the binary design** (+ the f·V_k frontier candidates)
   instead of the smooth-edge grey band: with QUBO's β schedule starting at 0.5 the grey
   band covers almost the whole domain, so the "trust region" was the whole design.
   `free_set="grey"` restores the design's definition.
4. **BESO history averaging** g_k ← (g_k + g_{k−1})/2 (default on): without it even the
   pure sorting control oscillated (100–200 flips per iteration at fixed volume, compliance
   0.27 vs 0.18 with averaging on cantilever MC 25).
5. **γ default 0** and γ in units of the mean |g| of the solid elements (not max|dc|, which
   is dominated by the load-point singularity): the perimeter term removes thin members and
   disconnected the load path on the coarse study meshes; F5 reports the sensitivity.
6. **Volume repair fills to V_k** (adds the cheapest elements even if the model energy
   rises), i.e. v·x = V_k up to one element, as the design text says; filling only while
   the model energy decreases left designs 5 % under-filled.
7. **Multisection instead of 12–20 sequential bisection steps** for batch-capable
   single-block backends (sa, tabu, exact, greedy): one geometric round of 12 λ values over
   [1e-5, 1]·λ_max, then three linear rounds of 7, each round one vectorised call (replicas
   split among the λ values) — 4 solver calls per iteration.
8. **Fixed λ in `volume="penalty"`** = the knapsack threshold of the first-order model
   (`lambda_source="sort"`, the exact multiplier of the linear binary problem); OC's
   continuous multiplier (`"oc"`) is available but is move-limit-biased early on.
9. **init="oc"**: the thresholded design is returned once (so its compliance is measured
   by the next FE solve), then QUBO iterations follow.
10. **Robust FE solves for 0/1 designs** (`robust_solve`): a design with a load path
    through void material (modulus 1e-9 E0) leaves direct solvers at a relative residual of
    1e-6…1e-4, which the core rejects as "singular"; iterative refinement is tried and a
    residual ≤ 1e-3 is accepted (compliance 10⁵–10⁸× larger — a legitimately terrible design
    that the optimiser and the study must see).  Only in QUBO mode / the new evaluations.
11. **Examples**: the design's D1–D4b box builders were not added; the study uses the
    registered generated examples (`cantilever_beam` = D1 role, `mbb_beam` = D2,
    `bridge_deck` = D3, `GE_bracket` = D4) at coarse MeshControl.
12. **Truss details.** (a) T1 ten-bar: Vmax = 65 % instead of 50 % — at 50 % no stable
    binary design exists (the lightest stable sub-truss needs 58.6 % of the total length;
    enumeration found only mechanisms).  (b) T2 / T2s / T4 keep the bars between fixed nodes
    (they carry no load) so the counts match the design (22 / 13 / 118).  The scratch value
    9.0897 {1, 2, 5, 6, 8, 10} of T2s is **not** a valid optimum: node (1, 0) is held only by
    two collinear bars (a kinematic mechanism whose zero-energy mode is orthogonal to the
    load) — see 14.  (c) The Taylor model is expanded
    with absent bars at A = 1e-3 (the scratch experiment's value; `amin_model`), feasibility
    and compliance use 1e-9: at 1e-9 the Hessian near mechanisms is ~1e9 and the model is
    useless.  (d) `trust_region=True`: if the proposed design is a mechanism, the iteration
    re-solves with a Hamming move penalty μ ∈ {0.1, 0.3, 1} and a halved volume step (up to
    5 extra QUBO solves).  (e) ER = 0.15 for trusses (scratch experiment), 0.05 continuum.
    (f) `round_qubo` expands around the continuous optimum, E = g·(x − a) + ½(x − a)ᵀH(x − a)
    (the design's formula omits the shift).
13. **D-Wave/IBM wrappers** are exercised only for the local samplers (dwave-samplers SA /
    tabu and qiskit-aer, installed into a scratch directory: dwave_sa / dwave_tabu reach the
    exact optimum on T2s and n = 10 QUBOs; qiskit-aer's sampled P(opt) = 0.030 vs 0.0285
    from the own simulator, approx. ratio 0.845 vs 0.843, which validates the RZZ/RZ/RX
    angle convention).  `dwave_qpu`, `dwave_hybrid`, `ibm` need tokens and are untested
    here; without SDK/token they raise `QuantumBackendUnavailable` (tested).
14. **Kinematic stability test for trusses (fix round, VERIFICATION M4).**  "Stable" now
    means: residual-checked finite solve, c ≤ 1e3 c_full, loaded nodes connected to a
    support, **and** positive-definite unit-area stiffness of the present bars on the free
    DOFs of the nodes they touch (min eigenvalue > 1e-9 × max; `TrussFE.kinematic`).  This
    is used by `evaluate`, `enumerate_exact`, the iteration's best-design bookkeeping and
    the trust-region test.  Re-pinned exact optima: T1 629.470 (unchanged, {0,2,3,6,7,8},
    5 stable designs at 65 %), **T2s 9.96376 {1,2,4,5,7,8,10}** (37 stable designs; equals
    the verifier's independent brute force), **T2 23.8915** {1,2,5,6,10,11,13,14,16,17,19}
    (was 23.159), **T3: at 45 % every candidate is a mechanism**; the lightest
    kinematically stable sub-truss needs 53.0 % → V_max = 55 %, optimum 7.84667 (292 stable
    designs).  Dangling bars (an unloaded, unsupported node touched by one bar) carry no
    force but make a design formally unstable; the iterative methods, OC + rounding and OC +
    QUBO rounding therefore `prune` them (compliance unchanged, volume decreases; the
    enumeration needs no pruning because the pruned design is enumerated too).  Consequence:
    on T2s every iterative method — QUBO with exact/SA/tabu/greedy *and* first-order sorting
    — ends at the same compliance 11.6569, +17.0 % above the exact optimum (QUBO bar set
    {0,1,2,4,5,8,10}, sorting {1,2,5,7,8,10}; equal compliance, different designs);
    the former claims "QUBO reaches the exact optimum on T2s, sorting is +111 %" were
    artefacts of accepting the collinear-hinge mechanism.  An optional nominal-load
    stabiliser (`stability_load` > 0: random load cases of weight × c_full added to the
    QUBO *model* only) was tried and did not change T2s/T3, so it is off by default.
15. **Stabilisation of the continuum update (fix round, VERIFICATION M3).**  On
    bridge_deck (MC 31: 840 elements, distributed load on *every* top node, the thin keep
    slab contains no element centre so keep_bc keeps nothing) the binary update removed
    the last solid element around loaded nodes (load in void: c jumps 0.0024 → 73 in one
    iteration) and never settled (50–84 flips/iteration).  Three defaults were added:
    (a) **load protection** (`protect_loads`): a greedy set cover of the loaded nodes by
    elements (prefer solid, then high |g|/v) is kept solid every iteration, so every
    loaded node keeps ≥ 1 solid element; (b) **adaptive move limit** (`move_limit` 0.25 of
    the free set, + the removals the volume step needs; the kept flips are chosen greedily
    on the model's Lagrangian, then the volume repair; halved on each rejection, ×1.25
    after an accepted step, floor `move_limit_min`); (c) **accept-if-improves guard**: if
    the FE compliance of the new design exceeds the previous accepted one by more than
    `guard_tol` = 50 % (volume still decreasing) or `guard_tol_target` = 25 % (at the
    target), the previous design is restored (one extra FE solve, so the factorisation used
    by the block Hessian is the right one), the volume target of the rejected step is
    restored and the update is redone; after `max_rejects` = 4 consecutive rejections the
    design is accepted.  At the target volume a rejection counts towards `patience`.  The
    returned design is still the **best design found at the target volume** (or the last
    one if better; `extra["returned_design"]`).  The guard tolerances were chosen on
    cantilever MC 25 / mbb MC 31 / bridge MC 31 (seed 0): 5 % at the target improved the
    bridge (0.0175 vs 0.0212) but stopped the cantilever early (0.231 vs 0.163); 25 % and
    50 % give identical results on all three, so 25 % is used.
16. **QAOA polish is explicit.**  The `qaoa` backend's greedy polish (`polish` /
    `QUBOOptions.qaoa_polish`, default True) is classical post-processing; every result
    reports the raw best shot (`best_shot_energy`, `best_shot_ratio`) and the polished
    energy, the study runs "QUBO-qaoa (raw)" and "(+greedy)" as separate methods, and F2
    shows the p = 0 (uniform superposition) and greedy-only baselines.  Timing: for all
    QAOA backends `timing["solver"]` is the whole variational loop *including* the
    classical COBYLA angle optimisation on the state-vector simulator (≈ 100 % of it is
    classical simulation); the polish is excluded (`info["polish_time"]`).
17. **Post-run evaluations are fail-safe** (VERIFICATION M5): `eval_beta`, `eval_binary`,
    `eval_crisp` and the QUBO final solve are wrapped; solver-library exceptions become
    `FreeTOError`, the metric is `inf`, the message is in `extra["eval_errors"]`, and the
    finished run is kept.  All text files are written with `encoding="utf-8"` (M6; the
    summary contains `−`, `±`, `β`, `⟨E⟩`, which cp1252 cannot encode).
18. **Connectivity repair (physics audit 2026-10, the internal audit report).**  The FE
    model of the QUBO update never sees the binary design x itself: it sees
    smoothedge3D(H x / Hs), and the density filter (rmin 1.5) plus the node averaging
    bridge one-element gaps.  A binary step can therefore cut a member, isolate the
    load-protection elements or detach the whole deck from its piers while the native
    compliance rises by only ~10 % (bridge_deck MC 31, seed 0, iteration 26: 70 % of the
    binary design ungrounded, c 0.0082 → 0.0092), so neither the model nor the guard
    notice.  The returned bridge designs had 22 (sa) / 33 (qaoa) face components with
    19–28 % of the solid touching no support; the common crisp evaluation then leaves
    3.6–4.4 % of the deck load on void (crisp c 179 / 230; a voxel FE of the binary
    design itself: 1.3e3 / 1.6e3).  `connectivity=True` (default) repairs every step
    (`QUBOUpdater.connect`): ungrounded components without keep / protected / loaded
    elements are removed, the others reconnected along the cheapest void path, and the
    added volume removed again from safe boundary elements (`_rebalance`).  Effect
    (seed 0, 1 thread): bridge sa crisp 179.5 → 0.0397, qaoa 230 → 0.0247 (MMA 0.0110);
    cantilever sa 0.166 → 0.153, qaoa 0.188 → 0.185; GE / l_bracket unchanged within
    0.3 %.  It does not add a length scale: QUBO designs still contain one-element
    members that the crisp proxy and the rendered surface thin out, and the bridge keep
    slab still contains no element centre at MC 31 (MusD empty for every optimizer; at
    MC 41 it holds 440 elements and the QUBO bridge stays connected with or without the
    repair).

## 4. Numerical checks (tests)

* `exact` = 2ⁿ numpy loop (n = 5, 10, 17 incl. the low/high split); Ising/BQM round trips.
* SA ≥ 95 %, tabu 100 %, greedy(16 restarts) ≥ 60 % exact hits on 24 random dense QUBOs
  (n = 10–14); chromatic SA on a 216-variable 3-D grid QUBO within 1 % of the best of 5 runs.
* QAOA: p = 1 two-qubit closed form ⟨Z₁Z₂⟩ = sin 4β sin 2γJ (1e-12), mixer = dense expm,
  n = 8, p = 3: mean approx ratio ≥ 0.6, P(opt) > 5/256.
* Block Gauss–Seidel with a recording `FakeBackend`: every block sees
  h_B + 2Q[B,¬B]y_¬B + λv_B + λ_q(v_B² + 2rv_B), global energy never increases, output =
  replayed sweep + repair.
* Continuum block Hessian vs central finite differences of c(x), ρ = Jx: p = 1 (`beso`, the
  Gram term is the exact Hessian) and p = 3 (`simp`, Gram − E″c); agrees with a dense
  reference Hessian to all digits and with FD to 2e-3 rel / 1e-4·max|H| abs.
* Truss: 2-bar analytic compliance and sensitivity, batched = single assembly, mechanism
  detection incl. the collinear-hinge design and dangling-bar pruning, Hessian and gradient
  vs FD (1e-5), pinned exact optima with the kinematic test (T1 629.470129 kip·in, T2
  23.8915297, T2s 9.96376183 {1,2,4,5,7,8,10} with 37 stable designs, T3 7.84666621 at
  55 %), QUBO-exact gap 0 on T1, QUBO-exact / sorting / SA / tabu / greedy all +16.99 % on
  T2s (regression), QAOA (whole problem) on T1, blocks and 2-bit areas, nbits = 0 rejected.
* Fix round: crisp evaluation at equal volume for MMA and BESO, a failing post-run
  evaluation keeps the run (`eval_errors`), QAOA raw/polished reporting and the uniform
  baseline, `dimod` required by the D-Wave backends, bridge_deck MC 31 with load
  protection + guard < 0.1 (unprotected 1.6–80), study smoke checks crisp/gap/β_final and
  that the summary needs UTF-8.
* End-to-end: cantilever_beam MC 25 QUBO-sa: feasible 0/1 design, volume within 0.02,
  improvement after the target is reached, binary design keeps its load path; init="oc" on a
  small box improves the thresholded start; QAOA with 8-qubit blocks end-to-end < 60 s;
  study smoke suite writes JSON/CSV/markdown and ≥ 5 figures.

## 5. Performance (this container: 2 vCPU x86-64, PARDISO)

Per-iteration cost of the QUBO update on `cantilever_beam` at **MeshControl 29** (28×9×4 =
1,008 active elements, 4,200 free DOFs, free set 490–750 elements, 25 iterations each):

| update | solver | update / iteration | of which Hessian | of which QUBO solver | FE solve | whole 25-iteration run |
|---|---|---|---|---|---|---|
| hessian none (+γ, SA) | pardiso | 0.13 s | – | 0.12 s | 0.02 s | 3.9 s |
| hessian diag, SA | pardiso | 0.74 s | – | 0.73 s | 0.01 s | 19 s |
| hessian block, SA (default) | pardiso | **1.87 s** | 0.55 s | 1.31 s | 0.01 s | 47 s |
| hessian block, SA | superlu | 1.94 s | 0.61 s | 1.33 s | 0.09 s | 51 s |
| hessian block, tabu | pardiso | 2.82 s | 0.60 s | 2.22 s | 0.03 s | 72 s |

(BESO sorting — hessian none, γ = 0 — is solved by sorting: < 1 ms.)  SA inside the update:
4 vectorised multisection calls per iteration, 100 sweeps, 32 replicas; with the 64-element
exact-Hessian patches the chromatic path uses ~70–130 colours.  The FE solve is negligible at
this size; at larger meshes the block Hessian's right-hand sides (one per free element and
load case) dominate — `hessian_max_rhs` (6,000) switches such iterations to `diag`, AMG
always uses `diag`.

Stand-alone solvers (random dense QUBOs): exact n = 22 in 0.05 s (n = 24: ~0.3 s); SA
n = 16 in 0.02 s; tabu n = 16 in 0.015 s; QAOA n = 12 / 16, p = 3: 0.9 s / 8 s (COBYLA 300
evaluations).  Truss: exact enumeration T2 (4.2 M masks, 760 k solved) 1.9 s, T3 3.5 s;
QUBO-sa on T4 (118 bars) 11 s per run.  Fix-round quick study (`--suite quick`, 202
runs, design-size meshes, 5 seeds): 67 min on this 2-vCPU container (OMP_NUM_THREADS=1);
QUBO-sa (block) 140–230 s per run at 1.9–2.2 k elements (cantilever MC 40, 3,042 el.:
~6 min), diag 31–45 s, MMA 3–5 s.  Tests: full suite 175 passed, 4 skipped (~3.5 min).


## 6. Findings worth knowing (fix-round quick study, `results/quick`, 2026-10)

> **Superseded for the continuum numbers by §8 (corrections of 2026-10-02).** The
> `results/quick` described here was rebuilt after the connectivity repair and the
> example corrections. The bridge_deck, l_bracket and mbb_beam numbers below come from the
> uncorrected setups. The truss, QAOA-scan and penalty findings are unchanged.

Protocol: §7.  All continuum numbers are the **common crisp evaluation at V\* = the
problem's volfrac**, mean ± sd over seeds; MMA (native β continuation, deterministic, one
run) is the baseline.  The previous headline ("QUBO-sa 0.163 beats MMA 0.173 on cantilever
MC 25", single seed, β = 8 re-smoothing, 768 elements) is withdrawn.

* **S3 headline (design-size meshes, 5 seeds).** cantilever_beam MC 36 (1,925 el., 5
  thick), V\* = 0.300: MMA 0.1296; QUBO-sa (block) **0.1277 ± 0.0009** (−1.4 %); QUBO-sa
  (diag) 0.1270 ± 0.0012 (−2.0 %); BESO-sort control 0.1354 (+4.5 %).  mbb_beam MC 46
  (2,205 el., 7 × 7): MMA 0.3470; QUBO-sa (block) **0.3453 ± 0.0014** (−0.5 %); diag
  0.3446 ± 0.0021 (−0.7 %); BESO-sort 0.3507 (+1.1 %).  Reading: under a crisp, equal-
  volume evaluation the binary QUBO update is **on par with MMA** (differences of 0.5–2 %;
  the seed sd is 0.3–0.9 %, but MMA is a single deterministic run whose sensitivity to its
  own settings is not measured, so this is not evidence of superiority); the second-order
  model gains 1.5–6 % over first-order BESO sorting; the exact block Hessian gives **no**
  gain over the cheap diag bound (and costs 4–6× the time: 157 s vs 38 s per run).
  **Accuracy of the proxy (round-2 verification):** a refined-voxel FE of the same crisp
  fine fields (cantilever MC 36 at 2×) gives MMA 0.0946, QUBO-sa diag 0.0953 (+0.8 %),
  BESO-sort 0.1020 (+7.9 %): the ranking holds, but the crisp proxy flatters the binary
  methods by ≈ 1–4 points, so the −0.5 … −2 % above mean *indistinguishable*, not better.
  On FreeTO's *native* metric QUBO is worse at equal volume: cantilever QUBO-sa block
  0.1638 ± 0.0004 @ V 0.283 vs MMA interpolated on its c–V curve at that volume 0.1455
  (**+12.6 %**; diag +14.5 %, BESO-sort +20.5 %); mbb 0.4224 @ 0.278 vs 0.4007 (**+5.4 %**;
  diag +7.5 %, BESO +7.2 %).  (The first version of this note said +22–29 %: the MMA (c, V)
  pairs mixed the compliance of the once-smoothed iterate with the volume of FreeTO's
  *twice-smoothed* returned field, ≈ 0.02 lower — N1 of the round-2 review; fixed in
  `study.py`, which now uses `hist["volfrac"][-2]` for OC/MMA and records the returned
  field's volume as `volume_fraction_returned`; results/quick was rebuilt with `python -m
  freeto.study --reprocess results/quick`.)  Side observation: FreeTO's returned OC/MMA
  `eleden` (smoothed twice at high β) is a much worse design than the one whose compliance
  it reports (`compliance_returned_design` 2–15× higher, e.g. cantilever MC 36 0.377 vs
  0.130; bridge 894 vs 0.039); F4's MMA surfaces show that twice-smoothed field.  The
  native metric smooths the binary design at β = 8 and SIMP-penalises its grey boundary,
  while MMA ends at β ≈ 30–35 nearly crisp.
* **Repair share (m7):** per iteration the QUBO-sa runs flip 42–55 elements (BESO-sort
  117–138) and the greedy volume repair adds only 0.4–0.8 of them (BESO 0.05–0.13): the
  steps are the QUBO's own choices (summary.md columns flips/it, repair+/it).
* **OC (FreeTO defaults)** stops at β_final = 2 (MC 36/46; 0.7 at MC 25) with a grey field;
  its crisp projection disconnects (2.36, 132 → "diverged").  Reported, flagged, never a
  baseline.
* **S4 robustness** (3 seeds; no MMA c–V curve; single MMA run).  l_bracket MC 30 (1,920
  el.): MMA 0.2733, QUBO-sa 0.2476 ± 0.0008 (−9.4 %), BESO-sort 0.2515 (−8 %).  GE_bracket
  MC 24 (776 el.): MMA 0.0845, QUBO-sa 0.0685 ± 0.0001 (−19 %), BESO-sort 0.0701 (−17 %).
  The binary BESO pipeline itself (not the QUBO) carries these margins, and MMA here
  stops after 36 (GE) / 78 (L) iterations with an iterate volume of 0.345 / 0.344 > 0.30
  (the stopping rule fired before the volume constraint was met) — these MMA runs are not
  converged at V\*, so the negative gaps must not be read as "QUBO beats MMA".  **bridge_deck MC 31 (840 el.) still fails under the crisp
  evaluation**: with load protection + move limit + guard the QUBO-sa *native* compliance
  is 0.019–0.022 (MMA native 0.039, MMA crisp 0.0110, i.e. ≈ 2× MMA crisp; before the fix
  1.6–80), all designs are connected in their own FE model, but the crisp projection of
  the binary field cuts the isolated deck elements that keep the loaded nodes attached
  (crisp c 14–180 → diverged).  Every top node is loaded and the deck is 1 element thick at
  this mesh; an element-binary design cannot represent MMA's sub-element deck.  BESO-sort
  diverges likewise.  Validated range of the continuum QUBO update: beam-like and bracket
  problems at 0.8–2.2 k elements; truss-like problems with distributed loads on thin
  regions remain open.
* **Block QAOA, penalty mode (the only QPU-sized formulation).**  It no longer diverges
  when started from the solid design: l_bracket crisp 0.287 (+5 % vs MMA), GE 0.0732
  (−13 %), bridge native 0.0254 (crisp diverged, as SA), 8-qubit blocks, p = 1, 1 seed.
  The earlier divergence (c 1e3–1e4) came from `init="oc"`: thresholding a 6-iteration grey
  OC field to a binary design at the target volume *is* the catastrophic step (c 1,194 /
  1,666 / 4.07 in the iteration after the threshold); the QUBO iterations then recover only
  partly (0.31 / 24.8 / 1.24).  `init="oc"` should not be used on these problems.
* **QAOA raw vs + greedy (S3q, cantilever MC 25, 10-qubit blocks, penalty):** identical
  designs (crisp 0.1879, +29 % vs MMA 0.1454; SA on the same block model 0.1664, +14 %);
  raw best shot = exact block optimum in 98 % of blocks, the polish never improved one —
  at n = 10, 1,000 shots nearly enumerate the 1,024 states.  In the truss loop raw and
  polished QAOA reach the same compliances (T1 exact, T2s +17 %), except that on T2s
  p = 3 **+greedy** ended in a mechanism in 1 of 3 seeds (raw: 0 of 3).
* **QAOA scans (F2) vs baselines.**  Approx. ratio of ⟨E⟩ 0.85–0.95 at p = 1–5 vs **0.52–
  0.56 for the uniform superposition (p = 0)**; P(opt) rises from the uniform fraction
  (6e-5 – 4e-3) to 3e-3 – 0.28.  The best of 1,000 shots is optimal on **every** instance
  (n = 8–14), as is greedy polish; greedy descent alone (16 random restarts, no QAOA) ends
  at the optimum in 69–100 % of its restarts.  QAOA here is a correct but expensive local
  sampler; nothing indicates an advantage.
* **Trusses with the kinematic stability test (S1/S2, 5 seeds).**  T1: every method exact.
  T2s: every iterative method (QUBO exact/SA/tabu/greedy/QAOA *and* sorting) ends at the
  same compliance 11.6569, **+17.0 %**, but not the same design: the QUBO variants return
  {0,1,2,4,5,8,10} (bar 0 joins two fixed nodes and carries no force), sorting
  {1,2,5,7,8,10}; OC + rounding and OC + QUBO rounding end in
  mechanisms.  T2 (22 bars): sorting +8.0 %, QUBO-tabu +9.9 % (1 of 5 feasible), QUBO-sa /
  greedy / exact-in-blocks and both OC roundings: mechanisms.  T3 (55 %): only enumeration
  finds a stable design (7.8467).  T4 (118 bars): QUBO-sa reaches the best known design in
  3 of 3 seeds, tabu +0…3 %, sorting and OC + rounding are mechanisms; T5: no method finds a
  stable design.  The Hessian helps only on T4; the previous "Hessian matters, sorting
  +111 % on T2s" was an artefact of the mechanism convention.
* **Penalty sensitivity (F5).**  Trusses: λ_q 0.1–3 × default gives the same result as
  bisection (T1 exact, T2s +17 %), 10 × degrades T2s to +92.5 %.  Continuum γ = 0.05 on
  cantilever MC 25 worsens the crisp compliance (0.175 vs 0.150) although it improves the
  native one (0.194 vs 0.218) — another case where the native metric misleads.
* **Reproducibility / thread count (N3).**  Runs are deterministic for a fixed seed *and*
  thread count only, and the effect is large: QUBO-sa diag seed 0 on cantilever MC 25 gives
  crisp 0.150 with 1 thread and 0.175 with 2 (+17 %; the SA/guard trajectory is chaotic in
  the last bits of the FE solution).  Single-seed MC 25 numbers (S3q, F5) are therefore one
  sample of a ≈ ±10 % distribution; only the MC 36/46 five-seed means have a measured
  spread.  The study now pins the thread count (`run_study(threads=1)` / `--threads`,
  threadpoolctl + OMP/OPENBLAS/MKL/NUMEXPR/VECLIB env vars) and records it in
  `results.json["machine"]["threads"]`; the quick run used OMP_NUM_THREADS=1.  F4 had to be rebuilt by re-running the lowest seeds
  with the default thread count (surfaces are not stored in results.json); the cantilever,
  mbb, L and GE designs are identical, but the chaotic bridge_deck runs differ (QUBO-sa
  seed 0 crisp 41.9 instead of 179.5, block QAOA 389 instead of 305 — both still
  "diverged"); F4's titles show the results.json values.  Also the S4 block-QAOA runs were
  re-run with `init="solid"` after the main run and merged into results/quick
  (`results.json["note"]`); `python -m freeto.study --suite quick` now produces them.
* **Timing (F6).**  CPU seconds of this 2-core container; QPU time n/a everywhere; QAOA
  "solver" time is classical simulation + angle optimisation.  No speed claim.

## 7. Continuum study protocol (fix round; `freeto.study`, `freeto/evaluate.py`)

1. Meshes: quick suite S3 at cantilever MC 36 (1,925 elements) and mbb MC 46 (2,205);
   S4 robustness at bridge_deck MC 41 (MC 31 before 2026-10-02, §8), l_bracket MC 30,
   GE_bracket MC 24; smoke stays
   tiny (MC 20).  The design's ≥ 8 elements through the thickness are only reached by
   the `full` suite meshes (MC 40/50) — not run here.
2. Seeds: 5 for the stochastic S3 methods, 3 in S4, mean ± sd (sample sd).
3. Baseline: MMA with its native β continuation (deterministic) + an MMA c–V curve at
   volfrac 0.26/0.28/0.32/0.34 (S3), used to give MMA's native compliance at each other
   method's native volume (log-linear interpolation, "MMA @ native V").  OC is run with the
   FreeTO defaults, its actual final β is recorded (`beta_final`), and it is flagged.
4. Headline metric: common crisp evaluation (`eval_crisp`): final pre-smoothing filtered
   field → node averaging → 4× fine grid (exactly smoothedge3D's xg) → 0/1 with the
   threshold bisected so that the window-averaged element densities have volume fraction
   V\* (achieved within 2e-4) → one FE solve, same SIMP model.  It is the β → ∞ limit of
   FreeTO's own volume-preserving projection at a common volume; it does not depend on each
   method's final β and does not cut sub-element members like element thresholding did.
   Diverged = crisp c > 10× MMA or singular (gap "n/a").  Native c and V are reported next
   to it; the former `eval_binary` column is no longer used by the study (it mixed two
   different quantities across families).  Native (c, V) of OC/MMA = the reported iterate
   (`hist["volfrac"][-2]`), not FreeTO's twice-smoothed returned field.  The proxy flatters
   binary designs by ≈ 1–4 points (refined-voxel check); the summary's verdicts compare MMA
   with the QUBO seed spread and call differences below 4 points indistinguishable — MMA is
   a single deterministic run, not a noise-free reference.
7. Reprocessing: `python -m freeto.study --reprocess DIR` rebuilds summary.md, results.csv
   and the figures from `DIR/results.json` (QUBO runs are never re-run; OC/MMA records from
   before the volume fix are re-run once, seconds each, to recover their volume history and
   their compliance is checked against the stored value; F4 is kept because surfaces are
   not stored).
5. Controls: BESO-sort (same pipeline, separable model solved by sorting); "QUBO-qaoa
   (raw)" vs "(+greedy)" vs SA on the same block-penalty model with `verify_exact`.
6. Remaining limits: MMA is one deterministic run per problem (no parameter study); S4 has
   no MMA curve and its MMA runs are not converged at V\*; bridge_deck is not solved by any
   binary method under the crisp evaluation; the quick suite takes ~70 min on 2 cores
   (S3 QUBO-sa block 140–230 s per run).

## 8. Corrections of 2026-10-02 (re-run of `results/quick`)

The physics audit (`docs/PHYSICS_AUDIT.md`) found one defect in the QUBO update and two
setup defects that hit every method. All issues found so far are corrected, and the quick
suite was re-run from scratch: 202 runs in 99 min on 2 cores with `--threads 1`, while
other jobs ran on the second core.

### What was corrected

1. **Connectivity repair of the binary update** (`QUBOOptions.connectivity = True`,
   §3.18 / PHYSICS_AUDIT §5). Every QUBO and BESO row of the study now uses it; the old
   rows predate it.
2. **Example setups** (`examples/make_examples.py`, `examples/EXAMPLES.md` "Corrections").
   Region slabs are now sized from the exact FreeTO grid, so that each contains the
   outermost element-centre and node layer at every mesh_control from 24 to 100.
   - **bridge_deck**
     - The kept deck slab and the support pads used to hold no element centre at
       MC 24/25/30/31, so MusD was 0 at the old study mesh.
     - The deck is now 7.41 mm deep, the pads 7.61 mm high and still 8.7 mm long.
     - **volfrac 0.2 → 0.35**, because the kept deck row is 17–33 % of the domain.
     - **Study mesh MC 31 → 41.** At MC 31 the captured deck is 25 % of the domain, which
       leaves under 9 % of the budget for the load path. At MC 41 the deck is 17 % (plus
       3 % for the pads), with 2,400 elements.
   - **l_bracket**: the support slab is clipped to the arm. It used to put 76 keep
     elements outside the domain at MC 30.
   - **mbb_beam** and **multi_load_beam**: the load and support patches now hold element
     centres at coarse meshes.
3. **Setup validation in the core.** WARNINGs are logged for a kept region with no
   element, for keep elements outside the domain, and for keep regions ≥ volfrac. This is
   only reported; the numerics are unchanged.
   - The OC/MMA FE solve now falls back to `robust_solve`'s iterative refinement. This
     applies only when the plain solve would abort the run as singular. The trigger was the
     mbb MMA c–V points at V = 0.26 and 0.28 (relative residual 3.5e-6 in MMA's near-void
     transient). These were re-run and merged; `results.json["note"]` records it.
   - `tests/compare_ref.py`: 6/6 PASS.
4. **Physics check (`freeto/audit.py`, `docs/AUDIT_API.md`).** Every run is audited on
   its crisp design at V\* for:
   - face-connected components and the floating share;
   - loads on solid material and on the supported body;
   - rigid-body restraint and captured regions;
   - the volume target.

   It reports `res.extra["audit"]`, `info["audit_live"]` (QUBO) and the summary columns
   *audit*, *comps*, *float %* and *loads solid*. A run that fails a physics check is
   labelled **physically invalid**.
   - Validated on the old broken case (bridge MC 31, old STLs, V = 0.2, `connectivity=False`,
     seed 0): the check reproduces the audit (crisp c 179.5, loads on solid 95.6 %, binary
     design 22 components, 19 % ungrounded) and fails it.
5. **Study** (`freeto/study.py`): bridge at MC 41, the audit fields in records and CSV,
   the protocol text and the labels. `--reprocess` still works on old and new results.

### New headline numbers

Crisp c at V\* (mean ± sd over seeds). MMA is one deterministic run. *audit* = runs that
pass every check. The QAOA rows are one seed, 8-qubit blocks in penalty mode; on the
cantilever the QAOA row is S3q at MC 25 (10-qubit blocks).

| problem (mesh, elements, V\*) | MMA | QUBO-sa (block) | QUBO-qaoa (+greedy) | BESO-sort | audit |
|---|---|---|---|---|---|
| cantilever MC 36 (1,925, 0.30) | 0.1296 | 0.1277 ± 0.0006 (−1.5 %); diag 0.1277 ± 0.0012 | MC 25: 0.1850 vs MMA 0.1454 (+27.2 %); SA on the same block model 0.1527 (+5.0 %) | 0.1489 (+15.0 %) | 16/17 (OC with FreeTO defaults: physically invalid, 4 crisp components, 7 % floating) |
| mbb MC 46 (2,205, 0.30) | 0.3500 | 0.3480 ± 0.0035 (−0.6 %); diag 0.3541 ± 0.0059 (+1.2 %) | – | 0.3531 (+0.9 %) | 17/17 |
| bridge_deck MC 41 (2,400, 0.35) | 0.003104 | 0.003394 ± 0.000019 (**+9.3 %**) | 0.004338 (+39.8 %) | 0.003412 (+9.9 %) | 6/6 |
| GE_bracket MC 24 (776, 0.30) | 0.08449 | 0.06830 ± 0.00006 (−19.2 %) | 0.08777 (+3.9 %) | 0.06937 (−17.9 %) | 5/6 (MMA: volume off target, native V 0.345) |
| l_bracket MC 30 (1,920, 0.30) | 0.2660 | 0.2535 ± 0.011 (−4.7 %) | 0.2664 (+0.1 %) | 0.2532 (−4.8 %) | 6/6 |

**Native volumes.** MMA is on target (0.300) except GE (0.345) and L (0.306). QUBO-sa:
cantilever 0.283, mbb 0.278, bridge 0.363, GE 0.315, L 0.295. At those volumes MMA's
native c–V curve gives +12.7 % (cantilever) and +6.3 % (mbb) for QUBO-sa (block) on the
native metric. The mbb curve points at V = 0.26 and 0.28 were missing in the first pass.

**Audit pass rates (all continuum runs).** MMA 13/14 (14/14 physics), QUBO-sa 32/32,
QUBO-qaoa 5/5, BESO-sort 5/5, OC 1/2. Every QUBO and QAOA design is one grounded body with
all loads on it. Exceptions:
- mbb QUBO-sa seeds 3 (block) and 2 (diag): a second crisp speck of 0.01 %.
- GE QAOA: a second component of 0.98 % on another bolt support. It is grounded and
  carries no load, so it is a separate supported body, not floating material. The 1 %
  floating rule passes it narrowly.

The binary designs of every QUBO and BESO run are a single component, except on GE: 2–3
components with 0.4–1.3 % on a separate bolt support, grounded.

### Reading

- **Bridge deck.** It is no longer a failure.
  - With the deck kept and the connectivity repair on, every binary method gives a
    connected, grounded design at MC 41.
  - QUBO-sa is **+9.3 % above MMA**, with all 3 seeds above it (+8.8 … +10.0 %), and equal
    to BESO-sort. So the QUBO model adds nothing over first-order sorting here, and MMA is
    better.
  - Block QAOA is +40 %.
  - The previous "diverged (crisp 14–180)" result was the empty keep region plus the
    disconnecting update. It was not a limit of binary designs.
- **Cantilever and mbb.** These remain **indistinguishable from MMA** (−1.5 % and −0.6 %,
  below the 1–4 point bias of the crisp proxy).
  - BESO-sort is now clearly worse on the cantilever (+15 % vs +4.5 % before). The
    connectivity repair changes its trajectory, so the second-order QUBO model now gains
    15 % over sorting there.
- **GE and L.** The negative gaps still do not mean "QUBO beats MMA".
  - GE: MMA stops off target (native V 0.345), and BESO-sort carries the same margin.
  - L: the corrected support no longer adds 76 phantom keep elements. MMA moves from
    0.2733 to 0.2660, QUBO-sa from 0.2476 ± 0.0008 to 0.2535 ± 0.011, so the L margin
    shrank from −9.4 % to −4.7 % and is within the seed spread of 4 points.
- **QAOA.** Block QAOA is never better than SA on the same pipeline.
  - S3q: raw and +greedy give identical designs; the raw best shot is the exact block
    optimum in 97.5 % of blocks.
- **Unchanged.** The truss (F1), QAOA-scan (F2) and penalty (F5) findings of §6 are
  unchanged by these corrections: truss and QAOA-scan code was not touched, and the
  numbers reproduce.

### Not changed or still open

- The web app's custom-study default mesh for bridge_deck is still MC 31
  (`webapp/server.py`, owned by the web-app side). The corrected example is feasible there,
  with 8 % free volume.
- FreeTO's volume constraint ignores the keep elements, which are set to 1 after
  filtering. OC on the corrected bridge at MC 70 ends at V = 0.382 against 0.35, and the
  audit reports "volume off target". That is MATLAB-faithful and was left unchanged.
- F4 still renders FreeTO's twice-smoothed returned fields. These can show pieces
  (L-bracket MMA, GE QAOA) that are attached in the crisp design the audit checks; see the
  audit figure of a job for the crisp view.

## 9. v2 changes (2026-10-04)

Specification: `paper/review/P0_CODE_SPEC.md` (manuscript v2).  New behaviour is opt-in
(`FreeTOConfig` options, `QUBOOptions.hessian="scalar"`, suite `quick2`); the
MATLAB-faithful defaults are unchanged (`tests/compare_ref.py` 6/6).  The one change of a
default is the truss ground structures (§9.4).  Tests: `tests/test_v2.py` (new),
`tests/test_truss.py` / `tests/test_quantum.py::test_study_smoke` updated.

### 9.1 Refined binary voxel evaluation (`freeto/evaluate.py`)

`refined_voxel_compliance(full_pre, Hn, Hns, nelx, nely, nelz, ele, target, F, fixeddof,
KE, E0, Emin, ngrid=4, f=2, solver="auto", steps=60, return_solid=False, bc_map="interp")`,
the scratch script `scratch_qverify2/refined_fe.py` as a module function:

* `xg = fine_field(full_pre, ..., ngrid)` (the crisp protocol's 4x node field); f^3 voxels
  per active coarse element; voxel value = mean of xg at its 8 corners (corner spacing
  ngrid/f fine points; f must divide ngrid); solid iff value > t, t bisected (60 steps) to
  the target solid fraction over the active region (achieved value returned).
* Binary moduli: solid `E0`, void `Emin`, no SIMP exponent.  `run_freeto` passes as `Emin`
  the modulus its FE model gives the crisp void density 0.001 (SIMP: 0.001 + 0.001^p
  (E0 − 0.001) ≈ 1e-9 E0 for the E0 = 210 GPa examples; SEMDOT: 0.001 E0 + 0.999·0.001 E0),
  i.e. the void of the crisp proxy and of the scratch script.  The package `Emin = 0.001`
  is an absolute modulus (FreeTO), so "E = Emin" literally would be a 5e-15 contrast at
  E0 = 210e9.
* Refined element stiffness `KE / f`; `freeto.fe.Assembler` + `make_solver(solver)`,
  `robust_solve`, then up to 3 iterative-refinement steps with the stored factorisation
  (`multi_rhs_solve`); returns `compliance, volfrac, threshold, n_voxels, ndof_free,
  residual, time`.
* **Boundary-condition mapping (deviation from the spec, which asked for coincident
  nodes).**  `bc_map="coincident"` applies every coarse nodal load and fixed DOF only at the
  coincident refined node: a clamped face becomes a grid of point supports and a
  distributed load a set of point loads on the refined mesh, and the compliance grows
  without bound under refinement.  Solid 6x3x3 cantilever (distributed tip load, clamped
  face): f = 1/2/4 → 11.386 / 20.721 / 34.923 (+82 %, then +69 %); with the default
  `bc_map="interp"` (a refined node is fixed in DOF d iff all coarse nodes of the smallest
  coarse vertex/edge/face/cell containing it are fixed in d; each load case's region is
  refined the same way and every coarse nodal load is spread over the refined region nodes
  with its trilinear weights, normalised so that it is conserved; a uniform distributed load
  stays uniform) 11.386 / 12.118 / 12.341 (+6.4 %, +1.8 %), so the spec's own acceptance
  test ("f = 1 vs f = 2 within 15 %") holds only with "interp".  Other blocks: 4x2x2 +12.6 %
  / +3.7 %, 8x4x4 +3.9 % / +1.1 %, 12x4x4 +3.4 % / +1.0 % (coincident: +47 … +123 %).
  The two mappings are identical at f = 1.  Note: refinement makes the model *softer*
  (higher compliance, more DOFs); the spec's "f = 2 is lower" was read as "softer".
* Cantilever MC 25, MMA (quick2 options, 79 it.): crisp proxy 0.14299; refined f = 1 / 2 / 4
  = 0.10324 / 0.09770 / 0.09660 (interp) and 0.10324 / 0.10817 / 0.12096 (coincident).
  MC 36: crisp 0.12694; refined f = 1 / 2 = 0.09645 / 0.09252 (interp), 0.09645 / 0.09706
  (coincident).  The refined voxel compliance is ~30 % below the crisp proxy (the proxy's
  window-averaged boundary elements are SIMP-penalised), and with "interp" f = 2 is within
  1.2 % of f = 4.  The values of `VERIFICATION_quantum_round2.md` §2 (MC 36, MMA 0.0946,
  coincident mapping, different MMA run) are of this size.
* Cost at f = 2 for the quick2 meshes: 0.7–4.4 s and < 2 GB per evaluation (PARDISO, 1
  thread; bridge_deck MC 41: 65 k free DOFs).  f = 4 at MC 36 exhausts the 7 GB container.
* `extra["full_pre"]` is float32 (spec): re-evaluating a saved field can move ties of the
  threshold; on the cantilever MC 25 MMA design the f = 2 compliance is identical to 6
  digits, on a symmetric 12x4x4 test box with loads on void it moved by 0.09 %.

### 9.2 `freeto/core.py` options

* `eval_refined=f` (None = off): after the crisp evaluation, `refined_voxel_compliance` on
  the same `full_pre`, target = the crisp target (`eval_crisp` value, else volfrac),
  `bc_map="interp"`, solver = `cfg.solver`; wrapped in `_optional_eval` (failure → inf and
  `eval_errors["eval_refined"]`).  Extras `refined_compliance, refined_volfrac,
  refined_threshold, refined_time, refined_f` (+ `refined_target, refined_residual,
  refined_ndof_free`).
* `mma_constraint="filtered"`: fval = Σ full_pre[ele] / (vol · nnele) − 1 with
  full_pre[ele] = (H x)/Hs and the active keep elements at 1.  Reason (code comment): the
  projected densities come from smoothedge3D, whose threshold preserves the volume of the
  whole bounding-box grid, not of the active elements, so the projected volume drifts from
  the one MMA controls on non-prismatic domains.  Gradient: the exact derivative
  dfdx = H (w / Hs) / (vol · nnele), w = 0 on keep elements, 1 elsewhere (H symmetric).
  This equals the spec's `H @ (1/Hs)` away from keep regions; the spec's "zero for keep
  elements" (dfdx of a keep design variable = 0) was replaced by the exact derivative, which
  differs only for elements within rmin of a keep region (a keep variable still feeds its
  non-keep neighbours' filtered densities).
* `mma_feasible_stop=True`: the change / topology tolerances end an MMA run only if
  |fval| ≤ 1e-3, where fval is evaluated on the design the loop would return (the new
  iterate) with the constraint definition MMA uses (filtered: full_pre; projected: the once-
  smoothed field); `max_iter` still ends the run.  `extra["mma_fval_final"]`.
* `init_perturb=a`, `init_seed=s`: x0 = volfrac + U(−a, a) (`default_rng(s).uniform(-a, a,
  nnele)`), clipped to [0.001, 1], keep positions left at volfrac; OC and MMA (QUBO starts
  solid and ignores it).
* `extra["full_pre"]` (float32 copy, length nele) for every run; MMA runs report
  `extra["mma_constraint"]`.
* Effect of the constraint (MMA, 100 it., quick2 options vs default; native V = projected
  iterate, filtered V = Σ full_pre[ele]/nnele):

  | example | constraint | it. | native c @ V | filtered V | crisp c | refined c (f = 2) |
  |---|---|---|---|---|---|---|
  | cantilever MC 25 | projected | 67 | 0.14723 @ 0.300 | – | 0.14545 | 0.09249 |
  | cantilever MC 25 | filtered + stop | 79 | 0.18220 @ 0.274 | 0.300 | 0.14299 | 0.09770 |
  | mbb MC 46 | projected | 86 | 0.35104 @ 0.300 | 0.327 | 0.35003 | 0.28462 |
  | mbb MC 46 | filtered + stop | 64 | 0.40406 @ 0.272 | 0.300 | 0.34339 | 0.28805 |
  | l_bracket MC 30 | projected | 78 | 0.26702 @ 0.306 | 0.322 | 0.26603 | 0.17927 |
  | l_bracket MC 30 | filtered + stop | 100 | 0.31253 @ 0.283 | 0.300 | 0.25993 | 0.17973 |

  With the projected constraint the filtered field FreeTO hands to smoothedge3D carries
  2–3 points more volume than V*, with the filtered constraint the projected iterate carries
  2–3 points less; crisp / refined compliances at the common V* move by −2 … +6 %.

### 9.3 `hessian="scalar"` (`freeto/quantum/update.py`, `options.py`)

Same curvature D as "diag" (element-local bound), then Q_c → diag(Q_c): all couplings
zeroed (separable control).  Added to `HESSIAN_MODES` (validated) and the CLI choices.
Test: the captured QUBO has no off-diagonal entries; "diag" has.

### 9.4 Truss ground structures (`freeto/truss/benchmarks.py`)

`drop_fixed=True` for every ground structure (`_grid_cantilever` default and gs_9x3;
tower3d / column3d already had it; ten_bar has no bar between its two pinned nodes —
checked).  A bar between two fully fixed nodes carries no force but its length entered the
volume budget `vmax_fraction · Σ L`, so the budgets shrink and the optima were re-pinned with
`enumerate_exact`:

| problem | bars | Σ L | V_max | C_EXACT (old → new) | optimal bars (new indices) | stable feasible designs |
|---|---|---|---|---|---|---|
| ten_bar | 10 | 4196.47 | 2727.70 | 629.4701293633094 (unchanged) | 0 2 3 6 7 8 | 5 |
| gs_3x2 | 13 → **12** | 17.129 → 16.129 | 8.5645 → 8.0645 | 9.963761829267339 → **11.656854216862358** | 0 1 3 4 7 9 | 37 → 23 |
| gs_4x2 | 22 → **21** | 33.754 → 32.754 | 13.5016 → 13.1016 | 23.891529700380538 → 23.89152970038057 (same design) | 0 1 4 5 9 10 12 13 15 16 18 | 629 |
| tower3d | 22 | 27.041 | 14.8726 | 7.846666208853116 (unchanged) | – | 292 |
| gs_9x3 | 118 → **116** | 185.64 → 183.64 | 64.97 → 64.27 | no exact reference | | |
| column3d | 66 | 87.21 | 26.16 | no exact reference | | |

The former gs_3x2 optimum (volume 8.243) exceeds the new budget.  Consequences on gs_3x2
(seed 0): the iterative QUBO update (exact / sa / tabu / greedy blocks) now ends on a
**mechanism** (bars 0 1 4 7 9 10 11; node (1, 0) held by two collinear bars; linear-analysis
compliance 11.6569 = the optimum, `stable=False`) → infeasible, gap n/a; first-order sorting
reaches an optimal design (bar 2-3 instead of the zero-force bar 1-2 of the enumerated
optimum) with gap 1.48e-9 — the difference is the 1e-9 void area of the stiffness model.
**Caution:** the study counts a run as an "exact hit" only if gap ≤ 1e-9, so this sort run
is not counted as a hit; a tolerance of 1e-6 would count it (not changed here).
`tests/test_truss.py` updated (counts, optima, mechanism / prune indices, the QUBO and
sorting regressions above, callback length 12); `test_study_smoke` now expects QUBO-exact
on gs_3x2 infeasible.

### 9.5 Study suite `quick2` (`freeto/study.py`)

`suite_spec("quick2")` = the "quick" list transformed by `_quick2` (264 runs from 114
entries; "quick": 203 / 99):

* every continuum run (S3, S3q, S4, F5 γ) gets `max_iter = 100` and
  `run_options = {"eval_refined": 2, "mma_constraint": "filtered",
  "mma_feasible_stop": True}` (new run key, passed through `_run_continuum` into
  FreeTOConfig; allowed keys `RUN_OPTION_KEYS`);
* S3q (cantilever MC 25) QUBO/QAOA controls: seeds 0–4 (the MMA baseline stays 1 run);
* the mbb_beam QAOA run (index 98 of "quick") sits after the mbb_beam S3 block;
* D5 controls after every S3/S4 example block (cantilever 36, mbb 46, bridge_deck 41,
  l_bracket 30, GE_bracket 24; not added to the S3q / F5 cantilever-25 runs):
  "BESO-sort (move 0.04)" = QUBO hessian "none", move_limit 0.04, 1 run, role control;
  "QUBO-sa (scalar)" with the seeds of the block variant (S3: 0–4, S4: 0–2), role control;
  "MMA (init s)" = one entry with seeds 0–4, `init_perturb = 0.05`, init_seed = run seed,
  role `baseline_spread` (excluded from the MMA c(V) curve and from F3/F4);
* truss and qaoa_scan runs are identical to "quick" (they use the new ground structures).

Records gain `refined_compliance, refined_volfrac, refined_threshold, refined_f,
refined_time, mma_constraint, mma_fval_final, init_perturb, init_seed, fields_file`;
`_postprocess` adds `c_ref_refined` / `gap_refined` (vs the MMA baseline's refined
compliance, same problem / mesh / V*; None if missing or > 10x, and None for the whole
example if the baseline's own refined c exceeds 10x its crisp c — seen on an unconverged
40-iteration MMA at cantilever MC 20, refined 1.07e5 vs crisp 0.464, which made every gap
−100 %); `_summary` rows add
`refined_mean/sd, refined_V_mean, gap_refined_mean/sd` and the MMA spread of the example
(`mma_spread_n, mma_spread_crisp_mean/sd, mma_spread_refined_mean/sd`).  `summary.md`:
two new table columns (refined c, gap vs MMA (refined)), an "MMA spread" table and the
refined gap in the verdicts; `results.csv` has the refined columns.  With `out_dir`, every
continuum run writes `fields/<run_id>.npz` (full_pre float32, binary_design uint8 for
QUBO/BESO, history arrays, grid; ~5 kB at MC 25).  CLI: `--suite quick2`, `--dry-run`.

Smoke (1 thread): S3q cantilever MC 25 MMA with the quick2 options — 79 iterations,
|fval| = 2.5e-6, crisp 0.14299 @ 0.3000, refined 0.09770 @ 0.29997 (0.6 s), field file
5.1 kB; truss gs_3x2 exact 11.656854216862358 (bars 0 1 3 4 7 9, 23 stable designs).

### 9.6 Truss kinematic repair (P0b; `freeto/truss/optimize.py`)

`kinematic_repair(fe, x, V_target)` and option `kinematic_repair: bool = False` of the
truss QUBO / sorting update (`solve_truss(..., kinematic_repair=True)`; binary designs only,
ignored for `nbits` > 1; not used by `exact`, `oc`, `oc_round`, `oc_qubo`).  After each
update (after the trust-region attempts have chosen the new design):

1. prune dangling bars (`TrussFE.prune`);
2. test = the package kinematic test (λmin > 1e-9 λmax of the unit-area stiffness on the
   free DOFs of the touched nodes) and every loaded node connected to a support; while it
   fails, add the absent bar with the largest λmin/λmax (ties: smallest volume).  When no
   single bar removes every zero mode the ratio is 0 for all candidates; the bar leaving the
   fewest zero modes and the most loaded nodes connected is then taken first (lexicographic
   key: passes, #zero modes, #loaded nodes connected, ratio, −L);
3. only if bars were added and the volume exceeds the iteration's target V_k: remove the
   present bar with the smallest |g_b|/L_b (g recomputed at the current design) whose
   removal (+ pruning) keeps the test passing; repeat until within target or nothing is
   removable.

A design that passes the test is returned unchanged (0 added / 0 removed).  Per-iteration
counts are stored as `qubo_stats[i]["kin_repair_added"/"kin_repair_removed"]`; study
records get `repair_added` / `repair_removed` (means over iterations; only for runs with
the repair on).  quick2 turns it on for every truss QUBO / sorting run (S1, S2, F5
penalty: 126 expanded runs), not for OC rounding or exact enumeration.

Results, seed 0 (c_exact gs_3x2 = 11.656854, gs_4x2 = 23.891530, tower3d = 7.846666):

| problem | method / backend | repair off | repair on | repair +/− per it. |
|---|---|---|---|---|
| gs_3x2 | QUBO exact / sa / tabu / greedy | infeasible (mechanism, bars 0 1 4 7 9 10 11) | **feasible, c = 11.656854, gap 0.0** (bars 0 1 3 4 7 9 = the exact optimum) for all four | 0.17 / 0.33 |
| gs_3x2 | sort | feasible, gap 1.5e-9 | same design, gap 1.5e-9 | 0.45 / 0 |
| gs_4x2 | QUBO sa | infeasible (c 29.99, V 0.361) | **feasible, c = 33.9879, gap +42.26 %** | 0.25 / 0.12 |
| gs_4x2 | sort | feasible, gap +15.04 % | unchanged (+15.04 %) | 0 / 0 |
| tower3d | QUBO sa | infeasible (c 2.6e7, mechanism) | **stable but infeasible: V 0.649 > 0.55**, c = 6.6486 | 1.60 / 0.20 |
| tower3d | sort | infeasible (c 2.8e9) | stable but infeasible: V 0.656, c = 22.84 | 2.33 / 0 |
| ten_bar | QUBO sa / sort | exact (gap 0) | unchanged | 0 / 0 |

tower3d: at V_max the update proposes a 12-bar mechanism (V 0.523), the repair adds 3
bars (V 0.649) and no single-bar removal keeps the design stable, so the run cycles until
`patience` and returns the last (over-volume) design.  A "cheapest repairing bar first"
add-back rule (tried, not adopted: the spec asks for the largest λmin/λmax) gives the
same outcome on tower3d (c 7.148, still over volume) and identical results on gs_3x2 /
gs_4x2.  Tests: `tests/test_truss.py::test_kinematic_repair_*`.

### 9.7 `QUBOOptions.hessian_scale` (P0b)

`hessian_scale: float = 1.0` (validated finite ≥ 0) multiplies Q_c (block and diag parts)
before the standard form; the perimeter term γ is not scaled.  quick2 control
"QUBO-sa (block, Qx3)" (hessian "block", hessian_scale 3.0, seeds 0–4, role control) on
the S3 examples cantilever MC 36 and mbb MC 46 only.  Test: the captured off-diagonal
QUBO at scale 3 equals 3x the one at scale 1.

### 9.8 Truss exact-hit tolerance (P0b)

`freeto.study.HIT_TOL = 1e-6`: a feasible truss run with gap ≤ 1e-6 counts as a hit (was
1e-9; designs that differ only in zero-force bars differ by ~1e-9 through the 1e-9 void
area, e.g. sorting on gs_3x2, gap 1.48e-9).  Summary rows keep `n_exact_hits` and add
`hit_tol`; the summary.md truss table header reads "hits (gap ≤ 1e-06)" and gains a
"repair +/− per it." column; CSV columns `repair_added`, `repair_removed`.

### 9.9 Parallel study execution (P0b)

`run_study(..., jobs=N)` / CLI `--jobs N`: the expanded runs go to a
`ProcessPoolExecutor` (spawn context) with at most N runs in flight; each worker pins its
BLAS/OpenMP/MKL pools to `--threads` (default 1) with `_ThreadPin` for its whole life and
runs exactly the serial code path (`_execute_run`, seeds from the spec; exact truss runs are
cached per worker), field files are written by the workers.  Records are collected in the
original run order; `results["machine"]["jobs"]` = N.  A run that raises is an `error`
record (as before); a crashed worker (BrokenProcessPool) turns the in-flight runs into
`error` records and the pool is replaced, so the study continues.  `stop_event` stops
submitting and waits for the running runs.  Both paths log a timestamped progress line per
finished run (`[YYYY-mm-dd HH:MM:SS] [k/n] #idx run_id: … (run s; study s)`) and rewrite
`results.json` every `PARTIAL_EVERY = 5` finished runs (`"partial": true`, `n_done`,
records so far in run order; atomic replace), before the run_end callback, so a crash loses
at most 4 runs; `reprocess` reads a partial file.  Spawned workers re-import `__main__`: a
script calling `run_study(jobs>1)` needs the usual `if __name__ == "__main__":` guard
(without it every worker dies — this was seen and is handled: all runs become error
records, the study still finishes).

Verification (spec of 7 runs: gs_3x2 QUBO-sa with repair ×2 seeds, cantilever MC 20 MMA
with the quick2 options, cantilever MC 20 QUBO-sa block ×2 seeds, one qaoa_scan, one
failing run): jobs = 2 gives records identical to jobs = 1 in every field except the
timings (wall/solver/update/hessian/fe/refined time; compared as JSON), identical field
files (array-equal), the failing run as an error record in both; 9.0 s vs 16.8 s.
`tests/test_v2.py::test_parallel_study_equals_serial` checks the same on three truss runs.

quick2 now lists 274 runs from 116 entries (130 continuum, 144 truss / QAOA scan).

### 9.10 Fresh worker processes and `--resume` (P0b, after the first quick2 launch)

The first `--jobs 2` quick2 launch lost runs to OOM kills: a long-lived worker grew to
4.7 GB, i.e. memory accumulates across runs in one process.  Changes in `freeto/study.py`:

* `run_study(..., fresh_workers=False)`; with True every run executes in a fresh spawned
  worker (`ProcessPoolExecutor(..., max_tasks_per_child=1)`, same initializer / thread
  pinning), also for `jobs=1` (a pool of one).  The CLI default is `--fresh-workers`
  (`--no-fresh-workers`: `--jobs 1` runs in-process as before, `--jobs N` reuses workers).
  The Python API default stays in-process serial (web app, tests).
  `results["machine"]["fresh_workers"]`.
* `--resume DIR` / `resume_records(DIR, spec)`: loads `DIR/results.json` (partial or
  complete), keeps every run of the current expanded spec whose records are all error-free
  (derived fields stripped and recomputed), runs only the missing and errored runs (same
  run ids), writes the complete `results.json` (`"partial": false`), `results.csv`,
  `summary.md` and figures to DIR with all records in spec order; field files of kept runs
  stay.  Spec: `--spec` / `--suite` if given, else the spec stored in results.json.  The log
  says how many runs were kept / are re-run (errored / missing).  F4 shows only the designs
  run in the resumed session (surfaces are not stored).
* Test `tests/test_v2.py::test_resume_runs_only_missing_and_failed` (3-run spec: one record
  deleted, then one replaced by an error record; each resume runs exactly that run in a
  fresh worker; final file has the 3 records in spec order, compliances unchanged).  CLI
  check: 3-run spec (truss sort, truss QUBO-sa with repair, cantilever MC 20 MMA), error
  injected into the QUBO record → "2 of 3 runs kept, 1 to run (1 with errors, 0
  missing)", 1 run executed, final `partial` false, MMA field file kept.

Usage for the interrupted study: `python -m freeto.study --resume results/quick2 --jobs 2`
(the stored quick2 spec is used).

### 9.11 Refined evaluation: loads act on the material present (2026-10-04, Fable)

`refined_voxel_compliance(..., load_on_solid=True)` (default): with `bc_map="interp"` the
trilinear weights that spread a coarse nodal load over its refined region are restricted to
refined nodes attached to at least one solid voxel (weights renormalised, load conserved); a
coarse node whose region touches no solid keeps the unrestricted weights. Reason: the cantilever
Qx3 seed-4 design has grey elements (rho 0.16-0.42) under part of the load patch; on the coarse
grid these carry the load with SIMP moduli, but at f = 2 some of their voxels are void and the
interpolated load then sat on void nodes (E = 1e-9 E0), giving c_ref = 149 against a proxy of
0.128. With the restriction c_ref = 0.0917; designs whose load nodes all touch solid voxels are
unchanged to all digits (MMA, QUBO-sa block, BESO-sort seed 0 checked). Records computed before
this change are recomputed from the stored fields by `scripts/recompute_refined.py`.
