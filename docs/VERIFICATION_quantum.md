# VERIFICATION — QUANTUM extension (independent adversarial review)

Reviewer: independent verifier (not the implementer). Scope: `freeto/quantum/*`,
`freeto/truss/*`, `freeto/study.py`, the QUBO hooks in `freeto/core.py`,
`tests/test_quantum.py`, `tests/test_truss.py`, `results/quick/*`, `docs/QUANTUM_*.md`,
`docs/NOTES_quantum.md`. Nothing in the code was modified; all checks are scripts in
`<workspace>/scratch_qverify/` (`check_qaoa.py`, `check_qaoa_baseline.py`,
`check_model.py`, `check_model2.py`, `check_truss.py`, `fairness.py` → `fairness.json`,
`robustness.py` → `robustness.json`, logs `*.log`). `webapp/` was not touched or reviewed.

## 0. Verdict in one paragraph

The code is largely correct as *software* (QAOA simulator, Metropolis kernel, tabu, block
Gauss–Seidel, truss FE, exact enumeration, volume handling all check out against independent
implementations; 171 tests pass; the original OC/MMA numerics are untouched, `compare_ref`
524/524). The *scientific claims* in `results/quick/summary.md` and `NOTES_quantum.md §6` are
overstated: the headline "QUBO-sa (block) 0.163 beats MMA 0.173 on cantilever MC 25" is a
single-seed result (seeds 1, 2 give 0.181 / 0.180, worse than MMA and BESO), obtained at a
mesh that is 4 elements thick (768 elements), compared through an evaluation protocol
(re-smoothing at β = 8 and element-level thresholding) that penalises the continuous methods
and flatters the binary ones, and against an OC baseline that stopped after 13 iterations at
β = 0.75 with an entirely grey design. Under FreeTO's own reported metric MMA dominates QUBO
on both study problems (cantilever 0.147 @ V 0.280 vs 0.163 @ 0.301; MBB 0.283 @ 0.270 vs
0.405 @ 0.292). On three problems the implementers did not use (bridge_deck, l_bracket,
GE_bracket at 776–960 elements) the default SA/block update is 50–100× worse than MMA on the
truss-like bridge and diverges with block QAOA in penalty mode on all three. Two real bugs
were found (a Windows crash when writing `summary.md`; un-caught solver exceptions in the
post-run `eval_beta/eval_binary` that destroy a finished run), plus a convention problem in
the truss "stability" test that makes the pinned T2s "exact optimum" a kinematic mechanism.

---------------------------------------------------------------------------
## 1. Findings (prioritised)

### BLOCKER

None that invalidates the code as a research tool, provided the claims below are corrected
before the study is used or published.

### MAJOR

**M1. Overstated headline claim (single seed; volume not equalised).**
`results/quick/summary.md` §Continuum, `docs/NOTES_quantum.md:193-199`. The S3 rows are
1 seed each (`freeto/study.py:116-128`, `_continuum_runs(..., [0], ...)`). Re-running
`QUBO-sa (block)` on cantilever MC 25 with seeds 0/1/2 gives c(β=8) = 0.1634 / 0.1813 /
0.1803 at V(β=8) = 0.3011 / 0.2955 / 0.2980 (`fairness.json`), i.e. mean 0.175 ± 0.010 —
indistinguishable from BESO sorting (0.1774 @ 0.302) and MMA re-smoothed at β = 8 (0.1732 @
0.3014). Volumes at β = 8 are not reported anywhere (the record stores only `volume_fraction`
= own-β volume; `volfrac_at_beta` is computed in `core.py:675` but dropped in
`study.py:339-349`). Fix: ≥ 5 seeds for every stochastic method, report mean ± sd, report
V(β=8) next to c(β=8), and state the seed-0 value only as one sample.

**M2. The evaluation protocol is not fair to OC/MMA and flatters QUBO/BESO.**
`study.py:326` (`eval_binary=True, eval_beta=8.0`), `core.py:670-683`, summary text
`study.py:712-717`.
* *Binary column.* Thresholding a fine-grid Heaviside design (MMA at β = 34) to the
  `volfrac·nnele` densest elements cuts every member thinner than one element: for MMA on
  cantilever MC 25 the 230 kept elements split into a 202-element component on the support
  and a 28-element island holding the load (diagnosed with a connectivity check), hence
  c = 1.07e6. This is an artefact of thresholding a design that has sub-element features at a
  4-element-thick mesh, not a property of MMA. For QUBO/BESO the same column *removes* the
  grey-band SIMP penalty of their smoothed design (0.1085 vs 0.1634), so "binary c" is a
  different quantity for the two families and must not be printed side by side as a quality
  measure. The sentence "Binary-thresholded OC and MMA designs lose the load path" (NOTES
  §6, summary) should be deleted or re-attributed to the protocol.
* *β = 8 column.* Re-smoothing MMA's β = 34 design at β = 8 *raises* its compliance from
  0.147 to 0.173 while raising its volume from 0.280 to 0.301: the re-smoothing adds grey
  material that is SIMP-penalised. The "common β" is therefore not neutral either: it is the
  QUBO run's own β and the point where MMA's design is evaluated worst.
* *OC row.* OC's 2.64 (and 0.51 at own β) is an artefact: with the FreeTO OC defaults the
  cantilever MC 25 run stops after 13 iterations by `change ≤ tolx = 0.003` while β is still
  0.75 (not 2, as stated in `summary.md`, `NOTES §6` and the F4 caption `study.py:946`);
  472 of 768 elements are in [0.1, 0.3) — no topology has formed. OC with the QUBO/MMA β
  schedule (0.5 → 8) diverges in this code (c → 1.7e4, NaN volume), so "run OC with
  continuation" is not available as a fix; the OC baseline should be reported as "stopped
  grey at β = 0.75, not comparable" or dropped, not shown as +1514 %.
* Fairest protocol: see §3 below.

**M3. Robustness: the default update fails on unseen problems.** (`robustness.json`)
| problem (nnele) | MMA c @ V (own β / β=8) | QUBO-sa block default | QUBO diag | BESO-sort | block-QAOA kb=8 penalty |
|---|---|---|---|---|---|
| bridge_deck MC 31 (840) | 0.0097 @ 0.189 / 0.0127 @ 0.200 | **1.64 @ 0.195** (best of 47 it; last 78.6; 50–84 flips/it at the end; thresholded design disconnected) | 0.547 | 61.0 | 1164 (diverged) |
| l_bracket MC 25 (960) | 1.53 @ 0.368 / 1.71 @ 0.360 | 0.49 @ 0.344 (good) | – | – | 1.06e4 (diverged) |
| GE_bracket MC 24 (776) | 0.088 @ 0.365 / 0.102 @ 0.343 | 0.110 @ 0.314 (ok) | – | – | 1.52 (diverged) |
The "truss-like continuum" case D3 of the design (bridge with keep-deck) is exactly where the
QUBO update is 50–100× worse than MMA with every Hessian option and never settles (the
returned design is the best-of-history). Every penalty-mode block-QAOA run — the only mode
usable on a QPU — diverges at MC 24–31, so the NOTES advice "use MC ≥ 25" (§6) is not
sufficient. Fix/claim: state that the update is validated only on cantilever/MBB at 750–770
elements; add bridge_deck to the quick suite so the failure is visible; treat penalty mode
as experimental.

**M4. Truss "stability" admits kinematic mechanisms; the pinned T2s optimum is one.**
`freeto/truss/fe.py:110-118` (`stable = residual ok and loaded nodes graph-connected and
c ≤ 1e3·c_full`), `optimize.py:128-133`. An independent brute force with a rank test on the
stiffness of the touched free DOFs (`check_truss.py`) gives for gs_3x2 (T2s) the optimum
c = 9.9638 with bars {1,2,4,5,7,8,10} (37 kinematically stable designs), whereas the package
reports c = 9.0897 with bars {1,2,5,6,8,10} (293 "feasible"). In the package's design node
(1,0) is joined only by the two collinear bars 0–2 and 2–4 (the two halves of the bar 0–4
that the overlap filter removed): its transverse DOF has zero stiffness (eigenvalue 1e-9 from
the area floor), i.e. the truss is a mechanism whose zero-energy mode happens to be
orthogonal to the load. Linear analysis cannot see it, so the check `c ≤ 1e3 c_full`
passes. ten_bar (T1) is unaffected (both methods give 629.470 with bars {0,2,3,6,7,8}).
Fix: either add a kinematic check (rank of K restricted to the DOFs of touched free nodes
equals their count, or Maxwell count + eigenvalues) to `TrussFE.evaluate`, `enumerate_exact`
and the trust-region test, and re-pin `C_EXACT`; or document that "stable" means
"finite compliance under the given load in linear analysis" and that collinear hinged
chains are accepted. The claims "QUBO-sa reaches the exact optimum on T2s" and "sorting is
+111 %" refer to this convention.

**M5. Un-caught solver exceptions in the post-run evaluations destroy finished runs.**
`core.py:635-646` (`_fe_compliance`) and `core.py:670-683` call `robust_solve` without the
`except Exception → FreeTOError` wrapper used in the loop (`core.py:510-519`), and without
try/except around the optional evaluations. Reproduced: OC with β → 8 on cantilever MC 25
raises `PyPardisoError(-4)` (PARDISO) / `RuntimeError: Factor is exactly singular`
(SuperLU) from `eval_beta`; MMA at volfrac 0.26 on mbb_beam MC 31 raises `FreeTOError` from
`eval_binary`. The optimisation itself had finished; `run_freeto` then raises and the
result (and the study record, `study.py:335`) is lost. Fix: wrap each optional evaluation
in try/except, store `inf`/`None` plus the message in `res.extra`, and wrap solver library
exceptions into `FreeTOError` as the loop does.

**M6. Windows: `summary.md` cannot be written.** `study.py:682` `open(pm, "w")` without
`encoding="utf-8"`; `summary_markdown` emits `−`, `∞`, `≤`, `β`, `⟨E⟩` (`study.py:661,
664, 696, 712-731`). Under the default Windows locale (cp1252; Python < 3.15) this raises
`UnicodeEncodeError: 'charmap' codec can't encode character '−'` after all runs
completed (verified by encoding the current summary with cp1252). The design requires
Windows laptops. Fix: `encoding="utf-8"` on every text `open()` in `study.py:468, 673, 676,
682, 1058` (and `newline=""` is already there for the CSV).

**M7. The `qaoa` backend's reported solution is largely the classical greedy polish.**
`backends.py:214-234`: after sampling, `greedy_descent` from the best shot is run and its
result replaces `x, E` (`info["polished"]=True`). Measured: n = 16 / 20, p = 1: best shot not
optimal, polished result exactly optimal (gap 2e-16). In the truss/continuum loops this
makes "QUBO-qaoa is exact on T1" (summary S1, NOTES §6) a statement about QAOA + greedy
descent; F2's ratios and P(opt) are pure QAOA and fine. Also, on the study's own instances
the uniform superposition (p = 0) already has approx ratio 0.51–0.66 and 16-restart greedy
descent finds the exact block optimum in 69–100 % of blocks (`check_qaoa_baseline.py`), so
"approx ratio 0.86–0.95" must be read against that baseline. Fix: expose `polish=False`,
report `best_shot_energy`/`best_shot_optimal` in the study tables, print the p = 0 baseline
ratio in F2, and say explicitly that the in-loop QAOA numbers include the greedy polish.

**M8. NOTES §3.1 gives a wrong reason for a (empirically) right decision.** The argument
"for E ∝ ρ³ at ρ = 1 the quadratic Taylor model predicts no stiffness loss (1 − 3 + 3 = 1)"
applies only to an *unfiltered* 0→1 change of ρ_e. With the density filter a design flip
changes each element density by J_ej ≤ ~0.33, where the ρ-space Taylor model is accurate.
Finite differences on a 192-element filtered problem (`check_model.py`, base = a QUBO design,
p = 3, truth = c(SIMP(Jx)) for single flips, adjacent pairs and 4–12-element clusters):

| model | single removal: median rel. err / model÷true | single addition | adjacent pair removal | 8-element cluster |
|---|---|---|---|---|
| `beso` + block (default) | 0.71 / 0.28 | 0.71 / 0.30 | 0.74 | 0.87 / 0.14 |
| `simp` + block | **0.05 / 0.94** | **0.08 / 1.05** | **0.12** | 0.49 / 0.56 |
| `secant` + block | 0.43 / 1.64 | 22 / −16 (wrong sign) | 0.28 | – |
| `beso` + none (BESO) | 0.75 / 0.24 | 0.67 / 0.35 | 0.80 | – |

The default `beso` weights (E′/p) underestimate every compliance change by ≈ 1/p; the
design-literal `simp` model is the accurate local model (rank correlations are high for
both, which is why sorting/λ-bisection still works). *However*, in the actual loop the
`simp` model is unstable — 4 of 6 seeds diverge on cantilever/MBB (c = 10–600) — because the
evaluated design is the Heaviside-projected one (β up to 8), not J x: a boundary flip then
changes the projected density by ≈ 1, which is precisely the regime where the ρ-space model
fails and the secant-modulus model is right; `secant` fails for additions (Neumann series
diverges in void). So the deviation is justified, but by the projection, not by the
"1 − 3 + 3" argument; §3.1 should be rewritten and the NOTES should also say that the
`beso` model is *not* "the exact secant expansion" for partial (filtered) density changes.
Also `NOTES §3.2` says cross-patch couplings of `hessian="block"` come "from the diag part":
the code sets `D = 0` for `block` unless `interp="simp"` (`update.py:446-453`), i.e. there
are no cross-patch couplings. Either the note or the code is wrong; the code's choice
(no double counting) is the sensible one, fix the note.

### MINOR

**m1.** `study.py:570-600`: continuum "gap" is measured against the best run of the same
(problem, MC). The MC 20 penalty demos both diverged, so `QUBO-qaoa demo` (c = 1004) is
printed with gap 0.00 % (summary line 79). Gaps should be relative to the best run of the
problem at any label, or the demos flagged infeasible (c > 10 × best) instead.

**m2.** Timing semantics differ between backends: for `qaoa`, `timing["solver"]`
(`backends.py:226`) includes the spectrum enumeration and the COBYLA parameter optimisation
(i.e. "solver s" for QUBO-qaoa p = 3 in the S1 table = 7.0 s is ~100 % classical
optimisation); for `qiskit_aer`/`ibm` the same classical optimisation runs *before* `t`
(`backends.py:399-405`) and is excluded from `solver` but included in `wall`. Document
("solver = whole variational loop for `qaoa`") or move the angle optimisation outside for
both. QPU fields are `None` for all local backends (verified) and rendered "n/a" in F6; the
disclaimer paragraph is present and correct; no reader of the tables could infer a speed-up
(the QAOA rows are the slowest), but the summary should say once that all "solver" times are
CPU seconds of a state-vector simulation.

**m3.** `backends.py:56-59`: `dwave_sa`/`dwave_tabu` list only `dwave.samplers` as required,
so `available_backends()` reports them available when `dimod` is missing; the call then
raises `QuantumBackendUnavailable("dwave-samplers is not installed")` — misleading message.
Add `"dimod"` to the module lists. A non-`ImportError` failure inside an optional SDK's
import (e.g. a broken wheel raising `RuntimeError`) propagates raw; consider `except
Exception` in the lazy imports.

**m4.** `core.py:695`: for OC/MMA `res.comp` is the compliance of the *previous* iterate
(FreeTO convention), for QUBO it is an extra solve on the returned design. Converged runs
make the difference negligible, but mixing the two in one table should be footnoted.

**m5.** `truss/optimize.py:236-241`: `nbits=0` is silently treated as 1; `nbits ≥ 2` gives
non-binary "on" (areas 1/3, 2/3) that `TrussResult.on = areas > 0.5` rounds; harmless but
document. Error handling for unknown backend / method / option names is clean (`ValueError`
with the list of known names; `QuantumBackendUnavailable` with install/token hints).

**m6.** `summary.md`/F4 caption: "OC stays at β = 2" — with the example configs OC reaches
β = 0.75 (cantilever, 13 it.) or 2 (MBB, 59 it.); "topo stop" is not what happened
(the `change ≤ tolx` criterion fired).

**m7.** `blockqubo.py:301`: the greedy repair fills to `V_k` even when the model energy
rises (NOTES §3.6); this makes the returned volume exact but means every iteration ends with
a few forced additions the QUBO did not choose — fine, but the study's `repair_added` counts
are not reported; they would tell how much of each step is the QUBO's and how much the
repair's.

**m8.** `study.py:1017-1041` (F6): bars are means over runs with very different sizes; the
per-run scatter would be more honest. `NOTES §5` timing table is consistent with what I
measured (SA block update ≈ 1.4 s / iteration at 768 elements).

**m9.** `examples.py` differs from the copy in `/mnt/user-data/outputs/FreeTO_Python.zip`
(six generated examples + `category` key) — that is the separate examples extension, not
this one; `mesh.py, filters.py, smoothedge.py, optimizers.py` are byte-identical and `fe.py`
differs only by the `self.last_lu = lu` line (`fe.py:270`). OK.

---------------------------------------------------------------------------
## 2. What was verified and how (with results)

| item | method | result |
|---|---|---|
| Tests | `pytest tests/test_quantum.py test_truss.py test_freeto.py test_examples.py` + webapp/playwright files | 129 + 42 passed, 4 skipped = **171 passed** (claim confirmed); quantum+truss 38 tests, 75 s |
| Original numerics | `FREETO_REF_DIR=… tests/compare_ref.py`; `diff` vs zip | **524/524** checks pass on 6 reference cases; core modules unchanged except `fe.py:270` |
| QUBO normal form, Ising, BQM | package tests + my QAOA script | round trips exact |
| SA Metropolis | analytic (accept iff `dE < −ln u/β` ⇔ `u < e^{−β dE}`) and empirical Boltzmann sampling at fixed β on a 4-variable QUBO, dense and chromatic paths | max |freq − Boltzmann| 0.014 / 0.009 = sampling noise |
| Pair-flip polish | analytic: ΔE(a,b) = ΔE_a + ΔE_b + 2 Q_ab s_a s_b | correct |
| Tabu | code read (aspiration, tenure, restart from perturbed best, exact re-evaluation of best) | correct |
| QAOA state | my own full-matrix (Pauli/expm) implementation, n = 2,3,4, p = 1..3, random angles | max |Δψ| = 4e-16; the RZZ/RZ/RX circuit convention in `qaoa_circuit` matches the diagonal cost (difference is a constant phase) |
| Approx ratio / P(opt) | definitions read; baselines computed | ratio = (E_max − ⟨E⟩)/(E_max − E_min), P(opt) counts all degenerate optima; uniform-state baseline 0.51–0.66 on the study's instances (M7) |
| QAOA at n = 20 | n = 16 p = 1 (50 evals) 2.1 s; n = 20 p = 1 (30 evals) 12.1 s; memory fine | "works up to n = 20" confirmed; best shot not optimal at n ≥ 16, polished result optimal (M7) |
| Exact solver | package test vs 2^n loop (n = 5, 10, 17) | confirmed |
| Block Gauss–Seidel / penalty bookkeeping | package `FakeBackend` test read line by line: block linear term h_B + 2 Q[B,¬B] y_¬B + λ v_B + λ_q(v_B² + 2 r v_B), acceptance only if block energy decreases | correct |
| λ-bisection / multisection / repair | code read; `test_volume_model_bisection_is_feasible_and_exact_small`; study volumes | returned v·y ≤ budget always; repair fills to within one element; final volumes 0.2955–0.3020 at β = 8 (the spread comes from smoothing, see M1) |
| Continuum Hessian (Gram term) | package FD test (p = 1 beso, p = 3 simp) read; my model-vs-truth FD (M8) | Gram term is the exact ∂²c/∂E² mapped through J; PSD within patches (by construction); the `diag` bound 2w²c_e/E_e² is a valid Loewner bound (derivation checked: c_e in the code is the unit-modulus strain energy) |
| Truss FE | 2-bar (c = P²√2/EA, dc/dx = −c/2, Hessian diag = c, off-diag = 0 — note the exact separable form c ∝ 1/x₁ + 1/x₂) and 3-bar (c = Px²/Kxx + Py²/Kyy) by hand | rel. err 2e-16; gradient/Hessian vs FD 3e-8 / 5e-10 |
| Exact enumeration | my own brute force with kinematic rank test | ten_bar identical (629.4701, {0,2,3,6,7,8}); gs_3x2 **differs** (M4) |
| Mechanism detection | 3-bar with only the vertical bar under an x-load | c = 1e9, `stable=False` (caught by c ≤ 1e3 c_full); collinear hinge chains are *not* caught (M4) |
| Truss error handling | unknown backend/method/option, cloud backends without SDK, `nbits` 0/2 with sa/exact/qaoa/sort | clean errors; multi-bit works (T2s exact with 2 bits, c = 9.0897) |
| Optional deps | dwave/qiskit absent here; fake broken packages injected | `available_backends()` never raises; cloud backends raise `QuantumBackendUnavailable` (m3) |
| Portability | grep for `/home/claude`, `/mnt`, POSIX APIs; `os.makedirs`, `os.path.join`; `matplotlib.use("Agg")` before pyplot (`study.py:753-755`) | no hard-coded paths, no POSIX-only calls; **UTF-8 bug (M6)** |
| Fairness re-runs | `fairness.py` (cantilever MC 25, MBB MC 31; OC/MMA/BESO/QUBO × interp × 3 seeds; MMA volume curve) | tables in M1/M2 and `fairness.json` |
| Robustness | `robustness.py` (bridge_deck 31, l_bracket 25, GE_bracket 24; SA-block and block-QAOA) | M3 |

Fairness table (cantilever_beam MC 25, 768 elements; `c` = FreeTO compliance, `V` = reported
volume fraction; `c8/V8` = re-smoothed at β = 8):

| method | c (own β) | V (own β) | c8 | V8 | it | β_end |
|---|---|---|---|---|---|---|
| OC default | 0.5125 | 0.3097 | 2.637 | 0.3018 | 13 | 0.75 |
| OC β 0.5→8 | diverges (c 1.7e4, NaN) | | crash in eval (M5) | | 19 | 8 |
| MMA default | **0.1472** | **0.2799** | 0.1732 | 0.3014 | 67 | 34 |
| MMA β ≤ 8 | 0.1749 | 0.2809 | 0.1749 | 0.3000 | 80 | 8 |
| MMA volfrac 0.32 | 0.1404 | 0.2959 | 0.1585 | 0.3253 | 80 | 40 |
| BESO-sort | 0.1774 | 0.3020 | 0.1774 | 0.3020 | 34 | 8 |
| QUBO-sa block, beso, s0/s1/s2 | 0.1634 / 0.1813 / 0.1803 | 0.3011 / 0.2955 / 0.2980 | same | same | 51/52/47 | 8 |
| QUBO-sa block, simp, s0/s1/s2 | 28.1 / 0.1895 / 10.6 | | | | | |
| QUBO-sa block, secant, s0 | 29.7 | | | | | |

MBB MC 31 (750 elements): MMA 0.2827 @ 0.2702 (β=8: 0.3195 @ 0.3077); BESO 0.3833 @
0.2949; QUBO-sa block beso 0.4064 / 0.4030 / 0.4052 @ 0.292–0.294; QUBO simp 2.1 / 595 /
258; OC 0.5846 @ 0.2907 (β = 2, 59 it.).

---------------------------------------------------------------------------
## 3. Scientific assessment

**What the study legitimately shows.**
1. On small trusses with an exact reference, a QUBO with the exact (Gram) Hessian and λ
   bisection, solved by SA/tabu/exact, reaches the enumeration optimum on T1 and T2s (under
   the package's stability convention, M4) where first-order sorting does not (T2s +111 %);
   on T2 (22 bars) it is +7–9 % while continuous OC + keep-largest rounding is exact on T1,
   T2s and T2 and beats every QUBO variant. The Hessian matters; the *solver* does not
   (SA = tabu = exact on these sizes), and the simplest classical pipeline (OC + rounding) is
   at least as good on every 2-D truss with an exact reference. On T3/T5 all iterative methods
   (QUBO and classical) end in mechanisms — those benchmarks are pathological (18 stable
   designs in 1.2 M) rather than discriminating.
2. The QAOA state-vector simulator is correct, and its behaviour is what the literature
   predicts: p ≤ 5 with 100·p COBYLA evaluations gives approx ratio 0.86–0.95 (uniform
   baseline 0.5–0.66) and P(opt) 0.003–0.35 at n = 8–14; deeper p is optimiser-limited; the
   in-loop usefulness comes from best-of-1000-shots plus greedy polish, and 16-restart greedy
   alone already solves 69–100 % of these blocks. Nothing here supports "how good quantum
   can be" beyond "a 13-qubit QAOA at p ≤ 5 is a weak, expensive local search"; a real QPU
   run (S5) was not done.
3. For continua the study shows that a binary BESO-like update with a second-order model can
   produce reasonable designs at 750 elements on beam problems, comparable to BESO sorting
   and to MMA re-evaluated under the QUBO's own β, within seed noise; it does not show that
   the QUBO update beats MMA (MMA dominates at own β and at equal volume on both problems),
   and it does not generalise to the truss-like bridge (M3). Penalty-mode block solving —
   the only formulation a QPU could take — fails on every continuum case tried.
4. No timing conclusion is possible or claimed; the disclaimer is adequate (m2).

**What cannot be concluded.** Any advantage of the QUBO update over MMA; any statement
about quantum hardware; anything at realistic mesh sizes (the S3 meshes are 4 elements thick
and 750–770 elements, the design specified 2,268–9,000); optimality of continuum designs.

**Fairest comparison protocol (recommended).**
1. *Resolution and baselines.* Run S3 at the design's mesh sizes (≥ 2,000 elements, ≥ 8
   elements through the thickness) so that members are resolvable; run OC with enough
   iterations to form a topology (its FreeTO stopping rule fires at β = 0.75 here) or state
   that OC is not a converged baseline; use MMA with its β continuation as the primary
   classical reference; ≥ 5 seeds for SA/tabu/QAOA and report mean ± sd.
2. *Equal volume, common evaluation.* For every method keep the final *pre-smoothing*
   filtered field and evaluate it in **three** ways, always reporting the volume of the
   evaluated field: (a) FreeTO's own output (own β); (b) a common crisp evaluation that does
   not depend on element-level thresholding — project the fine-grid (4×) Heaviside field at
   the same high β for all methods with the volume-preserving threshold set to the *same*
   fine-grid volume, and run the FE on the refined voxel model (or, cheaper, on the element
   averages but with a threshold of exactly `volfrac·nnele` applied after re-smoothing); (c)
   the element-binary evaluation only for methods whose design is element-binary. Never mix
   (c) values across families in one column.
3. *Volume normalisation.* Where volumes still differ by > 1 %, compare against a
   compliance–volume curve of the reference method (I ran MMA at volfrac 0.26–0.34; at the
   QUBO's V the interpolated MMA value is the fair opponent) or report c·V^a bounds.
4. *Controls.* Keep BESO sorting (same pipeline, no QUBO solver) and "same QUBO, SA vs exact
   vs QAOA-without-polish" as the two controls; report `repair_added` and the fraction of
   flips decided by the QUBO vs by the repair; report P(exact block optimum) per backend.
5. *Truss.* Add the kinematic stability test (M4), re-pin the exact optima, and report
   OC+rounding as the primary classical baseline in every truss table.
