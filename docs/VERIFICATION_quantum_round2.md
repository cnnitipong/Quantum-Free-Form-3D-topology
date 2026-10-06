# VERIFICATION — QUANTUM extension, round 2 (after the fix round)

Reviewer: independent verifier (not the implementer). Scope: the fix round described in
`docs/NOTES_quantum.md` (§3.1/3.2, §3.14–3.17, §6, §7), `docs/QUANTUM_API.md` (change list),
`freeto/evaluate.py`, the QUBO hooks in `freeto/core.py`, `freeto/study.py`, `freeto/truss/*`,
`freeto/quantum/*`, `results/quick/*`, and — new in this round — the web app (`webapp/`,
not modified). No code was changed. Scratch: `<workspace>/scratch_qverify2/` (`recompute.py`,
`crisp_check.py` → `crisp_check.json`, `refined_fe.py` → `refined_fe_MC25.json`,
`refined_fe_MC36.log`, `truss_check.py`, `api_runs.py` → `api_runs.json`, `ui_check.py` →
`shots/*.png`, `compare_ref.log`, `pytest_all.log`, `web/` = server workdir).

## 0. Verdict in one paragraph

All eight MAJOR findings of round 1 are fixed or honestly re-scoped; the minor ones are fixed
except m7 (repair statistics are recorded in `results.csv` but not shown in `summary.md`).
Tests: **175 passed, 4 skipped** (webapp/playwright skips) and `compare_ref` **524/524** on the
six reference cases (claims confirmed). The crisp evaluation protocol is implemented as
documented and reproduces independently (threshold, fine field and element densities equal to
3e-16; V\* = 0.29999 for every method); a refined-voxel FE of the same crisp fields confirms
the ranking (MMA ≈ QUBO-sa ≈ +1 %, BESO-sort +8 %). The summary and NOTES are now careful and
mostly accurate; the one new substantive error is in the *native*-metric comparison: the MMA
(c, V) pairs used for the MMA c–V curve mix the compliance of the once-smoothed iterate
(V ≈ V\*) with the volume of FreeTO's twice-smoothed returned field (V ≈ V\* − 0.02), so
"QUBO +22–29 % worse on FreeTO's native metric" is overstated — with consistent (c, V) pairs
it is **+5–14 %** (N1 below). The headline crisp comparison is unaffected. The web app works
end to end with the updated core (new callback fields, re-pinned truss optima, smoke study with
F1–F7); the UI issues found are cosmetic/semantic, none blocks use.

---------------------------------------------------------------------------
## 1. Status of the round-1 findings

| id | finding (round 1) | status | evidence |
|---|---|---|---|
| M1 | single-seed headline, volumes not reported | **fixed** | 5 seeds (`study.py:201-216`), mean ± sd (`_summary`, `study.py:817-822`), crisp V and native V in every row (`summary.md` table); recomputed from `results.csv`: cantilever block 0.12773 ± 0.00089, diag 0.12700 ± 0.00119, mbb block 0.34533 ± 0.00138, diag 0.34461 ± 0.00211, GE 0.06847 ± 0.00005, L 0.24759 ± 0.00075 — all match `summary.md`; gap mean/sd (−1.4 % ± 0.7) recomputed from the per-seed gaps ✓ |
| M2 | evaluation protocol unfair (β = 8 re-smoothing, element thresholding, OC +1514 %) | **fixed** | new common crisp evaluation `freeto/evaluate.py:41-65`, hooked at `core.py:774-791`; `eval_binary` no longer used by the study (`study.py:401-402` passes `eval_crisp=True`); OC flagged, never a baseline (`study.py:152-153`, `_verdicts` uses `role == "baseline"` only); the "OC/MMA lose the load path" sentence is gone. See §2 for the independent validation and the residual (native-metric) issue N1 |
| M3 | default update fails on unseen problems (bridge, penalty-mode block QAOA diverges) | **partially fixed / honestly re-scoped** | load protection, adaptive move limit, accept-if-improves guard (`update.py:199, 229, 271-296`); bridge native c now 0.019–0.022 (was 1.6–80) but **crisp c 14–180 → "diverged" for every binary method** (`results.csv`, S4 bridge rows; BESO-sort crisp 0.30 = 27× MMA); block QAOA in penalty mode no longer diverges when started solid (`study.py:223-230`, L 0.287, GE 0.073); NOTES §6 states the validated range explicitly ✓ |
| M4 | truss "stability" admitted mechanisms; T2s optimum 9.0897 is a mechanism | **fixed** | `truss/fe.py:133-150` (`kinematic`: PD test of the unit-area stiffness on the free DOFs of touched nodes, `KIN_RTOL = 1e-9`), used in `stable_mask` (`fe.py:171-181`); my own brute force with an independent rank test (`truss_check.py`) gives **9.96376183 {1,2,4,5,7,8,10}, 37 stable designs = package** (`enumerate_exact` 37); the old design {1,2,5,6,8,10} now evaluates `stable=False`; T1 unchanged (629.470) |
| M5 | un-caught solver exceptions in post-run evaluations | **fixed** | `core.py:716-728` `_optional_eval` → `inf` + `extra["eval_errors"]`; `_fe_compliance` wraps library exceptions (`core.py:704-710`); `results.json`: no `eval_errors` in 230 records, OC crisp values 2.36/132 computed without crash; test `test_crisp_evaluation_equal_volume_and_failed_eval_keeps_run` |
| M6 | `summary.md` cannot be written on Windows (cp1252) | **fixed** | `study.py:893, 896, 902` `encoding="utf-8"`; no other text `open()` without encoding in `freeto/` (grep) |
| M7 | `qaoa` result is the greedy polish | **fixed** | `backends.py:221-251` (`polish` option; `best_shot_energy/ratio`, `polished_energy`, `polish_applied`, `polish_time` always reported); study runs "(raw)" and "(+greedy)" separately (`study.py:120-127, 162-175`), F2 shows the uniform (p = 0) and greedy-16 baselines; NOTES §3.16 says the in-loop numbers include the polish. Data: S3q raw = polished (0.1879), `exact_match_raw` 0.98 ✓ |
| M8 | wrong reason in NOTES §3.1; §3.2 cross-patch statement contradicts code | **fixed** | §3.1 rewritten with the filtered/projected-flip argument; §3.2 now says "no cross-patch couplings", consistent with `update.py:573-580` (D = 0 in block mode unless `interp="simp"`) |
| m1 | continuum gap vs best run of same label | **fixed** | gap vs MMA baseline crisp, `None` when diverged (`study.py:735-757`, `DIVERGED` = 10×) |
| m2 | QAOA timing semantics | **fixed** | documented (NOTES §3.16, API change list); qiskit backends now include the angle optimisation (`backends.py:419-420`); summary says "All times are CPU seconds" |
| m3 | `dimod` missing from D-Wave requirements; broken SDKs raise | **fixed** | `backends.py:62-68`; `except Exception` in lazy imports (`backends.py:50, 272, 306, 344, 415`) |
| m4 | `res.comp` semantics differ OC/MMA vs QUBO | **fixed (documented)** | API §1, PROTOCOL paragraph; `compliance_returned_design` recorded. But see N1: the *volume* paired with MMA's `comp` is of a different design |
| m5 | `nbits=0` silently 1 | **fixed** | `truss/optimize.py:272-274` raises `ValueError` |
| m6 | "OC stays at β = 2" | **fixed** | `beta_final` recorded per run (OC 2.0 at MC 36/46), F4 caption "β_final ≤ 2" |
| m7 | `repair_added` not reported | **partially** | `repair_added_mean`, `flips_mean` are CSV columns (`study.py:862-863`); not in `summary.md` tables |
| m8 | F6 bars = means over different sizes | **fixed** | per-run scatter (`study.py:1364-1395`) |
| m9 | examples.py vs zip | n/a (other extension) | — |

## 2. The crisp evaluation protocol (independent validation)

What the code does (`evaluate.py:28-65`, `core.py:626, 741, 779`): the *pre-smoothing* filtered
field of the returned design (`full_pre` = `H·vxnew/Hs` with keep elements = 1; for QUBO the
best-at-target iteration's field, for OC/MMA the last update's) → node averaging `Hn/Hns` →
4× linear upsampling (same `upsample_linear` as `smoothedge3D`) → 0/1 at a threshold bisected so
that the **window-averaged element densities over the active elements** (same `window_reduce`
as `smoothedge3D`, void 0.001) have volume fraction V\* → one FE solve with the same SIMP model.

Checks (`crisp_check.py`, cantilever MC 25, MMA and QUBO-sa diag seed 0, OMP_NUM_THREADS=1):
* My own fine field (scipy `RegularGridInterpolator`, trilinear) and my own explicit-loop
  (f+1)³ window averages equal the package's to 3.3e-16; my bisection finds the identical
  threshold (0.45387 MMA, 0.40790 QUBO); achieved V = 0.299992 for **both** methods
  (the record values 0.1454 (S3q MMA) and 0.1500 (F5 γ = 0) reproduce exactly).
* MMA's design is evaluated by exactly the same pipeline (same `full_pre` definition, same
  filter radius, same threshold rule); nothing is method-specific.
* Precision of the docs: it is the β → ∞ limit of FreeTO's projection *with the volume
  measured on the window-averaged elements over active elements*; `smoothedge3D` itself
  preserves the fine-grid *point* mean over all elements (at β = 1e4 it gives ls = 0.4543 vs
  the crisp threshold 0.4539 for MMA, 0.4232 vs 0.4079 for QUBO). Same for every method, so
  harmless, but "exactly smoothedge3D's volume" would be wrong; "at a common volume" is right.
* **Is the proxy fair?** The crisp element field keeps 39–46 % grey (window-averaged boundary)
  elements that SIMP penalises; a method with more surface could be penalised more. Refined-
  voxel FE of the *same* crisp fine fields (`refined_fe.py`: fine cell solid iff mean of its
  8 corner values > t, t bisected to voxel volume 0.300, loads/supports at the coincident fine
  nodes, KE/f; residuals 1e-12; f = 1 reproduces the package compliance to all digits):

  | problem | method | crisp proxy (package) | refined-voxel FE, equal voxel volume |
  |---|---|---|---|
  | cantilever MC 36, 2× (15,400 cells) | MMA | 0.12956 | 0.09459 |
  | | QUBO-sa (diag, seed 0) | 0.12902 (−0.4 %) | 0.09534 (**+0.8 %**) |
  | | BESO-sort | 0.13536 (+4.5 %) | 0.10203 (+7.9 %) |
  | cantilever MC 25, 4× (49,152 cells) | MMA | 0.14545 | 0.11337 |
  | | QUBO-sa (diag, seed 0, 2 threads) | 0.17510 (+20.4 %) | 0.12866 (+13.5 %) |

  (4× at MC 36 needs > 6 GB and was OOM-killed.) The proxy's ranking holds; it flatters the
  binary methods by ~1–4 percentage points relative to a refined evaluation. "On par with MMA
  (0.5–2 %)" survives; "QUBO lower than MMA by more than one sd" (summary verdicts) does not
  survive the refined check and should not be read as a difference.

## 3. New findings (prioritised)

**N1 (major, documentation/analysis, not code-breaking). The native-metric comparison pairs
MMA's compliance with the wrong volume.** For OC/MMA `res.comp` is the compliance of the
*once-smoothed* iterate that entered the last iteration (its volume ≈ V\* by the MMA
constraint), while `res.finalvol`/`volume_fraction` is the volume of FreeTO's **twice-smoothed**
returned field (`core.py:635-638`), which is 0.02 lower and whose own compliance
(`compliance_returned_design`) is 2–15× higher (cantilever MC 36: 0.1305 @ hist V 0.300 vs
returned V 0.2766 with c 0.3774; bridge 0.039 vs 894; `results.csv` columns
`compliance`, `volume_fraction`, `compliance_returned_design`). The "MMA native c(V) curve"
(`study.py:694-703`) is therefore shifted left by ≈ 0.02 in V, and "MMA @ native V" is too
optimistic. Re-interpolating with the once-smoothed volumes (= the runs' volfrac targets,
verified `hist["volfrac"][-2]` = 0.2600/0.2799/0.2998/0.3199/0.3399):

| problem | method | native c @ V | summary: MMA @ V → gap | corrected: MMA @ V → gap |
|---|---|---|---|---|
| cantilever MC 36 | QUBO-sa (block) | 0.1638 @ 0.283 | 0.1266 → +29.3 % | 0.1456 → **+12.6 %** |
| | QUBO-sa (diag) | 0.1659 @ 0.283 | 0.1263 → +31.3 % | 0.1449 → +14.4 % |
| | BESO-sort | 0.1721 @ 0.286 | 0.1247 → +38.0 % | 0.1429 → +20.4 % |
| mbb MC 46 | QUBO-sa (block) | 0.4224 @ 0.278 | 0.3484 → +21.2 % | 0.4007 → **+5.4 %** |
| | QUBO-sa (diag) | 0.4352 @ 0.276 | 0.3518 → +23.7 % | 0.4048 → +7.5 % |
| | BESO-sort | 0.4253 @ 0.279 | 0.3445 → +23.5 % | 0.3969 → +7.2 % |

Fix: for OC/MMA records use `hist["volfrac"][-2]` (volume of the design whose compliance is
reported) as `volume_fraction` in the study, or report (`compliance_returned_design`,
`finalvol`) as the pair; correct NOTES §6 ("+22–29 %" → "+5–14 %"), the summary's "MMA @ native
V" column and F7's MMA native curve. Side observation worth a sentence in the docs: FreeTO's
returned MMA/OC `eleden` (twice-smoothed) is a much worse design than the one whose compliance
it reports; F4's MMA surfaces are of that twice-smoothed field.

**N2 (minor, docs).** NOTES §3.14/§6 "every iterative method converges to the same stable design
{1,2,4,5,8,10}" — the compliances are the same (11.6569, +17.0 %) but the bar sets are not:
QUBO-sa/exact return {0,1,2,4,5,8,10} (bar 0 joins two fixed nodes), sorting {1,2,5,7,8,10}.
Also "in the truss loop raw and polished QAOA give identical results": on T2s p = 3 the
polished variant has 1 of 3 seeds infeasible (summary S1 table). Minor wording.

**N3 (minor, reproducibility).** NOTES §6 says runs are deterministic for a fixed seed *and*
thread count. The magnitude is worth stating: the same QUBO-sa diag seed-0 run on cantilever
MC 25 gives crisp 0.1500 with OMP_NUM_THREADS=1 and 0.1751 with 2 threads (+17 %); at MC 36
the seed-0 value with 1 thread reproduces the record exactly. So MC 25 single-seed numbers
(S3q, F5) are one sample of a ≈ ±10 % distribution; the MC 36/46 five-seed means are the only
ones with a measured spread.

**N4 (minor, protocol).** The verdict rule "better than MMA if mean + sd < MMA" treats a
deterministic MMA run as noise-free; given N1-type sensitivity of MMA to its own settings and
the refined-voxel result (§2), the summary's "lower than MMA by more than one sd" lines should
read "indistinguishable from MMA (within 1–2 %)". NOTES §6 already says this; `summary.md`'s
auto-verdicts do not.

**N5 (minor, docs).** `evaluate.py` docstring / PROTOCOL: "the element densities ... have exactly
the target volume fraction" — achieved within 1e-4 (step function), as the code comment says;
fine. `smoothedge3D`'s own volume rule differs (point mean); see §2 bullet 3.

## 4. Web app against the updated core (server `--port 8777`, Playwright headless, Chromium)

Works: `/api/health` (quantum/truss/study usable), `/api/quantum/backends` (5 local
available, cloud ones unavailable with reasons), continuum QUBO job (cantilever MC 20, SA/diag,
`eval_binary`) — the per-iteration `qubo` dict with the new keys (`move_limit`, `flip_cap`,
`n_proposed_flips`, `truncated`, `guard`, `n_protected`, `n_protected_added`,
`exact_match_raw`, `best_shot_ratio`, `repair_*`) serialises and is shown in the log/status bar
without errors; truss gs_3x2 exact (9.9638, gap 0.00 %, feasible) and QUBO-sa (11.657,
+16.99 %, feasible) and oc_round (9.0897, feasible **no**, gap "—": `inf` → `null` handled);
`/api/truss/benchmarks` shows the re-pinned `c_exact` (T2s 9.9638, T2 23.8915, T3 7.8467 @
0.55); Study tab smoke suite: 14 runs, 15 records, **F1–F7 all rendered** (img naturalWidth
> 0), results.csv/json/summary.md download links present; no page errors; only console
error is a 404 for `preview.stl?iter=0` before the first iteration (pre-existing, harmless).
Screenshots: `scratch_qverify2/shots/0*.png`.

UI breakage / issues for the web agent (none blocking):
1. **Truss results table mislabels rows**: `app.js:1698-1702` reads the *current* benchmark/
   method/backend selects when the job finishes; changing a select during a run labels the
   finished run wrongly (observed: QUBO-sa run shown as "oc_round / —"). Use `result.method`,
   `result.backend`, `result.problem` from `truss_result` instead.
2. **Study results table lacks the fix-round columns**: `app.js:1929-1930` shows
   `compliance`/`gap`/`feasible` only; for continuum rows `gap` is the *crisp* gap vs MMA while
   `compliance` is the native value, with no `crisp_compliance`, `volfrac_target`, `diverged`
   or `beta_final` column — a reader cannot tell what the gap refers to. Add those columns
   (and `c_ref`), or label the gap column "gap (crisp vs MMA / exact)".
3. **Study log shows `gap=None` for every continuum run and `F2 None — starting`**
   (`jobs.py:415-427`): continuum gaps are only computed in post-processing, and qaoa_scan runs
   have no label. Print `crisp=` when present and fall back to `study`/`problem` for the label.
4. **Quantum settings γ default = 0.02** (`index.html:183`) while the core default is 0 and
   the study (F5) shows γ = 0.05 worsens the crisp result; the UI silently sends 0.02. Set the
   field to 0 / placeholder "0 (default)".
5. New `QUBOOptions` fields (`qaoa_polish`, `protect_loads`, `guard`, `guard_tol*`,
   `move_limit*`, `max_rejects`) and `eval_crisp` are not exposed (`server.py:504-528`,
   `JobCreateRequest`); the API change list says "additions only", so this is a feature gap,
   not breakage — but the UI's "eval_binary" is the metric the study abandoned; exposing
   `eval_crisp` (and showing `extra["crisp_compliance"]` in the Done line) would match the docs.
6. Pre-existing layout bug (also visible in `webapp/screenshots/04_done.png`): the right column
   of the Parameters and Quantum-settings cards overflows the 380 px side panel
   (`#p-optimizer` right edge 476 px, `#qubo-hessian` 488 px, `overflow-x: hidden`) at every
   viewport width — "Optimizer", "Hessian mode", "Frontier fraction" values are clipped.
7. Cosmetic: the top status bar keeps the last continuum job's `qubo:` stats while on the
   Truss/Study tabs.

## 5. Portability re-check (changed files)

No hard-coded paths, no POSIX-only APIs (`fork`, `signal`, `fcntl`, `resource`) in
`freeto/evaluate.py`, `core.py`, `study.py`, `truss/*`, `quantum/*`, `webapp/*`; all text
`open()` calls carry `encoding="utf-8"` (study JSON/CSV/markdown); `matplotlib.use("Agg")`
before pyplot; labels with "/" are sanitised in run ids (`sort-BESO`) and figure/file names are
fixed ASCII; console log lines of `freeto.study` are ASCII (non-ASCII only in `summary.md`, the
figures and web job logs, which are UTF-8 / rendered). `platform.processor()` /
`os.cpu_count()` are fine on Windows. Nothing new found.

## 6. Scientific assessment (for the researcher)

1. Trusses: with a kinematic stability test the QUBO update is exact on T1 and on the 118-bar
   T4 (3/3 seeds, best known), but on T2s every iterative method — QUBO with any solver *and*
   plain sorting — ends +17 % above the enumerated optimum; on T2/T3/T5 the iterative methods
   mostly end in mechanisms. The second-order model helps only on T4; the solver (SA = tabu =
   exact = QAOA) never matters at these sizes.
2. QAOA (state-vector simulation, p ≤ 5, 8–14 qubits) reaches approx. ratios 0.85–0.95 against
   a 0.52–0.56 uniform baseline, but 1,000 shots already enumerate the block, so raw QAOA, QAOA
   + greedy, SA and 16-restart greedy all give the same designs. Nothing here indicates an
   advantage, and no QPU was used; the "quantum" solvers are interchangeable with SA.
3. Continuum, design-size meshes (1.9–2.2 k elements, 5 seeds): under a common crisp,
   equal-volume evaluation the binary QUBO update (SA, block or diag Hessian) is
   indistinguishable from MMA (−0.5 … −2 % by the proxy; +1 % by a refined-voxel FE), beats
   first-order BESO sorting by 1.5–8 %, and the cheap diag Hessian is as good as the exact block
   one at 1/4 of the time. On FreeTO's native (smoothed, β = 8) metric QUBO is **+5–14 %**
   worse than MMA at equal volume (not +22–29 %: the MMA volumes in that comparison were of the
   wrong design, N1). The claimed negative gaps on L/GE brackets come from the BESO pipeline
   and un-converged MMA runs, as NOTES §6 says; bridge_deck is not solved by any binary method.
4. "How good can quantum be?" — on this evidence: as good as simulated annealing, which is as
   good as MMA on beam problems and worse on truss-like ones; the QUBO *formulation* (second-
   order model + volume bisection) is a competent BESO upgrade, and nothing in the study
   depends on the solver being quantum. Any hardware statement requires S5 (not run).
