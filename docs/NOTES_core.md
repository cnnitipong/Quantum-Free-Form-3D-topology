# FreeTO-Python core — engineering notes

Scope: `freeto/` (numerical core), `tests/`, `examples/benchmark.py`, core part of `README.md`,
core lines of `requirements.txt`.  Interface: `CONTRACT.md` (implemented exactly; a few extra,
optional `FreeTOConfig` fields and extra callback keys `nnele`, `fe_time`, `smooth_time`).

## 1. Architecture

| module | content |
|---|---|
| `stl_io.py` | binary/ASCII STL reader (float32 → float64, duplicate vertices merged with `np.unique`, face vertex order preserved), binary writer with unit normals |
| `inside.py` | grid ray casting replacing `intriangulation.m` (two modes, see §3) |
| `mesh.py` | MATLAB/Octave `linspace` and colon, `geomeshini` grid (incl. the \|coordinate\| axis quirk), `domainstokeep`, `domainprep` edof tables, `forcevec`, `supportDOFs` |
| `fe.py` | `lk_H8` (verbatim), `Assembler` (pattern built once, per-iteration data = one SpMV), solver back-ends cholmod / pardiso / superlu / amg |
| `filters.py` | `HHs3D`, `HnHns3D` via index offsets (incl. HnHns3D's asymmetric loop bounds) |
| `smoothedge.py` | `smoothedge3D`: exact separable ×4 linear upsampling (= `interp3` on the node grid), 17-step bisection, 5×5×5 window sum/min/max via strided reductions |
| `optimizers.py` | exact OC; own MMA (Svanberg 1987/2007 subproblem, primal-dual interior-point subsolver) |
| `core.py` | `FreeTOConfig` (+`validate`), `run_freeto` (SIMP/SEMDOT loop), `FreeTOResult` |
| `postprocess.py` | `FieldSnapshot`, symmetry with physical origin tracking, `smooth3` + marching cubes with capping |
| `examples.py`, `cli.py` | README examples, command line |

Internally everything uses MATLAB's (nely, nelx, nelz) layout, Fortran-order linear indices and
0-based ids (row index = descending y).  Conversion to (x, y, z) with physical origin/spacing is
done only for `FieldSnapshot` (`mesh.matlab_to_xyz`, a zero-copy view).

## 2. Faithfulness to the MATLAB code

Replicated exactly (verified against Octave dumps, §6):
grid construction (axis chosen by the largest *absolute* bounding-box coordinate; 1 % shrink on the
MeshControl axis; colon ranges centred on the other axes), membership/support/force node sets,
`domainstokeep` accumulation *including* the "−1 cancels an earlier 1" clamp quirk, edofMat column
order, `forcevec` (scalar magnitudes always written to column 0; point load = median node id),
`supportDOFs`, `freedofs = setdiff(n_vec, fixeddof)`, `KE = (aa/0.5)·lk_H8`, filters (restricted
`H(ele,ele)` row sums), SIMP `Emin = 0.001` absolute / SEMDOT `0.001·E0`, the SEMDOT pseudo
sensitivity, filtering `H*(dc./Hs)`, OC based on `vxPhys` with move 0.1, `MusD` forced to 1 before
smoothing (also for passive elements), `smoothedge3D` (target and mean over *all* elements/fine
points, floor 0.001, Terr over all elements / nnele), the second smoothing on termination (which
can un-terminate the loop), `fvol = sum(full)/nnele`, beta accumulated as repeated `beta+ER`
(38th step gives 2.000000000000001), log lines `It.:%5i Obj.:%11.3f ...` and "Parameter beta
increased to %g.".

Implementation notes that keep results bit-for-bit or ulp-close:
* `matlab_linspace` / `matlab_colon` implement MATLAB's (`colonop`) and Octave's (range with
  `tfloor`, symmetric linspace) arithmetic; `compat="octave"` reproduces the dump coordinates
  bit-exactly, the default `"matlab"` differs in the last bit of a few coordinates only.
* Assembly only stores the lower triangle (`M @ E` gives it), so K is exactly symmetric, like
  MATLAB's `(K+K')/2`.
* In `smoothedge3D` points with `xg == 0` / `xg == 1` are evaluated in closed form (exactly 0.001
  and 1, as the MATLAB formula gives with IEEE arithmetic); only the rest go through `tanh`.  An
  earlier variant using the tanh addition theorem was dropped because it broke `xgnew == 1`
  exactness and changed the Terr count (caught by the reference comparison).

Deliberate deviations (documented in README):
1. **STL output in original physical coordinates** (mm, true element size h); MATLAB's `stlgen`
   translates the design to the origin and scales by `max(del)/max(fn)`.  Surface = `smooth3`
   (3×3×3 box, replicate padding; MATLAB's ±1 thresholding lines are dead code) + marching cubes
   on the field padded with −1e3·max|φ|; the cap vertices lie ≤ 1e-3 voxel outside the grid
   boundary (flat caps, = isocaps), every edge is shared by exactly two faces, faces are
   oriented outward (signed volume check).  Coincident vertices are welded.
2. **Symmetry placement**: array semantics exactly as `symmetry.m` (boundary slice duplicated;
   `z-x`/`right` puts the copy on the −y side because rows run along −y), with the physical origin
   shifted for copies on the negative side, so the original half keeps its coordinates and the
   mirror plane is at the last grid plane ± h/8.  Note that (as in MATLAB) the grid ends up to
   ~h/2 inside the domain bounding box on non-MeshControl axes, so the mirror plane can sit up to
   ~h/2 inside the intended symmetry face.
3. **Errors instead of crashes** (`freeto.FreeTOError`, a ValueError subclass, propagates out of
   `run_freeto`): before the loop every connected part of the active domain is checked to have
   its six rigid-body modes restrained by the fixed DOFs (rank of the rigid-body modes at the
   fixed DOFs); after every solve non-finite displacements, solver-library exceptions (SuperLU
   "singular", PARDISO −4, ...) and direct-solve residuals > 1e-6 raise "The stiffness matrix is
   singular: the supports do not prevent rigid-body motion ...".  `validate()` also reads the
   STLs: empty files, a flat domain, a grid with no elements along an axis, and grids above
   `max_dofs` (default 3 M grid DOFs; `None` disables) are rejected; `mesh_control >= 4` (same
   as the web UI).  A non-watertight domain only logs a warning (the hand example's STL is not
   watertight and works).
4. **Input validation**: FreeTO.m's `all([Fmagx Fmagy Fmagz])` check (rejects loads whose
   components are all non-zero) is implemented as intended ("at least one non-zero"); added
   checks for missing files, load-case count vs force regions, empty force regions (MATLAB would
   index-error), no support on the active domain, etc.  `U` is sized from the non-zero load
   columns (MATLAB would size-error if a load column is all zero).
5. **MMA wiring**: the README says to comment out lines 79–84, 135, 137 and uncomment 44–56,
   86–99.  Taken literally, line 85 (`vxPhys = (H*vxnew)./Hs`) would stay and run *before* the MMA
   call, i.e. with an undefined `vxnew` in iteration 1 (crash) and with the previous design later.
   I treat line 85 as part of the OC block (MMA's own line 97 does the same update).  MMA uses
   `xval = vx`, `f0val = c`, `df0dx = dc` (filtered), `fval = sum(vxPhys)/(vol·nnele) − 1` with the
   ele-restricted smoothed `vxPhys`, `dfdx = dv'/(vol·nnele)`, a0 = 1, a = 0, c = 1e4, d = 0,
   asyinit 0.5, asyincr 1.2, asydecr 0.7, albefa 0.1, raa0 1e-5, move 0.5 (configurable).  With
   `optimizer="MMA"` the README settings apply: tolx = tol_thresh = 1e-3, beta = ER = 0.5 and the
   `if beta < 2` guard removed (beta grows without bound; `beta_max` caps it if wanted).
6. **Inside test default `inside_mode="robust"`** (see §3); `"matlab"` is bit-exact.

Reconciliation with SPEC.md (written concurrently; I agree with it except):
* SPEC §1.6 recommends OR-ing the keep regions; I kept MATLAB's accumulate-and-clamp (identical
  unless an inside test returns −1, faithful otherwise).
* SPEC §1.2 suggests a simplified colon formula; I implemented MATLAB's published `colonop`
  algorithm and Octave's range arithmetic instead (verified bit-exact vs. the Octave dumps).
* SPEC §8 asks to replicate intriangulation's rules; that is `inside_mode="matlab"`.  The default
  "robust" mode differs only for rays hitting non-axis-parallel facet edges exactly (see §3).

## 3. Point-in-mesh test (`inside.py`)

All test points are structured grids, so one ray per grid line is cast; crossings are computed
for all (ray, candidate facet) pairs at once (candidates from facet bounding boxes via
`searchsorted`, enumerated in chunks of 2M pairs), sorted per ray, and all points of a ray are
classified with cumulative crossing counts.

* `mode="matlab"`: operation-by-operation replica of VOXELISEinternal (strict bbox test, the three
  "predicted y" edge tests with IEEE ±Inf/NaN semantics, plane intersection with |C|<1e-14→0,
  mesh-limit filter, `round(z·1e10)/1e10` + unique, parity with strict inequalities) and of the
  z → x → y fallback chain.  Points undecided after all three axes are returned as **1
  (inside)**: intriangulation writes −1 into a *logical* array, which MATLAB/Octave coerce to
  true (confirmed in Octave; found by the independent verification).  `undecided=-1` exposes them.  `inside_points` is a
  per-point version used as a cross-check.  Found while testing: intriangulation misses rays that
  pass exactly through a *diagonal* edge shared by two facets (e.g. a unit cube, ray at x = y
  counts no crossing on the top/bottom faces → whole column "outside").
* `mode="robust"` (default): watertight edge rule — every projected edge is evaluated from its
  lexicographically ordered endpoints (both facets see identical values) and exact zeros are
  resolved by a symbolic perturbation p → p + ε(1, δ), so each edge/vertex hit is counted exactly
  once; no rounding/dedupe needed; rays along x, y and z; majority vote over the axes whose
  crossing count is even (non-watertight STLs); ties/undecided → outside.
* Validation: equal to a generalised-winding-number brute force on random grids for watertight
  STLs (GE_domain, quad_domain, air_force, hand_fixed); `matlab` mode equal to the per-point port
  on all tested meshes incl. the non-watertight hand/quad STLs; `matlab` and `robust` give
  identical node and element-centre masks for every README example STL at MeshControl 20…120;
  both reproduce the Octave dumps.
* Speed: 100×60×40 grid vs hand_domain.stl (58,944 facets): 0.04 s (matlab), 0.17 s (robust);
  300×200×150: 0.27 s / 1.3 s.

## 4. Linear solvers (`fe.py`)

`Assembler`: lower-triangle pattern over the active DOFs built once (`np.unique` on int64 keys);
`M` (nnz × nnele CSR, KE values) maps element moduli to pattern values, so an iteration costs one
SpMV (17 ms at MC80).  Views give K(free,free) as full CSR (SuperLU, CHOLMOD) or upper CSR
(PARDISO, analysis done once, numeric factorisation per iteration), and the active-DOF matrix with
fixed rows reduced to their diagonal (AMG).

* `auto`: cholmod (scikit-sparse) > pardiso (pypardiso) > superlu if nfree < 10k else amg.
  The threshold was lowered from the suggested 60k after measuring: SuperLU (single-threaded,
  3-D fill-in) needs 0.9 s at 15k, 3.2 s at 25k, 12 s at 38k and ~125 s at 110k free DOFs per
  iteration, while AMG needs 0.45 s / 0.85 s / 2.2 s / 7.5 s.
* `amg`: pyamg smoothed aggregation (6 rigid-body modes as near-nullspace, strength θ = 0,
  Jacobi-smoothed prolongation ω = 4/3, symmetric Gauss–Seidel, `splu` coarse solve which drops
  the zero rows of rank-deficient aggregates) as preconditioner of an own PCG loop (rtol 1e-8 on
  the true relative residual, warm start from the previous U).  The hierarchy is rebuilt each
  iteration early on; once the design settles the old hierarchy is tried first with a budget of
  1.3× the post-rebuild iteration count (exponential back-off after failures).  If PCG fails the
  solver falls back to a direct solver (never triggered in the tests).  With SIMP's absolute Emin
  the effective stiffness contrast is ~1e9 (element densities never drop below 0.001 because of
  the smooth-edge floor); PCG needed 25–40 iterations per load case throughout the GE runs.
  Several alternatives were measured and rejected: BSR/block Gauss–Seidel (slower V-cycle),
  Jacobi/Chebyshev smoothing (more iterations; own numpy Chebyshev V-cycle was not faster),
  energy-minimising prolongation (fewer iterations on real designs but 700+ on a speckled
  high-contrast test), θ > 0 (much higher operator complexity), `pinv` coarse solver (seconds
  when aggregation stalls at ~2k coarse DOFs).
* AMG vs direct: identical optimisation histories to ≤ 1.3e-10 relative (GE MC80, 73 iterations)
  and ≤ 4e-9 on the reference cases.

## 5. Performance

Machine: this container, 2 vCPU x86-64, 7 GB RAM (an M2 should be faster per core; pypardiso is
not available there, so the AMG column is the relevant one for a pip-only Mac install).
Full runs to convergence, README GE example (E = 210 GPa, 2 load cases):

| | MC 40 | MC 80 |
|---|---|---|
| grid / active elements | 39×13×23, 3,588 / 11,661 | 79×28×48, 31,747 / 106,176 |
| free DOFs | 14,724 | 110,451 |
| setup (STL read, inside tests, index sets, filters, assembly pattern) | 0.25–0.30 s | 1.36 s (pattern 0.95 s) |
| FE per iteration: pardiso / amg / superlu | 0.10 / 0.41 / 0.87 s | 2.43 / 7.55 / ~125 s |
| smooth-edge per iteration | 0.05 s | 0.42 s |
| sensitivities + OC per iteration | 0.003 s | 0.03 s |
| iterations to convergence | 49 | 73 |
| total: pardiso / amg / superlu | 8.4 / 22.8 / 46 s | 212 / 584 s / – |
| peak RSS: pardiso / amg | 0.27 / 0.29 GB | 1.8 / 1.3 GB |

Other examples at MeshControl 40 (auto → pardiso here): hand 46 it 7.1 s, quadcopter 51 it
9.2 s, air bracket 69 it 2.6 s.  GE MC40 with MMA: 53 iterations, 11.4 s (compliance 0.055 vs
0.379 with OC — OC's beta is capped at 2 so its designs stay grey and penalised, which is why the
FreeTO README recommends MMA).  Surface extraction + STL export: 0.1–0.4 s.
The Octave run of the original code needs 76 s for the 776-element reference case that the port
does in ~1 s.

Hot spots after optimisation: the linear solve dominates (≥ 80 % of an iteration).  Smooth-edge
is 17 bisection passes of `tanh` over the non-trivial fine-grid points (≈ 25 ms each at MC80);
the bisection must stay (it defines `ls` bit-for-bit).

## 6. Verification

`python tests/compare_ref.py` (Octave 8.4 dumps of the unmodified MATLAB algorithm in
`<ref-dir>/`, not shipped; see `tests/compare_ref.py --help`):

| case | checks | result |
|---|---|---|
| ge_simp (GE, SIMP, MC 24, 30 it.) | 85 | all pass |
| air_semdot (air bracket, SEMDOT, MC 32, 20 it., x-y symmetry) | 88 | all pass |
| quad_simp (quadcopter, SIMP, MC 20, point loads, keep_BCz, 2 symmetries; converged at it. 11) | 89 | all pass |
| hand_simp (independent verifier: keepdom, scalar Fmagx + vector Fmagz, 3 symmetries) | 92 | all pass |
| lever_semdot (verifier: zfixed + keep_BCx with empty xfixed, point loads, ν 0.33, rmin 2) | 85 | all pass |
| ge_semdot_keep (verifier: keep_BC='no' + keepdom, point loads, rmin 1.2) | 85 | all pass |

Exact: nelx/nely/nelz/nele/ndof, `aa`, node & centre coordinates (bit-exact with
`compat="octave"`, ≤ 1 ulp with "matlab"), `oute`, `outeM`, `sup_all/x/y/z`, force node sets,
`ele`, `fixeddof`, `freedofs`, `edofMatn`, `F`, `KE`, `H` pattern & values, `Hn`, iteration
counts, `lss` and the topology measure in every compared iteration, beta.  Floating point
(direct solver): `c`, `dc`, `dv`, `vxnew`, smoothed `vxPhys`, `xg`, `change`, `U`, compliance and
volume histories, `S.eleden`, `S.gridden`, pre- and post-symmetry `top`: max relative deviation
2e-12 (GE: 1e-13) over all iterations.  The same passes with the default robust inside test,
MATLAB arithmetic and the AMG solver (iteration tolerances relaxed to 1e-8; observed ≤ 4e-9).

`python -m pytest tests/test_freeto.py` (43 tests, ~55 s): KE symmetric PSD with exactly six zero
eigenvalues, reference values and patch/rigid-rotation checks; colon/linspace; SPEC's hand-computed
tiny-box grid; axis quirk; inside test vs winding number / per-point port / degenerate rays / mode
agreement / speed; STL round trip (binary + ASCII); filter row sums, restriction, Hn weights;
upsampling vs `RegularGridInterpolator`, window reductions; smooth-edge on a uniform field;
forcevec quirks; edof ordering; MMA on an analytic problem and on Svanberg's 2-constraint toy
problem (matches the published optimum to 1e-5); OC volume; closed/outward surfaces incl.
boundary caps; symmetry placement for all planes/directions; end-to-end box cantilever (setup
info, history, field snapshot frame, STL/NPZ export); all available solvers agree; SEMDOT+MMA with
symmetry; stop event (mid-run and before the first iteration); validation messages; examples;
under-constrained supports (pre-check and, with the pre-check bypassed, every solver's post-solve
detection), disconnected parts, MATLAB-mode undecided points, flat/empty/thin STLs, mesh_control
limits, open-domain warning.

## 7. Known limitations / notes for users

* The pip-only AMG path is ~3× slower than PARDISO on this machine; on macOS
  `brew install suite-sparse && pip install scikit-sparse` enables CHOLMOD (what MATLAB uses).
* Memory is dominated by the direct factor (1.8 GB at MC80 with PARDISO); the assembly map is
  300·nnele entries (≈ 110 MB at MC80).
* Per-iteration `FieldSnapshot`s are pre-symmetry (like MATLAB's live plot); `result.field` is
  post-symmetry.  `result.gridden` is pre-symmetry `xg` (as `S.gridden`).
* Callback `beta` is the value after the increment (as printed by MATLAB).
