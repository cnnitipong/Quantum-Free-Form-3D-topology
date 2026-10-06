# QUANTUM_DESIGN — QUBO / quantum design updates for FreeTO-Python

Status: design (implementation-ready), not yet implemented.  Scope: a third optimizer
`optimizer="QUBO"` for the continuum loop in `freeto/core.py`, a small truss ground-structure
module, pluggable QUBO solver backends (classical simulation by default, D-Wave / IBM optional),
example problems and a study driver.  Everything must run on a laptop (Apple M2 8–16 GB, Windows)
with pip-only dependencies; quantum SDKs are optional extras.

## 0. What "quantum" can mean here, and what we implement

Four families exist in the literature:

| family | idea | verdict for FreeTO |
|---|---|---|
| **Classical FE + QUBO design update** (Sukulthanasorn et al., CMAME 2025 / arXiv 2406.18833; Ye et al., arXiv 2301.11531 with Benders cuts; Honda et al., Sci. Rep. 2024 for trusses) | keep the FE solve classical, replace the OC/MMA update by a binary optimisation solved by an annealer | **implement** — it is the only family that reaches thousands of elements today and fits FreeTO's loop unchanged |
| Physics inside the QUBO (Mathematics 2024, arXiv 2311.18565: minimum complementary energy; Honda et al. displacement step) | encode forces/displacements with ~10–16 bits each so equilibrium is part of the QUBO | do not implement: qubit count ~10× DOFs, no FE reuse, accuracy 1e-2–1e-3 on 2–5 elements |
| Variational quantum eigensolver with a PDE register (Kim, Sul, Wang, QST 2025 / arXiv 2412.07099) | 2 registers, controlled rotation gates for topology constraints; ≤ 12 qubits, 30–60 % success | out of scope (needs a full circuit-simulator stack, ≤ 12 elements) |
| End-to-end Grover + QSVT (Hölscher et al., arXiv 2510.07280) | compliance of all designs in superposition | out of scope (theoretical, circuit simulation of toy MBB only) |

Hardware facts used below: D-Wave Advantage (Pegasus P16, 5,640 qubits, degree 15) embeds a
complete graph of at most K150 at full yield (K71 with chain length 10 at 95 % yield); Advantage2
(Zephyr, degree 20, native K8,8) is somewhat larger.  Dense QUBOs beyond ~100–150 variables must go
through the Leap hybrid BQM solver (classical decomposition + QPU, up to 1e6 variables, timed in
seconds).  Sukulthanasorn et al. ran their 3,200- and 8,000-element cases only on the hybrid solver
and on a GPU annealer, and their QPU cases had ≤ 51 qubits; Ye et al. embedded 47–67 logical
qubits into 172–616 physical qubits.  Gate-model QAOA (Farhi, Goldstone, Gutmann 2014) is limited
to ~20 qubits in exact state-vector simulation on a laptop (2^20 complex amplitudes = 16 MB, but
each energy evaluation touches the full vector; measured below).

Honest framing for the researcher: everything that runs locally is a *classical simulation of a
quantum algorithm* or a *quantum-inspired heuristic* (simulated annealing, tabu).  The study can
measure **solution quality** (gap to exact optimum, compliance vs OC/MMA/BESO) and **algorithmic
behaviour** (QAOA approximation ratio vs depth p, sensitivity to penalty weights, block sizes), and,
if a cloud account is added later, **real QPU quality and QPU time**.  It cannot demonstrate a
speed-up.

## A. Continuum QUBO design update

### A.1 Where it plugs in (core.py)

The loop in `run_freeto` computes, per iteration: `U`, `cval`, `Ee`, `ce` (per element, per
load case), filtered `dc = H(dc/Hs)`, `dv = H(1/Hs)`, then `vxnew = oc_update(...)` or MMA, then
`vxPhys = H vxnew / Hs`, `full[ele]=vxPhys; full[MusD]=1`, `smoothedge3D`, `change`.  The QUBO
optimizer replaces only the `vxnew = ...` line:

```python
if qubo_mode:
    vxnew = qupd.update(loop, vx, vxPhys, dc, dv, U, Ee, ce_list, cval)   # float array in {0,1}^nnele
```

Everything downstream is unchanged: the binary design is filtered (grey ring of width ~rmin at
the boundary), MusD keep elements are forced to 1, the Heaviside/level-set step gives the crisp
surface and preserves the volume, `change = Σ|vxnew − vx|/(vol·nnele)` becomes the (normalised)
Hamming distance between successive binary designs.  `QUBOUpdater` is constructed after setup with
references to `asm, solver, KE, edofMatn, H, Hs, ele, MusD, nelx, nely, nelz, vol, nnele, E0,
Emin, penal, simp` (all available in `run_freeto` at that point), plus the `FreeTOConfig.qubo`
options.

Base point.  As OC does, the linear model is expanded at `vxPhys` (the smooth-edge output used in
the FE solve), while the binary step is taken from the current binary design `x_k = vx`.  The
sensitivities `dc` already include the density-filter chain rule (`dc/dx_j = Σ_e dc/dρ_e H_ej/Hs_e`).

### A.2 The QUBO

Variables: `x ∈ {0,1}^m` over the *free set* `C_k ⊆ {1..nnele} \ MusD_pos` (A.4); all other
active elements keep their value from `x_k`; `MusD_pos` (positions in `ele` of keep elements)
are fixed to 1.  Energy to minimise (all terms in units of compliance, then divided by
`s = max|dc|` so the linear coefficients lie in [−1, 0]):

```
E(x) = g·x                                     (1) first-order compliance
     + ½ (x − x_k)ᵀ Q_c (x − x_k)              (2) second-order compliance (optional, block-exact or SIMP-diagonal)
     + γ Σ_{(e,f) ∈ N} (x_e − x_f)²            (3) perimeter / anti-checkerboard regulariser (sparse)
     + λ v·x                                   (4) volume, λ from bisection   — or —  λ_q (v·x − V_k)²  (quadratic penalty)
     + μ Σ_e (x_e − x_{k,e})²                  (5) move limit (Hamming penalty; linear because x² = x)
```

* `g = dc` (≤ 0), `v = dv` (filtered element volumes, ≈ 1).
* (2): the exact Hessian of compliance in the SIMP density is
  `∂²c/∂ρ_e∂ρ_f = 2 E'_e E'_f (k_e u_e)ᵀ K⁻¹ (k_f u_f) − δ_ef E''_e c_e`, with `E' = p ρ^{p−1}(E0−Emin)`,
  `E'' = p(p−1)ρ^{p−2}(E0−Emin)` (SEMDOT: `E' = E0(1−Emin/E0)`, `E'' = 0`), summed over load cases.
  In design space `Q_c = Jᵀ H_ρρ J`, `J = H diag(1/Hs)` restricted to the free elements.  Modes:
  - `hessian="none"`: linear model.  With (4) only, the QUBO is a knapsack solved exactly by sorting
    `g_e/v_e` — this is the BESO/sorting baseline (A.6) and must be reported as such.
  - `hessian="diag"`: keep only the `−δ_ef E''_e c_e` term (no extra solve).  Through the filter it
    couples each element to its filter neighbours (7 non-zeros per row at rmin = 1.5, 27 at
    rmin = 2.5): sparse, embeddable, free.
  - `hessian="block"` (default for block solvers): for each block B (A.4) build
    `Y = [Σ_e J_ej E'_e (k_e u_e)]_{j∈B}` (nfree × |B|, scatter via `edofMatn`), one multi-RHS
    solve `Z = K⁻¹ Y` with the already-factorised matrix (`solver.solve(data, Y, None)`), and
    `Q_c[B,B] = 2 Yᵀ Z + diag-term`.  Cost: one extra solve with |B| right-hand sides per block
    per iteration (direct solvers: factorisation is reused, so ~|B| back-substitutions; AMG: |B| PCG
    runs — use `hessian="diag"` with AMG).  All blocks' RHS are batched into one call.
  The Hessian is the physically justified source of non-trivial couplings: it is what tells the
  update that removing two neighbouring load-path elements together is much worse than the sum of
  their first-order costs.  Scratch experiment (`<workspace>/scratch_qdesign/truss_exp.py`, 13-bar
  ground structure, exact optimum by enumeration): the first-order sorting update converged to a
  mechanism (c = ∞); the block-exact-Hessian QUBO with λ-bisection reached the exact optimum
  (gap 0 %, 4 iterations); the analytic Hessian matched finite differences to 2e-4.
* (3): `N` = face-neighbour pairs among free elements (6-neighbourhood on the grid).
  `(x_e − x_f)² = x_e + x_f − 2 x_e x_f`.  Default `γ = 0.02` (in units of `s`); 0 disables.
* (4): default `volume="bisection"`: solve the QUBO for a given λ, bisect λ ∈ [0, λ_max] until
  `v·x ≤ V_k` with the smallest λ (monotone in practice; 12–20 QUBO solves, only for cheap backends).
  `V_k` is the iteration's volume target (A.5).  This keeps the QUBO sparse and gives exact volume
  control, like OC's bisection.  Alternative `volume="penalty"`: `λ_q (v·x − V_k)²` (dense
  `v vᵀ` coupling, Sukulthanasorn et al.); default `λ_q = 2 s / v_max²`, tunable; it is the mode to
  use when each QUBO solve is expensive (QPU, QAOA) — then λ is *fixed* to the value found by the OC
  bisection of the same iteration (`oc_update` is cheap and gives a Lagrange multiplier consistent
  with the continuous relaxation) followed by a greedy repair (add/remove elements with the smallest
  marginal `g_e + (Q_c x)_e` until `v·x = V_k`).  Both paths end with the repair, so the reported
  volume is always feasible.
* (5): `μ = 0` by default — the trust region is provided by the free-set selection (A.4), and the
  scratch experiment showed a Hamming penalty of 0.2 s trapping the truss in a 28 % worse design.
  Kept as an option (`move_penalty`).

Multi-bit densities (Sukulthanasorn et al. use 1 bit per element, α ∈ {0, 1.1}; multi-bit is a
straightforward extension `ρ_e = Σ_b 2^b q_{e,b}/(2^nb − 1)`) are **not** part of the first
implementation: the smooth-edge step already produces the grey transition, and 1 bit per element is
what makes the QUBO the same size as the design.

### A.3 Sizes

`nnele` is 1,000–12,000 for the study meshes (D.1–D.4).  Only the *frontier* is free each iteration:

* `frontier_fraction f` (default 0.25): the free set `C_k` is the union of (i) grey elements
  `0.02 < vxPhys < 0.98` (the smooth-edge boundary band), (ii) the `f·V_k` solid elements with the
  smallest `|g_e|/v_e` (candidates for removal), (iii) the `f·V_k` void elements with the largest
  `|g_e|/v_e` (candidates for addition).  Typically `m = |C_k| ≈ 0.3–0.6 nnele` early and a few
  hundred near convergence.
* Backend capacity: exact ≤ 24, QAOA (statevector) ≤ 20, QPU direct embedding ≤ ~100 dense /
  ~1,000 sparse, hybrid / SA / tabu: whole `C_k`.

### A.4 Block decomposition (for capacity-limited backends)

Given `C_k` and a block size `kb`, partition `C_k` into blocks of ≤ kb elements by **spatial
patching**: sort free elements by their grid coordinate along a Z-order (Morton) curve (`ele →
(j,i,k)`), cut the sorted list into runs of kb; runs are spatially compact so the Hessian /
filter couplings are mostly intra-block.  Alternative `blocks="rank"`: sort by `|g_e|/v_e` so each
block mixes removal and addition candidates around the threshold (cheaper, worse coupling capture).
Then run **block Gauss–Seidel** on the global QUBO:

```
x ← x_k
for sweep in 1..n_sweeps (default 2):
    for B in blocks (random order):
        h_B = g_B + λ v_B + μ(1−2x_k,B) + 2 Q_c[B, ¬B] (x_¬B − x_k,¬B) + γ-terms from fixed neighbours
        Q_B = Q_c[B,B] + γ L_B                      # L_B: graph Laplacian of N restricted to B
        x_B ← solve_qubo(Q_B, h_B)                  # backend
    if no block changed: break
repair volume (A.2)
```

The linear term of a block contains the current state of the other blocks, so the sweep is a
coordinate-descent on the global energy; every accepted block solve is guaranteed not to increase
the global model energy (accept only if `E_B(new) ≤ E_B(old)`, cheap to evaluate).  With
`blocks=None` (SA/tabu/hybrid) there is a single block = `C_k`.

### A.5 Schedule, initial design, convergence

* `init="solid"` (default): `x_0 = 1`, volume schedule `V_k = max(V_{k−1}(1 − ER), vol·nnele)`,
  `ER = 0.05` (BESO evolutionary ratio).  The FE solve on the solid domain is the first iteration.
* `init="oc"`: run `n_warm = 10` OC iterations (unchanged code path), then threshold `vxPhys` to
  the `vol·nnele` largest values → `x_0`; `V_k = vol·nnele` from then on.
* Heaviside: with binary input the smooth-edge step needs a sharper β than OC's cap of 2:
  defaults for QUBO mode `beta_init = 0.5, beta_step = 0.5, beta_max = 8` (the MMA-like schedule);
  `tolx = 1e-3`, `tol_thresh = 1e-3` as for MMA.
* Convergence: FreeTO's `change ≤ tolx` (Hamming distance ≤ tolx·vol·nnele, i.e. ≤ 1–3 flipped
  elements per 1,000 solid ones) **or** no compliance improvement for `patience = 8` iterations
  after the volume schedule reached `vol` (the updater keeps the best feasible design and returns
  it on exit).  `max_iter` still applies.
* Compliance reported is `cval` of the FE solve, as in FreeTO (evaluated on the previous
  iteration's design; the final logged value corresponds to the returned design after one last FE
  solve — add that solve in QUBO mode so `res.comp` matches `res.eleden`).

### A.6 Classical baselines that must be in the study

1. OC and MMA (existing).  2. **Sorting/BESO** = `hessian="none", volume="bisection"` (solved by
sorting, no QUBO solver at all) — the honest "is the QUBO doing anything?" control.  3. Same QUBO
solved by SA / tabu vs by QAOA / QPU — the "is the *quantum* solver doing anything?" control.

## B. Truss ground-structure module (`freeto/truss/`)

Small, self-contained, 2-D and 3-D; units free (default mm, N, MPa).

* `ground.py`: `GroundStructure(nodes (N,d), bars (M,2), L, E, A_full, supports (list of (node, dof)),
  loads (ndof,) or (ndof, nload))`; builders `grid2d(nx, ny, dx, dy, lmax=None)`,
  `grid3d(nx, ny, nz, ...)`, `from_lists(...)`; **overlap filter**: a candidate bar is dropped if
  another candidate node lies strictly inside it (collinear, `0 < (p−a)·d < |d|²`), and `lmax`
  limits bar length.  Bars between two fully fixed nodes are dropped.
* `fe.py`: `bar_stiffness` (unit-area 4×4 / 6×6 `E/L tᵀt`), batched assembly
  `assemble(areas (R, M)) → K (R, nfree, nfree)` (dense, `nfree ≤ ~60`), `solve(...)` batched
  `np.linalg.solve`, `compliance(areas)`, `grad(areas)` (`g_b = −u_bᵀ k_b u_b`), `hessian(areas)`
  (`2 Bᵀ K⁻¹ B`, `B = [k_b u_b]` scattered), stability check: residual `‖Ku − f‖ ≤ 1e-8‖f‖`, finite
  `u`, and a graph test that every loaded node is connected to a support through present bars.
  Areas are floored at `A_min = 1e-9 A_full` so absent bars never make K singular; a design is
  *infeasible* if `c > 1e3 × c(full ground structure)` or fails the checks.
* `optimize.py`:
  1. `enumerate_exact(gs, Vmax)`: all `2^M` masks (M ≤ 22 → 4.2 M) in chunks of 2^14: pre-filter
     by volume `L·x ≤ Vmax` and by "every loaded node touches a bar", batched solve
     (`(chunk, nfree, nfree)` → seconds for nfree ≤ 12, a few minutes at M = 22, nfree = 15),
     returns the global optimum, the sorted top-10 and the count of feasible designs.  Gray-code
     is unnecessary with batched solves.
  2. `oc_continuous(gs, Vmax)`: OC on continuous areas `A ∈ [A_min, A_full]` (move 0.2, λ bisection
     on `L·A ≤ Vmax`) → then `round_sorted` (keep largest areas until the volume is met) and
     `round_qubo` (B.4).
  3. `qubo_iterative(gs, Vmax, backend, ...)`: exactly the A.2/A.4 machinery with `v = L`,
     `H = I` (no filter), Hessian always exact (`M` RHS, trivial), `init="solid"` with the ER
     schedule; blocks only when `M` exceeds the backend capacity.  Multi-level areas
     (`nbits ≥ 2`, `A_b = A_full Σ_i 2^i q_{b,i}/(2^nbits − 1)`) supported by expanding the linear
     and quadratic coefficients (`x_b` → weighted sum of bits); default `nbits = 1`.
  4. `qubo_rounding(gs, A_cont)`: **one-shot** QUBO at the continuous optimum: `E(x) = g·x +
     ½ xᵀ Q_c x` (Hessian at `A_cont`) with λ bisection — a quadratic rounding of the relaxed
     solution.  Decision on a *true* one-shot formulation (Ye et al. Benders master problem, or
     encoding displacements/forces as bits as in Honda et al. / arXiv 2311.18565): **not
     implemented** — it needs 10–16 qubits per DOF or per bar force, its accuracy is 1e-2 at
     2–5 elements, and it offers nothing for the FreeTO pipeline.  The rounding step is the useful
     one-shot variant.
* Metrics (per run): final compliance, `gap = c/c_exact − 1` (or vs best known for M > 22),
  feasibility (stable, volume ≤ Vmax), number of QUBO solves, total FE solves, wall time,
  backend timing.

## C. Solver backends (`freeto/quantum/`)

Common convention: minimise `E(x) = xᵀ Q x + h·x + const`, `Q` symmetric with zero diagonal
(diagonal folded into `h`), `x ∈ {0,1}^n`.  Conversion helpers to/from dimod BQM (`quadratic
{(i,j): 2 Q_ij}`), to Ising (for QAOA), and `energy(Q, h, const, X)` vectorised over samples.

```python
@dataclass
class QUBOResult:
    x: np.ndarray          # best sample (n,), float 0/1
    energy: float
    samples: np.ndarray    # (R, n) all reads (or the sampled shots)
    energies: np.ndarray   # (R,)
    timing: dict           # {"wall": s, "solver": s, "qpu_access": s|None, "qpu_sampling": s|None, "hybrid_run_time": s|None}
    info: dict             # backend-specific (nfev, p, approx_ratio, p_opt, embedding chain stats, ...)

def solve_qubo(Q, h, const=0.0, *, backend="auto", num_reads=64, seed=None,
               initial_state=None, time_limit=None, **opts) -> QUBOResult
```

`backend="auto"`: `exact` if n ≤ 20, else `sa` (own) — never a cloud backend unless named.
`available_backends()` reports which optional ones import and whether tokens are set.

| name | module | notes |
|---|---|---|
| `exact` | `exact.py` | n ≤ 24; low 16 bits enumerated as a `(65536, n)` matrix, high bits looped; returns full spectrum for n ≤ 20 (used by QAOA metrics) |
| `sa` | `anneal.py` | own vectorised single-flip Metropolis: `R` replicas as `(R, n)` int8, local fields `f = Qx` kept incrementally, `ΔE_i = (1−2x_i)(h_i + 2 f_i)`, geometric β schedule from `β_0 = ln 2 / max|ΔE|` to `β_1 = ln 100 / min nonzero|ΔE|`, `sweeps = 200` (n ≤ 200) … 50 (n ≥ 5,000), random variable order per sweep; sparse `Q` via CSR rows.  ~1 μs per flip-replica; n = 3,000, R = 64, 100 sweeps ≈ 10 s |
| `tabu` | `anneal.py` | 1-flip tabu with tenure ~ n/10, restarts; the strongest classical control for dense QUBOs |
| `greedy` | `anneal.py` | steepest-descent 1-flip from `initial_state` — used for volume repair and as the block-solver fallback |
| `dwave_sa`, `dwave_tabu` | `backends.py` | `dwave.samplers.SimulatedAnnealingSampler / TabuSampler` if `dwave-samplers` installed (C++, faster) |
| `qaoa` | `qaoa.py` | own numpy statevector (C.1) |
| `qiskit_aer`, `ibm` | `backends.py` | same circuit via qiskit; `AerSimulator` (shot noise) or `qiskit-ibm-runtime` `SamplerV2` with `QISKIT_IBM_TOKEN`; classical angle optimisation done with our simulator when n ≤ 20, else the runtime loop (expensive; only for the study's "hardware demo") |
| `dwave_qpu`, `dwave_hybrid` | `backends.py` | `DWaveSampler()+EmbeddingComposite` (n ≤ ~100 dense / sparse embeddable; `chain_strength` = `uniform_torque_compensation`, `num_reads` 200–1000, `annealing_time` 20 μs) and `LeapHybridSampler` (`time_limit` ≥ 3 s); need `DWAVE_API_TOKEN`; raise `QuantumBackendUnavailable` with an install/token message otherwise |

### C.1 QAOA simulator (`qaoa.py`)

* Ising form `E = Σ J_ij z_i z_j + Σ b_i z_i + c` with `z = 1 − 2x`; energies of all 2^n
  bitstrings `E_all` computed once (vectorised, little-endian qubit index), scaled to [0, 1]
  (`Ẽ = (E − E_min)/(E_max − E_min)`, exact `E_min/E_max` known from the spectrum).
* State `ψ ∈ C^{2^n}`; cost layer `ψ ← exp(−iγ Ẽ) ψ` (element-wise); mixer `exp(−iβ Σ X_j)` applied
  axis by axis on `ψ.reshape([2]*n)` (in place, no `np.stack` copies).
* Depth `p` (default 3), parameters `(γ_1..γ_p, β_1..β_p)`; init `linear_ramp` (TQA-like,
  Sack & Serbyn 2021: `γ_k = (k/p) Δ, β_k = (1 − k/p) Δ`, `Δ = 0.75`) or `interp` (Zhou et al.
  2020: optimise p = 1, extend to p+1 by linear interpolation of the angle sequence) and
  `warm="previous"` (reuse the angles of the previous iteration's block — the sub-QUBOs are
  similar from iteration to iteration).  Classical optimiser: `scipy.optimize.minimize`
  (`COBYLA`, `maxiter = 100 p`; option `L-BFGS-B` with finite differences).
* Output: `shots` samples drawn from `|ψ|²` (default 1,000), best-of-shots `x`, and metrics
  `approx_ratio = (E_max − ⟨E⟩)/(E_max − E_min)`, `p_opt = |ψ_opt|²`, `p_opt_shots = 1 − (1 − p_opt)^shots`.
* Measured (scratch, random dense QUBO, COBYLA 200 evaluations, naive mixer):
  n = 12: p = 1/3/6 → ratio 0.73/0.85/0.81, 0.3–0.8 s; n = 16: 0.73/0.78/0.74, 2–11 s;
  n = 20: 0.53/0.80/0.69, 42–270 s and `p_opt_shots` 0.003–0.2.  Consequences: default block size
  for QAOA `kb = 14`, hard cap 20; depth p = 3; always take best-of-shots and fall back to
  `greedy` from the best shot; report the ratio.  Deeper p does not help without more optimiser
  evaluations (the p = 6 rows are optimiser-limited, not QAOA-limited — state this in the report).

### C.2 Honest timing

Every `QUBOResult.timing` has `wall` (Python call to return, includes network, embedding,
queue) and `solver` (time inside the sampler as reported by it).  D-Wave: copy
`sampleset.info["timing"]["qpu_access_time"]` and `qpu_sampling_time` (μs → s) and, for hybrid,
`info["run_time"]`, `info["qpu_access_time"]`.  IBM: job `metrics()["usage"]["quantum_seconds"]`
and queue time.  The study tables report `wall` and `qpu_access` side by side and never sum
"QPU time" alone into a speed claim (Sukulthanasorn et al.'s "time to find solution" excludes
0.5 s/solve of access overhead; Ye et al. note the wall clock is dominated by embedding).

## D. Example problems

All continuum geometry is generated programmatically (`freeto/quantum/examples.py` writes STLs
into a temp/example folder with the `box_mesh` triangulation of `tests/test_freeto.py`).  Grid
rules (SPEC §1.2): the MeshControl axis is the one with the largest |coordinate| — so put the
long axis of every box along +x starting at 0 and keep the other extents smaller; nodes on the MC
axis start `1 %` of a spacing inside the box, nodes on the other axes are centred with spacing
h.  **Slab rule**: a support/load/keep slab of thickness `t = 1.2 h` centred on a plane always
contains exactly one node plane on a non-MC axis (`h ≤ t < 2h`) and one on the MC axis when the
slab extends from `−h/2` outside to `0.6 h` inside the face.  `h` is known analytically:
`h = (L_x − 2·0.01·L_x/(MC−1))/(MC−1)`; the example builder computes it, builds the slabs, and
asserts `nnele`/support-node counts after `prepare_domain`.  Material E = 210 GPa, ν = 0.3 unless
stated; `keep_bc=True`; `loadtype="distributed"`.

| id | geometry (mm) | supports | loads | volfrac | MC → grid, active el., free DOFs | expected/target |
|---|---|---|---|---|---|---|
| D1 cantilever | box 60×20×20 | slab at x = 0 face (all DOFs) | patch at x = 60 face, y ∈ [8, 12], z ∈ [8, 12] (≥ 1 node plane each), Fy = −1000 N | 0.30 | 29 → 28×9×9, 2,268, 8.4 k free DOFs (SuperLU auto, ~0.3 s/solve); 41 → 40×13×13, 6,760, ~24 k (AMG) | I-beam-like web + flanges; OC/MMA reference |
| D2 MBB (half, 2 symmetries) | box 60×20×6 (half length, half thickness) | `xfixed` slab at x = 60 (symmetry), `zfixed` slab at z = 6 (symmetry), `yfixed` roller slab at bottom x ∈ [0, 1.2 h] | patch on top face x ∈ [60 − 1.2h, 60], Fy = −1000 N | 0.35 | 31 → 30×10×3, 900; 61 → 60×20×6, 7,200 | classic MBB truss-like web; `symmetry=[("y-z","right"),("x-y","right")]` |
| D3 bridge / deck | box 120×30×20, `keepdom` = deck slab y ∈ [15 − 0.6 h, 15 + 0.6 h] (= [13.2, 16.8] at MC 41; contains the node plane y = 15.0 and the element-centre layers 13.5, 16.5) full x,z | two bottom-corner slabs x ∈ [0, 1.2 h] and [120 − 1.2 h, 120], y ∈ [0, 1.2 h] (all DOFs) | force region = deck slab, Fy = −2000 N (distributed over all deck nodes) | 0.18 | 41 → 40×10×6, 2,400; 61 → 60×15×10, 9,000 (AMG) | arch/truss with vertical hangers — the "truss-like continuum" case |
| D4 advanced: GE bracket | existing `examples/STLs/GE_*.STL` | as `EXAMPLES["GE_bracket"]` | 2 load cases (Fz 1500 N, Fy −2000 N) | 0.30 | 40 → 3,588 active, 14.7 k free DOFs (0.4–0.9 s/solve, 49 OC iterations = 23 s AMG); 60 for the final figures | multi-load, real geometry; the FreeTO README case — compare to OC (0.379) and MMA (0.055) at MC 40 |
| D4b (optional) L-bracket | L-prism 40×40×10 with a 24×24 notch, generated as a 12-facet extruded polygon | top slab of the short leg | tip patch of the long leg, Fy | 0.30 | 41 → ~2,500 | stress-concentration corner; compare with Sukulthanasorn's L-shaped case |

Truss benchmarks (`freeto/truss/benchmarks.py`):

| id | nodes / bars | supports, loads | Vmax | exact? |
|---|---|---|---|---|
| T1 ten-bar truss (classic) | 6 nodes on 2×3 grid, 360 in (9,144 mm) spacing, 10 bars, E = 10^4 ksi, A_full = 10 in² | nodes 5, 6 fixed; 100 kip down at nodes 2 and 4 | 50 % of full | yes (1,024) |
| T2 small ground structure | 4×2 grid, 22 bars after overlap filter, 1 m spacing | left column fixed; unit load down at bottom-right | 40 % | yes (4.2 M, batched) |
| T2s QAOA-whole | 3×2 grid, 13 bars | same | 50 % | yes (8,192) — whole QUBO fits QAOA (13 qubits) |
| T3 3-D tower | base square 4 fixed nodes, 4 nodes at z = 1, apex at z = 2, `lmax = 1.5` → 22 bars | base fixed; apex Fx = 1, Fz = −1 | 45 % | yes |
| T4 large 2-D | 9×3 grid, `lmax = 2.3` → 118 bars | left column fixed; two loads on bottom chord | 25 % | no — best-known from tabu/SA multi-start; 118 dense variables is at the edge of a direct QPU embedding |
| T5 large 3-D | 2×2×4 column, `lmax = √3` → 66 bars (no base-base bars) | base fixed; lateral + vertical load at top | 30 % | no |

## E. Study protocol (`python -m freeto.study`)

Runs are declared in a small YAML/JSON matrix; every run writes `results/<study>/<run>.json`
(config, history, final metrics, timing, QUBO sizes per iteration, backend info) and the final
STL/NPZ; a `--report` pass builds tables (markdown + CSV) and figures (matplotlib, optional dep).

| study | problems | optimizers × backends | repeats |
|---|---|---|---|
| S1 truss exactness | T1, T2, T2s, T3 | exact enumeration; OC+round_sorted; OC+round_qubo (exact); iterative QUBO with backends exact, sa, tabu, qaoa (p = 1, 2, 3, 5; T2s/T1/T3 whole; T2 whole for exact/sa, blocks of 14 for qaoa); sorting/BESO | 10 seeds for stochastic backends |
| S2 truss scale | T4, T5 | sa, tabu, dwave_sa; qaoa blocks 14; sorting; OC+rounding; (dwave_hybrid / dwave_qpu if token) | 5 seeds |
| S3 continuum quality | D1, D2, D3 at the coarse MC | OC, MMA, BESO-sort, QUBO{hessian none/diag/block} × {sa, tabu}; QUBO-block × qaoa (kb = 14, p = 3) on D1 and D2 only (each iteration = ~100 block solves × ~5 s → minutes per iteration; cap `max_iter = 40`) | 3 seeds |
| S4 advanced | D4 at MC 40 (and MC 60 for OC/MMA/QUBO-sa only) | OC, MMA, QUBO-sa (hessian diag, block), QUBO-tabu | 3 seeds |
| S5 hardware (optional) | T2s, T3 on `dwave_qpu`; T4 on `dwave_hybrid`; D1 blocks on `dwave_qpu`; T2s on `ibm` | wall vs QPU time, chain breaks, quality vs `exact` | 3 |

Metrics per run: final compliance (after a final FE solve on the returned design, evaluated at
the same β so OC/MMA/QUBO are comparable — also report the *binary-thresholded* compliance for
OC/MMA to remove the grey-penalisation artefact), final volume fraction, gap to exact (truss) or
to the best value over all runs (continuum), iterations, FE solves, QUBO solves, mean/max QUBO
size, wall time, backend time, QPU access time, QAOA `approx_ratio` and `p_opt` per block
(distributions), and for SA the fraction of blocks where SA found the same energy as `exact`
(n ≤ 20 blocks are re-solved exactly offline for this — cheap and a strong quality check).

Figures: (F1) truss: gap-to-exact bar chart per method/backend, with seeds as points; (F2) QAOA
approximation ratio and `p_opt` vs p and vs n on T2s/T1/T3 and on D1 sub-QUBOs; (F3) continuum:
compliance history vs iteration for OC/MMA/BESO/QUBO-sa/QUBO-qaoa; (F4) final topologies (rendered
STL views, existing `surface()`); (F5) penalty-weight sensitivity (`λ_q`, γ) for `volume="penalty"`;
(F6) wall vs solver vs QPU time.

What can legitimately be concluded: whether the QUBO update with second-order couplings gives
better binary designs than sorting/BESO and than rounded OC/MMA at equal FE-solve budgets;
whether the *solver* matters (SA/tabu reach the exact block optimum in x % of blocks; QAOA at
p ≤ 5 reaches ratio r and P(opt) q); how QAOA quality degrades with n and depends on p; how many
qubits/couplers the real D-Wave/IBM runs would need and what their quality and QPU time are.
What cannot be concluded: any speed-up (the simulator is exponential in n; QPU access time of
~20 μs × reads excludes 0.1–1 s overheads; hybrid solvers are mostly classical); optimality of
continuum designs (no exact reference beyond ~22 variables).  The report must say so in its first
paragraph.

## F. Implementation plan

```
freeto/quantum/__init__.py     QUBOOptions, solve_qubo, available_backends
freeto/quantum/qubo.py         QUBO container, energy(), to_ising(), to_bqm(), Hamming/perimeter builders
freeto/quantum/exact.py        brute force, spectrum
freeto/quantum/anneal.py       sa, tabu, greedy (numpy)
freeto/quantum/qaoa.py         statevector QAOA + metrics
freeto/quantum/backends.py     registry, optional dwave/qiskit wrappers, timing extraction, errors
freeto/quantum/update.py       QUBOUpdater (A.2–A.5: free set, blocks, Hessian, λ bisection, repair)
freeto/quantum/examples.py     D1–D4b STL builders (box/L-prism), returning FreeTOConfig
freeto/truss/ground.py, fe.py, optimize.py, benchmarks.py   (B)
freeto/study.py                run matrix + report; `python -m freeto.study --study S1 --out results/`
```

API / config: `FreeTOConfig.optimizer` accepts `"QUBO"`; new field `qubo: QUBOOptions | dict | None`
with `backend="auto"`, `hessian="block"|"diag"|"none"`, `volume="bisection"|"penalty"`,
`lambda_q=None`, `gamma=0.02`, `move_penalty=0.0`, `frontier_fraction=0.25`, `block_size=None`
(auto by backend: qaoa 14, exact 20, dwave_qpu 80, others None), `blocks="morton"|"rank"`,
`sweeps=2`, `init="solid"|"oc"`, `er=0.05`, `n_warm=10`, `patience=8`, `num_reads=64`, `seed`,
`qaoa_p=3`, `qaoa_shots=1000`, `qaoa_init="linear_ramp"`, `time_limit=None`; `validate()` checks
the enum values and, for cloud backends, that the SDK imports and the token env var is set (clear
`FreeTOError` otherwise).  Defaults for `tolx/tol_thresh/beta_*` in QUBO mode as in A.5.  CLI:
`--optimizer QUBO --qubo-backend sa --qubo-hessian diag ...`.  Callback `iter` dict gains
`qubo: {"n_free", "n_blocks", "n_solves", "solver_time", "qpu_access_time", "approx_ratio"}`.
Web app: `JobCreateRequest` gets the same optional `qubo_*` fields; `index.html` adds `QUBO` to the
optimizer select and a collapsible "Quantum / QUBO" panel (backend select populated from
`/api/health` → `quantum_backends: {name: {"available": bool, "reason": str}}`, block size, hessian,
volume mode, reads, QAOA p); the log pane shows per-iteration QUBO stats.  The webapp stub core is
unaffected.

Dependencies: required stay `numpy, scipy` (plus the existing core deps).  Optional extras in a
new `requirements-quantum.txt`: `dwave-ocean-sdk>=7` (dimod, dwave-system, dwave-samplers — pure
wheels for macOS arm64 and Windows), `qiskit>=1.2`, `qiskit-aer`, `qiskit-ibm-runtime`,
`matplotlib` (report figures), `pyyaml` (study matrices).  All imports of these are lazy and
guarded.

Tests (`tests/test_quantum.py`, `tests/test_truss.py`):
* QUBO energy/Ising/BQM round trips on random instances; `exact` equals a 2^n numpy loop.
* `sa`, `tabu`, `greedy+restarts` find the exact optimum on 50 random dense QUBOs with n = 16
  (≥ 95 % of instances for sa, 100 % for tabu) and on sparse 3-D grid QUBOs with n = 200 within
  1 % energy of the best of 20 SA runs.
* QAOA: on n = 8 instances p = 3 gives `approx_ratio ≥ 0.6` and `p_opt` > 1/2^8·5; p = 1 angles
  reproduce the analytic single-layer expectation on a 2-qubit Ising (closed form).
* Truss FE: 2-bar analytic compliance; batched vs single assembly; mechanism detection.
* `enumerate_exact` on T1 reproduces the known binary ten-bar optimum (record the value at first
  run and pin it), T2s exact = 9.0897 (unit E, A; from the scratch experiment for the 3×2, Vmax =
  50 % structure with bars {1,2,5,6,8,10}).
* `qubo_iterative` with `exact` backend reaches gap 0 on T2s (scratch result) and ≤ 5 % on T1/T3;
  sorting/BESO documented gap on T2s (mechanism) is a regression test that the Hessian matters.
* Continuum: D1 at MC 16 (15×5×5 = 375 elements), `optimizer="QUBO"`, `backend="sa"`, 15
  iterations: compliance decreases monotonically after the volume schedule ends, final volume within
  0.02 of volfrac, design binary before smoothing, `res.comp` within 25 % of OC's thresholded
  compliance; `hessian="block"` Hessian equals finite differences on a 3-element block (1e-4 rel.);
  `backend="qaoa"` with `block_size=8` runs end-to-end in < 60 s.
* Backends: `available_backends()` never raises; cloud backends raise `QuantumBackendUnavailable`
  without a token; a `FakeBackend` (records calls, returns the exact solution) checks the block
  Gauss–Seidel bookkeeping (linear terms from fixed neighbours, monotone model energy).
* Study: `python -m freeto.study --study smoke` runs T2s + D1(MC 16) with exact/sa in < 2 min and
  writes the report files.

Risks and mitigations:
* **Penalty tuning** (`volume="penalty"`): known to need per-problem tuning (λ = 5 … 1e5 in
  Sukulthanasorn et al.); mitigated by making bisection the default and by the repair step.
* **Cost of the block Hessian with AMG** (large meshes): automatic downgrade to `hessian="diag"`
  when the solver is AMG, with a log line.
* **QAOA time**: n = 20 blocks take minutes; cap kb ≤ 16 by default, warm-start angles, and run
  QAOA only in S1/S3 sub-studies.  Memory: 2^20 complex128 = 16 MB — fine.
* **Chattering** of the binary update near convergence: patience-based stop with best-design
  memory; optional `move_penalty`.
* **Symmetry**: the QUBO acts on the half model like OC; nothing extra.
* **Reproducibility of cloud runs**: seeds do not apply; store raw samplesets (JSON) with the run.
* **Scientific over-claiming**: the report template starts with the "what this shows / does not
  show" paragraph of §E.

## References

* Ye, Z., Qian, X., Pan, W., *Quantum topology optimization via quantum annealing*, arXiv:2301.11531
  (Benders decomposition, reduced binary master problem, D-Wave Advantage + CQM hybrid; 120×40 …
  480×240 meshes, 47–67 logical / 172–616 physical qubits).
* Sukulthanasorn, N., Xiao, J., Wagatsuma, K., Nomura, R., Moriguchi, S., Terada, K., *A novel design
  update framework for topology optimization with quantum annealing: application to truss and
  continuum structures*, CMAME 437 (2025) 117746, arXiv:2406.18833 (multiplicative updater α,
  1 qubit/element, slack + squared volume penalty, D-Wave Advantage 6.4 / hybrid / Amplify; trusses
  6–29 bars, 2-D 50–3,200, 3-D 8,000 elements).
* Key, F., Freinberger, L., *A Formulation of Structural Design Optimization Problems for Quantum
  Annealing*, Mathematics 12(3) 482 (2024), arXiv:2311.18565 (complementary energy, forces as bits).
* Honda, R. et al., *Development of optimization method for truss structure by quantum annealing*,
  Sci. Rep. 14 (2024) (alternating displacement/area QA steps, Leap hybrid, 29/60 bars).
* Kim, J.E., Sul, J., Wang, Y., *Variational quantum algorithm for constrained topology optimization*,
  Quantum Sci. Technol. (2025), arXiv:2412.07099 (≤ 12 qubits, simulation).
* Hölscher et al., *End-to-end quantum algorithm for topology optimization in structural mechanics*,
  arXiv:2510.07280 (Grover + QSVT, simulation).
* Farhi, E., Goldstone, J., Gutmann, S., *A quantum approximate optimization algorithm*,
  arXiv:1411.4028; Zhou, L. et al., PRX 10 (2020) 021067 (INTERP); Sack, S., Serbyn, M., Quantum 5
  (2021) 491 (TQA / linear-ramp initialisation).
* D-Wave docs: *QPU topologies* and *Minor-embedding: best practices* (Pegasus: K150 at full yield,
  K71 at chain length 10; Zephyr degree 20) — docs.dwavequantum.com.
* Huang, X., Xie, Y.M., *Evolutionary topology optimization of continuum structures* (BESO, ER
  schedule and sensitivity-sorting baseline); Sigmund, O., Maute, K., *Topology optimization
  approaches* (SMO 2013) for the filter/Heaviside context.
