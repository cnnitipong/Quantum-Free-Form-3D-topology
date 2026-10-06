# Physics audit: QUBO / QUBO+QAOA designs compared with MMA (FreeTO-Python, quick suite, seed 0)

The 12 study cases were re-run with the exact `freeto.study._run_continuum` configurations: rows
S3q-66/67/69 and S4-70, 72, 73, 74, 76, 77, 78, 80, 81, with seed 0 and OMP_NUM_THREADS=1.
Every native and crisp compliance reproduces the internal reference results resA.pkl and resB.pkl (not included) to the
last digit.

Scripts are in the internal audit scratch folder (not included):
- `run_cases.py` re-runs a case and captures the fields.
- `analyze.py` turns the fields into the data contract.

Data is in the internal audit data folder (not included):
- `<problem>__<method>.npz`, `summary.csv`, `components.json`, `bc_check.json`
- `fixed/`: the same files after the fix
- `mc41/`: the bridge control at MC 41
- `fix_summary.md`

## 1. Verdict

**The user is right about one problem and wrong about the other three. There is no
support-location or support-constraint error anywhere.**

- **Bridge deck (QUBO-sa and QUBO-qaoa): the suspicion is right.** The binary designs the
  QUBO update returned were physically broken.
  - The SA design had 22 face-connected components, the QAOA design 33. Between 19 % and 28 %
    of the solid touched no support.
  - In the crisp design, 11 and 9 of the 248 loaded nodes sit on void.
  - A voxel FE of the binary design gives c = 1.3e3 and 1.6e3, against 0.011 for MMA.
  - This was a genuine defect in `quantum/update.py`. It is fixed (§5).
- **Cantilever: the suspicion is wrong.** The "detached rod" is not detached. It is a top
  chord, face-connected to the structure at x = 0–45 mm and x = 70–90 mm (§3). The QUBO
  cantilevers are physically valid but asymmetric (§4).
- **GE bracket and L-bracket: physically valid.** Every QUBO and QAOA design is one grounded
  body with all loads on solid material and all six rigid-body modes restrained.
- **Boundary conditions are bit-identical for MMA, QUBO-sa and QUBO-qaoa** (§2). Supports and
  loads are applied identically. Support elements are solid in the QUBO designs at least as
  often as in MMA's.
- **MMA is not flawless either.**
  - The MMA bridge's crisp deck has a 2.4 % sliver (x = 172–183 mm) that is not
    face-connected on the fine grid. Its rendered surface has 6 pieces.
  - The MMA iterates at steps 1–6 render a floating bar (§3).
  - At the element level, a voxel threshold of the MMA bridge and L-bracket gives c = 1.8e3
    and 2.3e5.

## 2. Boundary-condition identity (`bc_check.json`)

All of the following are identical across the three methods on all four problems
(`identical_across_methods: true`): fixeddofs, F, MusD (the keep elements), the active
elements, freedofs, edofMat, the grid and V\*. All are built by the same `prepare_domain` and
`support_dofs` code before the optimiser branch.

| problem | grid | active el. | support nodes (all 3 directions fixed) | loaded nodes | load resultant (N) | keep (MusD) |
|---|---|---|---|---|---|---|
| cantilever MC25 | 24×8×4 | 768 | 45 (x = 0 face) | 5 | (0, −1000, 0) | 40 (32 support + 8 load) |
| GE bracket MC24 | — | 776 | 16 | 4 | case 1 (0, 0, 1500), case 2 (0, −2000, 0) | 17 |
| L-bracket MC30 | — | 1920 | 150 | 30 | (0, −1000, 0) | 140, of which **76 lie outside the domain** |
| bridge deck MC31 | 30×4×7 | 840 | 32 (both bottom ends) | 248 (every top node) | (0, −1000, 0) | **0** |

Two setup facts apply to every method equally (FreeTO and Octave behaviour, not QUBO bugs):
- **Bridge:** `keepdom` deck slab elements are tested by their centre. The top element row's
  centres sit at y = 25.0 mm and the slab starts at y = 25.65 mm, so MusD is empty at MC 31.
  At MC 41 it holds 440 elements.
- **L-bracket:** 76 keep elements lie outside the design domain. They are written into the
  rendered field through `full[MusD] = 1`.

## 3. Findings per design

The crisp geometry is the fine-grid field above the crisp threshold, i.e. what the crisp FE
averages. "Binary" is the QUBO update's own 0/1 element variables. For MMA, "binary" means
x > 0.5 of the last iterate, which is not its physical design.

| design | crisp c | native c | crisp comps / float | load on solid | rb modes held | surface comps | binary comps / ungrounded | voxel c of binary x | lateral/vertical tip displacement |
|---|---|---|---|---|---|---|---|---|---|
| cantilever MMA | 0.1454 | 0.1472 | 1 / 0 | 1.00 | 6 | 1 | — | — | 0.00007 |
| cantilever QUBO-sa | 0.1664 | 0.1838 | 1 / 0 | 1.00 | 6 | 1 | 1 / 0 | 0.094 | 0.072 |
| cantilever QUBO-qaoa | 0.1879 | 0.1857 | 1 / 0 | 1.00 | 6 | 1 | 1 / 0 | 0.101 | 0.139 |
| GE MMA | 0.0845 | 0.0877 | 1 / 0 | 1.00 | 6 | 1 | — | — | 0.36 (geometry) |
| GE QUBO-sa | 0.0685 | 0.1098 | 1 / 0 | 1.00 | 6 | 1 | 2 grounded / 0 | 0.049 | 0.48 |
| GE QUBO-qaoa | 0.0725 | 0.1165 | 1 / 0 | 1.00 | 6 | 1 | 2 grounded / 0 | 0.051 | 0.49 |
| L MMA | 0.2733 | 0.2695 | 1 / 0 | 1.00 | 6 | 1 | — | — | 0.45 (geometry) |
| L QUBO-sa | 0.2484 | 0.3073 | 1 / 0 | 1.00 | 6 | 1 | 1 / 0 | 0.189 | 0.46 |
| L QUBO-qaoa | 0.2656 | 0.3262 | 1 / 0 | 1.00 | 6 | 1 | 1 / 0 | 0.201 | 0.45 |
| bridge MMA | 0.0110 | 0.0391 | 2 / 0.024 | 1.00 | 6 | 6 | — | — | 0.06 |
| **bridge QUBO-sa** | **179.5** | 0.0210 | 1 / 0 | **0.956** | 6 | 3 | **22 / 0.19** | **1258** | 0.02 |
| **bridge QUBO-qaoa** | **230.0** | 0.0190 | 3 / 0.018 | **0.964** | 6 | 7 | **33 / 0.28** | **1631** | 0.41 |

**Support and load elements.** Supports are solid wherever MMA's are:
- cantilever: 100 % in every design, because they are keep elements
- L-bracket: 100 %
- bridge (crisp): MMA 29 %, QUBO 29–32 %. Every design attaches to the piers only at the two
  side faces.
- GE (crisp): MMA 48 %, QUBO 45–53 %. 2–3 thin keep support elements are eroded by the
  smoothing in every method; this is the known smoothedge behaviour, the same in Octave.

The QUBO update never removed support elements that MMA keeps.

**The cantilever "rod".** It is the top layer of the QUBO-sa design:
- position: y = 35–40 mm, z = 10–20 mm (pages 2–3), x = 0–90 mm (columns 0–17)
- 2 of its elements are support keep elements (MusD)
- it is present in the binary x, in eleden_native, in the crisp field and in the surface;
  it is not a rendering artefact
- it is face-connected to the page-1 top flange along x = 0–45 mm and to the inclined web
  at x = 70–90 mm

It is a tension chord with a free span of 25 mm over a genuinely void web. QAOA has the same
chord, attached along x = 0–55 mm and 65–90 mm. Neither design has any floating material at
any level of description (crisp, surface, binary). The rod looks detached in
D_final_MMA_vs_quantum.png only because its attachments lie behind the visible web plane.

**MMA step-1 floating bar (A_design_evolution.png).** The MMA surfaces at steps 1–6 have
2 components (sizes 3519 and 425 fine points at step 1). At that point all x ≤ 0.53 and the
smoothed field lies between 0.015 and 0.88. It is the level set of a grey intermediate
field, not a design. MMA's final design is one component. The QUBO surfaces in the same run
are single-component at every step.

**Other floating pieces found:**
- Bridge, crisp field:
  - MMA: x = 172–183 mm deck sliver, 2.4 % of the solid
  - QAOA: x = 0–8 mm, 1.6 %; x = 23–37 mm, 0.2 %
- Bridge, surface: specks of 0.03–1.6 % for every method, all at loaded deck nodes
- GE binary: 0.4 % on a separate bolt support; it is grounded

Every component is listed with its mm extent in `components.json`.

## 4. Why the bridge failed, and what still differs physically

The deck carries a distributed load on all 248 top nodes, but it is not a keep region at
MC 31 (§2). The volume budget is V = 0.2, i.e. 168 elements, while the top layer alone has
210 elements. A 0/1 design therefore cannot have a full deck and must perforate it.

QUBO's load protection (a greedy set cover of the loaded nodes) keeps a checkerboard of
single top elements: every second element in x and every second page in z. The QUBO steps
then remove the connectors around them.

Per-iteration diagnostics (`qubo["connectivity"]`, `diagnostics=True`) show what happened on
SA, seed 0:
- 68 of 70 iterations ended with floating material.
- In 19 iterations the QUBO solve itself raised the floating share by more than 5 points.
- At iteration 26 the floating share went from 2.4 % to 70 %, because the whole deck and
  upper material were cut off from the piers. The native compliance rose only from 0.0082
  to 0.0092.

The native FE sees smoothedge(H x / Hs), not x. The density filter (rmin 1.5) and the node
averaging bridge one-element gaps, so neither the model nor the accept-if-improves guard
(tolerance 25–50 %) can see the break. The load path breaks along the deck between the
piers. In the crisp field, the loaded nodes on void are at x = 20, 160–180 mm (SA) and
x = 20, 87, 113–120, 173 mm (QAOA), i.e. in the deck spans between the arch feet.

**Remaining physical differences after the fix (modelling limitations, not bugs):**
- **No symmetry.** The cantilever problem is symmetric about the mid-planes. The MMA design
  is symmetric, with lateral tip displacement 7e-5 of the vertical. The QUBO and QAOA designs
  are not: 0.07 / 0.14 before the fix, 0.22 / 0.23 after. The stochastic block updates never
  enforce symmetry; the symmetry option in FreeTOConfig would.
- **No length scale.** Binary designs keep one-element members, for example the bridge deck
  strips. The filter, smoothing and crisp proxy thin these out. This is why the bridge crisp
  c after the fix is still 2–3.6× MMA (0.025–0.040 vs 0.011), while the voxel FE of the same
  binary design gives 0.016–0.018.

## 5. Root cause in `freeto/quantum/update.py`, and the fix

**Hypotheses tested** on bridge MC 31 SA, seed 0, with `connectivity=False` and
`diagnostics=True`. Columns: ungrounded share of the final binary design / mean
floating share over the iterations / crisp c.

| variant | final | mean | crisp c | conclusion |
|---|---|---|---|---|
| default | 0.16 | 0.255 | 179 | baseline |
| history_average off | 0.33 | 0.32 | 320 | sensitivity averaging is not the cause |
| protect_loads off | 0.38 | 0.20 | 230 | protection isolates elements but is not the cause; without it loads sit in void |
| gamma = 0.05 (perimeter penalty) | 0.25 | 0.12 | 523 | reduces the floating share, does not cure it |
| move limit and guard off | 0.26 | 0.25 | 134 | truncation and guard are not the cause |

Floating elements by source, summed over iterations:
- solid in the input and cut off by the step: 1386
- protected elements whose neighbours were removed: 1824
- added by the volume repair: 221

Support elements are in the free set unless they are keep elements, and nothing excludes
them from the frontier.

**Root cause.** The binary update has no connectivity constraint: the block, first-order and
Hessian model cannot see that removing a connector disconnects the structure. Its only FE
check is filter-smoothed, so it is blind to disconnection. Load protection then preserves
isolated load elements.

**Fix.** A new option, `QUBOOptions.connectivity = True` (default ON):
- `QUBOUpdater.connect` removes ungrounded components that hold no keep, protected or loaded
  element.
- It reconnects every other ungrounded component along the cheapest void path (Dijkstra on
  the face graph).
- `_rebalance` then removes the added volume from boundary elements with low |g|/v, but only
  where the removal keeps every required element grounded.
- Each iteration records `qubo["connectivity_repair"]`.
- A diagnostics option, `QUBOOptions.diagnostics` (default off), records the per-iteration
  connectivity statistics.
- `connectivity=False` reproduces the old trajectories bit for bit (checked on the
  cantilever).

**Results** (`data/fix_summary.md`). Values are crisp c.

| case | before | after | MMA |
|---|---|---|---|
| bridge QUBO-sa | 179.5 | **0.0397** | 0.0110 |
| bridge QUBO-qaoa | 230 | **0.0247** | 0.0110 |
| cantilever QUBO-sa | 0.1664 | 0.1527 | 0.1454 |
| cantilever QUBO-qaoa | 0.1879 | 0.1850 | 0.1454 |
| GE bracket QUBO-sa | 0.0685 | 0.0683 | 0.0845 |
| L-bracket QUBO-sa | 0.2484 | 0.2488 | 0.2733 |

After the fix:
- Every repaired binary design is grounded, with every load on solid material.
- The bridge binary volume is 0.205–0.206 against V\* = 0.20.
- Control at MC 41, where the deck keep exists: QUBO-sa is connected with and without the
  repair (crisp 0.098 / 0.094; MMA 0.018).

**Verification:**
- pytest: all pass, 175 + 4 skipped as before. `test_guard_and_load_protection_bridge_deck`
  now also asserts `floating_after == 0` and binary V within 0.02.
- compare_ref: 6/6 PASS. OC and MMA code is untouched; only the `quantum/` package and the
  docs changed (QUANTUM_API option table, NOTES_quantum §3.18).
- The QUBO rows in `results/quick` and in the figures predate the fix and are stale.

## 6. Data contract notes (for the plots)

- All arrays are in MATLAB layout (nely, nelx, nelz). Element (r, c, p) has its centre at
  (gx[c] + h/2, gy[nely−1−r] + h/2, gz[p] + h/2) mm. The keys `gx`, `gy`, `gz`, `h`,
  `origin`, `spacing` and `layout` are in every npz.
- `top_fine` is in (x, y, z) order, like FieldSnapshot; solid is `top_fine > 0`.
- `eleden_crisp` = 1 where the crisp element density (fine-window average) is ≥ 0.5. The
  fractional value is in `eleden_crisp_frac`.
- `comp_main` labels come from face-connectivity on the fine grid, the true crisp geometry,
  mapped to elements by majority:
  - 1 = the component with the most support elements
  - 0 = void, or an element less than half filled
  - The element-level ≥ 0.5 labels are in `comp_el05`. On its own that criterion falsely
    splits off the MMA cantilever load block.
- Extra fields:
  - `binary_design`, `comp_binary`, `comp_surface`, `comp_native`, `comp_crisp_fine`
  - `support_dir`: bitmask x = 1, y = 2, z = 4
  - `load_vec_cases`
- Extra `summary.csv` columns: `binary_c` (voxel FE of x), `voxel_c`, `ungrounded_frac`,
  `load_frac_on_solid`, `crisp_off_axis`, `keep_*`, `support_*` and `load_*` solidity.
- `keep` covers active MusD elements only. `n_keep_outside_domain` counts the rest (76 on
  the L-bracket).

## 7. Status (2026-10-02, after the corrections)

The setup defects of §2 are corrected in the examples:
- The bridge deck and pads now hold element centres at every mesh_control ≥ 24.
- The bridge volfrac is now 0.35 and its study mesh MC 41.
- The L-bracket support is clipped to the arm.

The core now warns about empty kept regions and about keep elements outside the domain.
The audit itself is `freeto/audit.py` (docs/AUDIT_API.md), and it runs on every study run.
`results/quick` was re-run with the connectivity repair: every QUBO and QAOA design passes
the physics check. The new numbers are in docs/NOTES_quantum.md §8.
