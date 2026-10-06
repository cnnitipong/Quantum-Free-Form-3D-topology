# FreeTO-Python — independent verification report

Verifier: separate agent, adversarial review of `` against the MATLAB
original (`<workspace>/FreeTO/`) and Octave 8.4 dumps.  Scratch material (new Octave drivers,
reference dumps, comparison and UI scripts, screenshots) is in `<workspace>/scratch_verify/`.

## 1. What was re-run / re-checked

| check | result |
|---|---|
| `python tests/compare_ref.py` (3 builder cases, superlu, compat=octave, inside=matlab) | 262/262 pass; worst rel. deviation 1.7e-11 (`it020 change`, air_semdot), everything else ≤ 7e-12; integer sets/grids exact |
| same with Mac-like settings (`--solver amg --compat matlab --inside robust`, clean venv) | 262/262 pass |
| `pytest tests/test_freeto.py` | 34 passed (56 s) |
| `pytest tests/test_webapp.py` / `test_playwright_smoke.py` | 19 + 1 passed |
| **3 new Octave reference cases the builders never saw** (see §2) | **456/456 checks pass**, worst rel. deviation 4.9e-11, both with reference settings and with the *default* settings (compat=matlab, inside=robust, solver=auto) |
| MMA KKT check (Svanberg toy problem, 2 constraints; separable 200-var compliance-like problem) | KKT stationarity residual 3e-10 / 5e-8 rel., constraints active to 1e-7, x within 7e-9 of the analytic optimum |
| output STL (GE MC40, quad MC30 + 2 symmetries, air MC40 + symmetry) with trimesh | watertight, winding consistent, positive volume, all vertices inside domain bbox ∪ mirrored extent; solid coincides with the input design where expected (bolt holes / clevis overlay verified visually in the UI) |
| API end-to-end with the real core (load example, run, poll, preview, stop, queue, download STL/NPZ, uploads with roles, symmetry) | works; stop latency 0.4 s, stopped job still yields STL/NPZ, queued job runs after |
| Playwright headless UI (example flow, user-upload flow with role selectors, arrow, chart, theme, narrow viewport) | works; screenshots in `<workspace>/scratch_verify/shots/`; arrow direction correct (F2 = −y points down), design correctly overlaid on inputs, chart correct |
| MC80 GE with AMG (pip-only Mac path) | setup 1.4 s, 8.0 s/iteration, 1.18 GB peak RSS on this 2-vCPU box |
| clean venv from `requirements.txt` only (numpy 2.4, scipy 1.17, skimage 0.26, pyamg 5.3) | core imports as `real`, solvers = superlu + amg, server starts, compare_ref passes |
| macOS arm64 wheels (`pip download --platform macosx_12_0_arm64`, cp310/cp311) | numpy, scipy, scikit-image, pyamg, fastapi, uvicorn, python-multipart all available as binary wheels |

`compare_ref.py` itself was read critically: it compares the right variables with correct
1-based→0-based conversion, shape checks precede value checks (so the `U[:, :nl]` slice and the
`edofMatn[:n]` slice cannot hide a column/row mismatch because `F` and `nnele` are checked
exactly first), tolerances are 1e-15 exact-ish for setup quantities, 1e-10 for the first three
iterations and 1e-7 for later ones (observed ≤ 2e-11, so not vacuous), and all iterations of
the compliance / volume histories plus the last iteration and the mirrored `top` are covered.
Nothing compares Python to itself.

## 2. New independent reference cases (Octave, original code via the instrumented drivers)

Drivers and dumps: `<workspace>/scratch_verify/ref/{driver_hand.m,driver_lever.m,driver_ge2.m}` →
`<workspace>/scratch_verify/ref/{hand_simp,lever_semdot,ge_semdot_keep}/`; comparison script
`<workspace>/scratch_verify/ref/mycompare.py` (independent of `tests/compare_ref.py`).

| case | what it exercises (all previously uncovered) |
|---|---|
| `hand_simp` (hand, SIMP, MC 26, 8 it.) | 5 load cases; **scalar `Fmagx=300` with vector `Fmagz`** (forcevec's "scalar goes to column 1 for every region" quirk); `keepdom` (= hand_force2.stl); MeshControl axis = y; 3 symmetries `z-x/right`, `x-y/left`, `y-z/left` (the three branches not covered by the builders' cases), compared after each stage |
| `lever_semdot` (lever STLs, never used by the builders; SEMDOT, MC 28, 8 it.) | `zfixed` + `keep_BCx='yes'` with **empty xfixed** + `keep_BCz='yes'` (the `{'yes','no','yes'}` branch of domainstokeep with an empty region), `loadtype='point'` with 2 load cases, ν = 0.33, rmin = 2.0 |
| `ge_semdot_keep` (GE, SEMDOT, MC 22, 6 it.) | `keep_BC='no'` + `keepdom` (the `else` branch of domainstokeep), point loads, scalar `Fmagx=100` with vector Fmagy/Fmagz, rmin = 1.2 |

All grid coordinates, `oute`, `outeM`, support/force node sets, `F` (including its column count),
`fixeddof`, `freedofs`, `edofMatn`, `KE`, `H`, `Hn`, per-iteration `c/dc/vxnew/vxPhys/xg/lss/tol/
change/beta/U`, histories, `S.*`, and `top` after every symmetry stage match (max rel. 4.9e-11).

## 3. Findings (prioritised)

No BLOCKER.  The numerical core is faithful; the issues are robustness/UX/portability.

### MAJOR

1. **Under-constrained model crashes with a cryptic error instead of a clear message.**
   `freeto/optimizers.py:28` — with a singular K (e.g. air bracket with only `zfixed`), SuperLU
   returns inf/NaN displacements, `dc` is NaN, the OC bisection never moves `l1`, `l2` underflows to
   0 and the loop dies with `ZeroDivisionError: float division by zero`; with PARDISO the user gets
   `PyPardisoError ... error code -4`; AMG falls back to the direct solver and reports the same
   (`fe.py:405-411`).  A pip-only Mac user will see "float division by zero" in the web UI.
   Fix: after every solve (`core.py:366`) check `np.all(np.isfinite(Uf))` (and catch the solver
   exceptions) and raise `ValueError("The stiffness matrix is singular: the supports do not
   prevent rigid-body motion (add fixed/xfixed/yfixed/zfixed regions on the active domain).")`;
   also guard `oc_update` against `l1 + l2 == 0`.
2. **Loading a second example (or uploading files after an example) keeps the previous example's
   files and roles.**  `webapp/static/js/app.js:510` (`applyPrefill`) never resets `state.files`;
   after GE → quadcopter there are two `domain` and two `fixed` files, `buildJobPayload` picks
   the first (GE_domain) with the quadcopter loads and the job fails with "force region 1
   (quad_force1.STL) contains no grid nodes" (reproduced).  Fix: on example load set every
   existing file's role to `unused` (and hide its mesh) or clear the list; when the user assigns
   `domain`/`fixed`/`keepdom`/... to a file, demote any other file holding that unique role.
3. **`run_app.command` / `run_app.sh` do not work offline and hide errors.**  `run_app.command:5,17`
   — `set -e` plus `pip install --quiet --upgrade pip` on *every* launch: without internet pip
   exits non-zero and the launcher aborts before starting the server; if the server fails,
   `set -e` also skips the "Press Enter to close" prompt (`:27`) so the Terminal window closes
   before the user can read the error.  Fix: only run pip when `.venv` was just created (or when
   `requirements.txt` is newer than a stamp file), drop `--upgrade pip` (or `|| true`), remove
   `set -e` around the server call and always reach the final `read`.  Also document that macOS
   Gatekeeper blocks a downloaded `.command` until the user right-click → Open (or
   `xattr -d com.apple.quarantine run_app.command`) — README says nothing about this.
4. **Live preview is generated while holding the lock the worker needs, at full resolution.**
   `webapp/jobs.py:155-175` holds `job._preview_lock` during `surface_from_field` +
   `_mesh_to_binary_stl` (measured 0.7 + 0.4 s at GE MC80, 1.1 + 0.3 s at quad MC70), and the
   worker's callback (`jobs.py:212`) blocks on the same lock, so every iteration can lose ~1 s;
   the preview STL is 24 MB (474k faces) per iteration at MC80 and is fetched/parsed by the
   browser on every poll.  Fix: take the field reference under the lock, generate outside it;
   build previews on a decimated field (`marching_cubes(..., step_size=2)` or subsampled `top`)
   and/or every N iterations; replace the per-face Python loop (`jobs.py:275`) with
   `freeto.stl_io.write_stl`'s vectorised record array (0.1 s).
5. **Silent fallback to a fake optimizer.**  `webapp/core_loader.py:33-40` — if `import freeto`
   fails for any reason (a broken numpy install, a missing dependency after a partial
   `pip install`), the app keeps working with `_stub_core.py`'s "shrinking sphere" and only a
   small badge says `core: stub`.  In a shipped tool this can produce meaningless "results".
   Fix: make the stub opt-in (`FREETO_ALLOW_STUB=1`), otherwise fail loudly at startup with the
   import error; at minimum show a red banner and refuse to run jobs on the stub.

### MINOR

6. `freeto/inside.py:331` + `mesh.py:308`: "matlab" mode maps undecided points (odd crossings
   along all three axes) to −1 → outside, but the real MATLAB/Octave code assigns `in(cl) = -1`
   into a *logical* array, which coerces to `true` → **inside** (see ref/README.md §3).  No README
   example produces undecided points at MC 20–120 (checked), so results are unaffected today, but
   the docstring/NOTES claim of "operation-by-operation replica" is wrong for this case.  Fix:
   return 1 for undecided in matlab mode (or document the deliberate deviation).
7. Non-watertight / degenerate STLs are accepted silently.  Uploading garbage (`server.py:116`
   swallows the parse error and registers a 0-triangle file with HTTP 200), a single-triangle
   "domain" only fails later with the misleading `MeshControl too small: the grid has no
   elements along at least one axis` (`mesh.py:273`), and an open box runs without warning
   (robust mode classifies the whole bbox as inside).  Fix: reject unparsable uploads with 400,
   check bbox extents > 0 and report "domain STL is flat/degenerate", and warn (log line + UI)
   when any ray test finds odd crossing counts (surface not closed).
8. No upper bound on `mesh_control` / problem size (`core.py:142` only checks ≥ 3; UI `min="4"`
   at `index.html:75` disagrees with the core's ≥ 3).  MC 200 on the GE bracket (~1.7 M elements, ~1.7 M DOFs) would need several GB
   (1.2 GB already at MC 80 with AMG).  Fix: estimate `nele`/`ndof` after `build_grid` and refuse (or warn in the UI) above a
   configurable limit; align the UI minimum with the core.
9. Load-case table is clipped in the left panel (`style.css:224-228`, fixed table layout): the
   Fx/Fy/Fz inputs show "150"/"-200" for 1500/−2000 and the ✕ button is cut off (see
   `shots/a1_quad_loaded.png`, `b2_user_roles_arrow.png`).  Fix: narrower inputs / allow
   horizontal scroll / drop the fixed 38 % file column.
10. Force arrow is scaled by the *load region's* bbox diagonal (`app.js:253`), so for small
    regions (quad_force1) the arrow is a barely visible stub.  Fix: scale by the domain bbox
    (e.g. 15 % of its diagonal) and always draw it above the region.
11. `webapp/server.py:438` `app = create_app()` runs at import, creating `~/.freeto_web` even when
    `--workdir` is given and when only the tests import the module.  Fix: create lazily in
    `main()` / use a factory for uvicorn.
12. `tests/compare_ref.py:34` and README line 114 hard-code `<workspace>/ref/out`; on the user's
    machine the script prints "reference not available – skipped" and exits 1.  Either ship the
    (small, 20 MB) dumps under `tests/ref/` or make the README say the reference data is not
    included.  `CONTRACT.md`/`SPEC.md`/`NOTES_core.md` also reference `/home/claude` (dev docs,
    acceptable, but say so).
13. README test instructions (`README.md:186`) omit `httpx` (needed by `fastapi.testclient`) and
    `requests`; `--break-system-packages` is Debian-specific noise on macOS.
14. Symmetry placement: the grid ends up to ~h/2 inside the bounding box on non-MeshControl axes
    (inherited from geomeshini), so the mirror plane sits at `z_last + h/8`, not at the domain
    face — the mirrored air bracket is 34.4 mm long instead of 2 × 19.05 = 38.1 mm at MC 40.
    This is documented in NOTES §2.2 but not in the README's user-facing "Differences" list;
    users overlaying the result on a CAD model will notice.  Consider mirroring about the bbox
    face instead (leaves a ≤h gap, which is what the physics of the half-model implies).
15. `webapp/jobs.py:63-66` log cursor: `deque(maxlen=4000)` shifts indices once it overflows, so
    `?since=` returns wrong/duplicated lines on very long runs (>2000 iterations); harmless at
    the default `max_iter`.  Track a monotonically increasing line counter instead.
16. Three.js geometries are never `dispose()`d (`app.js:768,775`): at MC80 each preview is a
    24 MB geometry per iteration; GPU memory is only reclaimed when the JS GC runs.  Call
    `geometry.dispose()` / `material.dispose()` on the replaced mesh.

### NIT

17. `examples/benchmark.py:10` imports `resource` (Unix-only; fails on Windows, fine on macOS).
18. Chart legend/tick colours are hard-coded for the dark theme (`app.js:842-851`); in the light
    theme the legend text (`#cbd5e1`) is nearly invisible (`shots/a1_quad_loaded.png`).
19. Theme toggle (`app.js:881`): with no stored theme and a light OS scheme the first click
    goes to `dark`, with a dark OS scheme the first click is a no-op (sets `dark` again).
    Derive `cur` from `matchMedia('(prefers-color-scheme: dark)')` when the attribute is unset.
20. Shipped tree contains `__pycache__/`, `.pytest_cache/`, `.pytest_freeto_web/`,
    `webapp/screenshots/`, and the 24 KB `_stub_core.py`; total 11 MB is fine, but clean the
    caches before delivery.
21. Job label is "Job" for user uploads unless an example is still selected in the dropdown
    (`app.js:584`); use the domain file name.
22. `_build_record` reads every uploaded STL fully (fine) but `read_stl` treats an ASCII file
    whose size happens to equal `84 + 50·ntri` as binary (`stl_io.py:31-37`) — astronomically
    unlikely, but a `solid` header + non-`facet` check would make it airtight.

## 4. Observations (not defects — faithful reproductions of the original algorithm)

* Kept regions (`MusD`, e.g. force regions with `keep_BC`) are set to density 1 *before*
  smoothing only; after the Heaviside/threshold step they are often void in the final STL
  (GE MC40: only 14 % of force-region element centres are solid; Octave reference ge_simp: 41 %).
  Identical in the Octave dumps, so this is the MATLAB algorithm, but users will be surprised.
* `Topo.` can exceed 1 (Terr counted over all elements / nnele), termination can be triggered by
  the second smoothing raising the volume (visible as the final jump in the volume chart) — both
  as in MATLAB.
* SIMP uses an absolute `Emin = 0.001` (stiffness contrast 2e14 with E = 210 GPa) — as in MATLAB.

## 5. Verdict on faithfulness

**Faithful.**  Beyond the builders' 262 checks, three fresh Octave runs of the unmodified
algorithm covering the previously untested code paths (5 load cases, scalar-magnitude quirk of
`forcevec`, `keepdom`, `keep_BC='no'`, `keep_BCx/z` with an empty region, point loads with
SEMDOT, all six symmetry branches, a never-used STL set) agree with the port to ≤ 5e-11 relative
in every intermediate and final quantity, with the default ("robust" inside test, MATLAB colon
arithmetic, auto solver) as well as the reference settings.  The MMA implementation converges to
KKT points on independent test problems.  The one semantic deviation found (undecided
ray-casting points → outside instead of MATLAB's accidental "inside") does not affect any
shipped example.  The deliberate deviations (physical-coordinate STL, symmetry placement,
stricter validation) are documented and reasonable.  The remaining work is robustness and UX
(items 1–5 above), not numerics.
