# AUDIT_API — physics / connectivity check (contract between core and web app)

Status: contract written by the planner; Opus implements `freeto/audit.py`, Sonnet codes the
web app against it. Additions only; do not rename.

## Core

```python
from freeto.audit import audit_result, audit_figure

rep = audit_result(res, cfg_or_setup)      # res: FreeTOResult of any optimizer (OC/MMA/QUBO)
```
`audit_result` returns a plain dict (JSON-serialisable):
```
{
  "ok": bool,                      # all checks below pass
  "checks": {                      # each: {"pass": bool, "value": ..., "detail": str}
     "bc_supports_present":   ...,  # ≥ 1 fixed DOF; 6 rigid-body modes restrained (reuse core's setup check)
     "supports_solid":        ...,  # fraction of support elements solid in the crisp design (value), pass if = 1
     "loads_solid":           ...,  # share of |F| acting on nodes adjacent to solid crisp material (pass if ≥ 0.99)
     "single_grounded_body":  ...,  # n_components, floating_frac (pass if floating_frac < 0.01 and loads on main)
     "load_on_main":          ...,
     "keep_regions_captured": ...,  # for keepdom/force/fixed region files: element/node counts > 0 (pass if all > 0)
     "volume_target":         ...   # |V_native − volfrac| ≤ 0.02
  },
  "n_components": int, "floating_frac": float, "crisp_compliance": float|None,
  "native_compliance": float, "volume_fraction": float,
  "components": [{"label": int, "n_elements": int, "grounded": bool, "has_load": bool, "bbox_mm": [[x0,y0,z0],[x1,y1,z1]]}, ...]
}
```
`audit_figure(res, cfg_or_setup, path_png, title=None)` writes the 3-panel overlay figure
(3D crisp design: main grey / floating orange / supports blue / loads red + arrow; mid-depth
slice of the native density; component map) — same look as (audit figures, not included) BC_overlay_*.png.

- `run_freeto(..., audit=True)` (new FreeTOConfig field `audit: bool = True`) stores the dict in
  `res.extra["audit"]`; a failed audit never raises, it is reported.
- The per-iteration callback gains `info["audit_live"] = {"n_components", "floating_frac"}` for
  QUBO runs (cheap component count of the binary design), optional for OC/MMA.
- Study records gain `audit_ok`, `n_components`, `floating_frac`, `loads_solid`; summary.md
  gets those columns; a run with `audit_ok == False` is labelled "physically invalid" in tables.

## Web app
- `GET /api/jobs/{id}/audit` → the dict above (404 until the job is done; 409 if audit disabled).
- `GET /api/jobs/{id}/audit.png` → the overlay figure (generated lazily, cached).
- `POST /api/jobs/{id}/audit/run` → (re)compute the audit for a finished job (useful for jobs run with audit off).
- Study status/results expose the new record fields; study results table shows a pass/fail pill.
- Setup tab: after a job finishes, a "Physics check" card: pass/fail per check, component list, and
  the overlay image; a "Download audit (JSON)" button.

## Implementation notes (core, 2026-10-02; additions only)

`freeto/audit.py` implements the contract above. Definitions:

- **Crisp design.** This is the common crisp evaluation of `freeto.evaluate`.
  - Field: the returned design's pre-smoothing field on the 4× fine grid.
  - Threshold: the one that gives V\* = `volfrac`. It is `extra["crisp_threshold"]`
    when the run used `eval_crisp`. Otherwise it is solved exactly as a weighted
    quantile.
  - Components are the face-connected components of the fine-grid solid
    (`scipy.ndimage.label`). Element labels are only for display.
  - **Main** is the component touching the most support elements. Support elements
    are the active elements that have a fixed node.
  - A node is *on* a component when one of its adjacent active elements has that
    component as the majority label of its fine window.
- **`supports_solid` pass rule.** The value is still the fraction of support
  elements that are solid in the crisp design.
  - The check passes when that fraction is above 0 **and** the fixed DOFs on the main
    body restrain all 6 rigid-body modes.
  - "Pass if = 1" would fail valid designs: MMA uses only 48 % of the GE bolt-hole
    elements and 29 % of the bridge pads. A design need not use its whole support area.
- **`volume_target`.** The value is the native volume, using the study's convention:
  history volfrac[-2] for OC/MMA and `res.finalvol` for QUBO.
  - The check passes when the native volume, the pre-smoothing field volume or (for
    QUBO) the binary design volume lies within 0.02 of `volfrac`.
  - FreeTO's returned field is smoothed twice, so its volume can be several points off
    even when the iterate is on target.
- **Extra keys:** `failed` (list of check names), `physics_ok` (every check except
  `volume_target`), `ungrounded_frac`, `crisp_volfrac`, `crisp_threshold`,
  `crisp_threshold_source`, `field_source`, `rigid_body_modes_main`, `load_cases`
  (centroid and resultant per case), `volumes`, `volume_fraction_returned_field`,
  `setup_warnings` and `n_components_listed`.
  - `components` lists at most 50 components, main first, each with extra keys
    `n_elements_touched` and `vol_frac`.
  - QUBO runs also get `binary`: the components of the 0/1 design variables
    (`n_components`, `floating_frac`, `ungrounded_frac`).
- **Without `res.audit_ctx`** (a result not produced by this core), the setup is
  rebuilt from the config. The audited geometry is then FreeTO's rendered level set,
  `res.gridden > res.ls` (`field_source` says so).
- **`info["audit_live"]`** is computed for QUBO runs only, from the binary design
  (`freeto.audit.LiveAudit`). It returns {n_components, floating_frac,
  ungrounded_frac}.
- **Study.** Records also carry `audit_physics_ok`, `audit_failed`, `load_on_main`,
  `supports_solid`, `binary_n_components` and `binary_floating_frac`.
  - Summary rows add `audit_pass`, `audit_n`, `audit_physics_fail`,
    `audit_volume_only_fail`, `n_components_max`, `floating_frac_max` and
    `loads_solid_min`.
  - A group with a physics failure is labelled **physically invalid**. A group that
    fails only `volume_target` is labelled "volume off target".
- **Setup validation** (`run_freeto`; only reported, MATLAB numerics unchanged).
  `setup_info["setup_warnings"]` and log WARNINGs cover:
  - a kept region with no element centre (keepdom, or fixed/force with `keep_bc`)
  - keep elements outside the active domain (count)
  - keep regions holding ≥ `volfrac` of the active elements

  `setup_info["regions"]` gives the node and element counts per region role. Empty
  force or fixed node sets still raise an error, as before.
- **Speed.** Measured on one thread: 0.35 s for 49,619 elements (torsion_bracket MC 60)
  with the stored crisp threshold, 0.7 s without it, and below 0.1 s for 1–2k elements.
