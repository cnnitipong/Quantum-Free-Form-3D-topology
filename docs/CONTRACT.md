# FreeTO-Python — shared interface contract

Original MATLAB source (read-only reference): the original FreeTO MATLAB repository (developer copy outside this package) (FreeTO.m, SIMP.m, SEMDOT.m, geomeshini.m, ...).
Example STLs: the `STLs/` folder of the original repository (copy into `freeto_py/examples/STLs/`).

## Layout
```
freeto_py/
  freeto/            # numerical core (Opus agent)  — pure Python: numpy, scipy, scikit-image, optional pyamg / scikit-sparse
    __init__.py      # exports FreeTOConfig, FreeTOResult, run_freeto, read_stl, write_stl, EXAMPLES
    stl_io.py        # read_stl(path)->(vertices float64 (n,3), faces int64 (m,3)) ascii+binary; write_stl(path, vertices, faces)
    inside.py        # points/grid-in-closed-mesh test (replaces intriangulation.m)
    mesh.py          # geomeshini + domainstokeep + domainprep
    fe.py            # lk_H8, assembly, linear solvers
    filters.py       # HHs3D, HnHns3D
    smoothedge.py    # smoothedge3D
    optimizers.py    # OC (faithful) + MMA (own implementation from Svanberg's published algorithm)
    core.py          # FreeTOConfig, FreeTOResult, run_freeto (SIMP / SEMDOT loop)
    postprocess.py   # symmetry, surface extraction (marching cubes, isocaps-equivalent), STL export
    examples.py      # EXAMPLES dict: the 4 README examples as FreeTOConfig factories
    cli.py           # `python -m freeto.cli ...`
  webapp/            # web app (Sonnet agent)
    server.py        # FastAPI app; `python -m webapp.server` or `uvicorn webapp.server:app`
    static/          # index.html, app.js, style.css, vendor/ (three.js etc. vendored, no CDN at runtime)
  examples/STLs/
  tests/
  requirements.txt
  README.md
  Start QFF-3D (macOS).command, run_app.sh (Linux), Start QFF-3D (Windows).bat
```

## Core API (the web app codes against exactly this)

```python
from freeto import FreeTOConfig, run_freeto, EXAMPLES, read_stl

cfg = FreeTOConfig(
    domain="path/domain.stl",           # required
    forces=["f1.stl", "f2.stl"],        # required, 1..10 load-case region STLs (force1..force10)
    mesh_control=80,                    # MeshControl: number of grid points along the longest-|coord| axis
    volfrac=0.3,
    fixed=None, xfixed=None, yfixed=None, zfixed=None,   # optional STL paths (at least one required)
    keepdom=None,                       # optional STL path
    keep_bc=True, keep_bcx=False, keep_bcy=False, keep_bcz=False,
    youngs_modulus=1.0, poisson_ratio=0.3,
    method="SIMP",                      # "SIMP" | "SEMDOT"
    optimizer="OC",                     # "OC" | "MMA"
    penal=3.0, rmin=1.5,
    fmagx=[0.0], fmagy=[0.0], fmagz=[0.0],   # lists (scalar allowed) — same semantics as MATLAB forcevec
    loadtype="distributed",             # "distributed" | "point"
    symmetry=[],                        # list of (plane, direction): plane in {"x-y","y-z","z-x"}, direction in {"left","right"}; max 3
    max_iter=500,
    solver="auto",                      # "auto" | "cholmod" | "pardiso" | "superlu" | "amg"
)
cfg.validate()   # raises ValueError with a human-readable message (same checks as FreeTO.m + file existence)

result = run_freeto(cfg, callback=None, stop_event=None, log=print)
```

* `callback(info: dict)` is called once after preprocessing with `info["stage"] == "setup"` and then once per iteration with `info["stage"] == "iter"`:
  - setup: `{"stage":"setup","nelx","nely","nelz","nele","nnele","ndof","nfree","nloads","h" (element size), "solver" (name actually used), "setup_time"}`
  - iter: `{"stage":"iter","iter","compliance","volfrac","change","topo","beta","elapsed","iter_time","field": FieldSnapshot}`
  - `FieldSnapshot` has attributes `top` (3D float array, level set φ = xg − ls, solid where φ>0), `origin` (3,), `spacing` (3,), i.e. array index (i,j,k) along axes (0,1,2) maps to physical point `origin + (i,j,k)*spacing` **with arrays already in (x,y,z) axis order** (core is responsible for the MATLAB (y-flipped, y,x,z) → (x,y,z) conversion). Callback must not mutate it.
* `stop_event` (threading.Event): checked every iteration; if set, the loop exits cleanly and a result is still returned with `result.stopped=True`.
* `log(str)`: per-iteration text line like MATLAB's fprintf.

`FreeTOResult` attributes:
`comp, finalvol, eleden (nely,nelx,nelz as in MATLAB), gridden (xg), elenum1, elenum2, history` (dict of lists: `compliance, volfrac, change, topo, beta`), `iterations, stopped, elapsed, field` (FieldSnapshot after symmetry — the final design in physical coords),
methods: `surface(smooth=True) -> (vertices float32 (n,3), faces int32 (m,3))` in physical STL coordinates (closed, capped surface), `write_stl(path)`, `save_npz(path)`.

`freeto.postprocess.surface_from_field(field, smooth=True) -> (vertices, faces)` — same as above for any FieldSnapshot (used for live previews).

`EXAMPLES`: dict name -> dict with keys `title`, `description`, `config_kwargs` (FreeTOConfig kwargs with STL paths **relative to** `examples/STLs/`, e.g. `"GE_domain.STL"`), `files` (dict role->filename for display). Names: `"GE_bracket"`, `"air_bracket"`, `"hand"`, `"quadcopter"`.

Units: like the MATLAB code, STL coordinates are assumed to be in **mm**; forces in N; Young's modulus in Pa; compliance in N·m.
