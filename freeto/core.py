"""FreeTO driver: configuration, SIMP / SEMDOT optimisation loop, result."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field, fields
from typing import Callable, Optional, Sequence

import numpy as np

from .errors import FreeTOError, SINGULAR_MSG
from .fe import (Assembler, lk_H8, make_solver, available_solvers,
                 _rigid_body_modes)
from .filters import HHs3D, HnHns3D
from .mesh import (build_grid, element_dofs, force_vectors, matlab_to_xyz,
                   prepare_domain, support_dofs)
from .optimizers import MMA, oc_update
from .quantum.options import QUBOOptions
from .postprocess import (FieldSnapshot, SYM_PLANES, apply_symmetry,
                          surface_from_field)
from .smoothedge import smoothedge3D
from .evaluate import crisp_projection, refined_voxel_compliance
from .stl_io import read_stl, write_stl as _write_stl

__all__ = ["FreeTOConfig", "FreeTOResult", "run_freeto", "FieldSnapshot",
           "FreeTOError", "MIN_MESH_CONTROL", "DEFAULT_MAX_DOFS"]

NGRID = 4
MIN_MESH_CONTROL = 4
DEFAULT_MAX_DOFS = 3_000_000
_AXES = "xyz"


def _open_edges(faces):
    """Number of edges not shared by exactly two faces (0 = watertight)."""
    f = np.asarray(faces)
    if f.size == 0:
        return 0
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return int(np.count_nonzero(cnt != 2))


def _check_rigid_body(nelx, nely, edofMatn, fixeddof):
    """Raise FreeTOError if some connected part of the active domain is not
    restrained against all six rigid-body motions by the fixed DOFs."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    nodes_e = edofMatn[:, 0::3] // 3
    act_nodes, inv = np.unique(nodes_e, return_inverse=True)
    inv = inv.reshape(nodes_e.shape)
    n = act_nodes.size
    rows = np.repeat(inv[:, 0], 7)
    cols = inv[:, 1:].ravel()
    G = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
    ncomp, lab = connected_components(G, directed=False)
    fd = np.asarray(fixeddof, dtype=np.int64)
    fnode = fd // 3
    pos = np.searchsorted(act_nodes, fnode)
    pos = np.minimum(pos, n - 1)
    ok = act_nodes[pos] == fnode
    fd, pos = fd[ok], pos[ok]
    comp_f = lab[pos]
    for comp in range(ncomp):
        sel = comp_f == comp
        nel_c = int(np.count_nonzero(lab[inv[:, 0]] == comp))
        rank = 0
        if np.any(sel):
            nd = act_nodes[pos[sel]]
            r = nd % (nely + 1)
            c = (nd // (nely + 1)) % (nelx + 1)
            p = nd // ((nely + 1) * (nelx + 1))
            Bm = _rigid_body_modes(np.stack([c, -r, p], axis=1).astype(float))
            rowsel = 3 * np.arange(nd.size) + fd[sel] % 3
            R = Bm[rowsel]
            sv = np.linalg.svd(R, compute_uv=False)
            rank = int(np.count_nonzero(sv > sv.max() * 1e-9)) if sv.size else 0
        if rank < 6:
            part = ("the structure" if ncomp == 1 else
                    f"a disconnected part of the domain ({nel_c} elements)")
            raise FreeTOError(
                f"The supports do not prevent rigid-body motion of {part}: only "
                f"{rank} of the 6 rigid-body motions (3 translations, 3 rotations) "
                "are restrained. Add or enlarge fixed / xfixed / yfixed / zfixed "
                "regions so that they overlap the domain.")


def _setup_warnings(dom, MusD, ele, vol, mesh_control):
    """Setup diagnostics (MATLAB-faithful: nothing is changed, only
    reported): kept regions without any element centre, keep elements outside
    the active domain, keep regions larger than the volume budget."""
    out = []
    seen = set()
    regs = dict(dom.regions or {})
    order = sorted(regs, key=lambda k: k != "keepdom")      # keepdom message first
    for role in order:
        d = regs[role]
        if not d.get("kept") or d.get("elements") is None:
            continue
        key = (d["file"], d["elements"])
        if d["elements"] == 0 and key not in seen:
            seen.add(key)
            if role == "keepdom":
                out.append(f"the keepdom region ({d['file']}) contains no element centre at "
                           f"mesh_control={mesh_control}: nothing is kept solid by it. Thicken "
                           "the region (>= 1 element size inside the domain) or refine the mesh.")
            else:
                out.append(f"the {role} region ({d['file']}) contains no element centre at "
                           f"mesh_control={mesh_control}, so keep_bc keeps none of its material: "
                           "its supports/loads act on design elements that may become void.")
    n_out = int(MusD.size - np.count_nonzero(np.isin(MusD, ele)))
    if n_out > 0:
        regs = sorted({d["file"] for d in (dom.regions or {}).values()
                       if d.get("elements_outside")})
        out.append(f"{n_out} keep element(s) lie outside the design domain"
                   + (f" (region(s) {', '.join(regs)})" if regs else "")
                   + ": they are not part of the FE model but are written as solid into the "
                     "returned field (full[MusD] = 1, as FreeTO.m). Clip the region to the domain.")
    nk = int(np.count_nonzero(np.isin(MusD, ele)))
    if ele.size and nk >= vol * ele.size:
        out.append(f"the keep regions hold {100.0 * nk / ele.size:.1f} % of the active elements, "
                   f"not less than volfrac = {vol:g}: the volume constraint cannot be met.")
    return out


def _flist(v):
    if v is None:
        return [0.0]
    if np.isscalar(v):
        return [float(v)]
    out = [float(a) for a in np.asarray(v, dtype=float).ravel()]
    return out if out else [0.0]


@dataclass
class FreeTOConfig:
    """Problem definition (mirrors FreeTO.m's inputs).

    STL coordinates are in mm, forces in N, Young's modulus in Pa.
    Extra (non-MATLAB) knobs: ``tolx``, ``tol_thresh``, ``beta_init``,
    ``beta_step``, ``beta_max`` (None -> method defaults, see below),
    ``mma_move``, ``compat`` ("matlab" | "octave" linspace/colon arithmetic),
    ``inside_mode`` ("robust" watertight ray casting with 3-axis vote, or
    "matlab" for the exact intriangulation.m rules), ``amg_rtol``.

    Defaults as SIMP.m/SEMDOT.m: tolx = tol_thresh = 3e-3, beta 0.1 -> +0.05
    per iteration while beta < 2.  With ``optimizer="MMA"`` the README's
    recommendation is used: tolx = tol_thresh = 1e-3, beta = ER = 0.5 and the
    ``if beta < 2`` guard removed (beta grows without bound; set ``beta_max``
    to cap it).

    ``optimizer="QUBO"`` (docs/QUANTUM_API.md): binary design update solved
    as a QUBO; options in ``qubo`` (:class:`freeto.quantum.QUBOOptions` or a
    dict).  Defaults then tolx = tol_thresh = 1e-3, beta 0.5 -> +0.5 per
    iteration up to 8.  ``eval_binary=True`` additionally reports the
    compliance of the binary-thresholded final design in
    ``result.extra["binary_compliance"]``; ``eval_beta=b`` re-smooths the
    final (filtered, pre-smoothing) design at Heaviside sharpness b and
    reports its compliance in ``result.extra["compliance_at_beta"]`` (common-
    beta comparison of OC / MMA / QUBO designs).  ``eval_crisp=V`` (or True =
    volfrac) evaluates the *common crisp design* (freeto/evaluate.py): the
    final pre-smoothing field projected crisp on the fine grid at volume
    fraction V -> ``extra["crisp_compliance"]``, ``["crisp_volfrac"]``, plus
    ``["compliance_returned_design"]`` (one FE solve on res.eleden).  Failed
    post-run evaluations never discard the run: the metric is inf and the
    message is in ``extra["eval_errors"]``.

    v2 options (docs/NOTES_quantum.md §9; all off by default):
    ``eval_refined=f`` additionally evaluates the same pre-smoothing field as a
    binary voxel design on a grid refined f times per element
    (:func:`freeto.evaluate.refined_voxel_compliance`) ->
    ``extra["refined_compliance"]``, ``["refined_volfrac"]``,
    ``["refined_threshold"]``, ``["refined_time"]``, ``["refined_f"]``.
    ``mma_constraint="filtered"`` computes MMA's volume constraint from the
    pre-projection filtered densities instead of the projected ones
    (default "projected" = FreeTO).  ``mma_feasible_stop=True`` lets MMA stop
    by the change / topology tolerances only when the returned design
    satisfies |fval| <= 1e-3 (max_iter still ends the run).
    ``init_perturb=a`` (> 0) starts OC/MMA from volfrac + U(-a, a) per active
    non-keep element (``numpy.random.default_rng(init_seed)``), clipped to
    [0.001, 1].  ``res.extra["full_pre"]`` is always a float32 copy of the
    evaluated pre-smoothing field.

    ``audit=True`` (default) runs the physics / connectivity check of
    :mod:`freeto.audit` on the returned design (docs/AUDIT_API.md) and stores
    its dict in ``result.extra["audit"]``; a failing audit never raises (a
    crash of the check itself is reported in ``extra["audit_error"]``).

    ``res.comp`` semantics: OC/MMA -- the compliance computed in the last
    iteration, i.e. of the design *entering* that iteration (FreeTO / MATLAB
    convention); QUBO -- one extra FE solve on the returned design.
    """
    domain: str = None
    forces: Sequence[str] = ()
    mesh_control: int = 80
    volfrac: float = 0.3
    fixed: Optional[str] = None
    xfixed: Optional[str] = None
    yfixed: Optional[str] = None
    zfixed: Optional[str] = None
    keepdom: Optional[str] = None
    keep_bc: bool = True
    keep_bcx: bool = False
    keep_bcy: bool = False
    keep_bcz: bool = False
    youngs_modulus: float = 1.0
    poisson_ratio: float = 0.3
    method: str = "SIMP"
    optimizer: str = "OC"
    penal: float = 3.0
    rmin: float = 1.5
    fmagx: Sequence[float] = (0.0,)
    fmagy: Sequence[float] = (0.0,)
    fmagz: Sequence[float] = (0.0,)
    loadtype: str = "distributed"
    symmetry: Sequence = ()
    max_iter: int = 500
    solver: str = "auto"
    # ---- extras -----------------------------------------------------------
    tolx: Optional[float] = None
    tol_thresh: Optional[float] = None
    beta_init: Optional[float] = None
    beta_step: Optional[float] = None
    beta_max: Optional[float] = None
    mma_move: float = 0.5
    compat: str = "matlab"
    inside_mode: str = "robust"
    amg_rtol: float = 1e-8
    max_dofs: Optional[int] = DEFAULT_MAX_DOFS
    qubo: Optional[object] = None
    eval_binary: bool = False
    eval_beta: Optional[float] = None
    eval_crisp: Optional[object] = None
    audit: bool = True
    # ---- v2 options (2026-10-04, opt-in; defaults = FreeTO behaviour) -------
    eval_refined: Optional[int] = None
    mma_constraint: str = "projected"
    mma_feasible_stop: bool = False
    init_perturb: float = 0.0
    init_seed: int = 0

    # ------------------------------------------------------------------
    def _paths(self):
        out = [("domain", self.domain)]
        out += [(f"force{i + 1}", f) for i, f in enumerate(self.forces or [])]
        for nm in ("fixed", "xfixed", "yfixed", "zfixed", "keepdom"):
            v = getattr(self, nm)
            if v:
                out.append((nm, v))
        return out

    def normalized(self):
        """Copy with canonical types (lists, upper-case method names...)."""
        c = FreeTOConfig(**{f.name: getattr(self, f.name) for f in fields(self)})
        if isinstance(c.forces, (str, os.PathLike)):
            c.forces = [c.forces]
        c.forces = [os.fspath(f) for f in (c.forces or []) if f]
        for nm in ("domain", "fixed", "xfixed", "yfixed", "zfixed", "keepdom"):
            v = getattr(c, nm)
            setattr(c, nm, os.fspath(v) if v else None)
        c.fmagx, c.fmagy, c.fmagz = (_flist(c.fmagx), _flist(c.fmagy),
                                     _flist(c.fmagz))
        c.method = str(c.method).upper()
        c.optimizer = str(c.optimizer).upper()
        c.loadtype = str(c.loadtype).lower()
        c.solver = str(c.solver).lower()
        sym = []
        for s in (c.symmetry or []):
            if isinstance(s, str):
                s = (s, "right")
            plane, direction = s
            sym.append((str(plane).lower(), str(direction).lower()))
        c.symmetry = sym
        c.mesh_control = int(c.mesh_control) if c.mesh_control is not None else None
        c.max_iter = int(c.max_iter)
        c.mma_constraint = str(c.mma_constraint).lower()
        if c.eval_refined is not None:
            c.eval_refined = int(c.eval_refined)
        if c.optimizer == "QUBO":
            try:
                c.qubo = QUBOOptions.from_any(c.qubo).normalized()
            except ValueError as e:
                raise FreeTOError(str(e)) from None
        return c

    def validate(self):
        """Raise ValueError with a readable message if the input is invalid
        (the checks of FreeTO.m plus file existence and consistency)."""
        c = self.normalized()
        if not c.domain:
            raise FreeTOError("A domain STL file is required.")
        if not c.forces:
            raise FreeTOError("At least one force (load region) STL is required.")
        if len(c.forces) > 10:
            raise FreeTOError("At most 10 force regions (force1..force10) are supported.")
        for role, p in c._paths():
            if not os.path.isfile(p):
                raise FreeTOError(f"{role}: file not found: {p}")
        if not (c.fixed or c.xfixed or c.yfixed or c.zfixed):
            raise FreeTOError("Please include at least one support condition "
                             "(fixed, xfixed, yfixed or zfixed).")
        if c.mesh_control is None or c.mesh_control < MIN_MESH_CONTROL:
            raise FreeTOError(f"mesh_control (MeshControl) must be an integer >= "
                              f"{MIN_MESH_CONTROL}.")
        if not (0.0 < float(c.volfrac) <= 1.0):
            raise FreeTOError("volfrac must be in (0, 1].")
        if c.method not in ("SIMP", "SEMDOT"):
            raise FreeTOError("method must be 'SIMP' or 'SEMDOT'.")
        if c.optimizer not in ("OC", "MMA", "QUBO"):
            raise FreeTOError("optimizer must be 'OC', 'MMA' or 'QUBO'.")
        if c.optimizer == "QUBO":
            try:
                c.qubo.validate()
            except (ValueError, RuntimeError) as e:
                raise FreeTOError(f"QUBO options: {e}") from None
        if c.loadtype not in ("distributed", "point"):
            raise FreeTOError("loadtype must be 'distributed' or 'point'.")
        if c.solver not in ("auto", "cholmod", "pardiso", "superlu", "amg"):
            raise FreeTOError("solver must be one of auto, cholmod, pardiso, superlu, amg.")
        if c.solver != "auto" and c.solver not in available_solvers():
            raise FreeTOError(f"solver '{c.solver}' is not installed "
                             f"(available: {', '.join(available_solvers())}).")
        lens = [len(c.fmagx), len(c.fmagy), len(c.fmagz)]
        nf = max(lens)
        for nm, ln in zip(("fmagx", "fmagy", "fmagz"), lens):
            if ln not in (1, nf):
                raise FreeTOError(f"{nm} has {ln} entries; each force component "
                                 f"must be a scalar or have {nf} entries (one per load case).")
        if nf > len(c.forces):
            raise FreeTOError(f"{nf} load cases are defined by fmagx/fmagy/fmagz but "
                             f"only {len(c.forces)} force region STL(s) were given.")
        if all(v == 0 for v in c.fmagx + c.fmagy + c.fmagz):
            raise FreeTOError("Please include at least one non-zero load definition.")
        if not all(math.isfinite(v) for v in c.fmagx + c.fmagy + c.fmagz):
            raise FreeTOError("Force magnitudes must be finite numbers.")
        if not (float(c.youngs_modulus) > 0):
            raise FreeTOError("youngs_modulus must be positive.")
        if not (-1.0 < float(c.poisson_ratio) < 0.5):
            raise FreeTOError("poisson_ratio must be in (-1, 0.5).")
        if c.method == "SIMP" and not (float(c.penal) > 0):
            raise FreeTOError("penal must be positive.")
        if not (float(c.rmin) > 0):
            raise FreeTOError("rmin must be positive.")
        if c.max_iter < 1:
            raise FreeTOError("max_iter must be >= 1.")
        if len(c.symmetry) > 3:
            raise FreeTOError("Symmetry can be specified at most 3 times.")
        for plane, direction in c.symmetry:
            if plane not in SYM_PLANES:
                raise FreeTOError(f"symmetry plane must be one of {SYM_PLANES}, got {plane!r}.")
            if direction not in ("left", "right"):
                raise FreeTOError(f"symmetry direction must be 'left' or 'right', got {direction!r}.")
        if c.inside_mode not in ("matlab", "robust"):
            raise FreeTOError("inside_mode must be 'matlab' or 'robust'.")
        if c.compat not in ("matlab", "octave"):
            raise FreeTOError("compat must be 'matlab' or 'octave'.")
        if c.mma_constraint not in ("projected", "filtered"):
            raise FreeTOError("mma_constraint must be 'projected' or 'filtered'.")
        if c.eval_refined is not None and not (c.eval_refined >= 1
                                               and NGRID % c.eval_refined == 0):
            raise FreeTOError(f"eval_refined must be a divisor of {NGRID} (1, 2 or 4) or None.")
        if not (0.0 <= float(c.init_perturb) < 1.0):
            raise FreeTOError("init_perturb must be in [0, 1).")
        # ---- geometry checks (cheap: reads the STLs, builds the 1-D grid)
        for role, p in c._paths():
            try:
                V, Fc = read_stl(p)
            except Exception as e:  # noqa: BLE001
                raise FreeTOError(f"{role}: cannot read STL file {os.path.basename(p)}: {e}")
            if Fc.shape[0] == 0:
                raise FreeTOError(f"{role}: STL file {os.path.basename(p)} contains no triangles.")
            if role == "domain":
                Vd = V
        ext = Vd.max(axis=0) - Vd.min(axis=0)
        if not np.all(ext > 1e-9 * max(float(ext.max()), 1e-300)):
            flat = ", ".join(_AXES[i] for i in range(3) if not ext[i] > 1e-9 * ext.max())
            raise FreeTOError(f"The domain STL is flat/degenerate (zero extent along "
                              f"{flat}); it must be a closed solid.")
        g = build_grid(Vd, c.mesh_control, c.compat)
        nel = (g.nelx, g.nely, g.nelz)
        if min(nel) < 1:
            ax = _AXES[int(np.argmin(nel))]
            raise FreeTOError(
                f"At mesh_control={c.mesh_control} the grid has no elements along "
                f"{ax} (element size {g.ssz:.3g} mm vs. domain extent "
                f"{ext['xyz'.index(ax)]:.3g} mm); increase mesh_control.")
        ndof = 3 * (g.nelx + 1) * (g.nely + 1) * (g.nelz + 1)
        if c.max_dofs is not None and ndof > c.max_dofs:
            raise FreeTOError(
                f"mesh_control={c.mesh_control} gives a {g.nelx}x{g.nely}x{g.nelz} grid "
                f"({g.nelx * g.nely * g.nelz:,} elements, {ndof:,} DOFs), above the "
                f"limit of {c.max_dofs:,} DOFs; this would need many GB of memory and "
                f"hours of run time. Reduce mesh_control (or raise max_dofs).")
        return True


@dataclass
class FreeTOResult:
    comp: float
    finalvol: float
    eleden: np.ndarray            # (nely, nelx, nelz), MATLAB layout
    gridden: np.ndarray           # xg, fine grid, MATLAB layout (pre-symmetry)
    elenum1: int                  # active elements
    elenum2: int                  # all elements
    history: dict
    iterations: int
    stopped: bool
    elapsed: float
    field: FieldSnapshot          # after symmetry, physical (x,y,z) frame
    ls: float = 0.0
    setup_info: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    config: Optional[FreeTOConfig] = None
    extra: dict = field(default_factory=dict)
    # setup arrays of the run kept for freeto.audit (references, not copies;
    # not serialised): grid size, active elements, keep elements, fixed DOFs,
    # F, the returned design's pre-smoothing field, region counts
    audit_ctx: Optional[dict] = field(default=None, repr=False, compare=False)

    def surface(self, smooth=True):
        """Closed, capped triangle surface in physical STL coordinates."""
        return surface_from_field(self.field, smooth=smooth)

    def write_stl(self, path, smooth=True):
        v, f = self.surface(smooth=smooth)
        _write_stl(path, v, f, header="FreeTO-Python optimized topology")
        return path

    def save_npz(self, path):
        h = {f"history_{k}": np.asarray(v) for k, v in self.history.items()}
        np.savez_compressed(
            path, comp=self.comp, finalvol=self.finalvol, eleden=self.eleden,
            gridden=self.gridden, elenum1=self.elenum1, elenum2=self.elenum2,
            iterations=self.iterations, stopped=self.stopped,
            elapsed=self.elapsed, ls=self.ls, top=np.asarray(self.field.top),
            origin=self.field.origin, spacing=self.field.spacing, **h)
        return path


def _fmt_g(v):
    """MATLAB %g."""
    return "%g" % v


def run_freeto(cfg: FreeTOConfig, callback: Optional[Callable] = None,
               stop_event=None, log: Optional[Callable] = print,
               _debug=None):
    """Run FreeTO (see CONTRACT.md).  ``_debug``: optional dict that receives
    internal arrays (used by the reference comparison)."""
    t_start = time.perf_counter()
    log = log or (lambda s: None)
    cfg.validate()
    c = cfg.normalized()
    timings = {}

    # ------------------------------------------------------------ setup
    t0 = time.perf_counter()
    dom = prepare_domain(c.mesh_control, c.domain, c.fixed, c.xfixed,
                         c.yfixed, c.zfixed, c.forces, c.keepdom, c.keep_bc,
                         c.keep_bcx, c.keep_bcy, c.keep_bcz, compat=c.compat,
                         inside_mode=c.inside_mode)
    timings.update(dom.timings)
    nelx, nely, nelz, nele, ndof = dom.nelx, dom.nely, dom.nelz, dom.nele, dom.ndof
    n_open = _open_edges(read_stl(c.domain)[1])
    if n_open:
        log(f"WARNING: the domain STL is not watertight ({n_open} open or "
            "non-manifold edges); grid points near the gaps may be misclassified "
            "(usually harmless for small gaps).")
    if ndof > 1_000_000:
        log(f"WARNING: large problem ({nele:,} elements, {ndof:,} grid DOFs); expect "
            "high memory use and long run times.")
    grid = dom.grid
    for i, fn in enumerate(dom.Fn[:max(len(c.fmagx), len(c.fmagy), len(c.fmagz))]):
        if len(fn) == 0:
            raise FreeTOError(f"force region {i + 1} ({os.path.basename(c.forces[i])}) "
                             "contains no grid nodes; increase mesh_control or "
                             "enlarge the region.")
    t1 = time.perf_counter()
    oute_f = dom.oute.ravel(order="F")
    ele = np.flatnonzero(oute_f != 0)
    nnele = int(ele.size)
    if nnele == 0:
        raise FreeTOError("The domain contains no elements at this mesh_control; "
                         "check the domain STL (closed surface?) or increase mesh_control.")
    edofMatn = element_dofs(nelx, nely, nelz, ele)
    n_vec = np.unique(edofMatn)
    MusD = np.flatnonzero(dom.outeM.ravel(order="F") == 1)
    F, nf = force_vectors(dom.Fn, ndof, c.fmagx, c.fmagy, c.fmagz, c.loadtype)
    fixeddof = support_dofs(dom.sup_all, dom.sup_x, dom.sup_y, dom.sup_z)
    freedofs = np.setdiff1d(n_vec, fixeddof)
    if freedofs.size == 0:
        raise FreeTOError("All degrees of freedom are fixed.")
    if F.shape[1] == 0 or not np.any(F[freedofs] != 0):
        raise FreeTOError("No load acts on a free degree of freedom (check that "
                         "the force regions overlap the domain and are not fully "
                         "supported).")
    if fixeddof.size == 0 or np.intersect1d(fixeddof, n_vec).size == 0:
        raise FreeTOError("No support node lies on the active domain; the "
                          "structure would be unconstrained.")
    _check_rigid_body(nelx, nely, edofMatn, np.intersect1d(fixeddof, n_vec))
    setup_warnings = _setup_warnings(dom, MusD, ele, float(c.volfrac), c.mesh_control)
    for w_ in setup_warnings:
        log("WARNING: " + w_)
    KE = (grid.aa / 0.5) * lk_H8(c.poisson_ratio)
    timings["domainprep"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    H, Hs = HHs3D(nelx, nely, nelz, c.rmin, ele, nele)
    Hn, Hns = HnHns3D(nelx, nely, nelz, 1)
    timings["filters"] = time.perf_counter() - t1
    t1 = time.perf_counter()
    asm = Assembler(edofMatn, KE, freedofs, ndof)
    timings["assembly_setup"] = time.perf_counter() - t1
    # node coordinates of the active dofs (for AMG rigid-body modes)
    act_nodes = asm.act[::3] // 3
    r = act_nodes % (nely + 1)
    cc_ = (act_nodes // (nely + 1)) % (nelx + 1)
    p = act_nodes // ((nely + 1) * (nelx + 1))
    node_coords = np.stack([cc_, -r, p], axis=1).astype(float)
    solver = make_solver(c.solver, asm, node_coords=node_coords, log=log)
    if isinstance(getattr(solver, "rtol", None), float):
        solver.rtol = c.amg_rtol
    setup_time = time.perf_counter() - t0
    timings["setup_total"] = setup_time

    # ------------------------------------------------------------ params
    mma_mode = c.optimizer == "MMA"
    qubo_mode = c.optimizer == "QUBO"
    mq = mma_mode or qubo_mode
    maxloop = c.max_iter
    tolx = c.tolx if c.tolx is not None else (1e-3 if mq else 0.003)
    tol_thresh = c.tol_thresh if c.tol_thresh is not None else (1e-3 if mq else 3e-3)
    beta = c.beta_init if c.beta_init is not None else (0.5 if mq else 0.1)
    ER = c.beta_step if c.beta_step is not None else (0.5 if mq else 0.05)
    beta_max = c.beta_max if c.beta_max is not None else (
        math.inf if mma_mode else (8.0 if qubo_mode else 2.0))
    E0 = float(c.youngs_modulus)
    Emin = 0.001
    penal = float(c.penal)
    vol = float(c.volfrac)
    simp = c.method == "SIMP"
    h = grid.ssz
    origin = np.array([grid.x[0], grid.y[0], grid.z[0]])
    spacing = np.full(3, h / NGRID)

    info = {"stage": "setup", "nelx": nelx, "nely": nely, "nelz": nelz,
            "nele": nele, "nnele": nnele, "ndof": ndof,
            "nfree": int(freedofs.size), "nloads": int(F.shape[1]), "h": h,
            "solver": solver.name, "setup_time": setup_time,
            "setup_warnings": list(setup_warnings), "regions": dict(dom.regions or {}),
            "n_keep": int(np.count_nonzero(np.isin(MusD, ele))),
            "n_keep_outside_domain": int(MusD.size - np.count_nonzero(np.isin(MusD, ele)))}
    qupd = None
    if qubo_mode:
        from .quantum.update import QUBOUpdater, robust_solve as _robust_solve
        qupd = QUBOUpdater(c.qubo, asm=asm, solver=solver, KE=KE, edofMatn=edofMatn,
                           freedofs=freedofs, H=H, Hs=Hs, ele=ele, MusD=MusD,
                           nelx=nelx, nely=nely, nelz=nelz, vol=vol, nnele=nnele,
                           E0=E0, Emin=Emin, penal=penal, simp=simp, ndof=ndof, log=log,
                           F=F)
        info["qubo"] = qupd.describe()
    live = None
    if qubo_mode and c.audit:
        try:
            from .audit import LiveAudit
            live = LiveAudit(nelx, nely, nelz, ele, np.intersect1d(fixeddof, n_vec))
        except Exception:  # noqa: BLE001 -- the live check is optional
            live = None
    log(f"FreeTO: grid {nelx}x{nely}x{nelz} (h = {h:.4g} mm), {nnele} active of "
        f"{nele} elements, {freedofs.size} free DOFs, {F.shape[1]} load case(s), "
        f"solver {solver.name}, setup {setup_time:.2f} s")
    if callback is not None:
        callback(dict(info))
    if _debug is not None:
        _debug.update(dict(dom=dom, ele=ele, edofMatn=edofMatn, n_vec=n_vec,
                           MusD=MusD, F=F, fixeddof=fixeddof, freedofs=freedofs,
                           KE=KE, H=H, Hs=Hs, Hn=Hn, Hns=Hns, iters=[]))

    # ------------------------------------------------------------ loop
    F_free = np.ascontiguousarray(F[freedofs])
    nload = F.shape[1]
    U = np.zeros((ndof, nload))
    Uf_prev = None
    vx = np.full(nnele, vol)
    keep_pos = np.isin(ele, MusD)           # active elements kept solid (full[MusD] = 1)
    if qubo_mode:
        vx = qupd.initial_design(vx)
    elif float(c.init_perturb) > 0:
        # perturbed start (v2): volfrac + U(-a, a) on the non-keep elements
        rng_ = np.random.default_rng(int(c.init_seed))
        a_ = float(c.init_perturb)
        pert = rng_.uniform(-a_, a_, nnele)
        vx = np.where(keep_pos, vx, np.clip(vol + pert, 0.001, 1.0))
    vxPhys = vx.copy()
    vxPhys_full = None
    xg = top = None
    lss = 0.0
    loop = 0
    change = 1.0
    tol = 1.0
    hist = {"compliance": [], "volfrac": [], "change": [], "topo": [], "beta": []}
    mma = MMA(nnele, 1, 0.0, 1.0, move=c.mma_move) if mma_mode else None
    mma_filtered = mma_mode and c.mma_constraint == "filtered"
    mma_fstop = mma_mode and bool(c.mma_feasible_stop)
    fval_new = 0.0           # MMA constraint value of the latest design (feasible stop)
    stopped = False
    t_fe = t_se = t_up = 0.0
    q_done = False
    q_prev = q_best = None
    q_best_c = math.inf
    n_fe = 0
    q_acc = None

    def _qubo_fe(vxP, U0):
        """FE solve + sensitivities of a design (QUBO mode only; same formulas
        as the loop body)."""
        Ee_ = Emin + vxP ** penal * (E0 - Emin) if simp else \
            vxP * E0 + (1 - vxP) * (Emin * E0)
        data_ = asm.data(Ee_)
        try:
            Uf_ = _robust_solve(solver, asm, data_, F_free, U0)
        except FreeTOError:
            raise
        except Exception as e:  # noqa: BLE001
            raise FreeTOError(f"{SINGULAR_MSG} (linear solver {solver.name}: "
                              f"{type(e).__name__}: {e})") from e
        if not np.all(np.isfinite(Uf_)):
            raise FreeTOError(SINGULAR_MSG + " (non-finite displacements)")
        Uf_ = np.asarray(Uf_).reshape(freedofs.size, nload)
        U_ = np.zeros((ndof, nload))
        U_[freedofs] = Uf_
        c_ = 0.0
        dc_ = np.zeros(nnele)
        for i_ in range(nload):
            Ue_ = U_[edofMatn, i_]
            ce_ = np.einsum("ij,ij->i", Ue_ @ KE, Ue_)
            if simp:
                c_ = c_ + np.sum(Ee_ * ce_)
                dc_ = dc_ - penal * vxP ** (penal - 1) * (E0 - Emin) * ce_
            else:
                c_ = c_ + np.sum(Ee_ * ce_)
                dc_ = dc_ - ((1 - vxP) * Emin + vxP) * E0 * ce_
        dv_ = H @ (np.ones(nnele) / Hs)
        dc_ = H @ (dc_ / Hs)
        return U_, data_, float(c_), dc_, dv_, Uf_

    beta_used = beta
    full_pre = None
    it_start_all = time.perf_counter()
    while ((not qubo_mode and ((change > tolx and tol > tol_thresh)
                               or (mma_fstop and abs(fval_new) > 1e-3)))
           or (qubo_mode and not q_done)) and loop < maxloop:
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        ti = time.perf_counter()
        loop += 1
        # FE analysis
        if simp:
            Ee = Emin + vxPhys ** penal * (E0 - Emin)
        else:
            Ee = vxPhys * E0 + (1 - vxPhys) * (Emin * E0)
        data = asm.data(Ee)
        try:
            if qubo_mode:
                Uf = _robust_solve(solver, asm, data, F_free, Uf_prev)
            else:
                # robust_solve returns solver.solve's result unchanged whenever it
                # passes the residual check; only a solve that would otherwise abort
                # the run as "singular" (relative residual in (1e-6, 1e-3], e.g. the
                # near-void transient of MMA on mbb_beam at V = 0.26) is refined
                from .quantum.update import robust_solve as _rs
                Uf = _rs(solver, asm, data, F_free, Uf_prev)
        except FreeTOError:
            raise
        except Exception as e:  # singular matrix inside the solver library
            raise FreeTOError(f"{SINGULAR_MSG} (linear solver {solver.name}: "
                              f"{type(e).__name__}: {e})") from e
        if not np.all(np.isfinite(Uf)):
            raise FreeTOError(SINGULAR_MSG + " (non-finite displacements)")
        Uf = np.asarray(Uf).reshape(freedofs.size, nload)
        Uf_prev = Uf
        n_fe += 1
        U[freedofs] = Uf
        t_fe_i = time.perf_counter() - ti
        tu = time.perf_counter()
        cval = 0.0
        dc = np.zeros(nnele)
        for i in range(nload):
            Ue = U[edofMatn, i]
            ce = np.einsum("ij,ij->i", Ue @ KE, Ue)
            if simp:
                cval = cval + np.sum(Ee * ce)
                dc = dc - penal * vxPhys ** (penal - 1) * (E0 - Emin) * ce
            else:
                cval = cval + np.sum((vxPhys * E0 + (1 - vxPhys) * (Emin * E0)) * ce)
                dc = dc - ((1 - vxPhys) * Emin + vxPhys) * E0 * ce
        dv = np.ones(nnele)
        dc = H @ (dc / Hs)
        dv = H @ (dv / Hs)
        dbg = None
        if _debug is not None:
            dbg = {"c": cval, "dc": dc.copy(), "dv": dv.copy(), "U": U.copy(),
                   "Ee": Ee.copy()}
        if qubo_mode:
            if q_prev is not None and q_acc is not None and qupd.guard_reject(cval,
                                                                            q_acc["c"]):
                # accept-if-improves guard: restore the design the rejected step
                # was computed from and redo its FE solve (refreshes U, dc and
                # the factorisation used by the block Hessian)
                vx, vxPhys = q_acc["vx"], q_acc["vxPhys"]
                U, data, cval, dc, dv, Uf_prev = _qubo_fe(vxPhys, Uf_prev)
                n_fe += 1
            elif q_prev is not None:
                # cval is the compliance of the design produced last iteration
                qupd.observe(cval)
                if q_prev["at_target"] and cval < q_best_c:
                    q_best_c, q_best = float(cval), q_prev
            q_acc = {"vx": np.array(vx, copy=True), "vxPhys": np.array(vxPhys, copy=True),
                     "c": float(cval)}
            vxnew = qupd.update(loop, vx, vxPhys, dc, dv, U, data, cval, stop_event,
                                oc_fallback=lambda: oc_update(vxPhys, dc, dv, vol, nnele))
        elif not mma_mode:
            vxnew = oc_update(vxPhys, dc, dv, vol, nnele)
        elif not mma_filtered:
            fval = np.sum(vxPhys) / (vol * nnele) - 1
            dfdx = (dv / (vol * nnele))[None, :]
            vxnew = mma.update(loop, vx, cval, dc, fval, dfdx)
        else:
            # mma_constraint="filtered" (v2): the volume constraint is evaluated on
            # the pre-projection filtered densities (H x)/Hs with the keep elements
            # at 1, i.e. on full_pre.  The projected densities vxPhys come from
            # smoothedge3D, whose threshold preserves the volume of the whole
            # bounding-box grid (void elements outside the domain included), not
            # of the active elements; on non-prismatic domains sum(vxPhys) then
            # drifts from the volume MMA believes it controls.
            vf_ = (H @ vx) / Hs
            vf_[keep_pos] = 1.0
            fval = float(np.sum(vf_)) / (vol * nnele) - 1
            # exact gradient of sum over the non-keep elements of (H x)/Hs
            # (H symmetric); away from keep regions this is H @ (1/Hs)
            dfdx = ((H @ (np.where(keep_pos, 0.0, 1.0) / Hs)) / (vol * nnele))[None, :]
            vxnew = mma.update(loop, vx, cval, dc, fval, dfdx)
        vxPhys = (H @ vxnew) / Hs
        full = np.zeros(nele)
        full[ele] = vxPhys
        full[MusD] = 1
        t_up += time.perf_counter() - tu
        ts = time.perf_counter()
        full_pre = full
        beta_used = beta
        full, xg, lss, top, tol = smoothedge3D(full, Hn, Hns, nelx, nely, nelz,
                                               nele, nnele, beta, NGRID)
        change = np.sum(np.abs(vxnew - vx)) / (vol * nnele)
        vx = vxnew
        if mma_fstop:
            # constraint value of the design just produced (the one returned if
            # the loop stops now), with the constraint definition MMA uses
            fval_new = (float(np.sum(full_pre[ele])) if mma_filtered
                        else float(np.sum(full[ele]))) / (vol * nnele) - 1
        if qubo_mode:
            q_done = qupd.should_stop(change, tolx) or (
                stop_event is not None and stop_event.is_set())
        elif (((change <= tolx or tol <= tol_thresh)
               and not (mma_fstop and abs(fval_new) > 1e-3)) or loop >= maxloop):
            full[MusD] = 1
            full, xg, lss, top, tol = smoothedge3D(full, Hn, Hns, nelx, nely,
                                                   nelz, nele, nnele, beta, NGRID)
        t_se_i = time.perf_counter() - ts
        t_se += t_se_i
        t_fe += t_fe_i
        fvol = np.sum(full) / nnele
        log("It.:%5i Obj.:%11.3f Vol.:%7.3f ch.:%7.5f Topo.:%7.5f"
            % (loop, cval, fvol, change, tol))
        if qubo_mode and qupd.last is not None:
            q = qupd.last
            log("  QUBO[%s/%s]: free %d, blocks %d, solves %d, V_k %.3f, flips %d, "
                "update %.2f s (Hessian %.2f s, solver %.2f s)"
                % (q["backend"], q["hessian"], q["n_free"], q["n_blocks"], q["n_solves"],
                   q["volume_target"], q["n_flips"], q["wall_time"], q["hessian_time"],
                   q["solver_time"]))
        if mq:
            if beta < beta_max:
                beta = min(beta + ER, beta_max) if math.isfinite(beta_max) else beta + ER
        else:
            if beta < beta_max:
                beta = beta + ER
        log("Parameter beta increased to %s." % _fmt_g(beta))
        vxPhys_full = full.reshape((nely, nelx, nelz), order="F")
        vxPhys = full[ele]
        hist["compliance"].append(float(cval))
        hist["volfrac"].append(float(fvol))
        hist["change"].append(float(change))
        hist["topo"].append(float(tol))
        hist["beta"].append(float(beta))
        if dbg is not None:
            dbg.update(vxnew=vxnew.copy(), vxPhys_full=full.copy(), xg=xg,
                       lss=lss, tol=tol, change=change, beta=beta, fvol=fvol)
            _debug["iters"].append(dbg)
        if qubo_mode:
            q_prev = {"vx": vx, "full": full, "xg": xg, "lss": lss, "top": top,
                      "tol": tol, "fvol": fvol, "iter": loop, "full_pre": full_pre,
                      "at_target": bool(qupd.at_target), "beta": float(beta_used)}
        if callback is not None:
            snap = FieldSnapshot(matlab_to_xyz(top), origin, spacing, lss)
            cb = {"stage": "iter", "iter": loop, "compliance": float(cval),
                  "volfrac": float(fvol), "change": float(change),
                  "topo": float(tol), "beta": float(beta),
                  "elapsed": time.perf_counter() - t_start,
                  "iter_time": time.perf_counter() - ti, "field": snap,
                  "fe_time": t_fe_i, "smooth_time": t_se_i}
            if qubo_mode and qupd.last is not None:
                cb["qubo"] = dict(qupd.last)
            if live is not None:
                try:
                    cb["audit_live"] = live(vx)
                except Exception:  # noqa: BLE001
                    pass
            callback(cb)

    if loop == 0:
        # stopped before the first iteration: report the initial design
        full = np.zeros(nele)
        full[ele] = vxPhys
        full[MusD] = 1
        full_pre = full.copy()
        full, xg, lss, top, tol = smoothedge3D(full, Hn, Hns, nelx, nely, nelz,
                                               nele, nnele, beta, NGRID)
        vxPhys_full = full.reshape((nely, nelx, nelz), order="F")
    extra = {}

    def _fe_compliance(rho):
        if simp:
            Ee_ = Emin + rho ** penal * (E0 - Emin)
        else:
            Ee_ = rho * E0 + (1 - rho) * (Emin * E0)
        data_ = asm.data(Ee_)
        from .quantum.update import robust_solve
        try:
            Uf_ = robust_solve(solver, asm, data_, F_free, Uf_prev)
        except FreeTOError:
            raise
        except Exception as e:  # singular matrix inside the solver library
            raise FreeTOError(f"{SINGULAR_MSG} (linear solver {solver.name}: "
                              f"{type(e).__name__}: {e})") from e
        Uf_ = np.asarray(Uf_).reshape(freedofs.size, nload)
        if not np.all(np.isfinite(Uf_)):
            raise FreeTOError(SINGULAR_MSG + " (non-finite displacements)")
        return float(np.sum(F_free * Uf_))

    eval_errors = {}

    def _optional_eval(name, fn):
        """Post-run evaluations must never destroy a finished run: a solver
        failure is logged, recorded in extra["eval_errors"] and the metric
        set to inf (the design is singular / disconnected under that model)."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            eval_errors[name] = msg
            log(f"WARNING: post-run evaluation '{name}' failed ({msg}); reported as inf")
            return math.inf

    comp_final = None
    if qubo_mode and loop > 0 and q_prev is not None:
        # one extra FE solve on the returned design, so res.comp matches res.eleden
        c_last = _optional_eval("last_design", lambda: _fe_compliance(q_prev["full"][ele]))
        n_fe += 1
        chosen, c_ch = q_prev, c_last
        if q_best is not None and q_best_c < c_last and (q_prev["at_target"]
                                                       or not q_best["at_target"]):
            chosen, c_ch = q_best, q_best_c
        full, xg, lss, top, tol = (chosen["full"], chosen["xg"], chosen["lss"],
                                   chosen["top"], chosen["tol"])
        full_pre = chosen["full_pre"]
        vxPhys_full = full.reshape((nely, nelx, nelz), order="F")
        comp_final = c_ch
        extra.update(final_compliance=c_ch, final_volfrac=float(chosen["fvol"]),
                     best_iteration=int(chosen["iter"]), last_compliance=c_last,
                     returned_design=("last" if chosen is q_prev else "best"),
                     beta_final=float(chosen.get("beta", beta)),
                     qubo_history=list(qupd.history), qubo_options=qupd.describe(),
                     binary_design=np.asarray(chosen["vx"]).copy())
        log(f"QUBO: returned design of iteration {chosen['iter']} (compliance "
            f"{c_ch:.6g}; last design {c_last:.6g})")
    if qubo_mode:
        extra.update(fe_solves=n_fe, qubo_solves=int(qupd.n_qubo_solves))
    if loop > 0:
        # beta of the Heaviside projection that produced the returned design
        # (hist["beta"] records the value *after* the per-iteration increment)
        if not qubo_mode or "beta_final" not in extra:
            extra["beta_final"] = float(beta_used)
    if c.eval_beta is not None and loop > 0:
        fb = smoothedge3D(np.array(full_pre, copy=True), Hn, Hns, nelx, nely, nelz, nele,
                          nnele, float(c.eval_beta), NGRID)[0]
        extra["compliance_at_beta"] = _optional_eval("eval_beta",
                                                     lambda: _fe_compliance(fb[ele]))
        extra["eval_beta"] = float(c.eval_beta)
        extra["volfrac_at_beta"] = float(np.sum(fb) / nnele)
    if c.eval_binary and vxPhys_full is not None:
        rho = vxPhys_full.ravel(order="F")[ele]
        xb = np.full(nnele, 0.001)          # void floor as in smoothedge3D
        order = np.argsort(-rho, kind="stable")
        xb[order[:int(round(vol * nnele))]] = 1.0
        xb[np.isin(ele, MusD)] = 1.0
        extra["binary_compliance"] = _optional_eval("eval_binary", lambda: _fe_compliance(xb))
        extra["binary_volfrac"] = float(np.mean(xb == 1.0))
    if c.eval_crisp is not None and loop > 0:
        vt = float(vol if c.eval_crisp is True else c.eval_crisp)
        crisp = {}

        def _crisp():
            rho_c, thr, v_c = crisp_projection(full_pre, Hn, Hns, nelx, nely, nelz, ele,
                                               vt, NGRID)
            crisp.update(crisp_volfrac=v_c, crisp_threshold=thr)
            return _fe_compliance(rho_c)
        extra["crisp_compliance"] = _optional_eval("eval_crisp", _crisp)
        extra.update(crisp_volfrac=crisp.get("crisp_volfrac"), crisp_target=vt,
                     crisp_threshold=crisp.get("crisp_threshold"))
        if vxPhys_full is not None:
            # compliance of the returned element densities (res.eleden) with one
            # FE solve -- for OC/MMA res.comp is the previous iterate's value
            extra["compliance_returned_design"] = _optional_eval(
                "returned_design",
                lambda: _fe_compliance(vxPhys_full.ravel(order="F")[ele]))
    if c.eval_refined is not None and loop > 0:
        # refined binary voxel evaluation of the same pre-smoothing field (v2);
        # void modulus = the modulus the FE model gives the crisp void density
        # 0.001 (SIMP: Emin + 0.001^p (E0 - Emin)), so that f = 1 reproduces the
        # package FE of the binarised element field
        vt_r = float(vol if (c.eval_crisp is None or isinstance(c.eval_crisp, bool))
                     else c.eval_crisp)
        E_void = (Emin + 0.001 ** penal * (E0 - Emin)) if simp else \
            (0.001 * E0 + (1 - 0.001) * (Emin * E0))
        ref = {}

        def _refined():
            r_ = refined_voxel_compliance(full_pre, Hn, Hns, nelx, nely, nelz, ele, vt_r,
                                          F, fixeddof, KE, E0, E_void, ngrid=NGRID,
                                          f=c.eval_refined, solver=c.solver)
            ref.update(r_)
            return r_["compliance"]
        extra["refined_compliance"] = _optional_eval("eval_refined", _refined)
        extra.update(refined_volfrac=ref.get("volfrac"), refined_threshold=ref.get("threshold"),
                     refined_time=ref.get("time"), refined_f=int(c.eval_refined),
                     refined_target=vt_r, refined_residual=ref.get("residual"),
                     refined_ndof_free=ref.get("ndof_free"))
    if full_pre is not None:
        extra["full_pre"] = np.asarray(full_pre, dtype=np.float32).copy()
    if mma_mode:
        extra["mma_constraint"] = c.mma_constraint
        if mma_fstop:
            extra["mma_fval_final"] = float(fval_new)
    if eval_errors:
        extra["eval_errors"] = dict(eval_errors)
    nit = max(loop, 1)
    timings.update({"fe_per_iter": t_fe / nit, "smoothedge_per_iter": t_se / nit,
                    "update_per_iter": t_up / nit,
                    "loop_total": time.perf_counter() - it_start_all})
    # ------------------------------------------------------------ post
    fld = FieldSnapshot(np.ascontiguousarray(matlab_to_xyz(top)), origin,
                        spacing, lss)
    for plane, direction in c.symmetry:
        fld = apply_symmetry(fld, plane, direction)
    elapsed = time.perf_counter() - t_start
    res = FreeTOResult(
        comp=hist["compliance"][-1] if hist["compliance"] else float("nan"),
        finalvol=hist["volfrac"][-1] if hist["volfrac"] else float("nan"),
        eleden=vxPhys_full, gridden=xg, elenum1=nnele, elenum2=nele,
        history=hist, iterations=loop, stopped=stopped, elapsed=elapsed,
        field=fld, ls=lss, setup_info=info, timings=timings, config=c, extra=extra)
    if comp_final is not None:
        res.comp = comp_final
        res.finalvol = extra["final_volfrac"]
    res.audit_ctx = {"nelx": nelx, "nely": nely, "nelz": nelz, "ele": ele, "MusD": MusD,
                     "fixeddof": np.intersect1d(fixeddof, n_vec), "F": F, "Hn": Hn, "Hns": Hns,
                     "full_pre": full_pre, "h": h, "origin": origin,
                     "regions": dict(dom.regions or {}), "volfrac": vol,
                     "binary_design": extra.get("binary_design"),
                     "mesh_control": c.mesh_control}
    if c.audit:
        try:
            from .audit import audit_result
            t_a = time.perf_counter()
            rep = audit_result(res, c)
            extra["audit"] = rep
            timings["audit"] = time.perf_counter() - t_a
            bad = [k for k, v in rep["checks"].items() if not v["pass"]]
            log("Physics check: " + ("PASS" if rep["ok"] else "FAIL (" + ", ".join(bad) + ")")
                + f"; {rep['n_components']} component(s), floating "
                  f"{100 * rep['floating_frac']:.1f} %, loads on solid "
                  f"{100 * rep['checks']['loads_solid']['value']:.1f} %")
        except Exception as e:  # noqa: BLE001 -- the audit never discards a run
            extra["audit_error"] = f"{type(e).__name__}: {e}"
            log(f"WARNING: physics check failed to run ({extra['audit_error']})")
    log(f"FreeTO finished: {loop} iterations, compliance {res.comp:.6g}, "
        f"volume fraction {res.finalvol:.4f}, {elapsed:.1f} s"
        + (" (stopped by user)" if stopped else ""))
    try:                       # release the PARDISO factorisation (fe.py close)
        solver.close()
    except Exception:
        pass
    return res
