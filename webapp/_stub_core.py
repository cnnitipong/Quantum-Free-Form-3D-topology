"""
Fake implementation of the `freeto` core package interface, used ONLY by the
web app while the real numerical core (freeto_py/freeto/) does not import
successfully yet. See CONTRACT.md for the interface this mirrors.

This lets the web app (upload, examples, job orchestration, live preview,
charts, downloads) be fully built and tested before the real core lands.
`webapp/core_loader.py` picks this module automatically when `import freeto`
fails, and switches to the real package the moment it becomes importable.

The "topology optimization" here is a deterministic, fast, fake: a sphere
level-set inscribed in the domain's bounding box that shrinks and wobbles
towards `volfrac` over iterations, with synthetic-but-plausible compliance /
change histories. It is good enough to exercise every code path in the web
app (setup info, per-iteration callback, stop, surfaces via marching cubes,
STL/NPZ export) with zero dependency on the real solver.
"""

from __future__ import annotations

import dataclasses
import io
import math
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from skimage import measure as _skmeasure
except Exception:  # pragma: no cover - scikit-image is a declared dependency
    _skmeasure = None


# --------------------------------------------------------------------------
# STL I/O (minimal, dependency-free; real core will likely have a more
# rigorous / welded version, this is enough for bbox/triangle-count/preview).
# --------------------------------------------------------------------------

def read_stl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read an ASCII or binary STL file.

    Returns (vertices float64 (n,3), faces int64 (m,3)). Vertices are not
    welded: each triangle gets 3 fresh vertex rows, faces are [3i,3i+1,3i+2].
    """
    with open(path, "rb") as fh:
        head = fh.read(84)
        fh.seek(0)
        data = fh.read()

    is_ascii = False
    stripped = data.lstrip()
    if stripped[:5].lower() == b"solid":
        # Could still be binary if a binary file happens to start with
        # "solid" (rare but real). Disambiguate using the binary triangle
        # count against the actual file size.
        if len(head) >= 84:
            ntri = struct.unpack("<I", head[80:84])[0]
            expected = 84 + ntri * 50
            if expected == len(data):
                is_ascii = False
            else:
                is_ascii = True
        else:
            is_ascii = True

    if is_ascii:
        return _read_stl_ascii(data)
    return _read_stl_binary(data)


def _read_stl_binary(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    ntri = struct.unpack("<I", data[80:84])[0]
    verts = np.empty((ntri * 3, 3), dtype=np.float64)
    offset = 84
    rec = struct.Struct("<12fH")
    for i in range(ntri):
        chunk = data[offset:offset + 50]
        vals = rec.unpack(chunk)
        # vals[0:3] normal, vals[3:6], vals[6:9], vals[9:12] vertices
        verts[3 * i + 0] = vals[3:6]
        verts[3 * i + 1] = vals[6:9]
        verts[3 * i + 2] = vals[9:12]
        offset += 50
    faces = np.arange(ntri * 3, dtype=np.int64).reshape(ntri, 3)
    return verts, faces


def _read_stl_ascii(data: bytes) -> Tuple[np.ndarray, np.ndarray]:
    text = data.decode("utf-8", errors="ignore")
    verts_list = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("vertex"):
            parts = s.split()[1:4]
            verts_list.append([float(x) for x in parts])
    verts = np.asarray(verts_list, dtype=np.float64)
    ntri = verts.shape[0] // 3
    verts = verts[: ntri * 3]
    faces = np.arange(ntri * 3, dtype=np.int64).reshape(ntri, 3)
    return verts, faces


def write_stl(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a binary STL file from an indexed triangle mesh."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    ntri = faces.shape[0]
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 80)
        fh.write(struct.pack("<I", ntri))
        rec = struct.Struct("<12fH")
        for tri in faces:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            n = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(n)
            if norm > 0:
                n = n / norm
            fh.write(rec.pack(*n.tolist(), *v0.tolist(), *v1.tolist(), *v2.tolist(), 0))


def stl_bbox(path: str) -> Tuple[List[float], List[float]]:
    verts, _ = read_stl(path)
    return verts.min(axis=0).tolist(), verts.max(axis=0).tolist()


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

_PLANES = {"x-y", "y-z", "z-x"}
_DIRECTIONS = {"left", "right"}
_METHODS = {"SIMP", "SEMDOT"}
_OPTIMIZERS = {"OC", "MMA"}
_SOLVERS = {"auto", "cholmod", "pardiso", "superlu", "amg"}
_LOADTYPES = {"distributed", "point"}


def _as_list(x) -> List[float]:
    if x is None:
        return [0.0]
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in x]
    return [float(x)]


@dataclass
class FreeTOConfig:
    domain: str
    forces: List[str]
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
    fmagx: Sequence[float] = field(default_factory=lambda: [0.0])
    fmagy: Sequence[float] = field(default_factory=lambda: [0.0])
    fmagz: Sequence[float] = field(default_factory=lambda: [0.0])
    loadtype: str = "distributed"
    symmetry: List[Tuple[str, str]] = field(default_factory=list)
    max_iter: int = 500
    solver: str = "auto"

    def validate(self) -> None:
        errors = []

        if not self.domain:
            errors.append("A domain STL is required.")
        elif not os.path.isfile(self.domain):
            errors.append(f"Domain STL not found: {self.domain}")

        if not self.forces or len(self.forces) == 0:
            errors.append("At least one load case (force1) is required.")
        elif len(self.forces) > 10:
            errors.append("At most 10 load cases (force1..force10) are supported.")
        else:
            for i, f in enumerate(self.forces, start=1):
                if not f:
                    errors.append(f"Load case {i} has no STL file selected.")
                elif not os.path.isfile(f):
                    errors.append(f"Load case {i} STL not found: {f}")

        support_files = [self.fixed, self.xfixed, self.yfixed, self.zfixed]
        if not any(support_files):
            errors.append(
                "Please include at least one support condition "
                "(Fixed, Fixed X, Fixed Y or Fixed Z)."
            )
        for name, p in (
            ("fixed", self.fixed),
            ("xfixed", self.xfixed),
            ("yfixed", self.yfixed),
            ("zfixed", self.zfixed),
            ("keepdom", self.keepdom),
        ):
            if p and not os.path.isfile(p):
                errors.append(f"{name} STL not found: {p}")

        fmagx = _as_list(self.fmagx)
        fmagy = _as_list(self.fmagy)
        fmagz = _as_list(self.fmagz)
        nf = len(self.forces) if self.forces else 1
        for name, lst in (("Fmagx", fmagx), ("Fmagy", fmagy), ("Fmagz", fmagz)):
            if len(lst) not in (1, nf):
                errors.append(
                    f"{name} must have 1 value (applied to all load cases) or "
                    f"{nf} values (one per load case); got {len(lst)}."
                )

        def _broadcast(lst):
            return lst if len(lst) == nf else lst * nf

        all_zero = True
        try:
            bx, by, bz = _broadcast(fmagx), _broadcast(fmagy), _broadcast(fmagz)
            for i in range(nf):
                if abs(bx[i]) > 0 or abs(by[i]) > 0 or abs(bz[i]) > 0:
                    all_zero = False
                    break
        except Exception:
            pass
        if all_zero:
            errors.append("Please include at least one non-zero load definition.")

        if not (isinstance(self.mesh_control, (int, np.integer)) and self.mesh_control >= 4):
            errors.append("MeshControl must be an integer >= 4.")

        if not (0.0 < float(self.volfrac) < 1.0):
            errors.append("volfrac must be between 0 and 1 (exclusive).")

        if float(self.youngs_modulus) <= 0:
            errors.append("Young's modulus must be positive.")

        if not (0.0 <= float(self.poisson_ratio) < 0.5):
            errors.append("Poisson ratio must be in [0, 0.5).")

        if self.method not in _METHODS:
            errors.append(f"method must be one of {sorted(_METHODS)}.")

        if self.optimizer not in _OPTIMIZERS:
            errors.append(f"optimizer must be one of {sorted(_OPTIMIZERS)}.")

        if float(self.penal) < 1.0:
            errors.append("penal must be >= 1.")

        if float(self.rmin) <= 0:
            errors.append("filter radius (rmin) must be positive.")

        if self.loadtype not in _LOADTYPES:
            errors.append(f"loadtype must be one of {sorted(_LOADTYPES)}.")

        if len(self.symmetry) > 3:
            errors.append("At most 3 symmetry planes are supported.")
        for plane, direction in self.symmetry:
            if plane not in _PLANES:
                errors.append(f"Unknown symmetry plane '{plane}' (expected one of {sorted(_PLANES)}).")
            if direction not in _DIRECTIONS:
                errors.append(f"Unknown symmetry direction '{direction}' (expected 'left' or 'right').")

        if int(self.max_iter) < 1:
            errors.append("max_iter must be >= 1.")

        if self.solver not in _SOLVERS:
            errors.append(f"solver must be one of {sorted(_SOLVERS)}.")

        if errors:
            raise ValueError(" ".join(errors))


# --------------------------------------------------------------------------
# FieldSnapshot / postprocess
# --------------------------------------------------------------------------

@dataclass
class FieldSnapshot:
    top: np.ndarray       # (nx,ny,nz) float, phi = xg - ls, solid where >0
    origin: np.ndarray    # (3,)
    spacing: np.ndarray   # (3,)


def _surface_from_field_impl(field: FieldSnapshot, smooth: bool = True):
    if _skmeasure is None:
        raise RuntimeError("scikit-image is required for surface extraction")
    vol = field.top
    lo, hi = float(vol.min()), float(vol.max())
    if not (lo < 0.0 < hi):
        # No zero crossing (e.g. fully solid or fully empty block) - synthesize
        # a tiny surface so callers get a valid (if degenerate) mesh.
        pad = np.pad(vol, 1, mode="constant", constant_values=lo - 1.0)
        vol = pad
        origin = np.asarray(field.origin) - np.asarray(field.spacing)
    else:
        origin = np.asarray(field.origin)
    sigma = 0.8 if smooth else 0.0
    if sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            vol = gaussian_filter(vol, sigma=sigma)
        except Exception:
            pass
    verts_idx, faces, _normals, _values = _skmeasure.marching_cubes(vol, level=0.0)
    spacing = np.asarray(field.spacing, dtype=np.float64)
    verts = origin + verts_idx * spacing
    return verts.astype(np.float32), faces.astype(np.int32)


class _PostprocessNamespace:
    @staticmethod
    def surface_from_field(field: FieldSnapshot, smooth: bool = True):
        return _surface_from_field_impl(field, smooth=smooth)


postprocess = _PostprocessNamespace()


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class FreeTOResult:
    comp: float
    finalvol: float
    eleden: np.ndarray
    gridden: np.ndarray
    elenum1: int
    elenum2: int
    history: dict
    iterations: int
    stopped: bool
    elapsed: float
    field: FieldSnapshot

    def surface(self, smooth: bool = True):
        return _surface_from_field_impl(self.field, smooth=smooth)

    def write_stl(self, path: str) -> None:
        verts, faces = self.surface()
        write_stl(path, verts, faces)

    def save_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            comp=self.comp,
            finalvol=self.finalvol,
            eleden=self.eleden,
            gridden=self.gridden,
            elenum1=self.elenum1,
            elenum2=self.elenum2,
            iterations=self.iterations,
            stopped=self.stopped,
            elapsed=self.elapsed,
            field_top=self.field.top,
            field_origin=self.field.origin,
            field_spacing=self.field.spacing,
            **{f"history_{k}": np.asarray(v) for k, v in self.history.items()},
        )


# --------------------------------------------------------------------------
# run_freeto (fake shrinking-sphere optimizer)
# --------------------------------------------------------------------------

def _domain_bbox(domain_path: str) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = stl_bbox(domain_path)
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def _grid_dims(bbox_lo: np.ndarray, bbox_hi: np.ndarray, mesh_control: int):
    extents = np.maximum(bbox_hi - bbox_lo, 1e-9)
    longest = float(extents.max())
    h = longest / max(int(mesh_control), 1)
    dims = np.maximum(np.round(extents / h).astype(int), 4)
    return dims, h


def run_freeto(
    cfg: "FreeTOConfig",
    callback: Optional[Callable[[dict], None]] = None,
    stop_event=None,
    log: Optional[Callable[[str], None]] = None,
) -> FreeTOResult:
    cfg.validate()
    t_start = time.time()
    log = log or (lambda s: None)

    bbox_lo, bbox_hi = _domain_bbox(cfg.domain)
    center = (bbox_lo + bbox_hi) / 2.0
    (nx, ny, nz), h = _grid_dims(bbox_lo, bbox_hi, cfg.mesh_control)
    # grid of sample points at cell centers, in (x,y,z) order
    nelx, nely, nelz = int(nx), int(ny), int(nz)
    nele = nelx * nely * nelz
    nnele = (nelx + 1) * (nely + 1) * (nelz + 1)
    ndof = 3 * nnele
    n_support = 1 if any([cfg.fixed, cfg.xfixed, cfg.yfixed, cfg.zfixed]) else 0
    nfree = max(ndof - 300 * n_support, ndof // 2)
    nloads = len(cfg.forces)

    solver = cfg.solver
    if solver == "auto":
        solver = "superlu"

    setup_info = {
        "stage": "setup",
        "nelx": nelx,
        "nely": nely,
        "nelz": nelz,
        "nele": nele,
        "nnele": nnele,
        "ndof": ndof,
        "nfree": nfree,
        "nloads": nloads,
        "h": float(h),
        "solver": solver,
        "setup_time": time.time() - t_start,
    }
    if callback:
        callback(dict(setup_info))
    log(
        f"[stub core] mesh {nelx}x{nely}x{nelz} ({nele} elements), "
        f"{nloads} load case(s), solver={solver}"
    )

    max_dim = float(np.max(bbox_hi - bbox_lo))
    r0 = 0.42 * max_dim
    r_target = r0 * math.sqrt(max(cfg.volfrac, 1e-3))

    grid_shape = (nelx + 1, nely + 1, nelz + 1)
    xs = bbox_lo[0] + np.arange(grid_shape[0]) * h
    ys = bbox_lo[1] + np.arange(grid_shape[1]) * h
    zs = bbox_lo[2] + np.arange(grid_shape[2]) * h
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)

    history = {"compliance": [], "volfrac": [], "change": [], "topo": [], "beta": []}
    rng = np.random.default_rng(1234)

    comp = 100.0
    change = 1.0
    last_field = None
    it_done = 0
    stopped = False

    n_report_iters = max(1, min(int(cfg.max_iter), 60))
    for it in range(1, int(cfg.max_iter) + 1):
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break

        t_iter = time.time()
        frac = it / n_report_iters
        frac = min(frac, 1.0)
        r = r0 + (r_target - r0) * (1 - math.exp(-3.0 * frac))
        wobble = 0.01 * r0 * math.sin(it * 0.7)
        top = (r + wobble) - dist  # phi = xg - ls -> solid where top>0

        vf = float((top > 0).mean())
        comp = 50.0 + 450.0 * math.exp(-0.12 * it) + rng.normal(0, 0.5)
        comp = max(comp, 1.0)
        change = max(0.5 * math.exp(-0.15 * it), 1e-4) + abs(rng.normal(0, 1e-4))
        beta = min(1.0 + 0.1 * it, 8.0) if cfg.method == "SEMDOT" else 1.0

        field = FieldSnapshot(top=top, origin=bbox_lo.copy(), spacing=np.array([h, h, h]))
        last_field = field
        it_done = it

        history["compliance"].append(comp)
        history["volfrac"].append(vf)
        history["change"].append(float(change))
        history["topo"].append(float(vf))
        history["beta"].append(float(beta))

        iter_time = time.time() - t_iter
        if callback:
            callback(
                {
                    "stage": "iter",
                    "iter": it,
                    "compliance": comp,
                    "volfrac": vf,
                    "change": float(change),
                    "topo": float(vf),
                    "beta": float(beta),
                    "elapsed": time.time() - t_start,
                    "iter_time": iter_time,
                    "field": field,
                }
            )
        log(
            f"it.:{it:4d}  obj.:{comp:10.4f}  vol.:{vf:6.4f}  ch.:{change:8.5f}"
        )

        if change < 1e-3 and it > 5:
            time.sleep(0.005)
            break
        time.sleep(0.01)

    if last_field is None:
        # stopped before first iteration; still produce a valid field
        top = r0 - dist
        last_field = FieldSnapshot(top=top, origin=bbox_lo.copy(), spacing=np.array([h, h, h]))

    final_field = _apply_symmetry_stub(last_field, cfg.symmetry)

    eleden = np.zeros((nely, nelx, nelz), dtype=np.float64)
    cx = (np.arange(nelx) + 0.5) * h + bbox_lo[0]
    cy = (np.arange(nely) + 0.5) * h + bbox_lo[1]
    cz = (np.arange(nelz) + 0.5) * h + bbox_lo[2]
    CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing="xy")
    cdist = np.sqrt((CX - center[0]) ** 2 + (CY - center[1]) ** 2 + (CZ - center[2]) ** 2)
    eleden[:] = np.clip((r_target - cdist) / h + 0.5, 0.0, 1.0)

    result = FreeTOResult(
        comp=float(history["compliance"][-1]) if history["compliance"] else comp,
        finalvol=float(history["volfrac"][-1]) if history["volfrac"] else float(cfg.volfrac),
        eleden=eleden,
        gridden=last_field.top,
        elenum1=int((eleden > 0.5).sum()),
        elenum2=int(eleden.size),
        history=history,
        iterations=it_done,
        stopped=stopped,
        elapsed=time.time() - t_start,
        field=final_field,
    )
    return result


def _apply_symmetry_stub(field: FieldSnapshot, symmetry) -> FieldSnapshot:
    """Very small stand-in for symmetry.m: mirrors the level-set field about
    the domain-center plane(s) requested, purely for visual plausibility."""
    if not symmetry:
        return field
    top = field.top.copy()
    axis_for_plane = {"y-z": 0, "z-x": 1, "x-y": 2}
    for plane, _direction in symmetry:
        ax = axis_for_plane.get(plane)
        if ax is None:
            continue
        n = top.shape[ax]
        mid = n // 2
        idx_hi = [slice(None)] * top.ndim
        idx_hi[ax] = slice(mid, n)
        idx_lo = [slice(None)] * top.ndim
        idx_lo[ax] = slice(0, n - mid)
        half = np.take(top, range(mid, n), axis=ax)
        mirrored = np.flip(half, axis=ax)
        take = min(mirrored.shape[ax], mid)
        src = [slice(None)] * top.ndim
        src[ax] = slice(mirrored.shape[ax] - take, mirrored.shape[ax])
        dst = [slice(None)] * top.ndim
        dst[ax] = slice(0, take)
        top[tuple(dst)] = np.take(mirrored, range(mirrored.shape[ax] - take, mirrored.shape[ax]), axis=ax)
    return FieldSnapshot(top=top, origin=field.origin, spacing=field.spacing)


# --------------------------------------------------------------------------
# EXAMPLES
# --------------------------------------------------------------------------

EXAMPLES = {
    "GE_bracket": {
        "title": "GE bracket",
        "description": "Optimize the GE bracket (the airplane bearing bracket of the FreeTO examples) under two load cases.",
        "config_kwargs": dict(
            domain="GE_domain.STL",
            forces=["GE_force.STL", "GE_force.STL"],
            mesh_control=80,
            volfrac=0.3,
            fixed="GE_fixed.STL",
            fmagx=[0.0, 0.0],
            fmagy=[0.0, -2000.0],
            fmagz=[1500.0, 0.0],
            youngs_modulus=210e9,
            poisson_ratio=0.3,
            method="SIMP",
        ),
        "files": {"domain": "GE_domain.STL", "force1": "GE_force.STL", "force2": "GE_force.STL", "fixed": "GE_fixed.STL"},
    },
    "air_bracket": {
        "title": "Airplane bracket",
        "description": "Optimize one half of an airplane bracket with symmetry and SEMDOT.",
        "config_kwargs": dict(
            domain="air_domain.STL",
            forces=["air_force.STL", "air_force.STL", "air_force.STL"],
            mesh_control=90,
            volfrac=0.2,
            fixed="air_fixed.STL",
            zfixed="air_zfixed.STL",
            fmagx=[1000.0, 1324.0, 0.0],
            fmagy=[0.0, -1324.0, -2500.0],
            fmagz=[0.0, 0.0, 0.0],
            youngs_modulus=210e9,
            poisson_ratio=0.3,
            method="SEMDOT",
            symmetry=[("x-y", "right")],
        ),
        "files": {
            "domain": "air_domain.STL",
            "force1": "air_force.STL",
            "force2": "air_force.STL",
            "force3": "air_force.STL",
            "fixed": "air_fixed.STL",
            "zfixed": "air_zfixed.STL",
        },
    },
    "hand": {
        "title": "Human hand",
        "description": "Optimize a human hand model with loads on five fingertips.",
        "config_kwargs": dict(
            domain="hand_domain.stl",
            forces=[
                "hand_force1.stl",
                "hand_force2.stl",
                "hand_force3.stl",
                "hand_force4.stl",
                "hand_force5.stl",
            ],
            mesh_control=90,
            volfrac=0.3,
            fixed="hand_fixed.stl",
            fmagx=[0.0],
            fmagy=[0.0],
            fmagz=[2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
            youngs_modulus=210e9,
            poisson_ratio=0.3,
            method="SIMP",
        ),
        "files": {
            "domain": "hand_domain.stl",
            "force1": "hand_force1.stl",
            "force2": "hand_force2.stl",
            "force3": "hand_force3.stl",
            "force4": "hand_force4.stl",
            "force5": "hand_force5.stl",
            "fixed": "hand_fixed.stl",
        },
    },
    "quadcopter": {
        "title": "Quadcopter arm (quarter)",
        "description": "Optimize a quarter of a quadcopter frame with two symmetry planes.",
        "config_kwargs": dict(
            domain="quad_domain.STL",
            forces=["quad_force1.STL", "quad_force2.STL", "quad_force3.STL"],
            mesh_control=70,
            volfrac=0.3,
            fixed="quad_fixed.STL",
            xfixed="quad_xfixed.STL",
            yfixed="quad_yfixed.STL",
            fmagx=[0.0],
            fmagy=[0.0],
            fmagz=[-1500.0, -1500.0, -1000.0],
            youngs_modulus=2e9,
            poisson_ratio=0.3,
            method="SIMP",
            symmetry=[("y-z", "right"), ("z-x", "left")],
            keep_bcz=True,
        ),
        "files": {
            "domain": "quad_domain.STL",
            "force1": "quad_force1.STL",
            "force2": "quad_force2.STL",
            "force3": "quad_force3.STL",
            "fixed": "quad_fixed.STL",
            "xfixed": "quad_xfixed.STL",
            "yfixed": "quad_yfixed.STL",
        },
    },
}
