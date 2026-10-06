"""FastAPI backend for the FreeTO-Python web app.

Run with:
    python -m webapp.server --port 8000 --open

Endpoints are documented inline below; everything is JSON except STL/NPZ
downloads and the static frontend.
"""

from __future__ import annotations

import argparse
import io
import logging
import json
import mimetypes
import os
import re
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import numpy as np

from webapp.core_loader import (
    CORE_SUPPORTS_QUBO,
    _config_has_field,
    CORE_USABLE,
    EXAMPLES,
    FreeTOConfig,
    TRUSS_METHODS,
    TRUSS_USABLE,
    STUDY_USABLE,
    audit_status,
    audit_unavailable_message,
    core_status,
    get_benchmark,
    quantum_backends_payload,
    quantum_status,
    read_stl,
    run_study,
    study_status,
    study_unavailable_message,
    suite_spec,
    truss_benchmarks_payload,
    truss_status,
    truss_unavailable_message,
    unavailable_message,
)
from webapp.jobs import JobManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("freeto.webapp.server")

# `mimetypes` seeds its table from the OS registry (`mimetypes.init()` calls
# `db.read_windows_registry()` on Windows), and registry entries *override*
# the Python-shipped defaults. A stray `HKEY_CLASSES_ROOT\.js` left behind by
# some other installer (a well-known Windows footgun) can make `.js` resolve
# to "text/plain", which browsers reject for ES module scripts / importmap
# targets — the app loads but silently does nothing. Force the types this
# app actually serves, unconditionally, so Starlette's StaticFiles/FileResponse
# (which call `mimetypes.guess_type`) always get a correct value regardless of
# the host's registry/mime.types state.
for _ext, _type in {
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".wasm": "application/wasm",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".html": "text/html",
    ".stl": "model/stl",
}.items():
    mimetypes.add_type(_type, _ext)

WEBAPP_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = WEBAPP_DIR.parent
STATIC_DIR = WEBAPP_DIR / "static"
EXAMPLES_STL_DIR = PACKAGE_ROOT / "examples" / "STLs"

DEFAULT_WORKDIR = Path(os.environ.get("FREETO_WEBAPP_DIR", "~/.freeto_web")).expanduser()


# ---------------------------------------------------------------------------
# In-memory file registry (uploads + registered example files)
# ---------------------------------------------------------------------------

class FileRecord:
    __slots__ = ("id", "name", "path", "triangles", "bbox_min", "bbox_max", "source", "watertight")

    def __init__(self, id_, name, path, triangles, bbox_min, bbox_max, source, watertight=True):
        self.id = id_
        self.name = name
        self.path = path
        self.triangles = triangles
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.source = source
        self.watertight = watertight

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "triangles": self.triangles,
            "bbox": {"min": self.bbox_min, "max": self.bbox_max},
            "source": self.source,
            "watertight": self.watertight,
        }


def _is_watertight(faces: np.ndarray) -> bool:
    """Cheap manifoldness check: a closed (watertight) surface has every
    undirected edge shared by exactly two triangles. Vectorised with numpy;
    O(n log n) in triangle count, no external mesh library needed."""
    if faces.shape[0] == 0:
        return True
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_filename(name: str) -> str:
    name = os.path.basename(name or "file.stl")
    name = _SAFE_NAME_RE.sub("_", name)
    return name or "file.stl"


class FileRegistry:
    def __init__(self, uploads_dir: Path):
        # NB: directory creation is lazy (on first actual upload), not here —
        # constructing a FileRegistry (e.g. at app-factory time) must not
        # touch the filesystem, so an unused --workdir is never created.
        self.uploads_dir = uploads_dir
        self._records: Dict[str, FileRecord] = {}
        self._lock = threading.Lock()

    def register_from_bytes(self, name: str, data: bytes) -> FileRecord:
        file_id = uuid.uuid4().hex[:12]
        safe = _safe_filename(name)
        # Parse (and validate) BEFORE touching disk, so a bad upload never
        # creates a stray file and always surfaces as a clear error to the
        # caller (which converts it to an HTTP 400).
        rec = self._build_record(file_id, name, data_or_path=data, source="upload")
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        dest = self.uploads_dir / f"{file_id}_{safe}"
        dest.write_bytes(data)
        rec.path = dest
        with self._lock:
            self._records[file_id] = rec
        return rec

    def register_example_file(self, filename: str) -> FileRecord:
        file_id = f"ex_{filename}"
        with self._lock:
            existing = self._records.get(file_id)
        if existing is not None:
            return existing
        path = EXAMPLES_STL_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Example STL not found: {path}")
        rec = self._build_record(file_id, filename, data_or_path=path, source="example")
        with self._lock:
            self._records[file_id] = rec
        return rec

    def _build_record(self, file_id: str, name: str, data_or_path, source: str) -> FileRecord:
        """Parse an STL (from raw bytes, for an upload not yet written to
        disk, or from an existing path, for a bundled example) and build its
        FileRecord. Raises ValueError with a user-facing message for an
        unparsable, empty, or degenerate STL — the caller decides whether
        that becomes an HTTP 400 (uploads) or a server-side error (examples,
        which should never actually be broken)."""
        if isinstance(data_or_path, (bytes, bytearray)):
            import tempfile
            # Use a TemporaryDirectory + a plain closed file, not
            # NamedTemporaryFile(delete=True): on Windows that opens the file
            # with O_TEMPORARY, and re-opening it by name (what read_stl does)
            # raises PermissionError there, even though it works fine on
            # POSIX. Writing to a normal file inside a temp dir and closing it
            # before parsing works identically on Windows, macOS and Linux.
            with tempfile.TemporaryDirectory(prefix="freeto_upload_") as td:
                tmp_path = os.path.join(td, "upload.stl")
                with open(tmp_path, "wb") as fh:
                    fh.write(data_or_path)
                verts, faces = self._parse_stl(tmp_path, name)
            path_placeholder = None
        else:
            path_placeholder = Path(data_or_path)
            verts, faces = self._parse_stl(str(path_placeholder), name)

        triangles = int(faces.shape[0])
        if triangles == 0:
            raise ValueError(f"'{name}' contains no triangles (empty STL).")
        bbox_min = verts.min(axis=0)
        bbox_max = verts.max(axis=0)
        extents = bbox_max - bbox_min
        if not np.all(np.isfinite(extents)) or np.any(extents <= 0):
            raise ValueError(
                f"'{name}' is flat or degenerate (bounding-box extent is zero along at "
                f"least one axis: {extents.tolist()}); this cannot be a valid domain/region STL."
            )
        watertight = _is_watertight(faces)
        if not watertight:
            logger.warning("STL '%s' is not watertight (surface is not closed).", name)
        return FileRecord(
            file_id, name, path_placeholder, triangles,
            bbox_min.tolist(), bbox_max.tolist(), source, watertight=watertight,
        )

    @staticmethod
    def _parse_stl(path: str, name: str):
        try:
            return read_stl(path)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"'{name}' could not be read as an STL file: {exc}") from exc

    def get(self, file_id: str) -> Optional[FileRecord]:
        with self._lock:
            return self._records.get(file_id)

    def resolve_path(self, file_id: Optional[str]) -> Optional[str]:
        if not file_id:
            return None
        rec = self.get(file_id)
        if rec is None:
            raise HTTPException(400, f"Unknown file id: {file_id}")
        return str(rec.path)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class LoadCaseIn(BaseModel):
    file_id: str
    fx: float = 0.0
    fy: float = 0.0
    fz: float = 0.0


class SymmetryIn(BaseModel):
    plane: str
    direction: str = "right"


class JobCreateRequest(BaseModel):
    label: Optional[str] = None
    domain_id: Optional[str] = None
    loads: List[LoadCaseIn] = Field(default_factory=list, max_length=10)
    fixed_id: Optional[str] = None
    xfixed_id: Optional[str] = None
    yfixed_id: Optional[str] = None
    zfixed_id: Optional[str] = None
    keepdom_id: Optional[str] = None
    mesh_control: int = 80
    volfrac: float = 0.3
    youngs_modulus: float = 1.0
    poisson_ratio: float = 0.3
    method: str = "SIMP"
    optimizer: str = "OC"
    penal: float = 3.0
    rmin: float = 1.5
    loadtype: str = "distributed"
    keep_bc: bool = True
    keep_bcx: bool = False
    keep_bcy: bool = False
    keep_bcz: bool = False
    symmetry: List[SymmetryIn] = Field(default_factory=list)
    max_iter: int = 500
    solver: str = "auto"
    # -- QUBO / quantum settings (optimizer="QUBO" only; see docs/QUANTUM_API.md
    # QUBOOptions). All optional — unset fields keep freeto.quantum's own
    # defaults. Ignored (and harmless) for optimizer="OC"/"MMA".
    qubo_backend: Optional[str] = None
    qubo_hessian: Optional[str] = None
    qubo_volume: Optional[str] = None
    qubo_lambda_q: Optional[float] = None
    qubo_gamma: Optional[float] = None
    qubo_move_penalty: Optional[float] = None
    qubo_frontier_fraction: Optional[float] = None
    qubo_block_size: Optional[int] = None
    qubo_blocks: Optional[str] = None
    qubo_sweeps: Optional[int] = None
    qubo_init: Optional[str] = None
    qubo_er: Optional[float] = None
    qubo_n_warm: Optional[int] = None
    qubo_patience: Optional[int] = None
    qubo_num_reads: Optional[int] = None
    qubo_seed: Optional[int] = None
    qubo_qaoa_p: Optional[int] = None
    qubo_qaoa_shots: Optional[int] = None
    qubo_qaoa_init: Optional[str] = None
    qubo_time_limit: Optional[float] = None
    qubo_verify_exact: Optional[bool] = None
    # Added with the QUBO fix round (docs/QUANTUM_API.md change list):
    qubo_interp: Optional[str] = None
    qubo_qaoa_polish: Optional[bool] = None
    qubo_move_limit: Optional[float] = None
    qubo_move_limit_min: Optional[float] = None
    qubo_protect_loads: Optional[bool] = None
    qubo_guard: Optional[bool] = None
    qubo_guard_tol: Optional[float] = None
    qubo_guard_tol_target: Optional[float] = None
    qubo_max_rejects: Optional[int] = None
    eval_binary: bool = False
    # Common crisp evaluation (FreeTOConfig.eval_crisp = True -> at the target
    # volfrac) for ANY optimizer; adds extra["crisp_compliance"] etc.
    eval_crisp: bool = False
    # Physics check (docs/AUDIT_API.md): run freeto.audit on the finished design.
    audit: bool = True


class TrussRunRequest(BaseModel):
    label: Optional[str] = None
    benchmark_id: str
    method: str = "exact"
    backend: Optional[str] = None
    seed: Optional[int] = None
    options: Dict[str, Any] = Field(default_factory=dict)


class StudyCustomIn(BaseModel):
    """Custom continuum study matrix: problems x methods x seeds, optional mesh override.
    Method names: see GET /api/study/custom_options."""
    problems: List[str] = Field(min_length=1)
    methods: List[str] = Field(min_length=1)
    seeds: List[int] = Field(default_factory=lambda: [0])
    mesh_control: Optional[int] = Field(default=None, ge=4)
    max_iter: Optional[int] = Field(default=None, ge=1)
    volfrac: Optional[float] = Field(default=None, gt=0, lt=1)
    include_baseline: bool = True


class StudyRunRequest(BaseModel):
    label: Optional[str] = None
    suite: Optional[str] = None
    spec: Optional[Dict[str, Any]] = None
    custom: Optional[StudyCustomIn] = None
    figures: bool = True


# Methods offered by the "custom" study builder (label -> run fields). They mirror the
# rows of the built-in quick suite (freeto/study.py), so a custom study is comparable.
CUSTOM_METHODS: Dict[str, Dict[str, Any]] = {
    "MMA": {"optimizer": "MMA", "role": "baseline"},
    "OC": {"optimizer": "OC", "label": "OC (FreeTO default, flagged)", "role": "flagged"},
    "BESO-sort": {"optimizer": "QUBO", "options": {"hessian": "none"}, "role": "control", "single_seed": True},
    "QUBO-sa (diag)": {"optimizer": "QUBO", "options": {"backend": "sa", "hessian": "diag"}},
    "QUBO-sa (block)": {"optimizer": "QUBO", "options": {"backend": "sa", "hessian": "block"}},
    "QUBO-tabu (block)": {"optimizer": "QUBO", "options": {"backend": "tabu", "hessian": "block"}},
    "QUBO-qaoa kb8 p1 penalty (+greedy)": {
        "optimizer": "QUBO",
        "options": {"backend": "qaoa", "hessian": "block", "block_size": 8, "qaoa_p": 1,
                    "qaoa_shots": 300, "volume": "penalty", "sweeps": 1, "verify_exact": True}},
}
# default mesh_control per continuum problem: the quick suite's sizes, read from
# freeto.study (QUICK_MAIN / QUICK_ROBUST) so they follow the core; fallback if absent.
def _quick_meshes() -> Dict[str, int]:
    try:
        from freeto import study as _st
        return {pb: int(mc) for pb, mc in tuple(_st.QUICK_MAIN) + tuple(_st.QUICK_ROBUST)}
    except Exception:  # noqa: BLE001
        return {"cantilever_beam": 36, "mbb_beam": 46, "bridge_deck": 41, "l_bracket": 30,
                "GE_bracket": 24}


CUSTOM_PROBLEMS: Dict[str, int] = _quick_meshes()


def build_custom_study_spec(c: "StudyCustomIn") -> Dict[str, Any]:
    """problems x methods x seeds -> a freeto.study spec dict (raises ValueError)."""
    unknown_m = [m for m in c.methods if m not in CUSTOM_METHODS]
    if unknown_m:
        raise ValueError(f"unknown method(s) {unknown_m}; choose from {list(CUSTOM_METHODS)}")
    runs: List[Dict[str, Any]] = []
    methods = list(c.methods)
    if c.include_baseline and "MMA" not in methods:
        methods.insert(0, "MMA")  # gaps are relative to MMA's crisp compliance
    for pb in c.problems:
        mc = c.mesh_control or CUSTOM_PROBLEMS.get(pb)
        if mc is None:
            raise ValueError(f"unknown problem {pb!r}; choose from {list(CUSTOM_PROBLEMS)} "
                             f"or give mesh_control")
        for m in methods:
            d = CUSTOM_METHODS[m]
            run = {"study": "custom", "kind": "continuum", "problem": pb, "mesh_control": int(mc),
                   "optimizer": d["optimizer"], "label": d.get("label", m),
                   "max_iter": int(c.max_iter or 100),
                   "seeds": [0] if (d["optimizer"] != "QUBO" or d.get("single_seed")) else list(c.seeds or [0]),
                   "role": d.get("role", "method")}
            if d.get("options") is not None:
                run["options"] = dict(d["options"])
            if c.volfrac is not None:
                run["volfrac"] = float(c.volfrac)
            runs.append(run)
    return {"name": "custom", "runs": runs}


_STUDY_ID_RE = re.compile(r"^[A-Za-z0-9_.~-]{1,128}$")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(workdir: Optional[Path] = None) -> FastAPI:
    # NB: no filesystem access here — directories under `workdir` (including
    # the default ~/.freeto_web) are created lazily on first actual upload
    # or job, so constructing an (unused) app/import never creates them.
    workdir = Path(workdir or DEFAULT_WORKDIR).expanduser()
    uploads_dir = workdir / "uploads"
    jobs_dir = workdir / "jobs"

    registry = FileRegistry(uploads_dir)
    manager = JobManager(jobs_dir)

    app = FastAPI(title="FreeTO-Python", version="0.1.0")
    app.state.registry = registry
    app.state.manager = manager
    app.state.workdir = workdir

    # -- health / meta ---------------------------------------------------

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "core": core_status(),
            "quantum": {**quantum_status(), "continuum_qubo_supported": CORE_SUPPORTS_QUBO},
            "truss": truss_status(),
            "study": study_status(),
            "audit": audit_status(),
            "workdir": str(workdir),
        }

    # -- uploads -----------------------------------------------------------

    @app.post("/api/upload")
    async def upload(files: List[UploadFile] = File(...)):
        out = []
        for f in files:
            data = await f.read()
            if not data:
                raise HTTPException(400, f"File '{f.filename}' is empty.")
            try:
                rec = registry.register_from_bytes(f.filename or "file.stl", data)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, str(exc))
            out.append(rec.to_dict())
        return out

    @app.get("/api/files/{file_id:path}")
    def get_file(file_id: str):
        # :path (not a plain segment) because example file ids embed the STL's
        # path under examples/STLs/ (e.g. "ex_generated/cantilever_domain.stl"
        # for the six added continuum examples, which live in a subdirectory
        # — see freeto/examples.py) and would otherwise 404 on the first "/".
        rec = registry.get(file_id)
        if rec is None:
            raise HTTPException(404, "Unknown file id")
        return FileResponse(rec.path, media_type="model/stl", filename=rec.name)

    # -- examples ------------------------------------------------------

    @app.get("/api/examples")
    def list_examples():
        return [
            {
                "name": name,
                "title": ex["title"],
                "description": ex["description"],
                # "paper" | "beam" | "truss-like" | "advanced" (freeto/examples.py);
                # tolerate an older core without the key.
                "category": ex.get("category", "paper"),
            }
            for name, ex in EXAMPLES.items()
        ]

    @app.post("/api/examples/{name}/load")
    def load_example(name: str):
        ex = EXAMPLES.get(name)
        if ex is None:
            if not CORE_USABLE:
                raise HTTPException(503, unavailable_message())
            raise HTTPException(404, f"Unknown example '{name}'")
        kwargs = ex["config_kwargs"]

        def reg(filename: Optional[str]):
            if not filename:
                return None
            rec = registry.register_example_file(filename)
            return rec.to_dict()

        domain_rec = reg(kwargs["domain"])
        force_recs = [reg(f) for f in kwargs.get("forces", [])]
        fixed_rec = reg(kwargs.get("fixed"))
        xfixed_rec = reg(kwargs.get("xfixed"))
        yfixed_rec = reg(kwargs.get("yfixed"))
        zfixed_rec = reg(kwargs.get("zfixed"))
        keepdom_rec = reg(kwargs.get("keepdom"))

        def broadcast(vals, n):
            vals = list(vals) if vals is not None else [0.0]
            if len(vals) == 1 and n > 1:
                return vals * n
            return vals

        n = len(force_recs)
        fx = broadcast(kwargs.get("fmagx", [0.0]), n)
        fy = broadcast(kwargs.get("fmagy", [0.0]), n)
        fz = broadcast(kwargs.get("fmagz", [0.0]), n)

        loads = [
            {"file_id": force_recs[i]["id"], "fx": fx[i], "fy": fy[i], "fz": fz[i]}
            for i in range(n)
        ]

        prefill = {
            "label": ex["title"],
            "domain_id": domain_rec["id"] if domain_rec else None,
            "loads": loads,
            "fixed_id": fixed_rec["id"] if fixed_rec else None,
            "xfixed_id": xfixed_rec["id"] if xfixed_rec else None,
            "yfixed_id": yfixed_rec["id"] if yfixed_rec else None,
            "zfixed_id": zfixed_rec["id"] if zfixed_rec else None,
            "keepdom_id": keepdom_rec["id"] if keepdom_rec else None,
            "mesh_control": kwargs.get("mesh_control", 80),
            "volfrac": kwargs.get("volfrac", 0.3),
            "youngs_modulus": kwargs.get("youngs_modulus", 1.0),
            "poisson_ratio": kwargs.get("poisson_ratio", 0.3),
            "method": kwargs.get("method", "SIMP"),
            "optimizer": kwargs.get("optimizer", "OC"),
            "penal": kwargs.get("penal", 3.0),
            "rmin": kwargs.get("rmin", 1.5),
            "loadtype": kwargs.get("loadtype", "distributed"),
            "keep_bc": kwargs.get("keep_bc", True),
            "keep_bcx": kwargs.get("keep_bcx", False),
            "keep_bcy": kwargs.get("keep_bcy", False),
            "keep_bcz": kwargs.get("keep_bcz", False),
            "symmetry": [
                {"plane": p, "direction": d} for (p, d) in kwargs.get("symmetry", [])
            ],
            "max_iter": kwargs.get("max_iter", 200),
            "solver": kwargs.get("solver", "auto"),
        }
        files_display = {
            "domain": domain_rec,
            **{f"force{i+1}": force_recs[i] for i in range(n)},
            "fixed": fixed_rec,
            "xfixed": xfixed_rec,
            "yfixed": yfixed_rec,
            "zfixed": zfixed_rec,
            "keepdom": keepdom_rec,
        }
        return {"prefill": prefill, "files": files_display}

    # -- jobs -------------------------------------------------------------

    def _build_config(req: JobCreateRequest) -> FreeTOConfig:
        domain_path = registry.resolve_path(req.domain_id)
        if domain_path is None:
            raise HTTPException(400, "A domain STL is required.")
        force_paths = []
        fmagx, fmagy, fmagz = [], [], []
        for lc in req.loads:
            p = registry.resolve_path(lc.file_id)
            if p is None:
                raise HTTPException(400, "Each load case needs an STL file.")
            force_paths.append(p)
            fmagx.append(lc.fx)
            fmagy.append(lc.fy)
            fmagz.append(lc.fz)

        extra_kwargs: Dict[str, Any] = {}
        if req.optimizer.strip().upper() == "QUBO":
            if not CORE_SUPPORTS_QUBO:
                raise HTTPException(
                    400,
                    "This installed 'freeto' core does not support optimizer=\"QUBO\" yet "
                    "(FreeTOConfig has no 'qubo' field) — use OC or MMA, or update freeto/.",
                )
            qubo_field_map = {
                "backend": req.qubo_backend,
                "hessian": req.qubo_hessian,
                "volume": req.qubo_volume,
                "lambda_q": req.qubo_lambda_q,
                "gamma": req.qubo_gamma,
                "move_penalty": req.qubo_move_penalty,
                "frontier_fraction": req.qubo_frontier_fraction,
                "block_size": req.qubo_block_size,
                "blocks": req.qubo_blocks,
                "sweeps": req.qubo_sweeps,
                "init": req.qubo_init,
                "er": req.qubo_er,
                "n_warm": req.qubo_n_warm,
                "patience": req.qubo_patience,
                "num_reads": req.qubo_num_reads,
                "seed": req.qubo_seed,
                "qaoa_p": req.qubo_qaoa_p,
                "qaoa_shots": req.qubo_qaoa_shots,
                "qaoa_init": req.qubo_qaoa_init,
                "time_limit": req.qubo_time_limit,
                "verify_exact": req.qubo_verify_exact,
                "interp": req.qubo_interp,
                "qaoa_polish": req.qubo_qaoa_polish,
                "move_limit": req.qubo_move_limit,
                "move_limit_min": req.qubo_move_limit_min,
                "protect_loads": req.qubo_protect_loads,
                "guard": req.qubo_guard,
                "guard_tol": req.qubo_guard_tol,
                "guard_tol_target": req.qubo_guard_tol_target,
                "max_rejects": req.qubo_max_rejects,
            }
            extra_kwargs["qubo"] = {k: v for k, v in qubo_field_map.items() if v is not None}
            extra_kwargs["eval_binary"] = req.eval_binary
        if req.eval_crisp and _config_has_field(FreeTOConfig, "eval_crisp"):
            extra_kwargs["eval_crisp"] = True
        if _config_has_field(FreeTOConfig, "audit"):
            extra_kwargs["audit"] = bool(req.audit)

        cfg = FreeTOConfig(
            domain=domain_path,
            forces=force_paths,
            mesh_control=req.mesh_control,
            volfrac=req.volfrac,
            fixed=registry.resolve_path(req.fixed_id),
            xfixed=registry.resolve_path(req.xfixed_id),
            yfixed=registry.resolve_path(req.yfixed_id),
            zfixed=registry.resolve_path(req.zfixed_id),
            keepdom=registry.resolve_path(req.keepdom_id),
            keep_bc=req.keep_bc,
            keep_bcx=req.keep_bcx,
            keep_bcy=req.keep_bcy,
            keep_bcz=req.keep_bcz,
            youngs_modulus=req.youngs_modulus,
            poisson_ratio=req.poisson_ratio,
            method=req.method,
            optimizer=req.optimizer,
            penal=req.penal,
            rmin=req.rmin,
            fmagx=fmagx,
            fmagy=fmagy,
            fmagz=fmagz,
            loadtype=req.loadtype,
            symmetry=[(s.plane, s.direction) for s in req.symmetry],
            max_iter=req.max_iter,
            solver=req.solver,
            **extra_kwargs,
        )
        return cfg

    @app.post("/api/jobs")
    def create_job(req: JobCreateRequest):
        if not CORE_USABLE:
            raise HTTPException(503, unavailable_message())
        cfg = _build_config(req)
        try:
            cfg.validate()
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"Invalid configuration: {exc}")

        label = req.label or "Untitled job"
        job = manager.submit(label, req.model_dump(), cfg, audit=req.audit)
        return {
            "job_id": job.id,
            "status": job.status,
            "queue_position": manager.queue_position(job.id),
        }

    @app.get("/api/jobs")
    def list_jobs():
        return manager.list_jobs()

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, since: int = 0):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job id")
        status = job.snapshot_status(log_since=since)
        status["queue_position"] = manager.queue_position(job_id)
        return status

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str):
        if not manager.stop(job_id):
            raise HTTPException(404, "Unknown job id")
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/preview.stl")
    def preview(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job id")
        data = manager.get_preview_stl(job_id)
        if data is None:
            raise HTTPException(404, "No preview available yet")
        return Response(content=data, media_type="model/stl")

    @app.get("/api/jobs/{job_id}/result.stl")
    def result_stl(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job id")
        path = jobs_dir / job_id / "result.stl"
        if not path.is_file():
            raise HTTPException(404, "Result not ready")
        fname = f"{_safe_filename(job.label)}_{job_id}.stl"
        return FileResponse(path, media_type="model/stl", filename=fname)

    @app.get("/api/jobs/{job_id}/result.npz")
    def result_npz(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job id")
        path = jobs_dir / job_id / "result.npz"
        if not path.is_file():
            raise HTTPException(404, "Result not ready")
        fname = f"{_safe_filename(job.label)}_{job_id}.npz"
        return FileResponse(path, media_type="application/octet-stream", filename=fname)

    # -- physics audit (docs/AUDIT_API.md) -----------------------------------

    def _finished_continuum_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job id")
        if job.kind != "continuum":
            raise HTTPException(400, "Audit is only available for continuum (Setup-tab) jobs")
        if job.status in ("queued", "running") or (job.status in ("done", "stopped") and job.result is None):
            raise HTTPException(404, "Job not finished yet - no audit available")
        if job.status == "error":
            raise HTTPException(404, "Job failed - no audit available")
        return job

    def _job_audit(job):
        aud = job.extra.get("audit")
        if aud is None:  # e.g. a stopped run: the core still stored its audit on the result
            raw = (getattr(job.result, "extra", None) or {}).get("audit")
            if isinstance(raw, dict):
                from webapp.jobs import _json_safe
                aud = job.extra["audit"] = _json_safe(raw)
        return aud

    @app.get("/api/jobs/{job_id}/audit")
    def job_audit(job_id: str):
        job = _finished_continuum_job(job_id)
        aud = _job_audit(job)
        if aud is not None:
            return aud
        if not audit_status()["usable"]:
            raise HTTPException(503, audit_unavailable_message())
        if not job.audit_enabled:
            raise HTTPException(409, "Audit was disabled for this job; POST /api/jobs/{id}/audit/run to compute it.")
        err = job.extra.get("audit_error")
        raise HTTPException(404, f"No audit available for this job{f' ({err})' if err else ''}; "
                                 "POST /api/jobs/{id}/audit/run to (re)compute it.")

    @app.get("/api/jobs/{job_id}/audit.png")
    def job_audit_png(job_id: str):
        job = _finished_continuum_job(job_id)
        if _job_audit(job) is None and not job.audit_enabled:
            raise HTTPException(409, "Audit was disabled for this job; POST /api/jobs/{id}/audit/run first.")
        if not audit_status()["usable"]:
            raise HTTPException(503, audit_unavailable_message())
        try:
            path = manager.audit_png_path(job)
        except Exception as exc:  # noqa: BLE001
            logger.exception("audit figure failed")
            raise HTTPException(500, f"Could not render the audit figure: {exc}")
        return FileResponse(path, media_type="image/png", filename=f"{_safe_filename(job.label)}_{job_id}_audit.png")

    @app.post("/api/jobs/{job_id}/audit/run")
    def job_audit_run(job_id: str):
        job = _finished_continuum_job(job_id)
        if not audit_status()["usable"]:
            raise HTTPException(503, audit_unavailable_message())
        try:
            return manager.compute_audit(job, persist=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("audit failed")
            raise HTTPException(500, f"Audit failed: {type(exc).__name__}: {exc}")

    # -- quantum / QUBO backends -------------------------------------------

    @app.get("/api/quantum/backends")
    def quantum_backends():
        payload = quantum_backends_payload()
        # Separate from per-backend availability: whether the *installed*
        # freeto core's continuum loop is wired to optimizer="QUBO" at all
        # (freeto/quantum/ can be ready before freeto/core.py is updated to
        # use it — see docs/QUANTUM_API.md). The Truss/Study tabs don't need
        # this; only the Setup panel's optimizer select does.
        payload["continuum_qubo_supported"] = CORE_SUPPORTS_QUBO
        return payload

    # -- truss benchmarks ---------------------------------------------------

    @app.get("/api/truss/benchmarks")
    def truss_benchmarks():
        return truss_benchmarks_payload()

    @app.get("/api/truss/benchmarks/{bench_id}")
    def truss_benchmark_detail(bench_id: str):
        if not TRUSS_USABLE:
            raise HTTPException(503, truss_unavailable_message())
        try:
            prob = get_benchmark(bench_id)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(404, f"Unknown truss benchmark '{bench_id}': {exc}")
        return {"problem": prob.to_dict()}

    def _find_truss_benchmark(benchmark_id: str) -> Optional[dict]:
        for b in truss_benchmarks_payload().get("benchmarks", []):
            if b.get("id") == benchmark_id or b.get("alias") == benchmark_id:
                return b
        return None

    @app.post("/api/truss/run")
    def truss_run(req: TrussRunRequest):
        if not TRUSS_USABLE:
            raise HTTPException(503, truss_unavailable_message())
        if TRUSS_METHODS and req.method not in TRUSS_METHODS:
            raise HTTPException(400, f"Unknown method '{req.method}'; choose from {TRUSS_METHODS}")
        bench = _find_truss_benchmark(req.benchmark_id)
        if bench is None:
            raise HTTPException(400, f"Unknown truss benchmark id '{req.benchmark_id}'")
        if req.method == "exact" and not bench.get("exact_available", False):
            raise HTTPException(
                400,
                f"Exact enumeration is not available for '{req.benchmark_id}' (too many bars); "
                f"choose a method such as 'oc_round' or 'qubo' instead.",
            )
        label = req.label or f"{bench.get('title', req.benchmark_id)} — {req.method}"
        job = manager.submit_truss(
            label, req.model_dump(), benchmark_id=bench["id"], method=req.method,
            backend=req.backend, seed=req.seed, options=req.options,
        )
        return {
            "job_id": job.id,
            "status": job.status,
            "queue_position": manager.queue_position(job.id),
        }

    @app.get("/api/jobs/{job_id}/truss_result")
    def truss_result(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job id")
        if job.kind != "truss":
            raise HTTPException(400, "Job is not a truss job")
        data = job.extra.get("truss_result")
        if data is None:
            raise HTTPException(404, "Result not ready")
        return data

    # -- study suites --------------------------------------------------------

    @app.post("/api/study/run")
    def study_run(req: StudyRunRequest):
        if not STUDY_USABLE:
            raise HTTPException(503, study_unavailable_message())
        if not req.suite and not req.spec and not req.custom:
            raise HTTPException(
                400, "Provide 'suite' ('smoke' / 'quick' / 'full'), a 'custom' matrix or a raw 'spec'.")
        if req.spec is not None:
            spec = req.spec
        elif req.custom is not None:
            try:
                spec = build_custom_study_spec(req.custom)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
        else:
            try:
                spec = suite_spec(req.suite)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(400, f"Unknown study suite '{req.suite}': {exc}")
        spec_name = spec.get("name") if isinstance(spec, dict) else None
        label = req.label or f"Study: {req.suite or spec_name or 'custom'}"
        run_id = uuid.uuid4().hex[:12]
        out_dir = workdir / "study" / run_id
        job = manager.submit_study(label, req.model_dump(), spec=spec, out_dir=out_dir, figures=req.figures)
        return {
            "job_id": job.id,
            "study_id": run_id,
            "n_runs": sum(len(r.get("seeds", [0])) for r in spec.get("runs", [])) if isinstance(spec, dict) else None,
            "status": job.status,
            "queue_position": manager.queue_position(job.id),
        }

    @app.get("/api/study/custom_options")
    def study_custom_options():
        """What the 'custom' study builder offers (problems with their default mesh, methods)."""
        return {
            "problems": [{"id": k, "default_mesh_control": v} for k, v in CUSTOM_PROBLEMS.items()],
            "methods": [{"id": k, "optimizer": v["optimizer"], "options": v.get("options")}
                        for k, v in CUSTOM_METHODS.items()],
            "request": "POST /api/study/run {\"custom\": {\"problems\": [...], \"methods\": [...], "
                       "\"seeds\": [0,1,2], \"mesh_control\": null, \"max_iter\": 100, "
                       "\"include_baseline\": true}}",
        }

    @app.get("/api/study/status/{job_id}")
    def study_job_status(job_id: str, since: int = 0):
        job = manager.get(job_id)
        if job is None or job.kind != "study":
            raise HTTPException(404, "Unknown study job id")
        status = job.snapshot_status(log_since=since)
        status["queue_position"] = manager.queue_position(job_id)
        return status

    # Past studies: every `<workdir>/study/<id>/` (and, read-only, `<repo>/results/<name>/`
    # written by `python -m freeto.study --out results/<name>`, listed as `cli~<name>`) is
    # listed from disk, so finished study outputs survive a server restart.
    study_roots = {"": workdir / "study", "cli~": PACKAGE_ROOT / "results"}

    def _study_dir(ident: str) -> Optional[Path]:
        """Study output dir for a study id (in-memory job id, directory name, or cli~name)."""
        job = manager.get(ident)
        if job is not None:
            if job.kind != "study":
                return None
            od = job.extra.get("out_dir")
            return Path(od) if od else None
        if not _STUDY_ID_RE.match(ident):
            return None
        for prefix, root in study_roots.items():
            if prefix and not ident.startswith(prefix):
                continue
            name = ident[len(prefix):]
            if not name or name.startswith("."):
                continue
            d = root / name
            if d.is_dir():
                return d
        return None

    def _study_files(job_key: str, d: Path) -> List[dict]:
        files = []
        base = d.resolve()
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = p.relative_to(base).as_posix()
                files.append({"path": rel, "name": p.name, "url": f"/api/study/file/{job_key}/{rel}"})
        return files[:500]  # sane cap for a "full" suite's per-run artefacts

    def _study_summary_row(sid: str, d: Path, source: str) -> Optional[dict]:
        rj = d / "results.json"
        if not rj.is_file() and not (d / "summary.md").is_file() and not any(d.glob("*.png")):
            return None
        row: Dict[str, Any] = {
            "id": sid, "source": source, "suite": None, "created": None, "n_records": None,
            "n_invalid": None, "n_errors": None, "has_audit": None, "elapsed": None,
            "complete": rj.is_file(), "n_figures": len(list(d.glob("*.png"))),
            "mtime": d.stat().st_mtime,
        }
        if rj.is_file():
            try:
                res = json.loads(rj.read_text(encoding="utf-8"))
                recs = res.get("records") or []
                cont = [r for r in recs if r.get("kind") == "continuum"]
                row.update(
                    suite=res.get("suite"), created=res.get("created"), n_records=len(recs),
                    elapsed=res.get("elapsed"),
                    n_errors=sum(1 for r in recs if r.get("error")),
                    n_invalid=sum(1 for r in recs if r.get("audit_ok") is False),
                    # stale = continuum rows without audit fields (run before the fix)
                    has_audit=(any("audit_ok" in r for r in cont) if cont else None),
                )
            except Exception as exc:  # noqa: BLE001
                row["read_error"] = f"{type(exc).__name__}: {exc}"
        return row

    @app.get("/api/study/list")
    def study_list():
        """Studies found on disk (newest first); `id` works with /api/study/results/{id}."""
        out = []
        for prefix, root in study_roots.items():
            if not root.is_dir():
                continue
            for d in root.iterdir():
                if not d.is_dir() or d.name.startswith("."):
                    continue
                row = _study_summary_row(prefix + d.name, d, "cli" if prefix else "web")
                if row is not None:
                    out.append(row)
        # annotate studies that are still queued/running in this server session
        for st in manager.list_jobs():
            if st.get("kind") == "study" and st.get("status") in ("queued", "running") and st.get("study_id"):
                for row in out:
                    if row["id"] == st["study_id"]:
                        row["running_job_id"] = st["id"]
        out.sort(key=lambda r: r["mtime"], reverse=True)
        return {"studies": out}

    @app.get("/api/study/results/{job_id}")
    def study_results(job_id: str):
        job = manager.get(job_id)
        d = _study_dir(job_id)
        if d is None:
            raise HTTPException(404, "Unknown study id")
        if job is not None and job.kind == "study":
            status, results = job.status, job.extra.get("study_results")
        else:
            status, results = "done" if (d / "results.json").is_file() else "incomplete", None
        if results is None and (d / "results.json").is_file():
            try:
                results = json.loads((d / "results.json").read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(500, f"Could not read results.json: {exc}")
        files = _study_files(job_id, d) if d.is_dir() else []
        summary_md = None
        if (d / "summary.md").is_file():
            summary_md = (d / "summary.md").read_text(encoding="utf-8", errors="replace")
        return {"status": status, "results": results, "files": files, "study_id": d.name,
                "summary_md": summary_md}

    @app.get("/api/study/file/{job_id}/{file_path:path}")
    def study_file(job_id: str, file_path: str):
        d = _study_dir(job_id)
        if d is None:
            raise HTTPException(404, "Unknown study id")
        base = d.resolve()
        target = (base / file_path).resolve()
        try:
            target.relative_to(base)
        except ValueError:
            raise HTTPException(404, "File not found")
        if not target.is_file():
            raise HTTPException(404, "File not found")
        media = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return FileResponse(target, media_type=media, filename=target.name)

    # -- static frontend --------------------------------------------------

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app


app = create_app()


def _pick_port(host: str, start: int, tries: int = 10) -> int:
    """Return the first free TCP port at/after `start` on `host`. A previous
    run (or any other dev server) can still own the default port, and uvicorn
    exiting with a raw 'address already in use' stack trace is both alarming
    and, in the launcher windows, easy to miss among the INFO lines above it."""
    import socket

    for p in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
            except OSError:
                continue
            return p
    raise SystemExit(
        f"ERROR: every port from {start} to {start + tries - 1} on {host} is already in "
        f"use; close the other FreeTO-Python window (or any other server on one of those "
        f"ports) or pass --port to pick a different one."
    )


def main():
    parser = argparse.ArgumentParser(description="FreeTO-Python web app server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workdir", default=None, help="Working directory for uploads/results (default ~/.freeto_web)")
    parser.add_argument("--open", action="store_true", help="Open the app in a browser once the server is up")
    args = parser.parse_args()

    import uvicorn

    global app
    app = create_app(Path(args.workdir) if args.workdir else None)

    # Pick a free port up front (falls forward from --port through the next 9
    # ports) instead of letting uvicorn.run() crash with a raw bind error, and
    # print the URL prominently so it's the first thing visible in a launcher
    # window full of INFO logging.
    args.port = _pick_port(args.host, args.port)
    url = f"http://{args.host}:{args.port}/"
    print("=" * max(60, len(url) + 22), flush=True)
    print(f"  FreeTO-Python web app:  {url}", flush=True)
    print("  (Ctrl+C in this window stops the server)", flush=True)
    print("=" * max(60, len(url) + 22), flush=True)

    if args.open:
        def _open():
            # Only open the browser once the server is actually answering
            # requests — opening immediately on a slow machine/first import
            # can land on a half-started connection, or (worse, if the port
            # was already someone else's before _pick_port ran) a stale tab.
            import urllib.request

            health_url = url + "api/health"
            for _ in range(80):  # up to ~20s
                try:
                    urllib.request.urlopen(health_url, timeout=1).close()
                    break
                except Exception:
                    time.sleep(0.25)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
