"""Public (hosted) mode of the QFF-3D web app.

Off by default, so a local ``python -m webapp.server`` behaves exactly as
before.  ``QFF3D_PUBLIC=1`` (set by the Dockerfile for the shared Hugging Face
Space behind https://nitipong.com/qff3d) turns on the limits below, which keep
one small shared CPU server usable for everybody.  Every limit can be changed
with an environment variable; the defaults are in brackets.

========================== ===================================================
QFF3D_PUBLIC               1 / true / yes / on enables public mode [off]
QFF3D_MAX_RUNNING          jobs running at the same time [1]
QFF3D_MAX_QUEUED           jobs waiting in the queue; more -> HTTP 429 [8]
QFF3D_MAX_JOBS_PER_CLIENT  active (queued + running) jobs per client IP [1]
QFF3D_MAX_MESH_CONTROL     largest mesh_control accepted [50]
QFF3D_MAX_ITER             largest max_iter accepted [300]
QFF3D_MAX_UPLOAD_MB        size of one uploaded STL, MB [20]
QFF3D_MAX_UPLOAD_FILES     files per upload request [6]
QFF3D_MAX_QAOA_BLOCK       largest QUBO block for the QAOA backends [12]
QFF3D_MAX_JOB_MINUTES      wall time of one job; then it is stopped cleanly [20]
QFF3D_TTL_MINUTES          job results and uploads are deleted after [120]
QFF3D_CLEANUP_MINUTES      how often the clean-up runs [5]
QFF3D_DIRECT_URL           optional direct URL of the app (e.g. the hf.space
                           address), shown for uploads larger than the proxy
                           allows
QFF3D_REPO_URL             link shown in the footer notice
                           [https://github.com/cnnitipong/Quantum-Free-Form-3D-topology]
========================== ===================================================

The paper settings (docs/PAPER_SETTINGS.md) fit inside the defaults: meshes
24-46, 300 iterations, QAOA blocks of 8.  None of the limits changes how a
run is computed; a request above a limit is rejected with a message, never
silently altered.  (The one exception is loading an example whose own mesh
is finer than the cap: the form is pre-filled with the cap and says so.)
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional

REPO_URL = "https://github.com/cnnitipong/Quantum-Free-Form-3D-topology"

#: QUBO backends whose block size is a qubit count (state-vector / circuit cost
#: grows as 2**block_size); see freeto/quantum/backends.py.
QAOA_BACKENDS = ("qaoa", "qiskit_aer", "ibm")
#: freeto.quantum's default block size of those backends when none is given
#: (backends._BLOCK); used to explain a rejection, never to change a run.
QAOA_DEFAULT_BLOCK = 14

_TRUE = {"1", "true", "yes", "on"}


def _env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    val = env.get(name)
    if val is None or not str(val).strip():
        return default
    return str(val).strip().lower() in _TRUE


def _env_num(env: Mapping[str, str], name: str, default, cast=int, minimum=None):
    val = env.get(name)
    if val is None or not str(val).strip():
        return default
    try:
        out = cast(float(val)) if cast is int else cast(val)
    except (TypeError, ValueError):
        raise ValueError(f"environment variable {name}={val!r} is not a number") from None
    if minimum is not None and out < minimum:
        raise ValueError(f"environment variable {name}={val!r} must be >= {minimum}")
    return out


@dataclass(frozen=True)
class PublicConfig:
    """Limits of the shared demo server (all inactive when ``enabled`` is False)."""

    enabled: bool = False
    max_running: int = 1
    max_queued: int = 8
    max_jobs_per_client: int = 1
    max_mesh_control: int = 50
    max_iter: int = 300
    max_upload_mb: float = 20.0
    max_upload_files: int = 6
    max_qaoa_block: int = 12
    max_job_minutes: float = 20.0
    ttl_minutes: float = 120.0
    cleanup_minutes: float = 5.0
    direct_url: Optional[str] = None
    repo_url: str = REPO_URL

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "PublicConfig":
        env = os.environ if env is None else env
        d = cls()
        return cls(
            enabled=_env_bool(env, "QFF3D_PUBLIC", False),
            max_running=_env_num(env, "QFF3D_MAX_RUNNING", d.max_running, int, 1),
            max_queued=_env_num(env, "QFF3D_MAX_QUEUED", d.max_queued, int, 0),
            max_jobs_per_client=_env_num(env, "QFF3D_MAX_JOBS_PER_CLIENT", d.max_jobs_per_client, int, 1),
            max_mesh_control=_env_num(env, "QFF3D_MAX_MESH_CONTROL", d.max_mesh_control, int, 4),
            max_iter=_env_num(env, "QFF3D_MAX_ITER", d.max_iter, int, 1),
            max_upload_mb=_env_num(env, "QFF3D_MAX_UPLOAD_MB", d.max_upload_mb, float, 0.01),
            max_upload_files=_env_num(env, "QFF3D_MAX_UPLOAD_FILES", d.max_upload_files, int, 1),
            max_qaoa_block=_env_num(env, "QFF3D_MAX_QAOA_BLOCK", d.max_qaoa_block, int, 1),
            max_job_minutes=_env_num(env, "QFF3D_MAX_JOB_MINUTES", d.max_job_minutes, float, 0.01),
            ttl_minutes=_env_num(env, "QFF3D_TTL_MINUTES", d.ttl_minutes, float, 0.01),
            cleanup_minutes=_env_num(env, "QFF3D_CLEANUP_MINUTES", d.cleanup_minutes, float, 0.01),
            direct_url=(env.get("QFF3D_DIRECT_URL") or "").strip() or None,
            repo_url=(env.get("QFF3D_REPO_URL") or "").strip() or REPO_URL,
        )

    # -- derived values ---------------------------------------------------

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def max_job_seconds(self) -> Optional[float]:
        return self.max_job_minutes * 60.0 if self.enabled else None

    @property
    def ttl_seconds(self) -> float:
        return self.ttl_minutes * 60.0

    @property
    def n_workers(self) -> int:
        """Worker threads of the job manager: QFF3D_MAX_RUNNING in public mode,
        one (the original behaviour) locally."""
        return self.max_running if self.enabled else 1

    def to_dict(self) -> Dict[str, Any]:
        """What GET /api/health reports (the UI reads it for its notices)."""
        out = asdict(self)
        out["ttl_hours"] = round(self.ttl_minutes / 60.0, 2)
        out["study_enabled"] = not self.enabled
        return out

    def notice(self) -> str:
        hours = self.ttl_minutes / 60.0
        when = f"{hours:g} hours" if hours >= 1 else f"{self.ttl_minutes:g} minutes"
        return (f"Shared demo server: one job at a time, runs are deleted after {when}. "
                f"For the full version run it locally ({self.repo_url}).")


class LimitError(Exception):
    """A request exceeds a public-mode limit -> HTTP ``status`` with ``message``."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status
        self.message = message


def client_ip(headers: Mapping[str, str], peer: Optional[str]) -> str:
    """Client address for the per-client limit: the first hop of
    X-Forwarded-For (the browser, as seen by the first proxy -- Vercel / the
    Hugging Face front end), else the TCP peer.  Header names are matched
    case-insensitively (Starlette's Headers already are)."""
    xff = None
    try:
        xff = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
    except Exception:  # noqa: BLE001
        xff = None
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    return peer or "unknown"


LOCAL_VERSION_MSG = ("Multi-run studies are available in the local version "
                     f"(see {REPO_URL}); the shared demo server runs single jobs only.")


def check_job_request(cfg: PublicConfig, req: Any) -> None:
    """Raise LimitError if a continuum job request (webapp.server.JobCreateRequest
    or any object with the same attributes) exceeds a public-mode limit."""
    if not cfg.enabled:
        return
    mc = int(getattr(req, "mesh_control", 0) or 0)
    if mc > cfg.max_mesh_control:
        raise LimitError(
            f"mesh_control = {mc} exceeds the shared demo server's limit of "
            f"{cfg.max_mesh_control} (the paper meshes are 24-46). Use a coarser mesh, "
            f"or run the local version for finer meshes.")
    it = int(getattr(req, "max_iter", 0) or 0)
    if it > cfg.max_iter:
        raise LimitError(
            f"max_iter = {it} exceeds the shared demo server's limit of {cfg.max_iter} "
            f"iterations (the paper uses 300).")
    if str(getattr(req, "optimizer", "") or "").strip().upper() == "QUBO":
        backend = str(getattr(req, "qubo_backend", None) or "").strip().lower()
        if backend in QAOA_BACKENDS:
            kb = getattr(req, "qubo_block_size", None)
            if kb is None:
                raise LimitError(
                    f"The QAOA backend '{backend}' uses blocks of {QAOA_DEFAULT_BLOCK} qubits by "
                    f"default; on the shared demo server set the QUBO block size to at most "
                    f"{cfg.max_qaoa_block} (the paper uses 8).")
            if int(kb) > cfg.max_qaoa_block:
                raise LimitError(
                    f"QUBO block size {int(kb)} exceeds the shared demo server's limit of "
                    f"{cfg.max_qaoa_block} qubits for the QAOA backends (the paper uses 8).")


def check_truss_request(cfg: PublicConfig, req: Any) -> None:
    """Raise LimitError if a truss request exceeds a public-mode limit."""
    if not cfg.enabled:
        return
    opts = dict(getattr(req, "options", None) or {})
    backend = str(getattr(req, "backend", None) or opts.get("backend") or "").strip().lower()
    kb = opts.get("block_size")
    if backend in QAOA_BACKENDS and kb is not None and int(kb) > cfg.max_qaoa_block:
        raise LimitError(
            f"QUBO block size {int(kb)} exceeds the shared demo server's limit of "
            f"{cfg.max_qaoa_block} qubits for the QAOA backends.")
    for key in ("max_iter", "iterations"):
        if opts.get(key) is not None and int(opts[key]) > cfg.max_iter:
            raise LimitError(f"{key} = {int(opts[key])} exceeds the shared demo server's "
                             f"limit of {cfg.max_iter}.")


def check_capacity(cfg: PublicConfig, manager: Any, client: str) -> None:
    """Raise LimitError (429) if the client already has an active job or the
    queue is full.  ``manager`` is a webapp.jobs.JobManager."""
    if not cfg.enabled:
        return
    mine = manager.active_count(client=client)
    if mine >= cfg.max_jobs_per_client:
        raise LimitError(
            f"You already have {mine} job{'s' if mine != 1 else ''} queued or running on the "
            f"shared demo server (limit {cfg.max_jobs_per_client} per visitor). Wait for it to "
            f"finish or stop it, then start the next one.", status=429)
    if manager.queued_count() >= cfg.max_queued:
        raise LimitError(
            f"The shared demo server is busy: {cfg.max_queued} jobs are already waiting. "
            f"Please try again in a few minutes, or run QFF-3D locally ({cfg.repo_url}).",
            status=429)
