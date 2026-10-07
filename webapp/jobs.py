"""In-process job manager for FreeTO runs.

A single background worker thread runs jobs strictly one at a time (FreeTO
itself is CPU/BLAS heavy and not designed for concurrent runs sharing a
process); further job submissions while one is running are queued and served
FIFO. Each job also gets its own `threading.Event` for cooperative stopping.

Public (hosted) mode (webapp/public.py) adds: an optional wall-time cap per
job (the job's stop event is set when it expires, so the run ends exactly as
if the user had pressed Stop), per-client bookkeeping, and deletion of
finished jobs after a time-to-live.  `n_workers` > 1 (QFF3D_MAX_RUNNING) runs
that many jobs side by side; that is possible but not recommended for the
reason above, and the default stays 1.
"""

from __future__ import annotations

import io
import json
import math
import os
import queue
import shutil
import struct
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
from collections import deque

import numpy as np

from webapp.core_loader import FreeTOConfig, run_freeto, surface_from_field
from webapp import core_loader

MAX_LOG_LINES = 4000
PREVIEW_MIN_INTERVAL_S = 0.75  # throttle preview *regeneration* (client-request driven)
# Preview meshes are decimated to keep marching-cubes + STL-encode cheap and
# to keep the payload the browser fetches/parses every poll small. This is a
# heuristic on total grid cells (not triangles directly, which depend on the
# shape), chosen so that even a fairly convoluted surface stays well under
# ~300k triangles.
PREVIEW_MAX_CELLS = 200_000


@dataclass
class Job:
    id: str
    label: str
    config_dict: Dict[str, Any]
    cfg: Optional["FreeTOConfig"] = None
    # "continuum" (the original FreeTO run_freeto job) | "truss" | "study".
    # The worker loop (_run_job) dispatches on this; everything else about
    # Job/JobManager (queueing, stop, log tailing, status polling) is shared
    # across kinds so truss/study jobs obey the same single-running-job rule.
    kind: str = "continuum"
    # Kind-specific run parameters that don't fit FreeTOConfig (benchmark id /
    # method / backend for truss, suite spec / out_dir for study).
    run_spec: Dict[str, Any] = field(default_factory=dict)
    status: str = "queued"  # queued | running | done | error | stopped
    stage: Optional[str] = None
    setup: Optional[dict] = None
    history: Dict[str, list] = field(default_factory=lambda: {
        "compliance": [], "volfrac": [], "change": [], "topo": [], "beta": []
    })
    log_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    last_iter: int = 0
    last_iter_time: Optional[float] = None
    # Coarse progress for multi-run jobs (study): {"index", "n_runs", "label", "study"}.
    progress: Optional[dict] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[Any] = None
    # Physics audit (docs/AUDIT_API.md): whether the job was submitted with
    # audit on; the audit dict itself lives in extra["audit"].
    audit_enabled: bool = True
    # Kind-specific extra payload not shaped like the continuum history/log
    # (a TrussResult.to_dict(), a study results dict, its out_dir, ...).
    extra: Dict[str, Any] = field(default_factory=dict)
    # Client address of the submitter (public mode: per-client limit; the job
    # list only shows a visitor's own jobs). None for local use.
    client: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Absolute (never-reset) count of log lines ever appended. The deque
    # itself is capped at MAX_LOG_LINES and silently drops old lines, which
    # would otherwise desync a client's `?since=<cursor>` index once a run
    # produces more than MAX_LOG_LINES iterations; tracking the absolute
    # count lets snapshot_status report a cursor/tail that stays consistent
    # (the client just never sees lines that already scrolled out).
    _log_total: int = 0

    # Live-preview field handoff: `_field_lock` is held only for the tiny,
    # fast reference swap/read of `_last_field` (by the worker's callback and
    # by preview requests) — it must NEVER be held across the expensive
    # marching-cubes + STL-encode work, or the worker stalls on every HTTP
    # preview request. `_preview_cache_lock` is a separate lock that DOES
    # wrap the (slow) generate-and-cache section, but only serializes
    # concurrent preview HTTP requests against each other, never the worker.
    _field_lock: threading.Lock = field(default_factory=threading.Lock)
    _last_field: Any = None
    _preview_cache_lock: threading.Lock = field(default_factory=threading.Lock)
    _preview_iter: int = -1
    _preview_bytes: Optional[bytes] = None
    _preview_generated_at: float = 0.0

    def append_log(self, line: str) -> None:
        with self._lock:
            self.log_lines.append(line)
            self._log_total += 1

    def snapshot_status(self, log_since: int = 0) -> dict:
        with self._lock:
            logs = list(self.log_lines)
            total = self._log_total
            dropped = total - len(logs)  # lines the deque already discarded
            start = max(0, log_since - dropped)
            tail = logs[start:]
            return {
                "id": self.id,
                "label": self.label,
                "kind": self.kind,
                "status": self.status,
                "stage": self.stage,
                "setup": self.setup,
                # list(...) first: the worker thread adds history keys
                # ("qubo", "audit_live", ...) without this lock, and iterating
                # the live dict would raise "dictionary changed size"
                "history": {k: list(v) for k, v in list(self.history.items())},
                "log_lines": tail,
                "log_cursor": total,
                "error": self.error,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "last_iter": self.last_iter,
                "last_iter_time": self.last_iter_time,
                "max_iter": self.cfg.max_iter if self.cfg is not None else None,
                "progress": self.progress,
                "has_result": self.result is not None,
                # What was *submitted* (truss: benchmark_id/method/backend/seed),
                # so clients label finished runs from the job itself, never
                # from whatever the UI selects show by the time it finishes.
                "spec": ({k: self.run_spec.get(k) for k in ("benchmark_id", "method", "backend", "seed")}
                         if self.kind == "truss" else None),
                # Scalar extras of a finished continuum run (crisp compliance,
                # beta_final, ...), see _run_continuum.
                "result_info": self.extra.get("result_info"),
                # paper context of a continuum job (webapp.paper_presets.paper_comparison):
                # matched paper run + its published values, MMA reference for the gap
                "paper": self.extra.get("paper") if self.kind == "continuum" else None,
                **self._audit_summary(),
            }

    def _audit_summary(self) -> dict:
        """Audit scalars for the status JSON (callers hold self._lock or not;
        extra/history reads here are plain dict lookups)."""
        out: dict = {"audit_enabled": self.audit_enabled if self.kind == "continuum" else None}
        aud = self.extra.get("audit")
        if isinstance(aud, dict):
            out["audit_ok"] = aud.get("ok")
            out["n_components"] = aud.get("n_components")
            out["floating_frac"] = aud.get("floating_frac")
        else:
            out["audit_ok"] = None
            out["n_components"] = None
            out["floating_frac"] = None
        live = self.history.get("audit_live")
        out["audit_live"] = live[-1] if live else None
        out["audit_error"] = self.extra.get("audit_error")
        if self.kind == "study" and self.extra.get("out_dir"):
            out["study_id"] = Path(self.extra["out_dir"]).name
        return out


ACTIVE_STATUSES = ("queued", "running")


class JobManager:
    def __init__(self, jobs_dir, n_workers: int = 1, max_wall_s: Optional[float] = None):
        self.jobs_dir = jobs_dir
        self.max_wall_s = float(max_wall_s) if max_wall_s else None
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._running_ids: set = set()
        self._global_lock = threading.Lock()
        self._workers = [threading.Thread(target=self._worker_loop, daemon=True,
                                          name=f"freeto-job-worker-{i}")
                         for i in range(max(1, int(n_workers)))]
        for w in self._workers:
            w.start()

    # -- public API ---------------------------------------------------

    def submit(self, label: str, config_dict: dict, cfg: "FreeTOConfig", audit: bool = True,
               paper: Optional[dict] = None, client: Optional[str] = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, label=label, config_dict=config_dict, cfg=cfg, kind="continuum",
                  audit_enabled=bool(audit), client=client)
        if paper is not None:
            job.extra["paper"] = paper
        return self._enqueue(job)

    def submit_truss(self, label: str, config_dict: dict, *, benchmark_id: str, method: str,
                      backend: Optional[str], seed: Optional[int], options: dict,
                      client: Optional[str] = None) -> Job:
        job_id = uuid.uuid4().hex[:12]
        run_spec = {
            "benchmark_id": benchmark_id,
            "method": method,
            "backend": backend,
            "seed": seed,
            "options": dict(options or {}),
        }
        job = Job(id=job_id, label=label, config_dict=config_dict, kind="truss", run_spec=run_spec,
                  client=client)
        return self._enqueue(job)

    def submit_study(self, label: str, config_dict: dict, *, spec: dict, out_dir, figures: bool) -> Job:
        job_id = uuid.uuid4().hex[:12]
        run_spec = {"spec": spec, "out_dir": str(out_dir), "figures": bool(figures)}
        job = Job(id=job_id, label=label, config_dict=config_dict, kind="study", run_spec=run_spec)
        job.extra["out_dir"] = str(out_dir)
        return self._enqueue(job)

    def _enqueue(self, job: Job) -> Job:
        with self._global_lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list_jobs(self, client: Optional[str] = None) -> List[dict]:
        """Status of every job, newest first (only `client`'s jobs if given)."""
        with self._global_lock:
            ids = list(self._order)
        out = []
        for jid in reversed(ids):
            job = self._jobs.get(jid)
            if job is None:
                continue
            if client is not None and job.client != client:
                continue
            out.append(job.snapshot_status())
        return out

    # -- counts (public-mode limits) -------------------------------------

    def _snapshot_jobs(self) -> List[Job]:
        with self._global_lock:
            return [self._jobs[j] for j in self._order if j in self._jobs]

    def active_count(self, client: Optional[str] = None) -> int:
        """Queued + running jobs (of `client` only, if given)."""
        return sum(1 for j in self._snapshot_jobs()
                   if j.status in ACTIVE_STATUSES and (client is None or j.client == client))

    def queued_count(self) -> int:
        return sum(1 for j in self._snapshot_jobs() if j.status == "queued")

    def running_count(self) -> int:
        return sum(1 for j in self._snapshot_jobs() if j.status == "running")

    def jobs_ahead(self, job_id: str) -> Optional[int]:
        """Jobs that will start or finish before a queued job can start
        (running jobs + queued jobs submitted earlier); None if not queued."""
        pos = self.queue_position(job_id)
        if pos is None:
            return None
        return self.running_count() + pos - 1

    # -- retention (public mode) ------------------------------------------

    def delete_job(self, job_id: str) -> bool:
        """Forget a finished job and delete its directory; refuses (False) for
        a queued / running job."""
        with self._global_lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in ACTIVE_STATUSES:
                return False
            del self._jobs[job_id]
            self._order = [j for j in self._order if j != job_id]
        shutil.rmtree(self.jobs_dir / job_id, ignore_errors=True)
        return True

    def purge_expired(self, ttl_s: float, now: Optional[float] = None) -> List[str]:
        """Delete every finished job older than `ttl_s` seconds (counted from
        when it finished, or was submitted if it never ran), plus job
        directories on disk that no live job owns and that were last modified
        more than `ttl_s` ago (e.g. left over from before a restart).
        Returns the deleted job ids."""
        now = time.time() if now is None else now
        cutoff = now - float(ttl_s)
        deleted = []
        for job in self._snapshot_jobs():
            if job.status in ACTIVE_STATUSES:
                continue
            t = job.finished_at or job.created_at
            if t <= cutoff and self.delete_job(job.id):
                deleted.append(job.id)
        root = Path(self.jobs_dir)
        if root.is_dir():
            for d in root.iterdir():
                if d.name in self._jobs:
                    continue
                try:
                    if d.stat().st_mtime <= cutoff:
                        if d.is_dir():
                            shutil.rmtree(d, ignore_errors=True)
                        else:
                            d.unlink()
                        deleted.append(d.name)
                except OSError:
                    pass
        return deleted

    def stop(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.stop_event.set()
        if job.status == "queued":
            job.status = "stopped"
            job.error = "Cancelled before it started."
        return True

    def queue_position(self, job_id: str) -> Optional[int]:
        """1-based position in the pending queue, or None if not queued."""
        job = self.get(job_id)
        if job is None or job.status != "queued":
            return None
        with self._global_lock:
            pending = [jid for jid in self._order if self._jobs[jid].status == "queued"]
        try:
            return pending.index(job_id) + 1
        except ValueError:
            return None

    # -- preview --------------------------------------------------------

    def get_preview_stl(self, job_id: str) -> Optional[bytes]:
        job = self.get(job_id)
        if job is None:
            return None

        # Fast, tiny critical section: just grab the current field reference
        # and iteration number. This lock is the same one the worker's
        # callback uses, so it must never be held across the expensive work
        # below — otherwise every preview request would stall the optimizer.
        with job._field_lock:
            field = job._last_field
            iter_now = job.last_iter

        if field is None:
            return None

        # Cache check and the actual (slow) generation happen under a
        # SEPARATE lock that only serializes concurrent preview requests
        # against each other — the worker thread never touches this lock.
        with job._preview_cache_lock:
            if job._preview_bytes is not None and job._preview_iter == iter_now:
                return job._preview_bytes
            now = time.time()
            if job._preview_bytes is not None and now - job._preview_generated_at < PREVIEW_MIN_INTERVAL_S:
                return job._preview_bytes
            try:
                small_field = _decimate_field(field, PREVIEW_MAX_CELLS)
                verts, faces = surface_from_field(small_field, smooth=True)
                data = _mesh_to_binary_stl(verts, faces)
            except Exception:
                return job._preview_bytes
            job._preview_bytes = data
            job._preview_iter = iter_now
            job._preview_generated_at = now
            return data

    # -- worker ---------------------------------------------------------

    def _worker_loop(self):
        while True:
            job_id = self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if job.status == "stopped":
                continue  # was cancelled while queued
            self._run_job(job)

    def _run_job(self, job: Job):
        """Common bookkeeping for every job kind; the actual work is
        dispatched to a `_run_<kind>` method. Any exception raised by that
        method (including one it deliberately re-raises after logging
        kind-specific context) is caught here exactly once, so every kind
        gets the same status/error/log handling."""
        job.status = "running"
        job.stage = "starting"
        job.started_at = time.time()
        with self._global_lock:
            self._running_ids.add(job.id)
        timer = None
        if self.max_wall_s:
            limit = self.max_wall_s

            def _expire():
                if job.status == "running" and not job.stop_event.is_set():
                    job.extra["wall_time_exceeded"] = True
                    job.append_log(f"Wall-time limit of the shared server reached "
                                   f"({limit / 60:g} min): stopping the job.")
                    job.stop_event.set()

            timer = threading.Timer(limit, _expire)
            timer.daemon = True
            timer.start()
        try:
            if job.kind == "continuum":
                self._run_continuum(job)
            elif job.kind == "truss":
                self._run_truss(job)
            elif job.kind == "study":
                self._run_study(job)
            else:
                raise RuntimeError(f"Unknown job kind: {job.kind!r}")
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.stage = "error"
            job.error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            job.append_log(f"ERROR: {exc}")
        finally:
            if timer is not None:
                timer.cancel()
            if job.extra.get("wall_time_exceeded") and job.status == "stopped":
                job.error = (f"Stopped after the shared server's wall-time limit of "
                             f"{self.max_wall_s / 60:g} min; the design reached so far is kept.")
            job.finished_at = time.time()
            with self._global_lock:
                self._running_ids.discard(job.id)

    def _run_continuum(self, job: Job):
        def log(line: str):
            job.append_log(line)

        def callback(info: dict):
            info = dict(info)
            if info.get("stage") == "setup":
                job.setup = {k: v for k, v in info.items() if k != "stage"}
                job.stage = "setup"
            elif info.get("stage") == "iter":
                field = info.pop("field", None)
                job.stage = "iter"
                job.last_iter = info.get("iter", job.last_iter)
                job.last_iter_time = info.get("iter_time")
                for key in ("compliance", "volfrac", "change", "topo", "beta"):
                    if key in info:
                        job.history[key].append(info[key])
                live = info.get("audit_live")
                if isinstance(live, dict):
                    # QUBO runs: cheap component count of the binary design
                    # (docs/AUDIT_API.md) -> status bar "components: N, floating: x %".
                    job.history.setdefault("audit_live", []).append(
                        {"n_components": live.get("n_components"),
                         "floating_frac": live.get("floating_frac")})
                qubo = info.get("qubo")
                if qubo:
                    # QUBO mode only (freeto.quantum's continuum design
                    # update): per-iteration solver stats, surfaced in the
                    # log so the researcher can see them without a separate
                    # UI element — the status bar reads the *last* line.
                    job.history.setdefault("qubo", []).append(qubo)
                    job.append_log(_format_qubo_log(job.last_iter, qubo))
                if field is not None:
                    # Cheap reference swap only — see _field_lock's docstring
                    # comment on the Job dataclass for why this must stay
                    # separate from (and never block on) preview generation.
                    with job._field_lock:
                        job._last_field = field

        # Pin the BLAS / OpenMP / MKL (PARDISO) thread pools like the study does
        # (freeto.study._ThreadPin, one thread by default): a binary design
        # trajectory depends on the floating-point summation order, so the same
        # inputs reproduce the study's result only with the same thread count.
        with _thread_pin() as pin:
            if pin is not None:
                job.append_log(f"Threads: {_pin_text(pin)} (FREETO_THREADS, as the study: 1)")
            result = run_freeto(job.cfg, callback=callback, stop_event=job.stop_event, log=log)
        job.result = result
        # Persist result files to disk BEFORE flipping the job to a
        # terminal status: a client polling status may act on "done"
        # immediately (e.g. downloading result.stl), so the files must
        # already exist by the time that status becomes visible.
        self._save_result(job)
        if result.stopped:
            job.status = "stopped"
            job.stage = "stopped"
            job.append_log("Run stopped by user.")
        else:
            job.status = "done"
            job.stage = "done"
            extra = getattr(result, "extra", None)
            if extra:
                job.extra["continuum_extra"] = extra
            self._collect_audit(job)
            info = _result_info(extra)
            info.update(_native_info(result, job.cfg))
            _paper_gap(info, job.extra.get("paper"))
            job.extra["result_info"] = info or None
            crisp = ""
            if info.get("crisp_compliance") is not None:
                crisp = f" | crisp compliance={info['crisp_compliance']:.4f}"
                if info.get("crisp_volfrac") is not None:
                    crisp += f" (crisp volfrac={info['crisp_volfrac']:.4f})"
            job.append_log(
                f"Done: compliance={result.comp:.4f} volfrac={result.finalvol:.4f} "
                f"iterations={result.iterations} elapsed={result.elapsed:.1f}s" + crisp
            )
            if info.get("refined_compliance") is not None:
                txt = (f"Refined binary voxel evaluation (f={info.get('refined_f')}): "
                       f"c={info['refined_compliance']:.6g} V={info.get('refined_volfrac'):.4f}")
                if info.get("gap_refined") is not None:
                    txt += (f" | gap to the paper's MMA reference "
                            f"({info['gap_reference']:.6g}): {100 * info['gap_refined']:+.2f} %")
                job.append_log(txt)

    # -- physics audit (docs/AUDIT_API.md) -----------------------------------

    def _collect_audit(self, job: Job) -> None:
        """Pick up the audit the core stored in res.extra["audit"]; for a core
        whose FreeTOConfig has no `audit` field, compute it here instead.
        Never raises: a failed audit is reported in job.extra["audit_error"]."""
        if not job.audit_enabled:
            return
        extra = getattr(job.result, "extra", None) or {}
        aud = extra.get("audit")
        if not isinstance(aud, dict):
            if core_loader._config_has_field(core_loader.FreeTOConfig, "audit"):
                # core ran with audit on but produced nothing: it reports a failure itself
                job.extra["audit_error"] = extra.get("audit_error") or "core produced no audit"
                job.append_log("Warning: physics audit produced no result.")
                return
            try:
                aud = self.compute_audit(job, persist=False)
            except Exception as exc:  # noqa: BLE001
                job.extra["audit_error"] = f"{type(exc).__name__}: {exc}"
                job.append_log(f"Warning: physics audit failed: {exc}")
                return
        job.extra["audit"] = _json_safe(aud)
        job.extra.pop("audit_error", None)
        self._persist_audit(job)
        job.append_log(_format_audit_log(job.extra["audit"]))

    def compute_audit(self, job: Job, persist: bool = True) -> dict:
        """(Re)compute the audit of a finished continuum job from its in-memory
        result. Raises RuntimeError if the module is unavailable / no result."""
        if not core_loader._load_audit():
            raise RuntimeError(core_loader.audit_unavailable_message())
        if job.result is None or job.cfg is None:
            raise RuntimeError("job has no result to audit")
        rep = _json_safe(core_loader.audit_result(job.result, job.cfg))
        if persist:
            job.extra["audit"] = rep
            job.extra.pop("audit_error", None)
            job.audit_enabled = True
            self._invalidate_audit_png(job)
            self._persist_audit(job)
        return rep

    def _persist_audit(self, job: Job) -> None:
        try:
            d = self.jobs_dir / job.id
            d.mkdir(parents=True, exist_ok=True)
            (d / "audit.json").write_text(json.dumps(job.extra["audit"], indent=1), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def _invalidate_audit_png(self, job: Job) -> None:
        try:
            (self.jobs_dir / job.id / "audit.png").unlink()
        except OSError:
            pass

    def audit_png_path(self, job: Job) -> Path:
        """Lazily render + cache the overlay figure; raises on failure."""
        if not core_loader._load_audit():
            raise RuntimeError(core_loader.audit_unavailable_message())
        path = self.jobs_dir / job.id / "audit.png"
        with _AUDIT_FIG_LOCK:  # matplotlib is not thread-safe
            if path.is_file() and path.stat().st_size > 0:
                return path
            if job.result is None or job.cfg is None:
                raise RuntimeError("job has no result to draw")
            path.parent.mkdir(parents=True, exist_ok=True)
            core_loader.audit_figure(job.result, job.cfg, str(path), title=job.label)
            if not path.is_file():
                raise RuntimeError("audit_figure did not write a file")
        return path

    def _run_truss(self, job: Job):
        spec = job.run_spec
        benchmark_id = spec["benchmark_id"]
        method = spec["method"]

        def callback(info: dict):
            info = dict(info)
            if info.get("stage") == "iter":
                job.stage = "iter"
                job.last_iter = info.get("iter", job.last_iter)
                for key in ("compliance", "volume", "volume_fraction"):
                    if key in info:
                        job.history.setdefault(key, []).append(info[key])
                qubo = info.get("qubo")
                if qubo:
                    job.history.setdefault("qubo", []).append(qubo)
                    job.append_log(_format_qubo_log(job.last_iter, qubo))

        try:
            prob = core_loader.get_benchmark(benchmark_id)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unknown truss benchmark {benchmark_id!r}: {exc}") from exc

        job.setup = {
            "benchmark_id": benchmark_id,
            "method": method,
            "backend": spec.get("backend"),
            "problem": prob.to_dict() if hasattr(prob, "to_dict") else None,
        }
        job.append_log(f"Running truss benchmark {benchmark_id!r} with method={method!r}"
                       + (f" backend={spec['backend']!r}" if spec.get("backend") else ""))

        kwargs = dict(spec.get("options") or {})
        if spec.get("backend"):
            kwargs["backend"] = spec["backend"]
        if spec.get("seed") is not None:
            kwargs["seed"] = spec["seed"]

        result = core_loader.solve_truss(
            prob, method=method, callback=callback, stop_event=job.stop_event, **kwargs
        )
        job.result = result
        data = result.to_dict() if hasattr(result, "to_dict") else result
        job.extra["truss_result"] = data
        job.status = "stopped" if job.stop_event.is_set() else "done"
        job.stage = job.status
        if isinstance(data, dict):
            job.append_log(
                f"Done: compliance={data.get('compliance')} "
                f"volume_fraction={data.get('volume_fraction')} gap={data.get('gap')} "
                f"feasible={data.get('feasible')}"
            )

    def _run_study(self, job: Job):
        spec = job.run_spec["spec"]
        out_dir = job.run_spec["out_dir"]
        figures = job.run_spec.get("figures", True)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        def callback(ev: dict):
            ev = dict(ev)
            event = ev.get("event")
            if event == "start":
                job.progress = {"index": 0, "n_runs": ev.get("n_runs", 0), "label": None, "study": None}
                job.append_log(f"Study {ev.get('suite')!r} started: {ev.get('n_runs')} runs")
            elif event == "run_start":
                # qaoa_scan runs carry no label/study: fall back to the run id.
                name = ev.get("label") or ev.get("run_id") or "run"
                study = ev.get("study") or "-"
                # 1-based "run i of n" (the core's event index is 0-based).
                pos = int(ev.get("index") or 0) + 1
                job.progress = {
                    "index": pos, "n_runs": ev.get("n_runs"),
                    "label": name, "study": study,
                }
                job.stage = "running"
                job.last_iter = pos
                job.append_log(f"[{pos}/{ev.get('n_runs')}] {study} {name} — starting")
            elif event == "run_end":
                record = ev.get("record") or {}
                job.history.setdefault("compliance", []).append(record.get("compliance"))
                job.history.setdefault("gap", []).append(record.get("gap"))
                err = record.get("error")
                parts = [f"native={_fmt_num(record.get('compliance'))}"]
                if record.get("crisp_compliance") is not None:
                    parts.append(f"crisp={_fmt_num(record.get('crisp_compliance'))}")
                # Continuum gaps are only filled in by the study's post-processing
                # (after the last run); a missing gap is simply not printed.
                if record.get("gap") is not None:
                    parts.append(f"gap={100 * float(record['gap']):.2f}%")
                if record.get("feasible") is False:
                    parts.append("infeasible")
                if record.get("audit_ok") is not None:
                    parts.append(
                        f"audit={'PASS' if record['audit_ok'] else 'FAIL (physically invalid)'}"
                        + (f" comps={record['n_components']}" if record.get("n_components") is not None else "")
                        + (f" floating={100 * float(record['floating_frac']):.1f}%"
                           if record.get("floating_frac") is not None else ""))
                job.append_log(
                    f"[{int(ev.get('index') or 0) + 1}/{ev.get('n_runs')}] {'ERROR' if err else 'done'}: "
                    + (f"{err}" if err else " ".join(parts))
                )
            elif event == "figure":
                job.append_log(f"Figure written: {ev.get('path')}")
            elif event == "done":
                job.extra["study_results"] = ev.get("results")

        results = core_loader.run_study(
            spec, callback=callback, out_dir=out_dir, stop_event=job.stop_event, figures=figures
        )
        job.result = results
        job.extra["study_results"] = results
        job.extra["out_dir"] = out_dir
        job.status = "stopped" if job.stop_event.is_set() else "done"
        job.stage = job.status
        n_runs = None
        if isinstance(results, dict):
            n_runs = len(results.get("records", []) or [])
        job.append_log(f"Study finished: {n_runs} runs." if n_runs is not None else "Study finished.")

    def get_evaluated_stl(self, job: Job) -> Optional[bytes]:
        """Binary STL of the evaluated (crisp) design of a finished continuum
        job, built once and cached in the job directory."""
        path = self.jobs_dir / job.id / "evaluated.stl"
        with _EVAL_STL_LOCK:
            if path.is_file() and path.stat().st_size > 84:
                return path.read_bytes()
            surf = evaluated_surface(job.result)
            if surf is None:
                return None
            data = _mesh_to_binary_stl(*surf)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            except OSError:
                pass
            return data

    def _save_result(self, job: Job):
        if job.result is None:
            return
        job_dir = self.jobs_dir / job.id
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            _save_fields(job_dir / "fields.npz", job.result)
        except Exception as exc:  # noqa: BLE001
            job.append_log(f"Warning: failed to write fields NPZ: {exc}")
        try:
            job.result.write_stl(str(job_dir / "result.stl"))
        except Exception as exc:  # noqa: BLE001
            job.append_log(f"Warning: failed to write result STL: {exc}")
        try:
            job.result.save_npz(str(job_dir / "result.npz"))
        except Exception as exc:  # noqa: BLE001
            job.append_log(f"Warning: failed to write result NPZ: {exc}")


_AUDIT_FIG_LOCK = threading.Lock()
_EVAL_STL_LOCK = threading.Lock()


def _thread_pin():
    """Context manager pinning the thread pools to FREETO_THREADS (default 1;
    "0" or "none" = leave them alone) with the study's own helper."""
    import contextlib
    n = os.environ.get("FREETO_THREADS", "1").strip().lower()
    if n in ("0", "none", ""):
        return contextlib.nullcontext(None)
    try:
        from freeto.study import _ThreadPin
    except Exception:  # noqa: BLE001 -- stub core / old core
        return contextlib.nullcontext(None)
    return _ThreadPin(int(n))


def _pin_text(pin) -> str:
    info = getattr(pin, "info", None) or {}
    pools = ", ".join(f"{p.get('api')}={p.get('num_threads')}" for p in info.get("pools") or [])
    return f"{info.get('requested')} per pool ({info.get('method')}" + (f": {pools})" if pools else ")")


def _save_fields(path, res) -> None:
    """Same layout as the study's fields/<run_id>.npz (freeto.study._save_fields):
    full_pre (float32), grid, binary_design (uint8, QUBO / BESO), history_*."""
    ex = getattr(res, "extra", None) or {}
    fp = ex.get("full_pre")
    si = getattr(res, "setup_info", None) or {}
    arrs = {"full_pre": np.asarray(fp if fp is not None else [], dtype=np.float32),
            "grid": np.array([si.get("nelx") or 0, si.get("nely") or 0, si.get("nelz") or 0],
                             dtype=np.int64)}
    if ex.get("binary_design") is not None:
        arrs["binary_design"] = (np.asarray(ex["binary_design"]) > 0.5).astype(np.uint8)
    for k, v in (getattr(res, "history", None) or {}).items():
        arrs[f"history_{k}"] = np.asarray(v, dtype=np.float64)
    np.savez_compressed(str(path), **arrs)


NGRID = 4  # FreeTO fine-grid factor (freeto.core.NGRID)


def evaluated_surface(res):
    """(vertices, faces) of the evaluated binary design of a continuum result:
    the final pre-smoothing field interpolated to the fine grid and projected
    crisp at the target volume fraction (the crisp evaluation's threshold),
    restricted to the active elements, lightly smoothed (Gaussian, 0.2 h) and
    contoured at 0.5 -- the rendering of the paper's Fig. 5 -- then mirrored
    like the STL output.  None if the result carries no evaluated field."""
    ctx = getattr(res, "audit_ctx", None) or {}
    full_pre = ctx.get("full_pre")
    if full_pre is None:
        return None
    from scipy.ndimage import gaussian_filter
    from freeto.evaluate import crisp_projection, fine_field
    from freeto.mesh import matlab_to_xyz
    from freeto.postprocess import FieldSnapshot, apply_symmetry
    nelx, nely, nelz = int(ctx["nelx"]), int(ctx["nely"]), int(ctx["nelz"])
    ele = np.asarray(ctx["ele"])
    Hn, Hns = ctx["Hn"], ctx["Hns"]
    extra = getattr(res, "extra", None) or {}
    thr = extra.get("crisp_threshold")
    if thr is None or not np.isfinite(thr):
        target = float(extra.get("crisp_target") or ctx.get("volfrac"))
        thr = crisp_projection(full_pre, Hn, Hns, nelx, nely, nelz, ele, target, NGRID)[1]
    xg = fine_field(full_pre, Hn, Hns, nelx, nely, nelz, NGRID)
    act = np.zeros(nelx * nely * nelz, dtype=bool)
    act[ele] = True
    act = act.reshape((nely, nelx, nelz), order="F")
    win = np.zeros(xg.shape, dtype=bool)
    for a in range(NGRID + 1):
        for b in range(NGRID + 1):
            for c in range(NGRID + 1):
                win[a:a + NGRID * nely:NGRID, b:b + NGRID * nelx:NGRID,
                    c:c + NGRID * nelz:NGRID] |= act
    solid = (xg > float(thr)) & win
    vol = gaussian_filter(solid.astype(np.float32), 0.8, mode="nearest")
    h = float(ctx["h"])
    fld = FieldSnapshot(np.ascontiguousarray(matlab_to_xyz(vol - 0.5)),
                        np.asarray(ctx["origin"], dtype=float), np.full(3, h / NGRID))
    cfg = getattr(res, "config", None)
    for plane, direction in (getattr(cfg, "symmetry", None) or []):
        fld = apply_symmetry(fld, plane, direction)
    return surface_from_field(fld, smooth=False)


def _native_info(res, cfg) -> dict:
    """Native compliance / volume with the study record's semantics
    (freeto.study._run_continuum): QUBO -- the returned design; OC / MMA -- the
    compliance of the iterate entering the last iteration and its volume."""
    hv = [float(v) for v in (getattr(res, "history", None) or {}).get("volfrac", [])]
    opt = str(getattr(cfg, "optimizer", "OC")).upper()
    v_nat = hv[-2] if (opt != "QUBO" and len(hv) >= 2) else float(res.finalvol)
    out = {"native_compliance": float(res.comp), "native_volfrac": v_nat,
           "iterations": int(res.iterations), "volfrac_target": float(getattr(cfg, "volfrac", 0))}
    return {k: (v if not isinstance(v, float) or math.isfinite(v) else None)
            for k, v in out.items()}


def _paper_gap(info: dict, paper: Optional[dict]) -> None:
    """Refined gap to the paper's MMA reference (same problem, f = 2 only)."""
    ref = (paper or {}).get("gap_reference")
    c = info.get("refined_compliance")
    if ref and c is not None:
        info["gap_reference"] = float(ref)
        info["gap_refined"] = float(c) / float(ref) - 1.0


def _json_safe(obj):
    """Round-trip through JSON (numpy scalars/arrays -> lists, nan/inf -> None)."""
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, np.generic):
            return clean(o.item())
        if isinstance(o, float):
            return o if math.isfinite(o) else None
        return o
    return clean(obj)


def _format_audit_log(aud: dict) -> str:
    bad = [k for k, v in (aud.get("checks") or {}).items()
           if isinstance(v, dict) and v.get("pass") is False]
    txt = f"Physics check: {'PASS' if aud.get('ok') else 'FAIL'}"
    if aud.get("n_components") is not None:
        txt += f" | components={aud['n_components']}"
    if aud.get("floating_frac") is not None:
        txt += f" floating={100 * float(aud['floating_frac']):.2f}%"
    if bad:
        txt += " | failed: " + ", ".join(bad)
    return txt


_RESULT_INFO_KEYS = ("crisp_compliance", "crisp_volfrac", "crisp_target", "beta_final",
                     "compliance_returned_design", "returned_design", "binary_compliance",
                     "binary_volfrac", "eval_errors", "refined_compliance", "refined_volfrac",
                     "refined_f", "refined_threshold", "compliance_at_beta", "volfrac_at_beta",
                     "eval_beta", "best_iteration", "mma_constraint", "mma_fval_final")


def _result_info(extra) -> dict:
    """JSON-safe scalar subset of FreeTOResult.extra shown by the UI (inf/nan -> None)."""
    out: dict = {}
    if not isinstance(extra, dict):
        return out
    for k in _RESULT_INFO_KEYS:
        v = extra.get(k)
        if v is None:
            continue
        if isinstance(v, bool) or isinstance(v, str):
            out[k] = v
        elif isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            out[k] = float(v) if math.isfinite(float(v)) else None
        elif isinstance(v, (dict, list)):
            out[k] = str(v)
    return out


def _fmt_num(v) -> str:
    try:
        return "n/a" if v is None else f"{float(v):.5g}"
    except (TypeError, ValueError):
        return str(v)


def _format_qubo_log(iter_no: int, qubo: dict) -> str:
    """One log line summarising a QUBO-mode iteration's solver stats (n_free,
    n_blocks, solver time, QPU time, QAOA approx ratio — see
    docs/QUANTUM_API.md's `iter` callback `qubo` dict), for the per-iteration
    log pane and (via the status bar reading the last such line) the header."""
    parts = [f"iter {iter_no} [qubo]"]
    if qubo.get("backend") is not None:
        parts.append(f"backend={qubo['backend']}")
    if qubo.get("n_free") is not None:
        parts.append(f"n_free={qubo['n_free']}")
    if qubo.get("n_blocks") is not None:
        parts.append(f"n_blocks={qubo['n_blocks']}")
    if qubo.get("n_solves") is not None:
        parts.append(f"n_solves={qubo['n_solves']}")
    st = qubo.get("solver_time")
    if st is not None:
        parts.append(f"solver_time={st:.3f}s")
    qt = qubo.get("qpu_time")
    if qt is not None:
        parts.append(f"qpu_time={qt:.3f}s")
    ar = qubo.get("approx_ratio")
    if ar is not None:
        parts.append(f"approx_ratio={ar:.3f}")
    return " ".join(parts)


def _decimate_field(field, max_cells: int):
    """Downsample a FieldSnapshot-like object (duck-typed: needs .top,
    .origin, .spacing) for a cheap preview, by simple strided subsampling —
    this is pure array indexing (no recompute), so it's effectively free
    compared to marching_cubes on the full grid. The final result download
    always uses the untouched, full-resolution field."""
    top = np.asarray(field.top)
    n_cells = top.size
    if n_cells <= max_cells:
        return field
    stride = int(np.ceil((n_cells / max_cells) ** (1.0 / 3.0)))
    stride = max(stride, 2)
    small_top = np.ascontiguousarray(top[::stride, ::stride, ::stride])
    small_spacing = np.asarray(field.spacing, dtype=np.float64) * stride
    return _DecimatedField(small_top, np.asarray(field.origin, dtype=np.float64), small_spacing)


class _DecimatedField:
    """Minimal duck-typed stand-in for a FieldSnapshot (top/origin/spacing),
    independent of whichever core (real or stub) produced the original
    field, so decimation works the same regardless of core."""

    __slots__ = ("top", "origin", "spacing")

    def __init__(self, top, origin, spacing):
        self.top = top
        self.origin = origin
        self.spacing = spacing


_STL_TRIANGLE_DTYPE = np.dtype([
    ("normal", "<f4", (3,)),
    ("v", "<f4", (3, 3)),
    ("attr", "<u2"),
])


def _mesh_to_binary_stl(vertices, faces) -> bytes:
    """Vectorised binary STL encoder (numpy structured array, no per-face
    Python loop) — writing ~500k triangles this way takes ~tens of ms
    instead of several seconds."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    m = faces.shape[0]
    arr = np.zeros(m, dtype=_STL_TRIANGLE_DTYPE)
    if m:
        v0s = vertices[faces[:, 0]]
        v1s = vertices[faces[:, 1]]
        v2s = vertices[faces[:, 2]]
        normals = np.cross(v1s - v0s, v2s - v0s)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr["normal"] = (normals / norms).astype(np.float32)
        arr["v"][:, 0, :] = v0s
        arr["v"][:, 1, :] = v1s
        arr["v"][:, 2, :] = v2s
    buf = io.BytesIO()
    buf.write(b"\x00" * 80)
    buf.write(struct.pack("<I", m))
    buf.write(arr.tobytes())
    return buf.getvalue()
