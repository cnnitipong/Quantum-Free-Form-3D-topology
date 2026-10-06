"""
Loads the FreeTO-Python numerical core.

By default this ONLY uses the real package (`freeto`, written separately in
freeto_py/freeto/) — there is no silent fallback to a fake solver. If the
real core isn't importable, the web app still starts (so `/api/health` and
the UI can show *why*), but `CORE_USABLE` is False, `EXAMPLES` is empty, and
`webapp/server.py` refuses to create jobs with a clear 503.

Setting the environment variable FREETO_WEB_STUB=1 opts into the built-in
fake "shrinking sphere" solver (`_stub_core.py`) on purpose, for developing
or testing the web app itself without the real core. This is never chosen
automatically.

Every other webapp module should import the core through this module:

    from webapp.core_loader import FreeTOConfig, run_freeto, EXAMPLES, read_stl, \
        surface_from_field, CORE_SOURCE, CORE_USABLE, CORE_IMPORT_ERROR, core_status
"""

from __future__ import annotations

import logging
import os
import traceback
from typing import Optional

from webapp import _stub_core as _stub_core_module
from webapp._stub_core import read_stl as _stub_read_stl

logger = logging.getLogger("freeto.webapp")

STUB_ENV_VAR = "FREETO_WEB_STUB"

# read_stl is harmless file-parsing (triangle count / bbox for the upload
# registry) rather than "solving" anything, so it's always available even
# when the real core can't be used — this default is overridden below the
# moment the real core imports successfully.
read_stl = _stub_read_stl

FreeTOConfig = None
run_freeto = None
surface_from_field = None
EXAMPLES: dict = {}

CORE_SOURCE = "unavailable"   # "real" | "stub" | "unavailable"
CORE_USABLE = False
CORE_IMPORT_ERROR = None
_core_module = None

_force_stub = os.environ.get(STUB_ENV_VAR, "") == "1"

if _force_stub:
    _core_module = _stub_core_module
    FreeTOConfig = _stub_core_module.FreeTOConfig
    run_freeto = _stub_core_module.run_freeto
    EXAMPLES = _stub_core_module.EXAMPLES
    read_stl = _stub_core_module.read_stl
    surface_from_field = _stub_core_module.postprocess.surface_from_field
    CORE_SOURCE = "stub"
    CORE_USABLE = True
    logger.warning(
        "FREETO_WEB_STUB=1: intentionally using the built-in fake solver "
        "(shrinking-sphere stub, NOT a real topology optimizer). This is "
        "meant for webapp development/tests only."
    )
else:
    try:
        import freeto as _real_core  # type: ignore
        from freeto import FreeTOConfig as _FreeTOConfig  # type: ignore
        from freeto import run_freeto as _run_freeto  # type: ignore
        from freeto import EXAMPLES as _EXAMPLES  # type: ignore
        from freeto import read_stl as _read_stl  # type: ignore
        from freeto import postprocess as _postprocess  # type: ignore

        _core_module = _real_core
        FreeTOConfig = _FreeTOConfig
        run_freeto = _run_freeto
        EXAMPLES = _EXAMPLES
        read_stl = _read_stl
        surface_from_field = _postprocess.surface_from_field
        CORE_SOURCE = "real"
        CORE_USABLE = True
        logger.info("Using real freeto core (%s)", getattr(_real_core, "__file__", "?"))
    except Exception as exc:  # noqa: BLE001 - the real core may be mid-development
        CORE_IMPORT_ERROR = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        CORE_SOURCE = "unavailable"
        CORE_USABLE = False
        logger.error(
            "The real 'freeto' core could not be imported (%s). The web app "
            "will start, but refuses to run jobs — there is no silent "
            "fallback to the fake stub solver. Set FREETO_WEB_STUB=1 to opt "
            "into the stub explicitly for webapp development/testing.",
            CORE_IMPORT_ERROR,
        )


# Whether the *installed* freeto core's FreeTOConfig actually has a `qubo`
# field yet — freeto/quantum/ (backends, options) can be ready before
# freeto/core.py's run_freeto loop is wired to use them (see
# docs/QUANTUM_API.md §1). Both must be true for the continuum "QUBO"
# optimizer to be offered in the Setup panel; the Truss tab and Study tab
# only need freeto.truss / freeto.study respectively, independent of this.
def _config_has_field(cls, name: str) -> bool:
    if cls is None:
        return False
    try:
        import dataclasses
        return any(f.name == name for f in dataclasses.fields(cls))
    except Exception:  # noqa: BLE001
        return False


CORE_SUPPORTS_QUBO = CORE_USABLE and _config_has_field(FreeTOConfig, "qubo")


def core_status() -> dict:
    return {
        "source": CORE_SOURCE,
        "usable": CORE_USABLE,
        "module_file": getattr(_core_module, "__file__", None),
        "import_error": CORE_IMPORT_ERROR,
        "stub_forced": _force_stub,
    }


# ---------------------------------------------------------------------------
# Quantum / QUBO extension (freeto.quantum) — optional, feature-detected.
#
# An Opus engineer implements freeto/quantum/, freeto/truss/ and freeto/study.py
# separately, against docs/QUANTUM_API.md. This module NEVER assumes any of
# them exist: every import is wrapped, and every *_USABLE flag defaults to
# False so the web app degrades gracefully (hide the controls, show a plain
# notice) when they're missing or only partially implemented.
# ---------------------------------------------------------------------------

QUANTUM_USABLE = False
QUANTUM_IMPORT_ERROR: Optional[str] = None
QUBOOptions = None
solve_qubo = None
available_backends = None
QuantumBackendUnavailable = Exception  # harmless fallback type for `except` clauses

try:
    from freeto.quantum import QUBOOptions as _QUBOOptions  # type: ignore
    from freeto.quantum import solve_qubo as _solve_qubo  # type: ignore
    from freeto.quantum import available_backends as _available_backends  # type: ignore
    from freeto.quantum import QuantumBackendUnavailable as _QBUnavailable  # type: ignore

    QUBOOptions = _QUBOOptions
    solve_qubo = _solve_qubo
    available_backends = _available_backends
    QuantumBackendUnavailable = _QBUnavailable
    QUANTUM_USABLE = True
    logger.info("freeto.quantum is available (QUBO/quantum backends enabled).")
except Exception as exc:  # noqa: BLE001 - freeto.quantum may not exist yet / be mid-development
    QUANTUM_IMPORT_ERROR = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    QUANTUM_USABLE = False
    logger.info(
        "freeto.quantum is not available (%s); QUBO/quantum controls will be hidden.",
        QUANTUM_IMPORT_ERROR,
    )


def quantum_status() -> dict:
    return {"usable": QUANTUM_USABLE, "import_error": QUANTUM_IMPORT_ERROR}


def quantum_backends_payload() -> dict:
    """GET /api/quantum/backends payload. Never raises."""
    if not QUANTUM_USABLE:
        return {"quantum_available": False, "import_error": QUANTUM_IMPORT_ERROR, "backends": []}
    try:
        backends = available_backends()
    except Exception as exc:  # noqa: BLE001
        return {
            "quantum_available": False,
            "import_error": f"available_backends() raised: {exc}",
            "backends": [],
        }
    return {"quantum_available": True, "import_error": None, "backends": backends}


QUANTUM_UNAVAILABLE_MESSAGE = (
    "The 'freeto.quantum' module is not available, so QUBO/quantum jobs cannot run. "
    "{detail}Install the optional extras with `pip install -r requirements-quantum.txt` for "
    "cloud backends (D-Wave / IBM); the classical/simulated backends (exact, sa, tabu, qaoa) "
    "need no extra install once freeto.quantum itself is present."
)


def quantum_unavailable_message() -> str:
    detail = f"Import error: {QUANTUM_IMPORT_ERROR} " if QUANTUM_IMPORT_ERROR else ""
    return QUANTUM_UNAVAILABLE_MESSAGE.format(detail=detail)


# ---------------------------------------------------------------------------
# Truss ground-structure module (freeto.truss) — same feature-detection.
# ---------------------------------------------------------------------------

TRUSS_USABLE = False
TRUSS_IMPORT_ERROR: Optional[str] = None
TrussProblem = None
list_benchmarks = None
get_benchmark = None
solve_truss = None
TrussResult = None
TRUSS_METHODS: list = []

try:
    from freeto.truss import TrussProblem as _TrussProblem  # type: ignore
    from freeto.truss import list_benchmarks as _list_benchmarks  # type: ignore
    from freeto.truss import get_benchmark as _get_benchmark  # type: ignore
    from freeto.truss import solve_truss as _solve_truss  # type: ignore
    from freeto.truss import TrussResult as _TrussResult  # type: ignore
    from freeto.truss import TRUSS_METHODS as _TRUSS_METHODS  # type: ignore

    TrussProblem = _TrussProblem
    list_benchmarks = _list_benchmarks
    get_benchmark = _get_benchmark
    solve_truss = _solve_truss
    TrussResult = _TrussResult
    TRUSS_METHODS = list(_TRUSS_METHODS)
    TRUSS_USABLE = True
    logger.info("freeto.truss is available (truss benchmarks enabled).")
except Exception as exc:  # noqa: BLE001
    TRUSS_IMPORT_ERROR = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    TRUSS_USABLE = False
    logger.info(
        "freeto.truss is not available (%s); the Truss tab will show a notice instead.",
        TRUSS_IMPORT_ERROR,
    )


def truss_status() -> dict:
    return {"usable": TRUSS_USABLE, "import_error": TRUSS_IMPORT_ERROR}


def truss_benchmarks_payload() -> dict:
    if not TRUSS_USABLE:
        return {"truss_available": False, "import_error": TRUSS_IMPORT_ERROR, "benchmarks": [], "methods": []}
    try:
        benches = list_benchmarks()
    except Exception as exc:  # noqa: BLE001
        return {
            "truss_available": False,
            "import_error": f"list_benchmarks() raised: {exc}",
            "benchmarks": [],
            "methods": [],
        }
    return {"truss_available": True, "import_error": None, "benchmarks": benches, "methods": TRUSS_METHODS}


TRUSS_UNAVAILABLE_MESSAGE = (
    "The 'freeto.truss' module is not available, so truss benchmarks cannot run. {detail}"
)


def truss_unavailable_message() -> str:
    detail = f"Import error: {TRUSS_IMPORT_ERROR} " if TRUSS_IMPORT_ERROR else ""
    return TRUSS_UNAVAILABLE_MESSAGE.format(detail=detail)


# ---------------------------------------------------------------------------
# Study runner (freeto.study) — same feature-detection.
# ---------------------------------------------------------------------------

STUDY_USABLE = False
STUDY_IMPORT_ERROR: Optional[str] = None
run_study = None
suite_spec = None
SUITES: list = []

try:
    from freeto.study import run_study as _run_study  # type: ignore
    from freeto.study import suite_spec as _suite_spec  # type: ignore
    from freeto.study import SUITES as _SUITES  # type: ignore

    run_study = _run_study
    suite_spec = _suite_spec
    SUITES = list(_SUITES)
    STUDY_USABLE = True
    logger.info("freeto.study is available (Study tab enabled).")
except Exception as exc:  # noqa: BLE001
    STUDY_IMPORT_ERROR = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    STUDY_USABLE = False
    logger.info(
        "freeto.study is not available (%s); the Study tab will show a notice instead.",
        STUDY_IMPORT_ERROR,
    )


def study_status() -> dict:
    return {"usable": STUDY_USABLE, "import_error": STUDY_IMPORT_ERROR}


STUDY_UNAVAILABLE_MESSAGE = (
    "The 'freeto.study' module is not available, so study suites cannot run. {detail}"
)


def study_unavailable_message() -> str:
    detail = f"Import error: {STUDY_IMPORT_ERROR} " if STUDY_IMPORT_ERROR else ""
    return STUDY_UNAVAILABLE_MESSAGE.format(detail=detail)


# ---------------------------------------------------------------------------
# Physics / connectivity audit (freeto.audit, docs/AUDIT_API.md) — same
# feature-detection: until the module exists the audit endpoints answer 503
# "audit module not available" and everything else keeps working.
# ---------------------------------------------------------------------------

AUDIT_USABLE = False
AUDIT_IMPORT_ERROR: Optional[str] = None
audit_result = None
audit_figure = None


def _load_audit():
    """(Re-)import freeto.audit; called at import time and lazily by the
    endpoints, so an audit module that appears while the server is running
    (or after `importlib.invalidate_caches`) is picked up without restart."""
    global AUDIT_USABLE, AUDIT_IMPORT_ERROR, audit_result, audit_figure
    if AUDIT_USABLE:
        return True
    if not CORE_USABLE or _force_stub:
        AUDIT_IMPORT_ERROR = "real freeto core not in use" if not CORE_USABLE else "stub core has no audit"
        return False
    try:
        import importlib
        importlib.invalidate_caches()
        mod = importlib.import_module("freeto.audit")
        audit_result = mod.audit_result
        audit_figure = mod.audit_figure
        AUDIT_USABLE = True
        AUDIT_IMPORT_ERROR = None
        logger.info("freeto.audit is available (physics check enabled).")
    except Exception as exc:  # noqa: BLE001
        AUDIT_IMPORT_ERROR = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        AUDIT_USABLE = False
    return AUDIT_USABLE


_load_audit()


def audit_status() -> dict:
    _load_audit()
    return {"usable": AUDIT_USABLE, "import_error": AUDIT_IMPORT_ERROR,
            "config_field": _config_has_field(FreeTOConfig, "audit")}


def audit_unavailable_message() -> str:
    detail = f" ({AUDIT_IMPORT_ERROR})" if AUDIT_IMPORT_ERROR else ""
    return f"audit module not available{detail}"


UNAVAILABLE_MESSAGE = (
    "The real FreeTO numerical core is not available, so jobs cannot run "
    "(the web app does not silently fall back to a fake solver). "
    "{detail}"
    "If you are developing the web app itself and want the built-in fake "
    "stub solver on purpose, restart the server with FREETO_WEB_STUB=1."
)


def unavailable_message() -> str:
    detail = f"Import error: {CORE_IMPORT_ERROR} " if CORE_IMPORT_ERROR else ""
    return UNAVAILABLE_MESSAGE.format(detail=detail)
