"""Stand-in for ``freeto.audit`` with the docs/AUDIT_API.md dict shape.

Only used by the web-app tests, to exercise the web app's own audit plumbing
(routing, caching, status fields, UI) independently of -- and before -- the real
module.  It never replaces the real audit in the app itself."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CHECK_NAMES = ("bc_supports_present", "supports_solid", "loads_solid", "single_grounded_body",
               "load_on_main", "keep_regions_captured", "volume_target")


def tiny_png() -> bytes:
    raw = b"\x00\xff\x00\x00"  # 1x1 RGB red pixel, filter 0

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def fake_report(res, cfg, ok=True):
    checks = {n: {"pass": True, "value": 1.0, "detail": f"fake {n}"} for n in CHECK_NAMES}
    if not ok:
        checks["single_grounded_body"] = {"pass": False, "value": {"n_components": 3},
                                          "detail": "floating"}
    return {
        "ok": ok, "checks": checks, "n_components": 1 if ok else 3,
        "floating_frac": 0.0 if ok else 0.2, "crisp_compliance": None,
        "native_compliance": float(res.comp), "volume_fraction": float(res.finalvol),
        "components": [{"label": 1, "n_elements": 10, "grounded": True, "has_load": True,
                        "bbox_mm": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]}],
    }


def install(core_loader):
    """Patch webapp.core_loader *and* the ``freeto.audit`` module (the core's run_freeto
    imports it when ``audit=True``); returns (calls, restore)."""
    import sys
    import types

    calls = {"report": 0, "figure": 0, "ok": True}

    def audit_result(res, cfg):
        calls["report"] += 1
        return fake_report(res, cfg, calls["ok"])

    def audit_figure(res, cfg, path, title=None):
        calls["figure"] += 1
        Path(path).write_bytes(tiny_png())
        return path

    saved_cl = {k: getattr(core_loader, k) for k in ("AUDIT_USABLE", "audit_result", "audit_figure")}
    had_mod = "freeto.audit" in sys.modules
    saved_mod = sys.modules.get("freeto.audit")
    import freeto
    saved_attr = getattr(freeto, "audit", None)

    fake_mod = types.ModuleType("freeto.audit")
    fake_mod.audit_result = audit_result
    fake_mod.audit_figure = audit_figure
    sys.modules["freeto.audit"] = fake_mod
    freeto.audit = fake_mod
    core_loader.AUDIT_USABLE = True
    core_loader.audit_result = audit_result
    core_loader.audit_figure = audit_figure

    def restore():
        for k, v in saved_cl.items():
            setattr(core_loader, k, v)
        if had_mod:
            sys.modules["freeto.audit"] = saved_mod
        else:
            sys.modules.pop("freeto.audit", None)
        if saved_attr is not None:
            freeto.audit = saved_attr
        elif hasattr(freeto, "audit"):
            del freeto.audit

    return calls, restore
