"""freeto.audit (docs/AUDIT_API.md) and the setup validation / example
corrections of 2026-10-02 (docs/PHYSICS_AUDIT.md, docs/NOTES_quantum.md §8)."""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from freeto import run_freeto                                   # noqa: E402
from freeto.audit import LiveAudit, audit_figure, audit_result  # noqa: E402
from freeto.core import FreeTOConfig                            # noqa: E402
from freeto.examples import example_config                     # noqa: E402
from freeto.geometry import box, l_shape, write_mesh           # noqa: E402
from freeto.mesh import prepare_domain                         # noqa: E402

CHECKS = {"bc_supports_present", "supports_solid", "loads_solid", "single_grounded_body",
          "load_on_main", "keep_regions_captured", "volume_target"}


@pytest.fixture(scope="module")
def mma_cantilever():
    cfg = example_config("cantilever_beam", mesh_control=25, optimizer="MMA", max_iter=100,
                         eval_crisp=True)
    return cfg, run_freeto(cfg, log=None)


def test_audit_mma_cantilever_passes(mma_cantilever):
    cfg, res = mma_cantilever
    rep = res.extra["audit"]
    assert set(rep["checks"]) == CHECKS
    assert rep["ok"], rep["failed"]
    assert rep["n_components"] == 1 and rep["floating_frac"] < 1e-9
    assert rep["components"][0]["grounded"] and rep["components"][0]["has_load"]
    assert rep["crisp_compliance"] == pytest.approx(res.extra["crisp_compliance"])
    assert abs(rep["crisp_volfrac"] - 0.3) < 2e-3
    assert rep["checks"]["bc_supports_present"]["value"]["rigid_body_modes_restrained"] == 6
    json.dumps(rep)                                          # JSON-serialisable
    # recomputing (also without the stored threshold) gives the same verdict
    r2 = copy.copy(res)
    r2.extra = {k: v for k, v in res.extra.items() if not k.startswith("crisp_")}
    rep2 = audit_result(r2, cfg)
    assert rep2["ok"] and abs(rep2["crisp_volfrac"] - rep["crisp_volfrac"]) < 2e-3


def test_audit_rebuilds_setup_from_config(mma_cantilever):
    cfg, res = mma_cantilever
    r2 = copy.copy(res)
    r2.audit_ctx = None
    rep = audit_result(r2, cfg)
    assert "rebuilt" in rep["field_source"]
    assert rep["checks"]["keep_regions_captured"]["pass"]
    assert rep["checks"]["single_grounded_body"]["pass"]


def test_audit_is_fast(mma_cantilever):
    cfg, res = mma_cantilever
    t = time.perf_counter()
    audit_result(res, cfg)
    assert time.perf_counter() - t < 1.0


def _synthetic(res, floating):
    """res with a hand-made pre-smoothing field: a solid bar from the
    support face to the load (rows around mid-height), optionally plus a
    solid block floating in the middle of the void."""
    ctx = dict(res.audit_ctx)
    nelx, nely, nelz = ctx["nelx"], ctx["nely"], ctx["nelz"]
    full = np.zeros((nely, nelx, nelz))
    full[2:6, :, :] = 1.0                      # bar y ~ 10-30 mm along the whole span
    full[2:6, 8:12, :] = 0.0                   # ... with a gap
    full[0:2, :, :] = 0.0
    if not floating:
        full[2:6, 8:12, :] = 1.0
    ctx["full_pre"] = full.ravel(order="F")
    ctx["binary_design"] = None
    r = copy.copy(res)
    r.audit_ctx = ctx
    r.extra = {}
    return r


def test_audit_disconnected_density_fails(mma_cantilever):
    cfg, res = mma_cantilever
    ok = audit_result(_synthetic(res, floating=False), cfg)
    assert ok["checks"]["single_grounded_body"]["pass"], ok["checks"]["single_grounded_body"]
    bad = audit_result(_synthetic(res, floating=True), cfg)
    assert not bad["ok"]
    assert not bad["checks"]["single_grounded_body"]["pass"]
    assert bad["n_components"] >= 2 and bad["floating_frac"] > 0.1
    assert not bad["checks"]["load_on_main"]["pass"]         # the loaded part floats
    assert any(not c["grounded"] and c["has_load"] for c in bad["components"])
    assert "single_grounded_body" in bad["failed"] and not bad["physics_ok"]


def test_audit_figure(tmp_path, mma_cantilever):
    pytest.importorskip("matplotlib")
    cfg, res = mma_cantilever
    p = audit_figure(res, cfg, str(tmp_path / "a.png"), title="cantilever MMA")
    assert os.path.getsize(p) > 10_000


def test_live_audit_counts_components():
    nelx, nely, nelz = 6, 3, 1
    ele = np.arange(nelx * nely * nelz)
    fixed_nodes = np.arange(nely + 1)                         # x = 0 face, z = 0 layer
    fixed_nodes = np.concatenate([fixed_nodes, fixed_nodes + (nelx + 1) * (nely + 1)])
    fd = np.concatenate([3 * fixed_nodes + k for k in range(3)])
    live = LiveAudit(nelx, nely, nelz, ele, fd)
    x = np.zeros((nely, nelx, nelz))
    x[1, :3, 0] = 1                                           # attached to x = 0
    x[1, 4:, 0] = 1                                           # floating
    r = live(x.ravel(order="F"))
    assert r["n_components"] == 2 and r["floating_frac"] == pytest.approx(2 / 5)
    assert r["ungrounded_frac"] == pytest.approx(2 / 5)


def test_qubo_run_reports_live_and_binary_audit():
    cfg = example_config("cantilever_beam", mesh_control=20, optimizer="QUBO", max_iter=12,
                         qubo={"backend": "sa", "hessian": "diag", "seed": 0})
    live = []
    res = run_freeto(cfg, log=None, callback=lambda i: live.append(i.get("audit_live")))
    live = [x for x in live if x]
    assert live and all({"n_components", "floating_frac"} <= set(x) for x in live)
    assert "binary" in res.extra["audit"]


def test_audit_off():
    cfg = example_config("cantilever_beam", mesh_control=20, max_iter=2, audit=False)
    res = run_freeto(cfg, log=None)
    assert "audit" not in res.extra and res.audit_ctx is not None


# ---------------------------------------------------------------------------
# example corrections + setup validation
# ---------------------------------------------------------------------------
def _domain(name, mc):
    c = example_config(name, mesh_control=mc).normalized()
    return prepare_domain(mc, c.domain, c.fixed, c.xfixed, c.yfixed, c.zfixed, c.forces,
                          c.keepdom, c.keep_bc, c.keep_bcx, c.keep_bcy, c.keep_bcz)


@pytest.mark.parametrize("mc", [24, 31, 41, 70])
def test_bridge_regions_captured(mc):
    dom = _domain("bridge_deck", mc)
    reg = dom.regions
    assert reg["keepdom"]["elements"] > 0 and reg["keepdom"]["elements_outside"] == 0
    assert reg["fixed"]["nodes"] > 0 and reg["fixed"]["elements"] > 0
    assert reg["force1"]["nodes"] > 0 and reg["force1"]["elements"] > 0
    keep = int(np.count_nonzero(dom.outeM))
    assert keep > 0
    if mc in (31, 41, 70):                    # feasible: keep below the 35 % budget
        assert keep / np.count_nonzero(dom.oute) < 0.35


@pytest.mark.parametrize("name", ["l_bracket", "mbb_beam", "multi_load_beam", "cantilever_beam",
                                  "torsion_bracket", "bridge_deck"])
@pytest.mark.parametrize("mc", [24, 30, 36])
def test_generated_examples_regions(name, mc):
    """No keep element outside the domain; every kept region holds element
    centres and every support / load region holds nodes at coarse meshes."""
    dom = _domain(name, mc)
    assert int(np.count_nonzero((dom.outeM > 0) & (dom.oute == 0))) == 0
    for role, d in dom.regions.items():
        if d["nodes"] is not None:
            assert d["nodes"] > 0, (role, d)
        if d["kept"]:
            assert d["elements"] > 0, (role, d)


def _l_case(tmp_path, keep_box):
    write_mesh(str(tmp_path / "dom.stl"), l_shape((0, 0, 0), (60, 60, 10), (25, 25, 0), (60, 60, 10)))
    write_mesh(str(tmp_path / "fix.stl"), box((-2, 55, -2), (25, 62, 12)))
    write_mesh(str(tmp_path / "load.stl"), box((55, -2, -2), (62, 12, 12)))
    write_mesh(str(tmp_path / "keep.stl"), box(*keep_box))
    return FreeTOConfig(domain=str(tmp_path / "dom.stl"), forces=[str(tmp_path / "load.stl")],
                        fixed=str(tmp_path / "fix.stl"), keepdom=str(tmp_path / "keep.stl"),
                        mesh_control=16, volfrac=0.3, fmagy=[-1.0], max_iter=1)


def test_warning_keep_outside_domain(tmp_path):
    cfg = _l_case(tmp_path, ((0, 40, -2), (60, 50, 12)))     # crosses the notch
    logs = []
    res = run_freeto(cfg, log=logs.append)
    w = [s for s in logs if "outside the design domain" in s]
    assert w and w[0].startswith("WARNING")
    assert res.setup_info["n_keep_outside_domain"] > 0
    assert res.extra["audit"]["checks"]["keep_regions_captured"]["value"]["n_keep_outside_domain"] > 0


def test_warning_empty_keepdom(tmp_path):
    cfg = _l_case(tmp_path, ((0, 59.9, -2), (25, 62, 12)))   # thinner than half an element
    logs = []
    res = run_freeto(cfg, log=logs.append)
    assert any(s.startswith("WARNING: the keepdom region") for s in logs)
    chk = res.extra["audit"]["checks"]["keep_regions_captured"]
    assert not chk["pass"] and "keepdom" in chk["detail"]
