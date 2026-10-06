"""Web app = paper: the settings the web app uses for the manuscript's five
examples equal the study's (suite "quick2" of freeto.study), and the server
forwards every option a study run sets (docs/PAPER_SETTINGS.md).

The reference configuration is captured from the study's own code path
(freeto.study._run_continuum with run_freeto intercepted), the web one from
POST /api/jobs (JobManager.submit intercepted), so no solve is run here.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import fields
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import freeto.core as fcore  # noqa: E402
import freeto.study as fstudy  # noqa: E402
from freeto import paper  # noqa: E402
from freeto.quantum.options import QUBOOptions  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PATHS = ("domain", "fixed", "xfixed", "yfixed", "zfixed", "keepdom")


class _Captured(Exception):
    pass


def study_cfg(problem, method, seed=0, monkeypatch=None):
    """FreeTOConfig the study builds for (problem, method, seed)."""
    spec = fstudy.suite_spec("quick2")
    runs = [(r, s) for _, r, s in fstudy._expand(spec)
            if r.get("kind") == "continuum" and r.get("study") in ("S3", "S4")
            and r["problem"] == problem and r["label"] == method and s == seed]
    assert len(runs) == 1
    box = {}

    def grab(cfg, *a, **k):
        box["cfg"] = cfg
        raise _Captured

    monkeypatch.setattr(fcore, "run_freeto", grab)
    with pytest.raises(_Captured):
        fstudy._run_continuum(runs[0][0], seed, keep_surface=False)
    return box["cfg"]


def canon(cfg):
    """Comparable dict of every FreeTOConfig field (real paths, loads per case,
    resolved QUBO options)."""
    c = cfg.normalized()
    out = {}
    n = len(c.forces)
    for f in fields(c):
        v = getattr(c, f.name)
        if f.name in PATHS:
            v = os.path.realpath(v) if v else None
        elif f.name == "forces":
            v = [os.path.realpath(p) for p in v]
        elif f.name in ("fmagx", "fmagy", "fmagz"):
            v = [float(x) for x in v]
            v = v * n if len(v) == 1 and n > 1 else v
        elif f.name == "qubo":
            v = QUBOOptions.from_any(v).to_dict() if c.optimizer == "QUBO" else None
        out[f.name] = v
    return out


@pytest.fixture()
def client(tmp_path):
    from webapp.server import create_app
    app = create_app(tmp_path / "web")
    with TestClient(app) as c:
        c.captured = []

        class _Job:
            id, status = "x", "queued"

        def submit(label, config_dict, cfg, audit=True, paper=None):
            c.captured.append((cfg, paper, config_dict))
            return _Job()

        app.state.manager.submit = submit
        app.state.manager.queue_position = lambda job_id: None
        yield c


def ui_payload(prefill, only_ui_qubo=False):
    """POST /api/jobs body built like the UI's buildJobPayload from a prefill."""
    from webapp.server import JobCreateRequest
    body = {k: v for k, v in prefill.items() if k in JobCreateRequest.model_fields}
    if only_ui_qubo:
        js = (REPO / "webapp" / "static" / "js" / "app.js").read_text()
        ui = set(re.findall(r"\b(qubo_[a-z_]+):", js))
        body = {k: v for k, v in body.items() if not k.startswith("qubo_") or k in ui}
    return body


def test_every_qubo_option_has_a_request_field():
    from webapp.server import QUBO_REQUEST_FIELDS
    assert set(QUBO_REQUEST_FIELDS) == {f.name for f in fields(QUBOOptions)}


def test_server_forwards_every_qubo_option(client):
    """Each QUBOOptions field set to a non-default value reaches FreeTOConfig.qubo."""
    pre = client.post("/api/examples/cantilever_beam/load").json()["prefill"]
    body = ui_payload(pre)
    changed = {"backend": "tabu", "hessian": "diag", "volume": "penalty", "lambda_q": 2.5,
               "gamma": 0.1, "move_penalty": 0.2, "frontier_fraction": 0.3, "block_size": 12,
               "blocks": "rank", "sweeps": 3, "init": "oc", "er": 0.07, "n_warm": 4,
               "patience": 5, "num_reads": 7, "seed": 11, "qaoa_p": 2, "qaoa_shots": 50,
               "qaoa_init": "interp", "qaoa_maxiter": 33, "time_limit": 9.0,
               "verify_exact": True, "hessian_block_size": 32, "hessian_max_rhs": 100,
               "hessian_scale": 3.0, "bisection_steps": 9, "lambda_source": "oc",
               "free_set": "grey", "history_average": False, "interp": "secant",
               "qaoa_polish": False, "move_limit": 0.1, "move_limit_min": 0.03,
               "protect_loads": False, "guard": False, "guard_tol": 0.4,
               "guard_tol_target": 0.2, "max_rejects": 2, "connectivity": False,
               "diagnostics": True, "backend_options": {"x": 1}}
    assert set(changed) == {f.name for f in fields(QUBOOptions)}
    body.update({f"qubo_{k}": v for k, v in changed.items()})
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 200, r.text
    q = QUBOOptions.from_any(client.captured[-1][0].qubo).to_dict()
    assert q == changed


def test_server_forwards_evaluation_and_mma_options(client):
    pre = client.post("/api/examples/cantilever_beam/load").json()["prefill"]
    body = ui_payload(pre)
    body.update(optimizer="MMA", eval_beta=4.0, eval_refined=4, mma_constraint="projected",
                mma_feasible_stop=False, init_perturb=0.05, init_seed=3)
    assert client.post("/api/jobs", json=body).status_code == 200
    cfg = client.captured[-1][0]
    assert (cfg.eval_beta, cfg.eval_refined, cfg.mma_constraint, cfg.mma_feasible_stop,
            cfg.init_perturb, cfg.init_seed) == (4.0, 4, "projected", False, 0.05, 3)


@pytest.mark.parametrize("problem", paper.PAPER_EXAMPLES)
def test_example_geometry_is_the_studys(problem):
    """The web app's example (files, loads, BCs, symmetry) is example_config's."""
    from webapp.core_loader import EXAMPLES
    from freeto.examples import EXAMPLES as STUDY_EXAMPLES
    assert EXAMPLES[problem]["config_kwargs"] == STUDY_EXAMPLES[problem]["config_kwargs"]


@pytest.mark.parametrize("problem", paper.PAPER_EXAMPLES)
def test_loaded_example_equals_study_config(problem, client, monkeypatch):
    """Load a paper example -> POST the prefill -> FreeTOConfig == the study's
    QUBO-SA (block) seed-0 config, field by field (UI subset of the QUBO fields
    and with every QUBO field)."""
    data = client.post(f"/api/examples/{problem}/load").json()
    assert data["paper"]["is_paper"] and data["paper"]["default_method"] == "QUBO-sa (block)"
    ref = canon(study_cfg(problem, "QUBO-sa (block)", 0, monkeypatch))
    for only_ui in (True, False):
        r = client.post("/api/jobs", json=ui_payload(data["prefill"], only_ui_qubo=only_ui))
        assert r.status_code == 200, r.text
        cfg, pap, _ = client.captured[-1]
        assert canon(cfg) == ref
        assert pap["method"] == "QUBO-sa (block)" and pap["seed"] == 0
        assert pap["gap_reference"] == paper.mma_reference(problem)
        assert pap["record"]["run_id"].endswith("QUBO-sa_(block)-s0")


@pytest.mark.parametrize("problem,method", [(p, m) for p in paper.PAPER_EXAMPLES
                                            for m in paper.paper_methods(p)])
def test_every_paper_method_preset_equals_study(problem, method, client, monkeypatch):
    """Every paper run offered by the 'Paper run' selector: preset -> POST ->
    config == the study's (seed 0)."""
    data = client.post(f"/api/examples/{problem}/load").json()
    pre = dict(data["prefill"], **data["paper"]["presets"][method])
    body = ui_payload(pre, only_ui_qubo=True)
    if body["optimizer"] != "QUBO":
        body = {k: v for k, v in body.items() if not k.startswith("qubo_")}
    assert client.post("/api/jobs", json=body).status_code == 200
    cfg, pap, _ = client.captured[-1]
    assert canon(cfg) == canon(study_cfg(problem, method, 0, monkeypatch))
    assert pap["method"] == method


def test_paper_settings_module_matches_study(monkeypatch):
    for pb in paper.PAPER_EXAMPLES:
        for m in paper.paper_methods(pb):
            assert canon(paper.paper_config(pb, m)) == canon(study_cfg(pb, m, 0, monkeypatch))


def test_paper_table1_values():
    """Mesh controls and volume fractions of manuscript Table 1; shared material."""
    t1 = {"cantilever_beam": (36, 0.30), "mbb_beam": (46, 0.30), "bridge_deck": (41, 0.35),
          "l_bracket": (30, 0.30), "GE_bracket": (24, 0.30)}
    for pb, (mc, vf) in t1.items():
        s = paper.paper_settings(pb)
        assert (s["mesh_control"], s["volfrac"]) == (mc, vf)
        assert (s["youngs_modulus"], s["poisson_ratio"], s["penal"], s["rmin"]) == \
            (210e9, 0.3, 3.0, 1.5)
        assert (s["max_iter"], s["eval_refined"], s["qubo"]["seed"]) == (300, 2, 0)
        assert (s["qubo"]["backend"], s["qubo"]["hessian"], s["qubo"]["er"],
                s["qubo"]["move_limit"]) == ("sa", "block", 0.05, 0.25)


def test_index_html_defaults_are_paper_values():
    """The form's static defaults (before any script runs) are the cantilever's
    paper values."""
    html = (REPO / "webapp" / "static" / "index.html").read_text()
    s = paper.paper_settings("cantilever_beam")

    def val(i):
        return re.search(rf'id="{i}"[^>]*value="([^"]*)"', html).group(1)
    assert int(val("p-mesh_control")) == s["mesh_control"]
    assert float(val("p-volfrac")) == s["volfrac"]
    assert float(val("p-youngs_modulus")) == s["youngs_modulus"]
    assert float(val("p-poisson_ratio")) == s["poisson_ratio"]
    assert float(val("p-penal")) == s["penal"]
    assert float(val("p-rmin")) == s["rmin"]
    assert int(val("p-max_iter")) == s["max_iter"]
    assert int(val("qubo-seed")) == s["qubo"]["seed"]
    assert float(val("p-eval_beta")) == s["eval_beta"]
    assert '<option value="2" selected>' in html  # refined f = 2
    assert "QUBO_QUICK_MESH" not in (REPO / "webapp" / "static" / "js" / "app.js").read_text()


def test_reference_values_match_results_json():
    """paper_reference.json == the study records (when results/quick2 is present)."""
    cands = [REPO / "results" / "quick2" / "results.json",
             Path("/home/claude/freeto_py/results/quick2/results.json")]
    src = next((p for p in cands if p.is_file()), None)
    if src is None:
        pytest.skip("results/quick2/results.json not available")
    recs = {r["run_id"]: r for r in json.loads(src.read_text())["records"]}
    ref = paper.paper_reference()
    n = 0
    for pb, ms in ref["records"].items():
        for m, seeds in ms.items():
            for sd, r in seeds.items():
                assert recs[r["run_id"]]["refined_compliance"] == r["refined_compliance"]
                assert recs[r["run_id"]]["iterations"] == r["iterations"]
                n += 1
        mma = [r for r in recs.values() if r["problem"] == pb and r["label"] == "MMA"
               and r["seed"] == 0 and r["study"] in ("S3", "S4")]
        assert ref["mma_reference_refined"][pb] == mma[0]["refined_compliance"]
    assert n > 50


def test_status_reports_paper_comparison(tmp_path):
    """A finished job's status carries native / refined values and, for a paper
    example, the gap to the paper's MMA reference (stubbed run, no solve)."""
    import types
    import numpy as np
    from webapp.jobs import _native_info, _paper_gap
    res = types.SimpleNamespace(comp=0.2, finalvol=0.3, iterations=5,
                                history={"volfrac": [0.5, 0.31, 0.3]})
    cfg = types.SimpleNamespace(optimizer="MMA", volfrac=0.3)
    info = _native_info(res, cfg)
    assert info["native_volfrac"] == 0.31 and info["native_compliance"] == 0.2
    info["refined_compliance"] = 0.1
    _paper_gap(info, {"gap_reference": 0.08})
    assert np.isclose(info["gap_refined"], 0.25)
