"""Tests of the v2 options (2026-10-04, docs/NOTES_quantum.md §9): refined
binary voxel evaluation, MMA constraint on the filtered field + feasible stop,
perturbed initial designs, hessian="scalar", the full_pre extra and the
study suite "quick2"."""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from freeto import FreeTOConfig, FreeTOError, run_freeto        # noqa: E402
from freeto.evaluate import refined_voxel_compliance              # noqa: E402
from freeto.fe import Assembler, lk_H8                            # noqa: E402
from freeto.filters import HnHns3D                                # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from freeto.geometry import box, face_slab, write_mesh
    d = tmp_path_factory.mktemp("tiny_v2")
    dom = ((0, 0, 0), (12, 4, 4))
    write_mesh(d / "dom.stl", box(*dom))
    write_mesh(d / "fix.stl", face_slab(dom, "x-", 1.2))
    write_mesh(d / "load.stl", face_slab(dom, "x+", 1.2))
    return dict(domain=str(d / "dom.stl"), forces=[str(d / "load.stl")],
                fixed=str(d / "fix.stl"), mesh_control=13, volfrac=0.4, fmagy=[-1.0],
                youngs_modulus=1.0, solver="superlu")


def _direct_compliance(edof, KE, free, ndof, E, F):
    asm = Assembler(edof, KE, free, ndof)
    K = asm.free_view().full(asm.data(E)).tocsc()
    Ff = np.asarray(F, dtype=float)[free]
    U = spla.spsolve(K, Ff).reshape(Ff.shape)
    return float(np.sum(Ff * U))


# ---------------------------------------------------------------------------
# 1. refined voxel evaluation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bc_map", ["interp", "coincident"])
def test_refined_f1_equals_package_fe_of_binarised_field(tiny, bc_map):
    """f = 1: the refined voxels are the coarse elements, so the result must be
    the package FE compliance of the binarised element field (to 1e-10)."""
    dbg = {}
    res = run_freeto(FreeTOConfig(**tiny, optimizer="MMA", max_iter=8, eval_refined=1),
                     log=None, _debug=dbg)
    s = res.setup_info
    nelx, nely, nelz = s["nelx"], s["nely"], s["nelz"]
    full_pre = res.extra["full_pre"].astype(float)
    assert res.extra["full_pre"].dtype == np.float32 and full_pre.size == s["nele"]
    E0, Ev = 1.0, 0.001 + 0.001 ** 3 * (1.0 - 0.001)      # crisp-model void modulus
    r = refined_voxel_compliance(full_pre, dbg["Hn"], dbg["Hns"], nelx, nely, nelz,
                                 dbg["ele"], 0.4, dbg["F"], dbg["fixeddof"], dbg["KE"], E0, Ev,
                                 f=1, solver="superlu", return_solid=True, bc_map=bc_map)
    solid = r["solid"]
    assert solid.size == dbg["ele"].size and r["n_voxels"] == solid.size
    assert abs(r["volfrac"] - 0.4) < 1.0 / solid.size + 1e-12
    c_pkg = _direct_compliance(dbg["edofMatn"], dbg["KE"], dbg["freedofs"], s["ndof"],
                               np.where(solid, E0, Ev), dbg["F"])
    assert r["compliance"] == pytest.approx(c_pkg, rel=1e-10)
    assert r["residual"] < 1e-8
    # the core's own eval_refined (same field, same void modulus)
    if bc_map == "interp":
        assert res.extra["refined_compliance"] == pytest.approx(c_pkg, rel=1e-10)
        assert res.extra["refined_f"] == 1
        assert res.extra["refined_volfrac"] == pytest.approx(r["volfrac"])
        assert res.extra["refined_threshold"] == pytest.approx(r["threshold"])
        assert res.extra["refined_time"] > 0


def _block(nelx, nely, nelz):
    nele = nelx * nely * nelz
    Hn, Hns = HnHns3D(nelx, nely, nelz, 1)
    ndof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    nn = np.arange(ndof // 3)
    c = (nn // (nely + 1)) % (nelx + 1)
    fixed = np.concatenate([3 * nn[c == 0] + k for k in range(3)])
    F = np.zeros((ndof, 1))
    tip = nn[c == nelx]
    F[3 * tip + 1, 0] = -1.0 / tip.size                  # distributed tip load
    return np.ones(nele), Hn, Hns, np.arange(nele), F, fixed


def test_refined_solid_block_mesh_refinement():
    """Solid stubby cantilever: f = 2 is softer (higher compliance, more DOFs) but
    within 15 % of f = 1, and f = 4 changes it by less than f = 2 did (the
    refined model converges).  With the coincident-node mapping of supports
    and loads it does not converge (point supports / point loads)."""
    nelx, nely, nelz = 6, 3, 3
    full, Hn, Hns, ele, F, fixed = _block(nelx, nely, nelz)
    KE = lk_H8(0.3)
    c = {}
    for f in (1, 2, 4):
        r = refined_voxel_compliance(full, Hn, Hns, nelx, nely, nelz, ele, 1.0, F, fixed, KE,
                                     1.0, 1e-9, f=f)
        assert r["volfrac"] == 1.0 and r["n_voxels"] == ele.size * f ** 3
        c[f] = r["compliance"]
    assert c[1] < c[2] < c[4]
    assert c[2] / c[1] - 1 < 0.15
    assert c[4] / c[2] - 1 < c[2] / c[1] - 1
    cc = {f: refined_voxel_compliance(full, Hn, Hns, nelx, nely, nelz, ele, 1.0, F, fixed, KE,
                                      1.0, 1e-9, f=f, bc_map="coincident")["compliance"]
          for f in (1, 2)}
    assert cc[1] == pytest.approx(c[1], rel=1e-12)
    assert cc[2] / cc[1] > 1.5
    # the interpolated load mapping conserves the total force
    from freeto.evaluate import _map_bcs
    act = np.arange((2 * nely + 1) * (2 * nelx + 1) * (2 * nelz + 1))
    Ff, _ = _map_bcs(F, fixed, nelx, nely, nelz, 2, act, "interp")
    assert Ff.sum() == pytest.approx(F.sum(), rel=1e-12)


def test_refined_rejects_bad_f():
    full, Hn, Hns, ele, F, fixed = _block(2, 1, 1)
    with pytest.raises(ValueError):
        refined_voxel_compliance(full, Hn, Hns, 2, 1, 1, ele, 1.0, F, fixed, lk_H8(0.3),
                                 1.0, 1e-9, f=3)


# ---------------------------------------------------------------------------
# 2. core options
# ---------------------------------------------------------------------------
def test_v2_option_validation(tiny):
    for bad in (dict(mma_constraint="nope"), dict(eval_refined=3), dict(init_perturb=-0.1),
                dict(init_perturb=1.0)):
        with pytest.raises(FreeTOError):
            FreeTOConfig(**tiny, **bad).validate()
    assert FreeTOConfig(**tiny, mma_constraint="FILTERED", eval_refined=2).validate()


def test_mma_filtered_constraint_and_feasible_stop(tiny):
    """mma_constraint='filtered': the returned pre-smoothing field holds the
    volume; mma_feasible_stop: the run only stops on the tolerances when
    |fval| <= 1e-3."""
    base = run_freeto(FreeTOConfig(**tiny, optimizer="MMA", max_iter=60), log=None)
    assert base.extra["mma_constraint"] == "projected" and "mma_fval_final" not in base.extra
    r = run_freeto(FreeTOConfig(**tiny, optimizer="MMA", max_iter=60,
                                mma_constraint="filtered", mma_feasible_stop=True), log=None)
    assert r.extra["mma_constraint"] == "filtered"
    fp = r.extra["full_pre"].astype(float)
    ele_vol = float(np.sum(fp)) / r.elenum1          # box domain: every element is active
    if r.iterations < 60:
        assert abs(r.extra["mma_fval_final"]) <= 1e-3
        assert ele_vol == pytest.approx(0.4, abs=1e-3 * 0.4 + 1e-6)
    else:
        assert math.isfinite(r.extra["mma_fval_final"])


def test_init_perturb_deterministic_and_off_by_default(tiny):
    kw = dict(**tiny, optimizer="OC", max_iter=3)
    r0 = run_freeto(FreeTOConfig(**kw), log=None)
    r00 = run_freeto(FreeTOConfig(**kw, init_perturb=0.0, init_seed=5), log=None)
    assert r0.history["compliance"] == r00.history["compliance"]
    a = run_freeto(FreeTOConfig(**kw, init_perturb=0.05, init_seed=1), log=None)
    b = run_freeto(FreeTOConfig(**kw, init_perturb=0.05, init_seed=1), log=None)
    c = run_freeto(FreeTOConfig(**kw, init_perturb=0.05, init_seed=2), log=None)
    assert a.history["compliance"] == b.history["compliance"]
    assert a.history["compliance"][0] != c.history["compliance"][0]
    assert a.history["compliance"][0] != r0.history["compliance"][0]
    dbg = {}
    run_freeto(FreeTOConfig(**dict(kw, max_iter=1, optimizer="MMA"), init_perturb=0.05,
                            init_seed=1), log=None, _debug=dbg)
    rng = np.random.default_rng(1)
    x0 = np.clip(0.4 + rng.uniform(-0.05, 0.05, dbg["ele"].size), 0.001, 1.0)
    keep = np.isin(dbg["ele"], dbg["MusD"])
    x0[keep] = 0.4
    Ee = 0.001 + x0 ** 3 * (1.0 - 0.001)          # first FE uses vxPhys = x0
    assert np.allclose(dbg["iters"][0]["Ee"], Ee, rtol=1e-12)


def test_full_pre_extra_and_refined_f2(tiny):
    dbg = {}
    r = run_freeto(FreeTOConfig(**tiny, optimizer="MMA", max_iter=10, eval_crisp=True,
                                eval_refined=2), log=None, _debug=dbg)
    fp = r.extra["full_pre"]
    assert fp.dtype == np.float32 and fp.size == r.elenum2
    for k in ("refined_compliance", "refined_volfrac", "refined_threshold", "refined_time"):
        assert math.isfinite(r.extra[k]) and r.extra[k] > 0
    assert r.extra["refined_f"] == 2
    # symmetric box problem: tied voxel values make the volume a coarse step function
    assert abs(r.extra["refined_volfrac"] - 0.4) < 5e-3
    # the evaluated field is the last iterate's filtered design with keep = 1
    ele, H, Hs = dbg["ele"], dbg["H"], dbg["Hs"]
    full = np.zeros(r.elenum2)
    full[ele] = (H @ dbg["iters"][-1]["vxnew"]) / Hs
    full[dbg["MusD"]] = 1.0
    assert np.array_equal(fp, full.astype(np.float32))
    s = r.setup_info
    ref = refined_voxel_compliance(full, dbg["Hn"], dbg["Hns"], s["nelx"], s["nely"],
                                   s["nelz"], ele, 0.4, dbg["F"], dbg["fixeddof"], dbg["KE"],
                                   1.0, 0.001 + 0.001 ** 3 * (1 - 0.001), f=2, solver="superlu")
    assert r.extra["refined_compliance"] == pytest.approx(ref["compliance"], rel=1e-10)


# ---------------------------------------------------------------------------
# 3. hessian="scalar"
# ---------------------------------------------------------------------------
def test_hessian_scalar_has_no_couplings(tiny):
    from freeto.quantum import QUBOOptions
    from freeto.quantum.update import QUBOUpdater
    QUBOOptions(hessian="scalar").validate()
    with pytest.raises(ValueError):
        QUBOOptions(hessian="scalars").validate()
    cap = []
    QUBOUpdater.capture = cap.append
    try:
        r = run_freeto(FreeTOConfig(**tiny, optimizer="QUBO", max_iter=4,
                                    qubo={"backend": "sa", "hessian": "scalar", "seed": 0}),
                       log=None)
    finally:
        QUBOUpdater.capture = None
    assert cap and all(abs(m["Q"]).sum() == 0 for m in cap)   # Q = off-diagonal part
    assert {q["hessian"] for q in r.extra["qubo_history"]} == {"scalar"}
    cap = []
    QUBOUpdater.capture = cap.append
    try:
        run_freeto(FreeTOConfig(**tiny, optimizer="QUBO", max_iter=2,
                                qubo={"backend": "sa", "hessian": "diag", "seed": 0}),
                   log=None)
    finally:
        QUBOUpdater.capture = None
    assert abs(cap[0]["Q"]).sum() > 0                         # diag mode has couplings


# ---------------------------------------------------------------------------
# 4. study suite quick2
# ---------------------------------------------------------------------------
def test_quick2_suite_spec():
    from freeto.study import suite_spec, _expand
    q = suite_spec("quick")
    q2 = suite_spec("quick2")
    co = [r for r in q2["runs"] if r["kind"] == "continuum"]
    assert co and all(r["max_iter"] == 300 for r in co)
    for r in co:
        o = r["run_options"]
        assert o == {"eval_refined": 2, "mma_constraint": "filtered",
                     "mma_feasible_stop": True} or (
            r["label"] == "MMA (init s)" and o == {"eval_refined": 2, "mma_constraint":
                                                   "filtered", "mma_feasible_stop": True,
                                                   "init_perturb": 0.05})
    s3q = [r for r in co if r["study"] == "S3q" and r["role"] != "baseline"]
    assert s3q and all(r["seeds"] == [0, 1, 2, 3, 4] for r in s3q)
    # QAOA on mbb_beam sits with the S3 runs of mbb_beam
    labs = [(r["study"], r["problem"], r["label"]) for r in co]
    i_mbb = labs.index(("S3", "mbb_beam", "QUBO-qaoa kb8 p1 penalty (+greedy)"))
    assert labs[i_mbb - 1][1] == "mbb_beam"
    assert labs[-1][0] == "F5"
    # D5 controls on every continuum example of the S3/S4 blocks
    probs = {(r["problem"], r["mesh_control"]) for r in co if r["study"] in ("S3", "S4")}
    for pb, mc in probs:
        g = [r for r in co if r["problem"] == pb and r["mesh_control"] == mc
             and r["study"] in ("S3", "S4")]
        bm = [r for r in g if r["label"] == "BESO-sort (move 0.04)"]
        assert len(bm) == 1 and bm[0]["options"] == {"hessian": "none", "move_limit": 0.04}
        sc = [r for r in g if r["label"] == "QUBO-sa (scalar)"]
        blk = [r for r in g if r["label"] == "QUBO-sa (block)"]
        assert len(sc) == 1 and sc[0]["seeds"] == blk[0]["seeds"]
        assert sc[0]["options"] == {"backend": "sa", "hessian": "scalar"}
        mi = [r for r in g if r["label"] == "MMA (init s)"]
        assert len(mi) == 1 and mi[0]["seeds"] == [0, 1, 2, 3, 4]
        assert mi[0]["role"] == "baseline_spread" and mi[0]["optimizer"] == "MMA"
        assert "init_seed" not in mi[0]["run_options"]     # init_seed = run seed
    # Qx3 control on the S3 examples only (P0b)
    qx3 = [r for r in co if r["label"] == "QUBO-sa (block, Qx3)"]
    assert sorted((r["study"], r["problem"], r["mesh_control"]) for r in qx3) == [
        ("S3", "cantilever_beam", 36), ("S3", "mbb_beam", 46)]
    for r in qx3:
        assert r["options"] == {"backend": "sa", "hessian": "block", "hessian_scale": 3.0}
        assert r["role"] == "control" and r["seeds"] == [0, 1, 2, 3, 4]
    # truss runs: those of "quick", with kinematic_repair on every QUBO / sorting run
    tq = [r for r in q["runs"] if r["kind"] != "continuum"]
    tq2 = [r for r in q2["runs"] if r["kind"] != "continuum"]
    assert len(tq) == len(tq2)
    for r, r2 in zip(tq, tq2):
        if r["kind"] == "truss" and r.get("method") in ("qubo", "sort"):
            assert r2["options"] == dict(r.get("options") or {}, kinematic_repair=True)
            assert {k: v for k, v in r2.items() if k != "options"} == \
                {k: v for k, v in r.items() if k != "options"}
        else:
            assert r2 == r
    ids = [rid for rid, _, _ in _expand(q2)]
    assert len(ids) == len(set(ids))


def test_run_continuum_options_and_fields(tmp_path, monkeypatch):
    """run_options reach FreeTOConfig (init_seed = run seed), refined columns are
    recorded and the field file is written."""
    import freeto.study as st
    seen = {}
    real = st.__dict__.get("_run_continuum")
    import freeto.core as core
    orig = core.run_freeto

    def spy(cfg, *a, **k):
        seen.update(init_seed=cfg.init_seed, init_perturb=cfg.init_perturb,
                    eval_refined=cfg.eval_refined, mma_constraint=cfg.mma_constraint,
                    mma_feasible_stop=cfg.mma_feasible_stop)
        return orig(cfg, *a, **k)
    monkeypatch.setattr(core, "run_freeto", spy)
    run = {"study": "S3", "kind": "continuum", "problem": "cantilever_beam",
           "mesh_control": 20, "optimizer": "MMA", "label": "MMA (init s)", "max_iter": 6,
           "seeds": [3], "role": "baseline_spread",
           "run_options": dict(st.V2_RUN_OPTIONS, init_perturb=0.05)}
    rec, _ = real(run, 3, keep_surface=False, out_dir=str(tmp_path), run_id="X-0-test-s3")
    assert seen == {"init_seed": 3, "init_perturb": 0.05, "eval_refined": 2,
                    "mma_constraint": "filtered", "mma_feasible_stop": True}
    assert math.isfinite(rec["refined_compliance"]) and rec["refined_volfrac"] > 0
    assert rec["fields_file"] == "fields/X-0-test-s3.npz"
    z = np.load(tmp_path / rec["fields_file"])
    assert z["full_pre"].dtype == np.float32 and "history_compliance" in z.files
    with pytest.raises(ValueError):
        real(dict(run, run_options={"bogus": 1}), 0, keep_surface=False)


def test_hessian_scale_scales_model(tiny):
    from freeto.quantum import QUBOOptions
    from freeto.quantum.update import QUBOUpdater
    with pytest.raises(ValueError):
        QUBOOptions(hessian_scale=-1.0).validate()
    Q = {}
    for sc in (1.0, 3.0):
        cap = []
        QUBOUpdater.capture = cap.append
        try:
            run_freeto(FreeTOConfig(**tiny, optimizer="QUBO", max_iter=1,
                                    qubo={"backend": "sa", "hessian": "diag", "seed": 0,
                                          "hessian_scale": sc}), log=None)
        finally:
            QUBOUpdater.capture = None
        Q[sc] = cap[0]["Q"].toarray()
    assert abs(Q[1.0]).sum() > 0
    assert np.allclose(Q[3.0], 3.0 * Q[1.0], rtol=1e-12, atol=0)


def test_truss_hit_tolerance_1e6():
    from freeto.study import HIT_TOL, _summary
    assert HIT_TOL == 1e-6
    recs = [{"kind": "truss", "study": "S1", "problem": "gs_3x2", "label": "sort/BESO",
             "feasible": True, "gap": g, "compliance": 11.66} for g in (1.48e-9, 5e-7, 2e-6)]
    row = _summary(recs)[0]
    assert row["n_exact_hits"] == 2 and row["hit_tol"] == 1e-6


def test_parallel_study_equals_serial(tmp_path):
    """jobs=2 (spawn pool) gives the same records as jobs=1, a failing run is an
    error record, records keep the run order, results.json is written."""
    import json as _json
    from freeto.study import run_study
    spec = {"name": "par", "runs": [
        {"study": "S1", "kind": "truss", "problem": "gs_3x2", "method": "qubo",
         "backend": "sa", "label": "QUBO-sa", "seeds": [0, 1],
         "options": {"kinematic_repair": True}},
        {"study": "S1", "kind": "truss", "problem": "nope", "method": "qubo",
         "backend": "sa", "label": "bad", "seeds": [0]},
        {"study": "S1", "kind": "truss", "problem": "ten_bar", "method": "sort",
         "label": "sort/BESO", "seeds": [0]}]}
    skip = {"wall_time", "solver_time", "traceback"}
    out = {}
    for j in (1, 2):
        res = run_study(spec, out_dir=str(tmp_path / f"j{j}"), figures=False, log=None, jobs=j)
        out[j] = [{k: v for k, v in r.items() if k not in skip} for r in res["records"]]
        assert res["machine"]["jobs"] == j
        assert (tmp_path / f"j{j}" / "results.json").is_file()
    assert _json.dumps(out[1], sort_keys=True, default=str) == \
        _json.dumps(out[2], sort_keys=True, default=str)
    assert [r["run_id"] for r in out[2]] == ["S1-0-gs_3x2-QUBO-sa-s0", "S1-0-gs_3x2-QUBO-sa-s1",
                                              "S1-1-nope-bad-s0", "S1-2-ten_bar-sort-BESO-s0"]
    assert out[2][2]["error"].startswith("ValueError")
    assert out[2][0]["feasible"] and out[2][0]["repair_added"] > 0


def test_resume_runs_only_missing_and_failed(tmp_path):
    """--resume: error-free runs of results.json are kept, a deleted and an
    errored run are re-run (each in a fresh worker process), the final file has
    every record in spec order and partial = False."""
    import json as _json
    from freeto.study import run_study, resume_records
    spec = {"name": "res", "runs": [
        {"study": "S1", "kind": "truss", "problem": p, "method": "sort", "label": "sort/BESO",
         "seeds": [0]} for p in ("ten_bar", "gs_3x2", "gs_4x2")]}
    d = tmp_path / "st"
    first = run_study(spec, out_dir=str(d), figures=False, log=None)
    ids = [r["run_id"] for r in first["records"]]
    for case in ("delete", "error"):
        data = _json.loads((d / "results.json").read_text(encoding="utf-8"))
        if case == "delete":
            data["records"] = [r for r in data["records"] if r["run_id"] != ids[1]]
        else:
            data["records"][2] = {"run_id": ids[2], "kind": "truss", "error": "boom"}
        data["partial"] = True
        (d / "results.json").write_text(_json.dumps(data), encoding="utf-8")
        kept = resume_records(str(d), spec, log=None)
        assert len(kept) == 2
        ran = []
        res = run_study(spec, out_dir=str(d), figures=False, log=None, resume_records=kept,
                        fresh_workers=True,
                        callback=lambda ev: ran.append(ev["run_id"])
                        if ev["event"] == "run_end" else None)
        assert ran == [ids[1] if case == "delete" else ids[2]]
        final = _json.loads((d / "results.json").read_text(encoding="utf-8"))
        assert not final.get("partial") and [r["run_id"] for r in final["records"]] == ids
        assert all(r.get("error") is None for r in final["records"])
        assert [r["compliance"] for r in final["records"]] == \
            [r["compliance"] for r in first["records"]]


def test_refined_load_on_solid_restricts_load_weights():
    """A coarse load node whose refined region is partly void: with
    load_on_solid the load acts only on refined nodes attached to solid voxels
    (the compliance stays finite); without it part of the load sits in void."""
    import numpy as np
    from freeto.evaluate import _map_bcs
    nelx, nely, nelz, f = 2, 1, 1, 2
    nnc = (nelx + 1) * (nely + 1) * (nelz + 1)
    F = np.zeros((3 * nnc, 1))
    F[3 * 0 + 1, 0] = -0.5           # load on the coarse edge of nodes 0 and 1
    F[3 * 1 + 1, 0] = -0.5           # (rows 0 and 1 of column 0, y direction)
    nnf = (f * nelx + 1) * (f * nely + 1) * (f * nelz + 1)
    act = np.arange(nnf)
    carry = np.zeros(nnf, dtype=bool)
    carry[[0, 2]] = True             # the coincident refined nodes touch solid, the mid node not
    Ff_all, _ = _map_bcs(F, np.zeros(0, dtype=np.int64), nelx, nely, nelz, f, act, "interp")
    Ff_sol, _ = _map_bcs(F, np.zeros(0, dtype=np.int64), nelx, nely, nelz, f, act, "interp",
                         carry=carry)
    assert abs(Ff_all.sum() + 1.0) < 1e-12 and abs(Ff_sol.sum() + 1.0) < 1e-12
    assert np.count_nonzero(Ff_all) == 3 and np.count_nonzero(Ff_sol) == 2
    assert Ff_sol[3 * 1 + 1, 0] == 0.0 and abs(Ff_sol[1, 0] + 0.5) < 1e-12
