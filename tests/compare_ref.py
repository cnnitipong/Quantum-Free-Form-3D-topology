#!/usr/bin/env python3
"""Compare the Python port against the Octave reference dumps of the
original MATLAB code (``<ref-dir>/<case>/``, see below).

Usage::

    python tests/compare_ref.py [case ...] [--ref-dir DIR[:DIR...]]
                                [--solver superlu] [--compat octave|matlab]

The reference dumps are not shipped with the package.  They are looked up
in ``--ref-dir`` / the ``FREETO_REF_DIR`` environment variable (both accept
several directories separated by the OS path separator), then in
``tests/ref/``; every sub-directory named like a case below is compared.
Without reference data the script reports "skipped" and exits 0.

For every case the same problem is run in Python (with the Octave
linspace/colon arithmetic by default, so that node coordinates are
bit-identical) and the setup quantities, the first iterations, the last
iteration and the whole history are compared.  Exit status 0 if every check
passes its tolerance.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.io as sio
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from freeto import FreeTOConfig, run_freeto          # noqa: E402
from freeto.examples import STL_DIR                   # noqa: E402
from freeto.mesh import matlab_to_xyz                 # noqa: E402
from freeto.postprocess import FieldSnapshot, apply_symmetry  # noqa: E402

DEFAULT_REF_DIRS = [os.path.join(HERE, "ref")]


def S(name):
    return os.path.join(STL_DIR, name)


CASES = {
    "ge_simp": dict(
        domain=S("GE_domain.STL"), forces=[S("GE_force.STL"), S("GE_force.STL")],
        mesh_control=24, volfrac=0.3, fixed=S("GE_fixed.STL"),
        fmagx=[0.0], fmagy=[0, -2000], fmagz=[1500, 0], youngs_modulus=210e9,
        method="SIMP", max_iter=30),
    "air_semdot": dict(
        domain=S("air_domain.STL"),
        forces=[S("air_force.STL")] * 3, mesh_control=32, volfrac=0.2,
        fixed=S("air_fixed.STL"), zfixed=S("air_zfixed.STL"),
        fmagx=[1000, 1324, 0], fmagy=[0, -1324, -2500], fmagz=[0.0],
        youngs_modulus=210e9, method="SEMDOT", max_iter=20,
        symmetry=[("x-y", "right")]),
    "quad_simp": dict(
        domain=S("quad_domain.STL"),
        forces=[S("quad_force1.STL"), S("quad_force2.STL"), S("quad_force3.STL")],
        mesh_control=20, volfrac=0.3, fixed=S("quad_fixed.STL"),
        xfixed=S("quad_xfixed.STL"), yfixed=S("quad_yfixed.STL"),
        fmagx=[0.0], fmagy=[0.0], fmagz=[-1500, -1500, -1000],
        youngs_modulus=2e9, method="SIMP", loadtype="point", keep_bcz=True,
        max_iter=20, symmetry=[("y-z", "right"), ("z-x", "left")]),
    # independent verification cases (other code branches)
    "hand_simp": dict(
        domain=S("hand_domain.stl"), forces=[S(f"hand_force{i}.stl") for i in range(1, 6)],
        fixed=S("hand_fixed.stl"), keepdom=S("hand_force2.stl"), mesh_control=26,
        volfrac=0.3, youngs_modulus=210e9, fmagx=300, fmagy=0, fmagz=[2000.0] * 5,
        method="SIMP", max_iter=8,
        symmetry=[("z-x", "right"), ("x-y", "left"), ("y-z", "left")]),
    "lever_semdot": dict(
        domain=S("lever_domain.STL"), forces=[S("lever_force1.STL"), S("lever_force2.STL")],
        fixed=S("lever_fixed.STL"), zfixed=S("lever_zfixed.STL"), keep_bcx=True,
        keep_bcz=True, mesh_control=28, volfrac=0.35, youngs_modulus=70e9,
        poisson_ratio=0.33, rmin=2.0, fmagx=[0, -500], fmagy=[800, 0], fmagz=[0, -200],
        loadtype="point", method="SEMDOT", max_iter=8),
    "ge_semdot_keep": dict(
        domain=S("GE_domain.STL"), forces=[S("GE_force.STL")] * 2, fixed=S("GE_fixed.STL"),
        keepdom=S("GE_force.STL"), keep_bc=False, mesh_control=22, volfrac=0.25,
        youngs_modulus=210e9, rmin=1.2, fmagx=100, fmagy=[0, -2000], fmagz=[1500, 0],
        loadtype="point", method="SEMDOT", max_iter=6),
}


class Report:
    def __init__(self, case):
        self.case = case
        self.rows = []

    def add(self, name, ok, detail):
        self.rows.append((name, ok, detail))
        flag = "ok  " if ok else "FAIL"
        print(f"  [{flag}] {name:<34s} {detail}")

    def exact(self, name, a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            self.add(name, False, f"shape {a.shape} vs {b.shape}")
            return False
        nd = int(np.count_nonzero(a != b))
        self.add(name, nd == 0, f"exact; {nd} of {a.size} differ" if nd else
                 f"exact ({a.size} values)")
        return nd == 0

    def close(self, name, a, b, rtol):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.shape != b.shape:
            self.add(name, False, f"shape {a.shape} vs {b.shape}")
            return False
        scale = max(np.abs(b).max() if b.size else 0.0, 1e-300)
        err = np.abs(a - b).max() if a.size else 0.0
        rel = err / scale
        ok = rel <= rtol
        self.add(name, ok, f"max|d| = {err:.3e}  rel = {rel:.2e}  (tol {rtol:.0e})")
        return ok

    @property
    def ok(self):
        return all(r[1] for r in self.rows)


def _dense(v):
    if sp.issparse(v):
        v = v.toarray()
    return np.asarray(v, dtype=float)


def _scalar(v):
    return float(np.asarray(v).ravel()[0])


def _idx(v):
    v = np.asarray(v).ravel()
    return (v.astype(np.int64) - 1) if v.size else np.zeros(0, np.int64)


def compare_case(case, refdir, solver, compat, inside="matlab", iters_rtol=1e-7):
    d = os.path.join(refdir, case)
    print(f"\n=== {case} ({d}) ===")
    rep = Report(case)
    if not os.path.isfile(os.path.join(d, "setup.mat")):
        print("  reference not available - skipped")
        return None
    st = sio.loadmat(os.path.join(d, "setup.mat"))
    st = {kk: (_dense(vv) if sp.issparse(vv) else vv) for kk, vv in st.items()}
    kw = dict(CASES[case])
    maxit = int(_scalar(st["maxloop_used"])) if "maxloop_used" in st else kw["max_iter"]
    kw["max_iter"] = maxit
    cfg = FreeTOConfig(solver=solver, compat=compat, inside_mode=inside, **kw)
    dbg = {}
    res = run_freeto(cfg, log=lambda s: None, _debug=dbg)
    dom = dbg["dom"]
    g = dom.grid

    # ---- grid ----------------------------------------------------------
    for nm in ("nelx", "nely", "nelz", "nele", "ndof"):
        rep.exact(nm, getattr(dom, nm), int(_scalar(st[nm])))
    rep.exact("aa", g.aa, _scalar(st["aa"]))
    if "gx1" in st:
        # bit-exact with compat="octave"; MATLAB's linspace/colon differ from
        # Octave's in the last bit of some coordinates
        coord = (rep.exact if compat == "octave"
                 else (lambda n, a, b: rep.close(n, a, b, 1e-15)))
        coord("x nodes", g.x, st["gx1"][0, :, 0])
        coord("y nodes", g.y, st["gy1"][:, 0, 0])
        coord("z nodes", g.z, st["gz1"][0, 0, :])
        coord("x centres", g.xc, st["gx1_cen"][0, :, 0])
        coord("y centres", g.yc, st["gy1_cen"][:, 0, 0])
        coord("z centres", g.zc, st["gz1_cen"][0, 0, :])
    for nm, dd in (("del_x", 0), ("del_y", 1), ("del_z", 2)):
        rep.exact(nm, g.del_xyz[dd], _scalar(st[nm]))
    # ---- membership ----------------------------------------------------
    rep.exact("oute", dom.oute, st["oute"])
    rep.exact("outeM", dom.outeM, st["outeM"])
    for nm in ("sup_all", "sup_x", "sup_y", "sup_z"):
        rep.exact(nm, getattr(dom, nm), _idx(st[nm]))
    Fn = st["Fn_full"]
    for i in range(Fn.shape[1]):
        col = Fn[:, i]
        rep.exact(f"Fn[{i}]", dom.Fn[i], col[col != 0].astype(np.int64) - 1)
    # ---- index sets / vectors -----------------------------------------
    rep.exact("ele", dbg["ele"], _idx(st["ele"]))
    rep.exact("nnele", dbg["ele"].size, int(_scalar(st["nnele"])))
    rep.exact("fixeddof", dbg["fixeddof"], _idx(st["fixeddof"]))
    rep.exact("freedofs", dbg["freedofs"], _idx(st["freedofs"]))
    ed = st["edofMatn_save"].astype(np.int64) - 1
    rep.exact("edofMatn", dbg["edofMatn"][:ed.shape[0]], ed)
    rep.exact("F", dbg["F"], st["F_full"])
    rep.close("KE", dbg["KE"], st["KE"], 1e-15)
    nn = dbg["ele"].size
    Href = sp.csr_matrix((st["Hv"].ravel(), (_idx(st["Hi"]), _idx(st["Hj"]))),
                         shape=(nn, nn))
    dH = abs(dbg["H"] - Href)
    rep.add("H", dH.nnz == 0 or dH.max() == 0,
            f"nnz {dbg['H'].nnz} vs {Href.nnz}, max|d| = {dH.max() if dH.nnz else 0:.2e}")
    rep.close("Hs", dbg["Hs"], st["Hs_full"].ravel(), 1e-15)
    Hnref = sp.csr_matrix((st["Hnv"].ravel(), (_idx(st["Hni"]), _idx(st["Hnj"]))),
                          shape=dbg["Hn"].shape)
    dHn = abs(dbg["Hn"] - Hnref)
    rep.add("Hn", dHn.nnz == 0 or dHn.max() == 0,
            f"nnz {dbg['Hn'].nnz} vs {Hnref.nnz}, max|d| = {dHn.max() if dHn.nnz else 0:.2e}")
    rep.close("Hns", dbg["Hns"], st["Hns_full"].ravel(), 1e-15)

    # ---- iterations ----------------------------------------------------
    files = sorted(f for f in os.listdir(d) if f.startswith("iter_"))
    its = dbg["iters"]
    for f in files:
        k = int(f[5:8])
        it = sio.loadmat(os.path.join(d, f))
        it = {kk: (_dense(vv) if sp.issparse(vv) else vv) for kk, vv in it.items()}
        if k > len(its):
            rep.add(f"iter {k}", False, f"python stopped after {len(its)} iterations")
            continue
        py = its[k - 1]
        tag = f"it{k:03d}"
        # direct solvers: ~1e-12 agreement; AMG-PCG (rtol 1e-8) ~1e-9
        rtol = (1e-10 if solver != "amg" else 1e-8) if k <= 3 else iters_rtol
        rep.close(f"{tag} c", py["c"], _scalar(it["c"]), rtol)
        rep.close(f"{tag} dc (filtered)", py["dc"], it["dc"].ravel(), rtol)
        rep.close(f"{tag} dv", py["dv"], it["dv"].ravel(), 1e-14)
        rep.close(f"{tag} vxnew", py["vxnew"], it["vxnew"].ravel(), rtol)
        rep.close(f"{tag} vxPhys (smoothed)", py["vxPhys_full"], it["vxPhys"].ravel(), rtol)
        rep.close(f"{tag} xg", py["xg"], it["xg"], rtol)
        rep.close(f"{tag} lss", py["lss"], _scalar(it["lss"]), 1e-15 if k <= 3 else rtol)
        rep.close(f"{tag} tol (topology)", py["tol"], _scalar(it["tol"]), 1e-15 if k <= 3 else rtol)
        rep.close(f"{tag} change", py["change"], _scalar(it["change"]), rtol)
        rep.exact(f"{tag} beta", py["beta"], _scalar(it["beta"]))
        if "U" in it:
            nl = py["U"].shape[1]
            rep.close(f"{tag} U", py["U"], it["U"][:, :nl], rtol)

    # ---- history / final ----------------------------------------------
    fp = os.path.join(d, "final.mat")
    if os.path.isfile(fp):
        fin = sio.loadmat(fp, squeeze_me=False)
        cc = fin["cc_hist"].ravel()
        fv = fin["fvol_hist"].ravel()
        rep.exact("iterations", res.iterations, cc.size)
        n = min(cc.size, len(res.history["compliance"]))
        rep.close("history compliance", res.history["compliance"][:n], cc[:n], iters_rtol)
        rep.close("history volfrac", res.history["volfrac"][:n], fv[:n], iters_rtol)
        Sres = fin["S"][0, 0]
        rep.close("S.comp", res.comp, _scalar(Sres["comp"]), iters_rtol)
        rep.close("S.finalvol", res.finalvol, _scalar(Sres["finalvol"]), iters_rtol)
        rep.close("S.eleden", res.eleden, Sres["eleden"], iters_rtol)
        rep.close("S.gridden (xg)", res.gridden, Sres["gridden"], iters_rtol)
        rep.exact("S.elenum1", res.elenum1, int(_scalar(Sres["elenum1"])))
        rep.exact("S.elenum2", res.elenum2, int(_scalar(Sres["elenum2"])))
        sym = cfg.normalized().symmetry
        pre_key = "top_presym" if "top_presym" in fin else ("top" if not sym else None)
        top_py = res.gridden - res.ls
        if pre_key:
            rep.close("top (pre-symmetry)", top_py, fin[pre_key], iters_rtol)
        if sym and "top_presym" in fin:
            # symmetry.m array semantics, applied in the (x,y,z) frame
            fld = FieldSnapshot(matlab_to_xyz(top_py), np.zeros(3), np.ones(3))
            steps = []
            for pl, di in sym:
                fld = apply_symmetry(fld, pl, di)
                steps.append(np.asarray(fld.top))
            for i in range(1, len(steps)):
                key = f"top_after_sym{i}"
                if key in fin:
                    rep.close(f"top after symmetry {i}", steps[i - 1],
                              matlab_to_xyz(fin[key]), iters_rtol)
            rep.close("top after all symmetries", steps[-1],
                      matlab_to_xyz(fin["top"]), iters_rtol)
            # the Python result field must hold the same mirrored array
            rep.close("result.field.top (mirrored)", np.asarray(res.field.top),
                      matlab_to_xyz(fin["top"]), iters_rtol)
    status = "PASS" if rep.ok else "FAIL"
    nfail = sum(1 for r in rep.rows if not r[1])
    print(f"  --> {case}: {status} ({len(rep.rows) - nfail}/{len(rep.rows)} checks)")
    return rep


def find_cases(dirs, wanted):
    """(case, directory) pairs for every case sub-directory found."""
    out = []
    for root in dirs:
        if not root or not os.path.isdir(root):
            continue
        for case in wanted:
            d = os.path.join(root, case)
            if os.path.isfile(os.path.join(d, "setup.mat")):
                out.append((case, root))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cases", nargs="*", default=list(CASES))
    ap.add_argument("--ref-dir", "--ref", dest="ref_dir",
                    default=os.environ.get("FREETO_REF_DIR"),
                    help="directory (or os.pathsep-separated list) holding "
                         "<case>/setup.mat ... dumps [env FREETO_REF_DIR]")
    ap.add_argument("--solver", default="superlu")
    ap.add_argument("--compat", default="octave")
    ap.add_argument("--inside", default="matlab", choices=["matlab", "robust"])
    a = ap.parse_args(argv)
    dirs = a.ref_dir.split(os.pathsep) if a.ref_dir else DEFAULT_REF_DIRS
    found = find_cases(dirs, [c for c in a.cases if c in CASES])
    if not found:
        print("No Octave reference data found (looked in: %s) - skipped.\n"
              "The dumps are not part of the package; pass --ref-dir or set "
              "FREETO_REF_DIR." % ", ".join(dirs))
        return 0
    reps = [compare_case(c, root, a.solver, a.compat, a.inside) for c, root in found]
    reps = [r for r in reps if r is not None]
    print("\nSummary:")
    for r in reps:
        nfail = sum(1 for x in r.rows if not x[1])
        print(f"  {r.case:<15s} {'PASS' if r.ok else 'FAIL'}  "
              f"({len(r.rows) - nfail}/{len(r.rows)} checks)")
    return 0 if reps and all(r.ok for r in reps) else 1


if __name__ == "__main__":
    sys.exit(main())
