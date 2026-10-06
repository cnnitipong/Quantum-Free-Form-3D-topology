"""Physics / connectivity audit of a finished FreeTO design (docs/AUDIT_API.md).

    from freeto.audit import audit_result, audit_figure
    rep = audit_result(res, cfg)            # plain JSON-serialisable dict
    audit_figure(res, cfg, "audit.png")     # 3-panel overlay figure

What is checked (docs/PHYSICS_AUDIT.md, the audit that found the QUBO
connectivity defect):

* the **crisp design** is the common crisp evaluation of the returned design
  (:mod:`freeto.evaluate`): the pre-smoothing filtered field on FreeTO's 4x
  fine grid, thresholded so that the element densities have the target volume
  fraction (the threshold of ``extra["crisp_threshold"]`` when the run was
  evaluated with ``eval_crisp``, otherwise the same threshold computed here
  exactly by a weighted quantile instead of the bisection);
* components are the face-connected components of that fine-grid solid
  (``scipy.ndimage.label``), i.e. of the true crisp geometry; element-level
  labels (majority of the element's fine window) are used only for display.
  Labelling the element grid at rho >= 0.5 instead would falsely split thin
  members (it splits the MMA cantilever's load block off, PHYSICS_AUDIT §6);
* the **main** component is the one touching the most support elements
  (active elements with a fixed node); floating = crisp solid outside it;
* a node is *on* a component when one of its adjacent active elements has
  that component as the majority label of its fine window (the FE element
  then has a crisp density above void), as in the audit scripts.

The audit only reports; it never changes a design.  It needs numpy and scipy;
the figure needs matplotlib (imported lazily, Agg backend).
"""
from __future__ import annotations

import math
import os

import numpy as np

__all__ = ["audit_result", "audit_figure", "LiveAudit", "audit_context"]

NGRID = 4
VOID = 0.001
LOAD_TOL = 0.99          # loads_solid / load_on_main pass threshold
FLOAT_TOL = 0.01         # single_grounded_body: floating volume fraction
VOL_TOL = 0.02           # volume_target
MAX_COMPONENTS_LISTED = 50

_NOFF = None


def _face():
    from scipy import ndimage
    return ndimage.generate_binary_structure(3, 1)


def _node_offsets(nelx, nely):
    nyp, nxy = nely + 1, (nelx + 1) * (nely + 1)
    return np.array([1, 1 + nyp, nyp, 0, nxy + 1, nxy + 1 + nyp, nxy + nyp, nxy])


def _element_nodes(e, nelx, nely):
    """(len(e), 8) node indices (MATLAB node order) of full-grid elements e."""
    nyp, nxy = nely + 1, (nelx + 1) * (nely + 1)
    r, c, p = e % nely, (e // nely) % nelx, e // (nely * nelx)
    n0 = r + nyp * c + nxy * p
    return n0[:, None] + _node_offsets(nelx, nely)[None, :]


def _rb_rank(fd, nelx, nely):
    """Rigid-body modes (of 6) restrained by the fixed DOFs fd."""
    from .fe import _rigid_body_modes
    fd = np.asarray(fd, dtype=np.int64)
    if fd.size == 0:
        return 0
    nd = fd // 3
    r = nd % (nely + 1)
    c = (nd // (nely + 1)) % (nelx + 1)
    p = nd // ((nely + 1) * (nelx + 1))
    Bm = _rigid_body_modes(np.stack([c, -r, p], axis=1).astype(float))
    R = Bm[3 * np.arange(nd.size) + fd % 3]
    sv = np.linalg.svd(R, compute_uv=False)
    return int(np.count_nonzero(sv > sv.max() * 1e-9)) if sv.size and sv.max() > 0 else 0


# ---------------------------------------------------------------------------
# setup context
# ---------------------------------------------------------------------------
def audit_context(res, cfg_or_setup=None):
    """The setup arrays the audit needs.  Taken from ``res.audit_ctx`` (set
    by :func:`freeto.core.run_freeto`); if absent, rebuilt from the config
    (prepare_domain etc.; the returned smoothed design then stands in for the
    pre-smoothing field), or taken from ``cfg_or_setup`` if it is a dict."""
    ctx = getattr(res, "audit_ctx", None)
    if isinstance(cfg_or_setup, dict) and "ele" in cfg_or_setup:
        ctx = cfg_or_setup
    if ctx is not None:
        out = dict(ctx)
        out.setdefault("field_source", "pre-smoothing field of the returned design")
        return out
    if cfg_or_setup is None:
        cfg_or_setup = getattr(res, "config", None)
    if cfg_or_setup is None:
        raise ValueError("audit_result needs the run's FreeTOConfig (or res.audit_ctx)")
    from .filters import HnHns3D
    from .mesh import element_dofs, force_vectors, prepare_domain, support_dofs
    c = cfg_or_setup.normalized()
    dom = prepare_domain(c.mesh_control, c.domain, c.fixed, c.xfixed, c.yfixed, c.zfixed,
                         c.forces, c.keepdom, c.keep_bc, c.keep_bcx, c.keep_bcy, c.keep_bcz,
                         compat=c.compat, inside_mode=c.inside_mode)
    ele = np.flatnonzero(dom.oute.ravel(order="F") != 0)
    MusD = np.flatnonzero(dom.outeM.ravel(order="F") == 1)
    F, _ = force_vectors(dom.Fn, dom.ndof, c.fmagx, c.fmagy, c.fmagz, c.loadtype)
    fixeddof = support_dofs(dom.sup_all, dom.sup_x, dom.sup_y, dom.sup_z)
    n_vec = np.unique(element_dofs(dom.nelx, dom.nely, dom.nelz, ele))
    Hn, Hns = HnHns3D(dom.nelx, dom.nely, dom.nelz, 1)
    full = np.asarray(res.eleden, dtype=float).ravel(order="F")
    g = dom.grid
    return {"nelx": dom.nelx, "nely": dom.nely, "nelz": dom.nelz, "ele": ele, "MusD": MusD,
            "fixeddof": np.intersect1d(fixeddof, n_vec), "F": F, "Hn": Hn, "Hns": Hns,
            "full_pre": full, "h": g.ssz, "origin": np.array([g.x[0], g.y[0], g.z[0]]),
            "regions": dict(dom.regions or {}), "volfrac": float(c.volfrac),
            "binary_design": (getattr(res, "extra", None) or {}).get("binary_design"),
            "mesh_control": c.mesh_control, "field_is_returned": True,
            "field_source": "returned (smoothed) design; setup rebuilt from the config"}


# ---------------------------------------------------------------------------
# crisp field + components
# ---------------------------------------------------------------------------
def _window_weights(act, f=NGRID):
    """Number of active element windows containing each fine-grid point."""
    nely, nelx, nelz = act.shape
    w = np.zeros((f * nely + 1, f * nelx + 1, f * nelz + 1), dtype=np.int16)
    a8 = act.astype(np.int16)
    for a in range(f + 1):
        for b in range(f + 1):
            for d in range(f + 1):
                w[a:a + f * nely:f, b:b + f * nelx:f, d:d + f * nelz:f] += a8
    return w


def _threshold_for_volume(xg, w, nn, target, f=NGRID):
    """Threshold t such that the crisp element volume (window averages of
    xg > t, void 0.001) over the nn active elements is closest to target --
    the objective of freeto.evaluate.crisp_projection, solved exactly."""
    sel = w > 0
    vals = xg[sel]
    wt = w[sel].astype(np.float64)
    order = np.argsort(-vals, kind="stable")
    vals, wt = vals[order], wt[order]
    cw = np.concatenate([[0.0], np.cumsum(wt)])
    m = (f + 1) ** 3 * nn
    vol = (cw * (1.0 - VOID) + VOID * m) / m          # volume when the k largest are solid
    # admissible k: thresholds strictly between distinct values
    ok = np.ones(vals.size + 1, dtype=bool)
    ok[1:-1] = vals[:-1] > vals[1:]
    k = int(np.flatnonzero(ok)[np.argmin(np.abs(vol[ok] - target))])
    if k == 0:
        return float(vals[0]) + 1e-12 if vals.size else 0.0
    if k == vals.size:
        return float(vals[-1]) - 1e-12
    return 0.5 * (float(vals[k - 1]) + float(vals[k]))


def _gather_windows(lab, rows, cols, pages, f=NGRID):
    """(n, (f+1)^3) labels of the fine windows of elements (rows, cols, pages)."""
    W = np.empty((rows.size, (f + 1) ** 3), dtype=lab.dtype)
    o = 0
    for a in range(f + 1):
        for b in range(f + 1):
            for d in range(f + 1):
                W[:, o] = lab[f * rows + a, f * cols + b, f * pages + d]
                o += 1
    return W


def _row_labels(W):
    """Per row of window labels: the most frequent positive label (0 if
    none) and, for the few rows holding several labels, their label sets.
    Returns (mode, mixed_rows, mixed_sets)."""
    big = np.iinfo(W.dtype).max
    Wp = np.where(W > 0, W, big)
    mn = Wp.min(axis=1)
    mx = W.max(axis=1)
    mode = np.where(mx > 0, mn, 0).astype(np.int64)
    mixed = np.flatnonzero((mx > 0) & (mn != mx))
    sets = []
    for i in mixed:
        v = W[i][W[i] > 0]
        u, cnt = np.unique(v, return_counts=True)
        mode[i] = u[np.argmax(cnt)]
        sets.append(u)
    return mode, mixed, sets


def _labels_in_rows(rows, mode, mixed, sets, n):
    """count[k] = number of the given rows whose window holds label k."""
    cnt = np.zeros(n + 1, dtype=np.int64)
    if rows.size == 0:
        return cnt
    sel = np.zeros(mode.size, dtype=bool)
    sel[rows] = True
    is_mixed = np.zeros(mode.size, dtype=bool)
    is_mixed[mixed] = True
    uni = sel & ~is_mixed & (mode > 0)
    cnt += np.bincount(mode[uni], minlength=n + 1)
    for i, u in zip(mixed, sets):
        if sel[i]:
            cnt[u] += 1
    return cnt


def _analyse(res, cfg_or_setup):
    from scipy import ndimage
    from .evaluate import fine_field
    from .smoothedge import window_reduce
    ctx = audit_context(res, cfg_or_setup)
    nelx, nely, nelz = int(ctx["nelx"]), int(ctx["nely"]), int(ctx["nelz"])
    shape = (nely, nelx, nelz)
    nele = nelx * nely * nelz
    ele = np.asarray(ctx["ele"], dtype=np.int64)
    nn = ele.size
    vt = float(ctx["volfrac"])
    extra = getattr(res, "extra", None) or {}
    act_f = np.zeros(nele, dtype=bool)
    act_f[ele] = True
    act = act_f.reshape(shape, order="F")
    # ---- crisp fine field --------------------------------------------
    full_pre = ctx.get("full_pre")
    field = full_pre if full_pre is not None else np.asarray(res.eleden, dtype=float).ravel(order="F")
    xg = fine_field(np.asarray(field, dtype=float), ctx["Hn"], ctx["Hns"],
                    nelx, nely, nelz, NGRID)
    w = _window_weights(act)
    gd = getattr(res, "gridden", None)
    if ctx.get("field_is_returned") and gd is not None and np.shape(gd) == xg.shape:
        # no pre-smoothing field: audit FreeTO's rendered geometry instead
        xg = np.asarray(gd, dtype=float)
        thr = float(getattr(res, "ls", 0.0))
        extra = dict(extra, crisp_threshold=thr, crisp_target=vt)
        ctx = dict(ctx, field_is_returned=False,
                   field_source="FreeTO's rendered level set (res.gridden > res.ls); setup "
                                "rebuilt from the config")
    thr = extra.get("crisp_threshold")
    if ctx.get("field_is_returned") or full_pre is None or thr is None or extra.get("crisp_target") is None or \
            abs(float(extra["crisp_target"]) - vt) > 1e-9:
        thr = _threshold_for_volume(xg, w, nn, vt)
        thr_src = "computed by the audit (weighted quantile)"
    else:
        thr_src = ("res.ls" if "level set" in str(ctx.get("field_source")) else
                   "extra['crisp_threshold'] (eval_crisp)")
    thr = float(thr)
    fm = (xg > thr) & (w > 0)
    rho_c = (window_reduce(np.where(fm, 1.0, VOID), NGRID, "sum")
             / float((NGRID + 1) ** 3))                       # (nely, nelx, nelz)
    rho_c = np.where(act, rho_c, 0.0)
    v_crisp = float(rho_c[act].sum() / max(nn, 1))
    # ---- BC sets -----------------------------------------------------
    nnodes = (nelx + 1) * (nely + 1) * (nelz + 1)
    fd = np.asarray(ctx["fixeddof"], dtype=np.int64)
    fixed_node = np.zeros(nnodes, dtype=bool)
    fixed_node[fd // 3] = True
    F = ctx["F"]
    F = F.toarray() if hasattr(F, "toarray") else np.asarray(F)
    F = F.reshape(F.shape[0], -1)
    Fn_case = np.abs(F).reshape(nnodes, 3, F.shape[1]).sum(axis=1)   # (nnodes, ncase)
    Fmag = Fn_case.sum(axis=1)
    load_node = Fmag > 0
    nodes_act = _element_nodes(ele, nelx, nely)                     # (nn, 8)
    sup_rows = np.flatnonzero(fixed_node[nodes_act].any(axis=1))   # active rows
    load_rows = np.flatnonzero(load_node[nodes_act].any(axis=1))
    r_, c_, p_ = ele % nely, (ele // nely) % nelx, ele // (nely * nelx)
    # ---- components of the crisp fine-grid solid -----------------------
    lab, n = ndimage.label(fm, structure=_face())
    lab = lab.astype(np.int32)
    W = _gather_windows(lab, r_, c_, p_)
    size = np.bincount(lab.ravel(), minlength=n + 1)[1:].astype(float)
    mode, mixed, sets = _row_labels(W) if n else (np.zeros(nn, dtype=np.int64),
                                                   np.zeros(0, dtype=np.int64), [])
    del W
    nsup = _labels_in_rows(sup_rows, mode, mixed, sets, n)
    nload = _labels_in_rows(load_rows, mode, mixed, sets, n)
    n_touch = _labels_in_rows(np.arange(nn), mode, mixed, sets, n)
    if n:
        order = sorted(range(1, n + 1), key=lambda i: (-nsup[i], -size[i - 1]))
        remap = np.zeros(n + 1, dtype=np.int32)
        for k, i in enumerate(order):
            remap[i] = k + 1
        inv = np.argsort(remap)          # new label -> old label (inv[0] = 0)
        lab = remap[lab]
        mode = remap[mode]
        size = size[inv[1:] - 1]
        nsup, nload, n_touch = nsup[inv], nload[inv], n_touch[inv]
    tot = float(fm.sum())

    def node_flags(el_mask):
        out = np.zeros(nnodes, dtype=bool)
        out[nodes_act[el_mask].ravel()] = True
        return out
    mainN = node_flags(mode == 1)
    anyN = node_flags(mode > 0)
    lw = Fmag[load_node]
    lsum = float(lw.sum()) or 1.0
    load_on_main = float(lw[mainN[load_node]].sum() / lsum)
    load_on_solid = float(lw[anyN[load_node]].sum() / lsum)
    grounded = {int(k) for k in np.flatnonzero(nsup) if k > 0}
    loaded = {int(k) for k in np.flatnonzero(nload) if k > 0}
    main_pts = float(size[0]) if n else 0.0
    floating_frac = float(1.0 - main_pts / tot) if tot > 0 else 1.0
    ung = sum(size[k - 1] for k in range(1, n + 1) if k not in grounded)
    ungrounded_frac = float(ung / tot) if tot > 0 else 1.0
    rank_main = _rb_rank(fd[mainN[fd // 3]], nelx, nely) if n else 0
    rank_all = _rb_rank(fd, nelx, nely)
    # element-level labels (display): majority label of elements >= half full
    rho_act = rho_c.ravel(order="F")[ele]
    lab_el_act = np.where(rho_act >= 0.5, mode, 0)
    lab_el = np.zeros(nele, dtype=np.int32)
    lab_el[ele] = lab_el_act
    lab_el = lab_el.reshape(shape, order="F")
    # components list (main first, then by size)
    comps = []
    hf = float(ctx["h"]) / NGRID
    org = np.asarray(ctx["origin"], dtype=float)
    ytop = org[1] + nely * float(ctx["h"])
    sl = ndimage.find_objects(lab) if n else []
    n_el = np.bincount(lab_el_act, minlength=n + 1)
    for k in sorted(range(1, n + 1), key=lambda k: (k != 1, -size[k - 1]))[:MAX_COMPONENTS_LISTED]:
        s = sl[k - 1]
        x0, x1 = org[0] + hf * s[1].start, org[0] + hf * (s[1].stop - 1)
        y0, y1 = ytop - hf * (s[0].stop - 1), ytop - hf * s[0].start
        z0, z1 = org[2] + hf * s[2].start, org[2] + hf * (s[2].stop - 1)
        comps.append({"label": int(k), "n_elements": int(n_el[k]),
                      "n_elements_touched": int(n_touch[k]),
                      "vol_frac": float(size[k - 1] / tot) if tot else 0.0,
                      "grounded": bool(k in grounded), "has_load": bool(k in loaded),
                      "bbox_mm": [[round(float(x0), 4), round(float(y0), 4), round(float(z0), 4)],
                                  [round(float(x1), 4), round(float(y1), 4), round(float(z1), 4)]]})
    sup_el_mask = np.zeros(nele, dtype=bool)
    sup_el_mask[ele[sup_rows]] = True
    load_el_mask = np.zeros(nele, dtype=bool)
    load_el_mask[ele[load_rows]] = True
    supports_solid = float(np.mean(rho_act[sup_rows] >= 0.5)) if sup_rows.size else 0.0
    # ---- per-case load arrows (centroid of the loaded nodes, resultant) --
    nyp, nxp = nely + 1, nelx + 1
    load_cases = []
    for i in range(F.shape[1]):
        nd = np.flatnonzero(Fn_case[:, i] > 0)
        if nd.size == 0:
            continue
        rr, cc, pp = nd % nyp, (nd // nyp) % nxp, nd // (nyp * nxp)
        h = float(ctx["h"])
        ctr = [org[0] + h * cc.mean(), ytop - h * rr.mean(), org[2] + h * pp.mean()]
        vec = F[:, i].reshape(nnodes, 3).sum(axis=0)
        load_cases.append({"centroid_mm": [float(v) for v in ctr], "resultant_N": [float(v) for v in vec]})
    return {"ctx": ctx, "shape": shape, "act": act, "rho_c": rho_c, "v_crisp": v_crisp,
            "thr": thr, "thr_src": thr_src, "n": int(n), "floating_frac": floating_frac,
            "ungrounded_frac": ungrounded_frac, "load_on_main": load_on_main,
            "load_on_solid": load_on_solid, "rank_main": rank_main, "rank_all": rank_all,
            "n_fixed_dofs": int(fd.size), "lab_el": lab_el, "comps": comps,
            "grounded": grounded, "sup_el": sup_el_mask.reshape(shape, order="F"),
            "load_el": load_el_mask.reshape(shape, order="F"),
            "supports_solid": supports_solid, "n_support_el": int(sup_rows.size),
            "n_load_nodes": int(np.count_nonzero(load_node)), "load_cases": load_cases,
            "main_has_support": bool(1 in grounded)}


# ---------------------------------------------------------------------------
# live (per-iteration) check of a binary design
# ---------------------------------------------------------------------------
class LiveAudit:
    """Cheap component statistics of a 0/1 element design (QUBO binary
    variables): face-connected components on the element grid; main = the one
    with the most support elements.  ``live(x_active)`` -> {"n_components",
    "floating_frac", "ungrounded_frac"}."""

    def __init__(self, nelx, nely, nelz, ele, fixeddof):
        self.shape = (nely, nelx, nelz)
        self.nele = nelx * nely * nelz
        self.ele = np.asarray(ele, dtype=np.int64)
        nnodes = (nelx + 1) * (nely + 1) * (nelz + 1)
        fixed = np.zeros(nnodes, dtype=bool)
        fixed[np.asarray(fixeddof, dtype=np.int64) // 3] = True
        sup = np.zeros(self.nele, dtype=bool)
        sup[self.ele] = fixed[_element_nodes(self.ele, nelx, nely)].any(axis=1)
        self.sup = sup.reshape(self.shape, order="F")
        self.structure = _face()

    def __call__(self, x_active):
        from scipy import ndimage
        s = np.zeros(self.nele, dtype=bool)
        s[self.ele] = np.asarray(x_active) > 0.5
        s = s.reshape(self.shape, order="F")
        lab, n = ndimage.label(s, structure=self.structure)
        tot = int(s.sum())
        if n == 0 or tot == 0:
            return {"n_components": 0, "floating_frac": 1.0, "ungrounded_frac": 1.0}
        size = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        nsup = np.bincount(lab[self.sup], minlength=n + 1)[1:]
        main = int(np.lexsort((-size, -nsup))[0])
        grounded = nsup > 0
        return {"n_components": int(n), "floating_frac": float(1.0 - size[main] / tot),
                "ungrounded_frac": float(size[~grounded].sum() / tot)}


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _chk(passed, value, detail):
    return {"pass": bool(passed), "value": value, "detail": str(detail)}


def _regions_check(ctx, n_load_nodes):
    regs = ctx.get("regions") or {}
    bad, info = [], {}
    for role, d in regs.items():
        info[role] = {k: d.get(k) for k in ("file", "nodes", "elements", "elements_outside", "kept")}
        if role == "keepdom":
            if not d.get("elements"):
                bad.append(f"keepdom ({d.get('file')}) holds no element")
        elif d.get("nodes") is not None and d["nodes"] == 0:
            bad.append(f"{role} ({d.get('file')}) holds no grid node")
    MusD = np.asarray(ctx.get("MusD", []), dtype=np.int64)
    ele = np.asarray(ctx["ele"], dtype=np.int64)
    n_keep = int(np.count_nonzero(np.isin(MusD, ele)))
    n_out = int(MusD.size - n_keep)
    notes = []
    for role, d in regs.items():
        if role != "keepdom" and d.get("kept") and d.get("elements") == 0:
            notes.append(f"{role} keeps no element (keep_bc)")
    if n_out:
        notes.append(f"{n_out} keep element(s) outside the design domain")
    value = {"regions": info, "n_keep": n_keep, "n_keep_outside_domain": n_out,
             "n_support_nodes": int(np.unique(np.asarray(ctx["fixeddof"]) // 3).size),
             "n_load_nodes": int(n_load_nodes)}
    if not regs:
        detail = "region counts not available (setup without region diagnostics)"
    else:
        detail = ("; ".join(bad) if bad else "every keepdom region holds elements and every "
                  "support / load region holds grid nodes")
    if notes:
        detail += " (note: " + "; ".join(notes) + ")"
    return _chk(not bad and n_load_nodes > 0 and value["n_support_nodes"] > 0, value, detail)


def audit_result(res, cfg_or_setup=None):
    """Physics / connectivity audit of a finished FreeTO run (any optimizer).
    Returns a JSON-serialisable dict (docs/AUDIT_API.md)."""
    A = _analyse(res, cfg_or_setup)
    ctx = A["ctx"]
    extra = getattr(res, "extra", None) or {}
    ele = np.asarray(ctx["ele"])
    vt = float(ctx["volfrac"])
    eleden = getattr(res, "eleden", None)
    v_ret = (float(np.mean(np.asarray(eleden).ravel(order="F")[ele])) if eleden is not None
             else float(getattr(res, "finalvol", float("nan"))))
    # native volume as in freeto.study: OC/MMA report the compliance of the
    # iterate entering the last iteration, whose volume is history volfrac[-2]
    # (FreeTO's returned field is smoothed once more); QUBO/BESO return the
    # design of res.finalvol
    cfg_ = getattr(res, "config", None)
    hv = list((getattr(res, "history", None) or {}).get("volfrac", []))
    opt = str(getattr(cfg_, "optimizer", "") or "").upper()
    if opt in ("OC", "MMA") and len(hv) >= 2:
        v_nat, v_src = float(hv[-2]), "volume of the reported iterate (history volfrac[-2])"
    elif math.isfinite(float(getattr(res, "finalvol", float("nan")))):
        v_nat, v_src = float(res.finalvol), "res.finalvol"
    else:
        v_nat, v_src = v_ret, "mean density of res.eleden"
    checks = {}
    checks["bc_supports_present"] = _chk(
        A["n_fixed_dofs"] > 0 and A["rank_all"] == 6,
        {"n_fixed_dofs": A["n_fixed_dofs"], "rigid_body_modes_restrained": A["rank_all"]},
        f"{A['n_fixed_dofs']} fixed DOF(s) on the active domain restrain "
        f"{A['rank_all']} of 6 rigid-body modes")
    checks["supports_solid"] = _chk(
        A["supports_solid"] > 0 and A["rank_main"] == 6, round(A["supports_solid"], 6),
        f"{100 * A['supports_solid']:.1f} % of the {A['n_support_el']} support elements are solid "
        f"(crisp rho >= 0.5); the main body is held by fixed DOFs restraining {A['rank_main']} "
        "of 6 rigid-body modes (pass: some support material solid and all 6 modes held; a "
        "design need not use the whole support area)")
    checks["loads_solid"] = _chk(
        A["load_on_solid"] >= LOAD_TOL, round(A["load_on_solid"], 6),
        f"{100 * A['load_on_solid']:.2f} % of |F| acts on nodes next to crisp material "
        f"(pass >= {100 * LOAD_TOL:.0f} %)")
    checks["load_on_main"] = _chk(
        A["load_on_main"] >= LOAD_TOL, round(A["load_on_main"], 6),
        f"{100 * A['load_on_main']:.2f} % of |F| acts on the main (supported) component")
    sgb = A["floating_frac"] < FLOAT_TOL and checks["load_on_main"]["pass"] and A["n"] > 0
    checks["single_grounded_body"] = _chk(
        sgb, {"n_components": A["n"], "floating_frac": round(A["floating_frac"], 6),
              "ungrounded_frac": round(A["ungrounded_frac"], 6)},
        f"{A['n']} face-connected crisp component(s); {100 * A['floating_frac']:.2f} % of the "
        f"crisp solid lies outside the main component ({100 * A['ungrounded_frac']:.2f} % touches "
        f"no support); pass: floating < {100 * FLOAT_TOL:.0f} % and loads on the main body")
    checks["keep_regions_captured"] = _regions_check(ctx, A["n_load_nodes"])
    # the design's own volume measures: the native one (study convention), the
    # pre-smoothing filtered field (what the filter/volume constraint act on,
    # keep elements = 1) and, for QUBO/BESO, the binary design variables
    fp = ctx.get("full_pre")
    vols = {"native": v_nat}
    if fp is not None:
        vols["pre_smoothing_field"] = float(np.mean(np.asarray(fp, dtype=float)[ele]))
    bd_ = ctx.get("binary_design")
    if bd_ is not None and np.size(bd_) == ele.size:
        vols["binary_design"] = float(np.mean(np.asarray(bd_) > 0.5))
    dv = {k: abs(v - vt) for k, v in vols.items() if math.isfinite(v)}
    best = min(dv, key=dv.get) if dv else "native"
    checks["volume_target"] = _chk(
        bool(dv) and dv[best] <= VOL_TOL, round(v_nat, 6),
        f"native volume {v_nat:.4f} ({v_src}); "
        + ", ".join(f"{k} {v:.4f}" for k, v in vols.items() if k != "native")
        + f"; returned field {v_ret:.4f}; volfrac {vt:g} (pass: one of the design's own volume "
          f"measures within {VOL_TOL:g}; closest: {best})")
    cc = extra.get("crisp_compliance")
    rep = {"ok": all(c["pass"] for c in checks.values()), "checks": checks,
           "n_components": A["n"], "floating_frac": float(A["floating_frac"]),
           "crisp_compliance": (float(cc) if cc is not None and math.isfinite(float(cc)) else None),
           "native_compliance": float(getattr(res, "comp", float("nan"))),
           "volume_fraction": v_nat, "components": A["comps"],
           # additions (docs/AUDIT_API.md "implementation notes")
           "ungrounded_frac": float(A["ungrounded_frac"]),
           "volume_fraction_returned_field": v_ret,
           "crisp_volfrac": A["v_crisp"], "crisp_threshold": A["thr"],
           "crisp_threshold_source": A["thr_src"], "field_source": ctx.get("field_source"),
           "rigid_body_modes_main": A["rank_main"], "load_cases": A["load_cases"],
           "n_components_listed": len(A["comps"]),
           "setup_warnings": list((getattr(res, "setup_info", None) or {}).get("setup_warnings", []))}
    bd = ctx.get("binary_design")
    if bd is not None and np.size(bd) == ele.size:
        live = LiveAudit(int(ctx["nelx"]), int(ctx["nely"]), int(ctx["nelz"]), ele, ctx["fixeddof"])
        rep["binary"] = live(bd)
    rep["failed"] = [k for k, c in checks.items() if not c["pass"]]
    # physics = everything except the volume bookkeeping (a design that is
    # off its volume target is not physically invalid)
    rep["physics_ok"] = all(c["pass"] for k, c in checks.items() if k != "volume_target")
    rep["volumes"] = vols
    return rep


# ---------------------------------------------------------------------------
# figure (vendored from the audit's plot_audit.py)
# ---------------------------------------------------------------------------
C_MAIN, C_FLOAT, C_SUP, C_LOAD = "#8a8a8a", "#ff8c00", "#1f5fbf", "#d00000"
_FACES = {
    (0, 1): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (0, -1): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (1, 1): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (1, -1): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (2, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (2, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
}
_PLANES = {"x": (0, 2, 1, ("z (mm)", "y (mm)")), "y": (1, 0, 2, ("x (mm)", "z (mm)")),
           "z": (2, 0, 1, ("x (mm)", "y (mm)"))}
_VIEW = dict(elev=24, azim=-120)


def _to_xyz(a):
    return np.transpose(np.asarray(a)[::-1], (1, 0, 2))


def _voxel_faces(mask, colors, origin, h, cam):
    light = np.array([0.35, 0.85, 0.45])
    light /= np.linalg.norm(light)
    polys, cols = [], []
    M = np.pad(mask, 1)
    for (ax, sg), corners in _FACES.items():
        if cam[ax] * sg < 0:
            continue
        sl = [slice(1, -1)] * 3
        sl[ax] = slice(2, None) if sg > 0 else slice(0, -2)
        idx = np.argwhere(mask & ~M[tuple(sl)])
        if len(idx) == 0:
            continue
        verts = (idx[:, None, :] + np.array(corners, float)[None]) * h + origin
        nrm = np.zeros(3)
        nrm[ax] = sg
        shade = 0.55 + 0.45 * max(float(nrm @ light), 0.0) if sg * light[ax] > 0 else 0.55
        polys.append(verts)
        cols.append(np.clip(colors[tuple(idx.T)] * shade, 0, 1))
    if not polys:
        return np.zeros((0, 4, 3)), np.zeros((0, 3))
    return np.concatenate(polys), np.concatenate(cols)


def _outline(ax, m, ext, color, lw=2.5, ls="-", z=5):
    from matplotlib.collections import LineCollection
    if not m.any():
        return
    nh, nv = m.shape
    dx, dy = (ext[1] - ext[0]) / nh, (ext[3] - ext[2]) / nv
    P = np.pad(m, 1)
    segs = []
    for i, j in np.argwhere(P[1:, 1:-1] != P[:-1, 1:-1]):
        x = ext[0] + i * dx
        segs.append([(x, ext[2] + j * dy), (x, ext[2] + (j + 1) * dy)])
    for i, j in np.argwhere(P[1:-1, 1:] != P[1:-1, :-1]):
        y = ext[2] + j * dy
        segs.append([(ext[0] + i * dx, y), (ext[0] + (i + 1) * dx, y)])
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, linestyles=ls, zorder=z))


def _slice(A, plane, idx):
    fa, ha, va, _ = _PLANES[plane]
    S = np.take(A, idx, axis=fa)
    rem = [a for a in range(3) if a != fa]
    return S if rem == [ha, va] else S.T


def _proj(m, plane):
    fa, ha, va, _ = _PLANES[plane]
    P = m.any(axis=fa)
    rem = [a for a in range(3) if a != fa]
    return P if rem == [ha, va] else P.T


def audit_figure(res, cfg_or_setup=None, path_png="audit.png", title=None, report=None):
    """Write the 3-panel physics-check figure: (a) 3D crisp design (main grey,
    floating orange, supports blue, loads red + load arrows), (b) mid-depth
    slice of the returned (native) density with the support/load outlines,
    (c) the component map on the same slice.  Returns ``path_png``."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, to_rgb
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as e:  # pragma: no cover
        raise ImportError("audit_figure needs matplotlib (pip install matplotlib)") from e
    A = _analyse(res, cfg_or_setup)
    rep = report if report is not None else audit_result(res, cfg_or_setup)
    ctx = A["ctx"]
    h = float(ctx["h"])
    hv = np.full(3, h)
    org = np.asarray(ctx["origin"], dtype=float)
    comp = _to_xyz(A["lab_el"])
    sup = _to_xyz(A["sup_el"])
    load = _to_xyz(A["load_el"])
    eleden = getattr(res, "eleden", None)
    dens = _to_xyz(eleden if eleden is not None else A["rho_c"])
    crisp = comp > 0
    shape = np.array(comp.shape)
    ext = shape * h
    fs = 12
    wb = float(np.clip(5.0 * ext[0] / ext[1], 4.6, 7.5))
    rowh = float(np.clip(wb * ext[1] / ext[0] + 2.4, 4.4, 6.8))
    wa = 1.35 * wb
    fig = plt.figure(figsize=(wa + 2 * wb + 1.6, rowh + 1.9))
    head = title or "FreeTO physics check"
    verdict = "PASS" if rep["ok"] else "FAIL: " + ", ".join(rep["failed"])
    fig.suptitle(f"{head} — {verdict}", fontsize=fs + 3, fontweight="bold", x=0.01, ha="left",
                 y=0.985, color=("#1b7f3b" if rep["ok"] else "#b00020"))
    cc = rep.get("crisp_compliance")
    sub = (f"native c = {rep['native_compliance']:.4g}"
           + (f", crisp c = {cc:.4g}" if cc is not None else "")
           + f", V = {rep['volume_fraction']:.3f} (native) / {rep['crisp_volfrac']:.3f} (crisp),"
           f" components = {rep['n_components']}, floating = {100 * rep['floating_frac']:.1f} %,"
           f" loads on solid = {100 * rep['checks']['loads_solid']['value']:.1f} %,"
           f" on main = {100 * rep['checks']['load_on_main']['value']:.1f} %")
    fig.text(0.01, 0.925, sub, fontsize=fs - 1, ha="left")
    gs = fig.add_gridspec(1, 3, width_ratios=[wa, wb, wb], left=0.0, right=0.985, top=0.84,
                          bottom=0.14, wspace=0.2)
    # (a) 3D
    ax = fig.add_subplot(gs[0], projection="3d")
    col = np.ones(comp.shape + (3,))
    col[comp == 1] = to_rgb(C_MAIN)
    col[comp > 1] = to_rgb(C_FLOAT)

    def tint(c):
        return 0.45 * np.array(to_rgb(c)) + 0.55
    col[sup & crisp] = to_rgb(C_SUP)
    col[sup & ~crisp] = tint(C_SUP)
    col[load & crisp] = to_rgb(C_LOAD)
    col[load & ~crisp] = tint(C_LOAD)
    P = lambda p: np.asarray(p)[..., [0, 2, 1]] * np.array([1, -1, 1])  # noqa: E731
    ax.set_proj_type("ortho")
    srt = np.sort(ext)
    ax.set_box_aspect((ext[0], ext[2], ext[1]) / ext.max(),
                      zoom=1.45 if srt[2] > 2.5 * srt[1] else 1.05)
    ax.view_init(elev=_VIEW["elev"], azim=_VIEW["azim"])
    e_, a_ = np.radians(_VIEW["elev"]), np.radians(_VIEW["azim"])
    eye = np.array([np.cos(e_) * np.cos(a_), np.cos(e_) * np.sin(a_), np.sin(e_)])
    cam = np.sign(np.array([eye[0], eye[2], -eye[1]]))
    polys, cols = _voxel_faces(crisp | sup | load, col, org, hv, cam)
    if len(polys):
        ax.add_collection3d(Poly3DCollection(P(polys), facecolors=cols,
                                             edgecolors=np.clip(cols * 0.6, 0, 1),
                                             linewidths=0.3 if len(polys) < 40000 else 0.0))
    ax.set_xlim(org[0], org[0] + ext[0])
    ax.set_ylim(-(org[2] + ext[2]), -org[2])
    ax.set_zlim(org[1], org[1] + ext[1])
    for lc in rep.get("load_cases", []):
        vec = np.asarray(lc["resultant_N"], float)
        nv = np.linalg.norm(vec)
        if nv <= 0:
            continue
        u = vec / nv
        L = 0.2 * ext.max()
        ctr = np.asarray(lc["centroid_mm"], float)
        q = ax.quiver(*P(ctr - u * L), *P(u * L), color=C_LOAD, linewidth=3, arrow_length_ratio=0.3,
                      pivot="tail", zorder=20)
        q.set_clip_on(False)
        ax.text(*P(ctr - u * L * 1.1), f"{nv:.4g} N", color=C_LOAD, fontsize=fs - 3,
                fontweight="bold", ha="center", va="bottom", zorder=21, clip_on=False)
    ax.set_xlabel("x (mm)", labelpad=8)
    ax.set_ylabel("z (mm)", labelpad=8)
    ax.set_zlabel("y (mm)", labelpad=10)
    for a, k in ((ax.xaxis, 5), (ax.yaxis, 3), (ax.zaxis, 4)):
        a.set_major_locator(plt.MaxNLocator(k))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{(0.0 if abs(v) < 1e-9 else -v):.3g}"))
    ax.tick_params(labelsize=fs - 4, pad=1)
    ax.set_title("(a) crisp design: main grey, floating orange", fontsize=fs, loc="left")
    # (b) / (c) mid-depth z slice
    kz = comp.shape[2] // 2
    ext2 = [org[0], org[0] + ext[0], org[1], org[1] + ext[1]]
    zpos = org[2] + (kz + 0.5) * h

    def bc(axx):
        for m, c in ((sup, C_SUP), (load, C_LOAD)):
            _outline(axx, _proj(m, "z"), ext2, c, lw=1.4, ls=(0, (3, 2)), z=4)
            _outline(axx, _slice(m, "z", kz), ext2, c, lw=2.8, z=6)
    axb = fig.add_subplot(gs[1])
    im = axb.imshow(_slice(dens, "z", kz).T, origin="lower", extent=ext2, cmap="viridis",
                    vmin=0, vmax=1, interpolation="nearest", aspect="equal")
    bc(axb)
    axb.set_title(f"(b) returned density, z = {zpos:.3g} mm", fontsize=fs)
    cb = fig.colorbar(im, ax=axb, fraction=0.045, pad=0.02, shrink=0.8)
    cb.ax.tick_params(labelsize=fs - 4)
    axc = fig.add_subplot(gs[2])
    c3 = np.where(comp == 0, 0, np.where(comp == 1, 1, 2))
    axc.imshow(_slice(c3, "z", kz).T, origin="lower", extent=ext2,
               cmap=ListedColormap(["#ffffff", C_MAIN, C_FLOAT]), vmin=0, vmax=2,
               interpolation="nearest", aspect="equal")
    bc(axc)
    axc.set_title(f"(c) crisp components, z = {zpos:.3g} mm", fontsize=fs)
    for axx in (axb, axc):
        axx.set_xlabel("x (mm)")
        axx.set_ylabel("y (mm)")
        axx.xaxis.set_major_locator(plt.MaxNLocator(5))
        axx.yaxis.set_major_locator(plt.MaxNLocator(4))
    hs = [Patch(fc=C_MAIN, ec="k", label="main component"),
          Patch(fc=C_FLOAT, ec="k", label="floating component"),
          Patch(fc="none", ec=C_SUP, lw=2.5, label="support elements (pale = void; dashed = other depths)"),
          Patch(fc="none", ec=C_LOAD, lw=2.5, label="load elements (pale = void; dashed = other depths)"),
          Line2D([0], [0], color=C_LOAD, lw=3, marker=">", ms=8, label="load resultant per case")]
    fig.legend(handles=hs, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.005), frameon=False,
               fontsize=fs - 3)
    os.makedirs(os.path.dirname(os.path.abspath(path_png)), exist_ok=True)
    fig.savefig(path_png, dpi=100)
    plt.close(fig)
    return path_png
