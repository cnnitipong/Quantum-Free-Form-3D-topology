"""Recompute the refined binary voxel evaluation of every continuum record of a
finished study from its stored field file (docs/NOTES_quantum.md §9.11), then
rebuild gaps, summary and CSV with ``freeto.study.reprocess``.

usage: python scripts/recompute_refined.py results/quick2 [--f 2] [--dry-run]

The set-up arrays (filters, loads, supports, KE) of each (problem, mesh_control)
are obtained once from a 1-iteration MMA run with the record's volfrac target
(the mesh and boundary conditions do not depend on the optimizer).  Records whose
value changes by more than 1e-9 relative are listed; the old value is kept in
``refined_compliance_v1``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def setup_for(problem, mc, volfrac):
    from freeto.core import run_freeto
    from freeto.examples import example_config
    dbg = {}
    cfg = example_config(problem, mesh_control=mc, optimizer="MMA", max_iter=1,
                         eval_crisp=False, volfrac=volfrac)
    res = run_freeto(cfg, log=None, _debug=dbg)
    si = res.setup_info
    E0 = float(cfg.youngs_modulus)
    return dict(nelx=si["nelx"], nely=si["nely"], nelz=si["nelz"], ele=np.asarray(dbg["ele"]),
                Hn=dbg["Hn"], Hns=dbg["Hns"], F=dbg["F"], fixeddof=dbg["fixeddof"],
                KE=dbg["KE"], E0=E0, Evoid=1e-3 + 0.001 ** 3 * (E0 - 1e-3))  # as core.py (SIMP)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--f", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from freeto.evaluate import refined_voxel_compliance
    out = Path(a.out_dir)
    d = json.loads((out / "results.json").read_text())
    recs = d["records"]
    cache = {}
    changed, n_done, t0 = [], 0, time.time()
    for r in recs:
        if r.get("kind") != "continuum" or r.get("error") or not r.get("fields_file"):
            continue
        fp = out / r["fields_file"]
        if not fp.exists():
            print("missing fields", r["run_id"])
            continue
        key = (r["problem"], r.get("mesh_control"), round(float(r["volfrac_target"]), 6))
        if key not in cache:
            cache[key] = setup_for(*key)
        S = cache[key]
        full = np.load(fp)["full_pre"].astype(float)
        res = refined_voxel_compliance(full, S["Hn"], S["Hns"], S["nelx"], S["nely"], S["nelz"],
                                       S["ele"], float(r["volfrac_target"]), S["F"],
                                       S["fixeddof"], S["KE"], S["E0"], S["Evoid"], f=a.f)
        old = r.get("refined_compliance")
        new = float(res["compliance"])
        n_done += 1
        if old is None or abs(new - old) > 1e-9 * max(abs(old), 1e-300):
            changed.append((r["run_id"], old, new))
            if not a.dry_run:
                r["refined_compliance_v1"] = old
                r["refined_compliance"] = new
                r["refined_volfrac"] = float(res["volfrac"])
                r["refined_threshold"] = float(res["threshold"])
                r["refined_recomputed"] = True
    print(f"{n_done} records evaluated in {time.time() - t0:.0f} s; {len(changed)} changed:")
    for rid, o, n in changed:
        print(f"  {rid}: {o} -> {n}")
    if not a.dry_run and changed:
        d["note"] = (d.get("note") or "") + " refined evaluation recomputed with load_on_solid (NOTES §9.11)."
        (out / "results.json").write_text(json.dumps(d, indent=1, default=str))
        from freeto.study import reprocess
        reprocess(str(out), threads=1, rerun_baselines=False)


if __name__ == "__main__":
    main()
