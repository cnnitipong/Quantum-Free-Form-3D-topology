"""Write freeto/paper_reference.json from the study results of the manuscript.

usage:  python scripts/export_paper_reference.py [results/quick2/results.json]

Copies, for the paper methods of freeto.paper (S3/S4 runs of suite "quick2"),
the per-seed published values (iterations, native compliance / volume, crisp
and refined binary voxel compliance / volume, refined gap, wall time) and the
MMA reference run's refined compliance of every example.  The web app shows
them next to a run with the paper's settings (docs/PAPER_SETTINGS.md).
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from freeto.paper import METHOD_ORDER, PAPER_EXAMPLES, paper_methods, paper_run  # noqa: E402

KEYS = ("run_id", "iterations", "compliance", "volume_fraction", "crisp_compliance",
        "crisp_volfrac", "refined_compliance", "refined_volfrac", "refined_f",
        "gap_refined", "c_ref_refined", "wall_time", "audit_ok")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    src = argv[0] if argv else os.path.join(ROOT, "results", "quick2", "results.json")
    res = json.load(open(src, encoding="utf-8"))
    recs = res["records"]
    out = {"source": "FreeTO-Python study suite 'quick2' (manuscript v2), results.json "
                     f"created {res.get('created')}",
           "records": {}, "mma_reference_refined": {}}
    for pb in PAPER_EXAMPLES:
        for m in paper_methods(pb):
            run = paper_run(pb, m)
            for r in recs:
                if (r.get("kind") == "continuum" and r.get("study") == run["study"]
                        and r.get("problem") == pb and r.get("label") == m
                        and int(r.get("mesh_control")) == int(run["mesh_control"])
                        and not r.get("error")):
                    out["records"].setdefault(pb, {}).setdefault(m, {})[str(r["seed"])] = {
                        k: r.get(k) for k in KEYS}
        mma = out["records"].get(pb, {}).get("MMA", {}).get("0")
        if mma:
            out["mma_reference_refined"][pb] = mma["refined_compliance"]
    dst = os.path.join(ROOT, "freeto", "paper_reference.json")
    with open(dst, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=False)
        fh.write("\n")
    n = sum(len(s) for p in out["records"].values() for s in p.values())
    print(f"wrote {dst}: {n} records, methods {list(METHOD_ORDER)}")


if __name__ == "__main__":
    main()
