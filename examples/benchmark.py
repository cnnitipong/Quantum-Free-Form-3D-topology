#!/usr/bin/env python3
"""Time the phases of a FreeTO run (setup, FE solve, smooth-edge step).

    python examples/benchmark.py GE_bracket --mesh 40 --solver auto --iters 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from freeto import EXAMPLES, example_config, run_freeto  # noqa: E402
from freeto.fe import available_solvers                  # noqa: E402


def _peak_rss_mb():
    try:
        import resource  # POSIX only
    except ImportError:
        return float("nan")
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 2 if sys.platform == "darwin" else 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example", choices=sorted(EXAMPLES), nargs="?", default="GE_bracket")
    ap.add_argument("--mesh", type=int, default=40)
    ap.add_argument("--solver", default="auto")
    ap.add_argument("--iters", type=int, default=10, help="max iterations (0 = full run)")
    ap.add_argument("--out", help="write the resulting STL here")
    a = ap.parse_args()
    kw = dict(mesh_control=a.mesh, solver=a.solver)
    if a.iters:
        kw["max_iter"] = a.iters
    cfg = example_config(a.example, **kw)
    fe_t, sm_t, it_t = [], [], []
    setup = {}

    def cb(info):
        if info["stage"] == "setup":
            setup.update(info)
            print(f"grid {info['nelx']}x{info['nely']}x{info['nelz']}  active elements "
                  f"{info['nnele']}/{info['nele']}  free DOFs {info['nfree']}  loads "
                  f"{info['nloads']}  solver {info['solver']}  setup {info['setup_time']:.2f} s",
                  flush=True)
        else:
            fe_t.append(info["fe_time"])
            sm_t.append(info["smooth_time"])
            it_t.append(info["iter_time"])
            print(f"  it {info['iter']:4d}  c = {info['compliance']:.6g}  FE {info['fe_time']:.2f} s"
                  f"  smooth-edge {info['smooth_time']:.2f} s  total {info['iter_time']:.2f} s",
                  flush=True)
    print(f"solvers available: {', '.join(available_solvers())}")
    t0 = time.perf_counter()
    res = run_freeto(cfg, callback=cb, log=None)
    total = time.perf_counter() - t0
    print("\nphase timings (s):")
    for k, v in res.timings.items():
        print(f"  {k:<22s} {v:8.3f}")
    print(f"mean per iteration: FE {np.mean(fe_t):.3f}  smooth-edge {np.mean(sm_t):.3f}  "
          f"total {np.mean(it_t):.3f}")
    print(f"{res.iterations} iterations, compliance {res.comp:.6g}, volume fraction "
          f"{res.finalvol:.4f}, wall {total:.1f} s, peak RSS "
          f"{_peak_rss_mb():.0f} MB")
    if a.out:
        t = time.perf_counter()
        res.write_stl(a.out)
        print(f"surface + STL export {time.perf_counter() - t:.2f} s -> {a.out}")


if __name__ == "__main__":
    main()
