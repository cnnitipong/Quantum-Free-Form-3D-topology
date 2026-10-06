"""Command-line interface.

Examples::

    python -m freeto.cli --example GE_bracket --mesh 40 --out ge.stl
    python -m freeto.cli --domain GE_domain.STL --force GE_force.STL \\
        --force2 GE_force.STL --mesh 80 --volfrac 0.3 --fixed GE_fixed.STL \\
        --Fmagy 0 -2000 --Fmagz 1500 0 --YoungsModulus 210e9 --out ge.stl

MATLAB name-value argument names (``--YoungsModulus``, ``--PoissonRatio``,
``--optimization``, ``--penaltySIMP``, ``--filterRadius``, ``--Fmagx``,
``--keep_BC``, ``--Symmetry1``/``--direction1``, ``--modelName`` ...) are
accepted as aliases of the Python-style flags.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading

from .core import FreeTOConfig, FreeTOError, run_freeto
from .examples import EXAMPLES, example_config
from .fe import available_solvers


def _yesno(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "y", "true", "1", "on"):
        return True
    if s in ("no", "n", "false", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected yes/no, got {v!r}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m freeto.cli",
        description="FreeTO: 3D freeform topology optimisation with smooth "
                    "boundaries (Python port).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--example", choices=sorted(EXAMPLES),
                   help="run one of the README examples (other flags override)")
    p.add_argument("--list-examples", action="store_true",
                   help="list the built-in examples and exit")
    g = p.add_argument_group("geometry (STL files, mm)")
    g.add_argument("--domain", help="design domain STL")
    g.add_argument("--force", "--force1", dest="force1", help="load region 1 STL")
    for i in range(2, 11):
        g.add_argument(f"--force{i}", dest=f"force{i}", help=argparse.SUPPRESS)
    g.add_argument("--forces", nargs="+", help="all load-region STLs (alternative)")
    g.add_argument("--fixed", help="region fixed in x, y and z")
    g.add_argument("--xfixed", help="region fixed in x")
    g.add_argument("--yfixed", help="region fixed in y")
    g.add_argument("--zfixed", help="region fixed in z")
    g.add_argument("--keepdom", help="region kept solid")
    g.add_argument("--keep-bc", "--keep_BC", dest="keep_bc", type=_yesno)
    g.add_argument("--keep-bcx", "--keep_BCx", dest="keep_bcx", type=_yesno)
    g.add_argument("--keep-bcy", "--keep_BCy", dest="keep_bcy", type=_yesno)
    g.add_argument("--keep-bcz", "--keep_BCz", dest="keep_bcz", type=_yesno)
    o = p.add_argument_group("optimisation")
    o.add_argument("--mesh", "--MeshControl", dest="mesh_control", type=int,
                   help="MeshControl (grid points along the chosen axis)")
    o.add_argument("--volfrac", type=float)
    o.add_argument("--method", "--optimization", dest="method",
                   choices=["SIMP", "SEMDOT", "simp", "semdot"])
    o.add_argument("--optimizer", choices=["OC", "MMA", "QUBO", "oc", "mma", "qubo"])
    q = p.add_argument_group("QUBO design update (--optimizer QUBO; docs/QUANTUM_API.md)")
    q.add_argument("--qubo-backend", help="exact | sa | tabu | greedy | qaoa | dwave_sa | "
                   "dwave_tabu | dwave_qpu | dwave_hybrid | qiskit_aer | ibm (default auto = sa)")
    q.add_argument("--qubo-hessian", choices=["auto", "block", "exact-block", "diag", "scalar", "none"])
    q.add_argument("--qubo-volume", choices=["bisection", "penalty"])
    q.add_argument("--qubo-block-size", type=int)
    q.add_argument("--qubo-init", choices=["solid", "oc"])
    q.add_argument("--qubo-er", type=float, help="volume reduction ratio per iteration")
    q.add_argument("--qubo-gamma", type=float, help="perimeter penalty weight")
    q.add_argument("--qubo-reads", type=int, dest="qubo_num_reads")
    q.add_argument("--qubo-seed", type=int)
    q.add_argument("--qubo-p", type=int, dest="qubo_qaoa_p", help="QAOA depth")
    q.add_argument("--list-qubo-backends", action="store_true",
                   help="list QUBO backends and whether they are usable, then exit")
    o.add_argument("--E", "--YoungsModulus", dest="youngs_modulus", type=float)
    o.add_argument("--nu", "--PoissonRatio", "--PoissonsRatio",
                   dest="poisson_ratio", type=float)
    o.add_argument("--penal", "--penaltySIMP", dest="penal", type=float)
    o.add_argument("--rmin", "--filterRadius", dest="rmin", type=float)
    o.add_argument("--fmagx", "--Fmagx", dest="fmagx", type=float, nargs="+")
    o.add_argument("--fmagy", "--Fmagy", dest="fmagy", type=float, nargs="+")
    o.add_argument("--fmagz", "--Fmagz", dest="fmagz", type=float, nargs="+")
    o.add_argument("--loadtype", choices=["distributed", "point"])
    o.add_argument("--max-iter", dest="max_iter", type=int)
    o.add_argument("--solver", choices=["auto", "cholmod", "pardiso", "superlu", "amg"])
    o.add_argument("--tolx", type=float, help="convergence tolerance on the design change")
    o.add_argument("--tol-thresh", dest="tol_thresh", type=float,
                   help="convergence threshold on the topology (grey) measure")
    o.add_argument("--beta-init", dest="beta_init", type=float)
    o.add_argument("--beta-step", dest="beta_step", type=float)
    o.add_argument("--inside-mode", dest="inside_mode", choices=["robust", "matlab"])
    s = p.add_argument_group("post-processing")
    s.add_argument("--symmetry", action="append", default=None,
                   help="PLANE[:DIRECTION], e.g. x-y:right (repeat up to 3 times)")
    for i in (1, 2, 3):
        s.add_argument(f"--Symmetry{i}", dest=f"sym{i}", help=argparse.SUPPRESS)
        s.add_argument(f"--direction{i}", dest=f"dir{i}", help=argparse.SUPPRESS)
    s.add_argument("--out", "--modelName", dest="out", help="output STL path")
    s.add_argument("--no-smooth", action="store_true",
                   help="skip the smooth3 box filter before iso-surfacing")
    s.add_argument("--npz", help="also save the result arrays to this .npz")
    s.add_argument("--quiet", action="store_true")
    return p


def config_from_args(a):
    if a.example:
        cfg = example_config(a.example)
        kw = {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}
    else:
        kw = {}
    forces = list(a.forces or [])
    if a.force1 or any(getattr(a, f"force{i}") for i in range(2, 11)):
        forces = [getattr(a, f"force{i}") for i in range(1, 11)]
        forces = [f for f in forces if f]
    if forces:
        kw["forces"] = forces
    for name in ("domain", "fixed", "xfixed", "yfixed", "zfixed", "keepdom",
                 "keep_bc", "keep_bcx", "keep_bcy", "keep_bcz", "mesh_control",
                 "volfrac", "method", "optimizer", "youngs_modulus",
                 "poisson_ratio", "penal", "rmin", "fmagx", "fmagy", "fmagz",
                 "loadtype", "max_iter", "solver", "tolx", "tol_thresh",
                 "beta_init", "beta_step", "inside_mode"):
        v = getattr(a, name, None)
        if v is not None:
            kw[name] = v
    sym = []
    for s in a.symmetry or []:
        plane, _, direction = s.partition(":")
        sym.append((plane, direction or "right"))
    for i in (1, 2, 3):
        pl = getattr(a, f"sym{i}")
        if pl:
            sym.append((pl, getattr(a, f"dir{i}") or "right"))
    if sym:
        kw["symmetry"] = sym
    qopts = {}
    for name in ("backend", "hessian", "volume", "block_size", "init", "er", "gamma",
                 "num_reads", "seed", "qaoa_p"):
        v = getattr(a, f"qubo_{name}", None)
        if v is not None:
            qopts[name] = v
    if qopts:
        kw["qubo"] = qopts
    return FreeTOConfig(**kw)


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.list_qubo_backends:
        from .quantum import available_backends
        for b in available_backends():
            print(f"{b['name']:13s} {'yes' if b['available'] else 'no ':3s} "
                  f"{b['description']}" + (f"  [{b['reason']}]" if b["reason"] else ""))
        return 0
    if a.list_examples:
        for name, ex in EXAMPLES.items():
            print(f"{name:12s} {ex['title']}\n{'':12s} {ex['description']}")
        return 0
    try:
        cfg = config_from_args(a)
        cfg.validate()
    except (ValueError, TypeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    stop = threading.Event()

    def on_sigint(sig, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        print("\nstopping after the current iteration (Ctrl-C again to abort)",
              file=sys.stderr)
        stop.set()
    signal.signal(signal.SIGINT, on_sigint)
    log = (lambda s: None) if a.quiet else (lambda s: print(s, flush=True))
    if not a.quiet:
        log(f"available solvers: {', '.join(available_solvers())}")
    try:
        res = run_freeto(cfg, stop_event=stop, log=log)
    except FreeTOError as e:
        # problems only detectable during setup (e.g. a load region without
        # grid nodes at this mesh_control): same clean message as validate()
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = a.out
    if out:
        root, ext = os.path.splitext(out)
        if ext.lower() != ".stl":
            out = root + ".stl"
        res.write_stl(out, smooth=not a.no_smooth)
        log(f"wrote {out}")
    if a.npz:
        res.save_npz(a.npz)
        log(f"wrote {a.npz}")
    log(f"comp = {res.comp:.6g}  finalvol = {res.finalvol:.4f}  elenum1 = "
        f"{res.elenum1}  elenum2 = {res.elenum2}  iterations = {res.iterations}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
