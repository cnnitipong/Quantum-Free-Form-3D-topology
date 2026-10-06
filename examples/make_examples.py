"""Generate the STL geometry for the new continuum example problems.

Idempotent: re-running this script regenerates every file in
``examples/STLs/generated/`` from scratch (deterministic pure-Python
geometry, no randomness), so running it twice leaves the directory
byte-for-byte identical. Run it from anywhere:

    python examples/make_examples.py

It writes domain / fixed / force STLs for the six examples registered in
``freeto/examples.py`` (cantilever_beam, mbb_beam, bridge_deck,
torsion_bracket, l_bracket, multi_load_beam) and prints each domain's bbox,
element size at its recommended mesh_control, and a watertightness check for
every generated file.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from freeto.geometry import (box, l_shape, combine, write_mesh,   # noqa: E402
                             is_watertight, face_slab, patch, bbox_of)
from freeto.mesh import build_grid                                # noqa: E402
from freeto.stl_io import read_stl                                 # noqa: E402

OUT_DIR = os.path.join(HERE, "STLs", "generated")

# Region slabs must contain grid NODES (supports / loads act on nodes) and
# element CENTRES (keepdom / keep_bc regions keep the elements whose centre is
# inside) at every mesh_control the examples are used with.  _capture computes
# the depth a slab must reach inward from a domain face so that it contains the
# outermost node layer and element-centre layer for every mesh_control in
# CAPTURE_MCS, from the exact FreeTO grid (freeto.mesh.build_grid) -- the
# grid's 1 % shrink and centring make the outermost centre lie anywhere between
# h/2 and h from the face, so a fixed "1.5 h at the recommended mesh" is not
# enough at coarse meshes (2026-10-02 correction, docs/PHYSICS_AUDIT.md §2).
CAPTURE_MCS = range(24, 101)
CAPTURE_MARGIN = 0.3   # mm beyond the deepest required layer (avoids ties)


def _h(domain_lo, domain_hi, mesh_control):
    """Approximate FreeTO element size at this mesh_control (for sizing
    region-slab thickness): the true value needs build_grid on the actual
    STL, this is the same axis-selection rule from freeto.mesh.build_grid
    applied to the raw bbox corners."""
    lo, hi = np.asarray(domain_lo, float), np.asarray(domain_hi, float)
    stt = np.abs([lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]])
    ax = int(np.argmax(stt)) // 2
    return (hi[ax] - lo[ax]) / (mesh_control - 1)


def _capture(lo, hi, axis, side, mcs=CAPTURE_MCS, margin=CAPTURE_MARGIN):
    """Inward depth (mm) from face ``axis``/``side`` of the box domain bbox
    (lo, hi) that contains the outermost element-centre layer (and therefore
    the outermost node layer, which is h/2 closer to the face) at every
    mesh_control in ``mcs``."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    corners = np.array([lo, hi])
    need = 0.0
    for mc in mcs:
        g = build_grid(corners, mc)
        c = (g.xc, g.yc, g.zc)[axis]
        need = max(need, (c[0] - lo[axis]) if side == "-" else (hi[axis] - c[-1]))
    return float(need + margin)


def _emit(name, mesh, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.stl")
    write_mesh(path, mesh)
    ok = is_watertight(mesh)
    lo, hi = bbox_of(mesh)
    print(f"  {name:28s} {'OK ' if ok else 'BAD'} tris={mesh[1].shape[0]:4d} "
          f"bbox=[{lo[0]:.2f},{hi[0]:.2f}]x[{lo[1]:.2f},{hi[1]:.2f}]x"
          f"[{lo[2]:.2f},{hi[2]:.2f}]")
    if not ok:
        raise RuntimeError(f"{name}: generated mesh is not watertight")
    return path


def make_cantilever_beam(mesh_control=50):
    print("cantilever_beam (mesh_control=%d)" % mesh_control)
    lo, hi = (0.0, 0.0, 0.0), (120.0, 40.0, 20.0)
    h = _h(lo, hi, mesh_control)
    t = max(1.5 * h, 2.0)
    _emit("cantilever_domain", box(lo, hi))
    _emit("cantilever_fixed", face_slab((lo, hi), "x-", thickness=t))
    # tip load: small patch at the free end, mid-height, full depth (a line
    # load across z), pushing -y
    force = box((120.0 - t, 17.0, -t), (120.0 + t, 23.0, 20.0 + t))
    _emit("cantilever_force", force)


def make_mbb_beam(mesh_control=80):
    print("mbb_beam (mesh_control=%d)" % mesh_control)
    lo, hi = (0.0, 0.0, 0.0), (150.0, 25.0, 25.0)
    h = _h(lo, hi, mesh_control)
    t = max(1.5 * h, 2.0)
    _emit("mbb_domain", box(lo, hi))
    # roller support: y-fixed patch near the bottom of the FAR end (x=0),
    # full depth, a couple of element widths wide (restrains ty and, being
    # spread across the full z range, rotation about x too)
    _emit("mbb_yfixed", box((-t, -t, -t), (2.0 * t, 6.0, 25.0 + t)))
    # symmetry plane: x = max face, xfixed (only the x-DOF is restrained —
    # symmetry.m mirrors about this plane, see apply_symmetry("y-z","right"));
    # a full planar x-fixed face also restrains rotation about y and z (any
    # such rotation would need a varying u_x within that plane)
    _emit("mbb_xfixed", face_slab((lo, hi), "x+", thickness=t))
    # a single small z-fixed patch at one corner: tx/ty/rx/ry/rz are already
    # restrained by xfixed + yfixed above, this adds the last DOF (tz,
    # translation along the depth/width axis) so the structure is fully
    # constrained against rigid-body motion
    _emit("mbb_zfixed", box((-t, -t, -t), (2.0 * t, 2.0 * t, t)))
    # load: top of the symmetry plane, pushing -y (the classic MBB centre
    # point load, split across the mirror by symmetry)
    # (corrected 2026-10-02: the slab now reaches the outermost element-centre
    # layer in x and y at every mesh_control >= 24, so the loaded material is a
    # keep region at every mesh; it was empty at MC 20-41 except 31)
    dx = max(t, _capture(lo, hi, 0, "+"))
    dy = max(3.0, _capture(lo, hi, 1, "+"))
    _emit("mbb_force", box((150.0 - dx, 25.0 - dy, -t), (150.0 + t, 25.0 + t, 25.0 + t)))


def make_bridge_deck(mesh_control=70):
    print("bridge_deck (mesh_control=%d)" % mesh_control)
    lo, hi = (0.0, 0.0, 0.0), (200.0, 30.0, 50.0)
    h = _h(lo, hi, mesh_control)
    t = max(1.5 * h, 2.0)
    _emit("bridge_domain", box(lo, hi))
    # Corrected 2026-10-02 (docs/PHYSICS_AUDIT.md §2): the deck slab was
    # y >= 25.65 mm and the support pads y <= 4 mm, so at MC 24, 25, 30, 31 no
    # element centre lay inside them (keepdom empty: the "kept" deck was not
    # kept, MusD = 0).  Both now reach the outermost element-centre layer at
    # every mesh_control >= 24 (deck: y >= 30 - 7.41 mm, pads: y <= 7.61 mm);
    # the load is still the distributed downward deck load and the supports
    # are still the two bottom ends (pads 8.7 mm long, full width).  The
    # captured deck is one element row (1/nely of the domain: 17-33 %), so
    # volfrac was raised 0.2 -> 0.35 to keep the problem feasible
    # (freeto/examples.py).
    # margin 0.1 mm: 7.41 mm captures exactly one row at MC 24-47 (MC 41's
    # second row centre is 7.51 mm deep); finer meshes keep two or three rows
    d_deck = _capture(lo, hi, 1, "+", margin=0.1)
    d_pad = max(4.0, _capture(lo, hi, 1, "-"))
    l_pad = max(2.0 * t, _capture(lo, hi, 0, "-"))
    left = box((-t, -t, -t), (l_pad, d_pad, 50.0 + t))
    right = box((200.0 - l_pad, -t, -t), (200.0 + t, d_pad, 50.0 + t))
    _emit("bridge_fixed", combine(left, right))
    # deck slab: the top layer, kept solid AND the load region (a
    # distributed downward load over the whole span)
    deck = box((-t, 30.0 - d_deck, -t), (200.0 + t, 30.0 + t, 50.0 + t))
    _emit("bridge_deck_slab", deck)


def make_torsion_bracket(mesh_control=28):
    print("torsion_bracket (mesh_control=%d)" % mesh_control)
    lo, hi = (0.0, 0.0, 0.0), (60.0, 30.0, 30.0)
    h = _h(lo, hi, mesh_control)
    t = max(1.5 * h, 2.0)
    _emit("torsion_domain", box(lo, hi))
    _emit("torsion_fixed", face_slab((lo, hi), "x-", thickness=t))
    # load case 1 (bending): centred patch at the free end, pushing -y
    bend = box((60.0 - t, 12.0, 12.0), (60.0 + t, 18.0, 18.0))
    _emit("torsion_force_bend", bend)
    # load case 2 (torsion): an off-axis (lever-arm) patch near one corner
    # of the free end, pushing +z — offset from the section centroid on
    # BOTH y and z so a single transverse force there produces a genuine
    # torque about the bracket's long (x) axis, not just extra bending
    torque = box((60.0 - t, 22.0, 22.0), (60.0 + t, 28.0, 28.0))
    _emit("torsion_force_torque", torque)


def make_l_bracket(mesh_control=45):
    print("l_bracket (mesh_control=%d)" % mesh_control)
    lo, hi = (0.0, 0.0, 0.0), (100.0, 100.0, 15.0)
    notch_lo, notch_hi = (35.0, 35.0, 0.0), (100.0, 100.0, 15.0)
    h = _h(lo, hi, mesh_control)
    t = max(1.5 * h, 2.0)
    _emit("L_domain", l_shape(lo, hi, notch_lo, notch_hi))
    # fixed: the top face of the vertical arm (y = 100, x in [0, 35]).
    # Corrected 2026-10-02: the slab used to span the whole bbox face
    # (x in [0, 100]), so its elements over the notch (x > 35) were keep
    # elements outside the design domain (76 at MC 30); it is now clipped to
    # the arm (outward margin only beyond the outer faces x = 0, y = 100, z).
    ty = max(t, _capture(lo, hi, 1, "+"))
    m = max(0.2 * ty, 0.05 * (hi[1] - lo[1]))
    _emit("L_fixed", box((-m, 100.0 - ty, -m), (notch_lo[0], 100.0 + m, 15.0 + m)))
    # force: the far right tip of the horizontal arm, downward
    tx = max(t, _capture(lo, hi, 0, "+"))
    force = box((100.0 - tx, -t, -t), (100.0 + t, 20.0, 15.0 + t))
    _emit("L_force", force)


def make_multi_load_beam(mesh_control=60):
    print("multi_load_beam (mesh_control=%d)" % mesh_control)
    lo, hi = (0.0, 0.0, 0.0), (120.0, 20.0, 20.0)
    h = _h(lo, hi, mesh_control)
    t = max(1.5 * h, 2.0)
    _emit("multiload_domain", box(lo, hi))
    # corrected 2026-10-02: support pads and load patches reach the outermost
    # element-centre layer at every mesh_control >= 24 (pads were y <= 4 mm
    # and patches y >= 16.95 mm: no keep element at MC 24, 28, 30, 36)
    d_pad = max(4.0, _capture(lo, hi, 1, "-"))
    d_load = max(t, _capture(lo, hi, 1, "+"))
    left = box((-t, -t, -t), (2.0 * t, d_pad, 20.0 + t))
    right = box((120.0 - 2.0 * t, -t, -t), (120.0 + t, d_pad, 20.0 + t))
    _emit("multiload_fixed", combine(left, right))
    for x, tag in ((30.0, "a"), (60.0, "b"), (90.0, "c")):
        f = box((x - 3.0, 20.0 - d_load, -t), (x + 3.0, 20.0 + t, 20.0 + t))
        _emit(f"multiload_force_{tag}", f)


def main():
    print(f"writing generated STLs to {OUT_DIR}")
    make_cantilever_beam()
    make_mbb_beam()
    make_bridge_deck()
    make_torsion_bracket()
    make_l_bracket()
    make_multi_load_beam()
    print("done.")


if __name__ == "__main__":
    main()
