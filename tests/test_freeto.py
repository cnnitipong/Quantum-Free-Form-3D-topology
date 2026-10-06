"""Unit and end-to-end tests for the FreeTO Python port (pytest)."""
from __future__ import annotations

import os
import sys
import threading

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from freeto import FreeTOConfig, run_freeto, read_stl, write_stl, EXAMPLES  # noqa: E402
from freeto.examples import STL_DIR, example_config                        # noqa: E402
from freeto.fe import lk_H8, available_solvers                              # noqa: E402
from freeto.filters import HHs3D, HnHns3D                                  # noqa: E402
from freeto.inside import inside_grid, inside_points                        # noqa: E402
from freeto.mesh import (matlab_colon, matlab_linspace, build_grid,        # noqa: E402
                         force_vectors, element_dofs, element_nodes,
                         xyz_to_matlab, matlab_to_xyz)
from freeto.optimizers import MMA, oc_update                                # noqa: E402
from freeto.postprocess import (FieldSnapshot, apply_symmetry,              # noqa: E402
                                surface_from_field)
from freeto.smoothedge import smoothedge3D, upsample_linear, window_reduce  # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def box_mesh(lo, hi):
    """Closed, outward-oriented triangulated box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5],
                  [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
                  [3, 0, 4], [3, 4, 7]])
    return V, F


def write_box(path, lo, hi):
    V, F = box_mesh(lo, hi)
    write_stl(path, V, F)
    return str(path)


def winding_number(V, F, P, chunk=256):
    """Generalised winding number (brute-force inside reference)."""
    out = np.zeros(len(P))
    a0, b0, c0 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        a, b, c = a0[None] - p, b0[None] - p, c0[None] - p
        la, lb, lc = (np.linalg.norm(a, axis=2), np.linalg.norm(b, axis=2),
                      np.linalg.norm(c, axis=2))
        det = np.einsum("ijk,ijk->ij", a, np.cross(b, c))
        den = (la * lb * lc + np.einsum("ijk,ijk->ij", a, b) * lc
               + np.einsum("ijk,ijk->ij", b, c) * la
               + np.einsum("ijk,ijk->ij", c, a) * lb)
        out[s:s + chunk] = np.arctan2(det, den).sum(axis=1) / (2 * np.pi)
    return out


def edge_manifold(faces):
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                faces[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return np.all(cnt == 2)


def signed_volume(V, F):
    V = V.astype(float)
    return np.einsum("ij,ij->", V[F[:, 0]],
                     np.cross(V[F[:, 1]], V[F[:, 2]])) / 6.0


# ----------------------------------------------------------------------------
# element stiffness
# ----------------------------------------------------------------------------
def test_lk_H8_properties():
    KE = lk_H8(0.3)
    assert KE.shape == (24, 24)
    assert np.array_equal(KE, KE.T)
    w = np.linalg.eigvalsh(KE)
    assert w.min() > -1e-13
    assert np.sum(np.abs(w) < 1e-10) == 6
    assert KE[0, 0] == pytest.approx(0.23504273504273504, rel=1e-15)
    assert np.trace(KE) == pytest.approx(5.6410256410256405, rel=1e-14)
    # local node coordinates of the top3d hexahedron (SPEC section 2)
    X = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    for c in range(3):
        t = np.zeros(24)
        t[c::3] = 1
        assert np.abs(KE @ t).max() < 1e-13
    u = np.zeros(24)
    u[0::3] = X[:, 0]
    assert u @ KE @ u == pytest.approx(0.7 / (1.3 * 0.4), rel=1e-12)
    rot = np.zeros(24)
    rot[0::3] = -X[:, 1]
    rot[1::3] = X[:, 0]
    assert np.abs(KE @ rot).max() < 1e-13


# ----------------------------------------------------------------------------
# grid arithmetic
# ----------------------------------------------------------------------------
def test_colon_and_linspace():
    for compat in ("matlab", "octave"):
        assert len(matlab_colon(0, 0.1, 0.3, compat)) == 4
        assert len(matlab_colon(0, 0.1, 1.0, compat)) == 11
        assert matlab_colon(0, 0.1, 1.0, compat)[-1] == 1.0
        assert len(matlab_colon(1, 2, 0.5, compat)) == 0
        x = matlab_linspace(0.25, 7.5, 11, compat)
        assert x[0] == 0.25 and x[-1] == 7.5 and len(x) == 11


def test_tiny_box_grid_matches_spec():
    V, F = box_mesh((0, 0, 0), (10, 4, 4))
    g = build_grid(V, 11)
    assert g.axis == 0
    assert g.ssz == pytest.approx(0.998, abs=1e-12)
    np.testing.assert_allclose(g.x[:2], [0.01, 1.008], atol=1e-12)
    np.testing.assert_allclose(g.y, [0.004, 1.002, 2.0, 2.998, 3.996], atol=1e-12)
    np.testing.assert_allclose(g.z, g.y, atol=0)
    assert (g.nelx, g.nely, g.nelz) == (10, 4, 4)


def test_axis_selection_quirk():
    # x in [100,110] (small extent) but largest |coordinate| -> x axis
    V, F = box_mesh((100, 0, 0), (110, 50, 5))
    g = build_grid(V, 11)
    assert g.axis == 0
    assert g.nely > 40


def test_frame_roundtrip():
    a = np.random.default_rng(1).random((4, 5, 6))
    assert np.array_equal(xyz_to_matlab(matlab_to_xyz(a)), a)


# ----------------------------------------------------------------------------
# inside test
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("name,n", [("GE_domain.STL", 14), ("quad_domain.STL", 18),
                                    ("air_force.STL", 18), ("hand_fixed.stl", 16)])
def test_inside_vs_winding_number(name, n):
    V, F = read_stl(os.path.join(STL_DIR, name))
    rng = np.random.default_rng(42)
    lo, hi = V.min(0), V.max(0)
    pad = 0.05 * (hi - lo)
    g = [np.sort(rng.uniform(lo[i] - pad[i], hi[i] + pad[i], n)) for i in range(3)]
    R = inside_grid(V, F, *g)
    X, Y, Z = np.meshgrid(*g, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    w = winding_number(V, F, P)
    assert np.all((w > 0.5) == (R.ravel() == 1))
    # identical to the per-point port of intriangulation
    assert np.array_equal(inside_points(V, F, P), R.ravel())


@pytest.mark.parametrize("name", ["hand_force5.stl", "quad_force1.STL"])
def test_inside_non_watertight_matches_port(name):
    # these example STLs are not watertight: no winding-number ground truth,
    # but the grid ray caster must equal the per-point intriangulation port
    V, F = read_stl(os.path.join(STL_DIR, name))
    rng = np.random.default_rng(7)
    lo, hi = V.min(0), V.max(0)
    g = [np.sort(rng.uniform(lo[i], hi[i], 15)) for i in range(3)]
    R = inside_grid(V, F, *g)
    X, Y, Z = np.meshgrid(*g, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    assert np.array_equal(inside_points(V, F, P), R.ravel())


def test_inside_rays_through_vertices_and_edges():
    # rays exactly through vertices / edges / faces of a unit cube
    V, F = box_mesh((0, 0, 0), (1, 1, 1))
    g = np.array([-0.5, 0.0, 0.25, 0.5, 1.0, 1.5])
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    strict = ((X > 0) & (X < 1) & (Y > 0) & (Y < 1) & (Z > 0) & (Z < 1))
    outside = (X < 0) | (X > 1) | (Y < 0) | (Y > 1) | (Z < 0) | (Z > 1)
    # "matlab" mode == per-point intriangulation port (including its blind
    # spot: z-rays through the face diagonals (x == y) miss both facets)
    Rm = inside_grid(V, F, g, g, g, mode="matlab")
    assert np.array_equal(inside_points(V, F, P), Rm.ravel())
    assert np.all(Rm[strict & (X != Y)] == 1)
    assert np.all(Rm[strict & (X == Y)] == 0)
    # "robust" mode classifies every strictly interior/exterior point right
    Rr = inside_grid(V, F, g, g, g, mode="robust")
    assert np.all(Rr[strict] == 1) and np.all(Rr[outside] == 0)
    # tetrahedron with slanted faces hit along edges and vertices
    V2 = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], float)
    F2 = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    g2 = np.linspace(-0.5, 2.5, 13)
    X, Y, Z = np.meshgrid(g2, g2, g2, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    R2 = inside_grid(V2, F2, g2, g2, g2, mode="matlab")
    assert np.array_equal(inside_points(V2, F2, P), R2.ravel())
    inner = (X > 0) & (Y > 0) & (Z > 0) & (X + Y + Z < 2)
    out2 = (X < 0) | (Y < 0) | (Z < 0) | (X + Y + Z > 2)
    R2r = inside_grid(V2, F2, g2, g2, g2, mode="robust")
    assert np.all(R2r[inner] == 1) and np.all(R2r[out2] == 0)


@pytest.mark.parametrize("name", ["GE_domain.STL", "hand_domain.stl", "quad_force1.STL"])
def test_inside_modes_agree_on_examples(name):
    V, F = read_stl(os.path.join(STL_DIR, name))
    g = build_grid(read_stl(os.path.join(STL_DIR, name))[0], 40)
    assert np.array_equal(inside_grid(V, F, g.x, g.y, g.z, mode="matlab") == 1,
                          inside_grid(V, F, g.x, g.y, g.z, mode="robust") == 1)


def test_inside_speed():
    import time
    V, F = read_stl(os.path.join(STL_DIR, "hand_domain.stl"))
    lo, hi = V.min(0), V.max(0)
    g = [np.linspace(lo[i], hi[i], n) + 1e-3 for i, n in enumerate((100, 60, 40))]
    t = time.perf_counter()
    inside_grid(V, F, *g)
    assert time.perf_counter() - t < 5.0


# ----------------------------------------------------------------------------
# STL io
# ----------------------------------------------------------------------------
def test_stl_roundtrip_binary_and_ascii(tmp_path):
    V, F = box_mesh((0, 0, 0), (1, 2, 3))
    p = tmp_path / "b.stl"
    write_stl(p, V, F)
    V2, F2 = read_stl(p)
    assert V2.shape == (8, 3) and F2.shape == (12, 3)
    np.testing.assert_allclose(V2[F2], V[F], atol=1e-6)
    lines = ["solid t"]
    for f in F:
        lines += ["facet normal 0 0 0", "outer loop"]
        lines += ["vertex %r %r %r" % tuple(float(a) for a in V[i]) for i in f]
        lines += ["endloop", "endfacet"]
    lines.append("endsolid t")
    pa = tmp_path / "a.stl"
    pa.write_text("\n".join(lines))
    V3, F3 = read_stl(pa)
    assert V3.shape == (8, 3)
    np.testing.assert_allclose(V3[F3], V[F])


# ----------------------------------------------------------------------------
# filters / smoothedge
# ----------------------------------------------------------------------------
def test_filters():
    nelx, nely, nelz = 6, 5, 4
    nele = nelx * nely * nelz
    ele = np.arange(nele)
    H, Hs = HHs3D(nelx, nely, nelz, 1.5, ele, nele)
    assert abs(H - H.T).max() == 0
    e = 2 + nely * 2 + nely * nelx * 2        # interior element
    assert Hs[e] == pytest.approx(5.529437251522859, rel=1e-15)
    assert H[e].nnz == 19
    sub = ele[::2]
    H2, Hs2 = HHs3D(nelx, nely, nelz, 1.5, sub, nele)
    Hf = H.tocsr()[sub][:, sub]
    assert abs(H2 - Hf).max() == 0
    Hn, Hns = HnHns3D(nelx, nely, nelz, 1)
    assert Hn.shape == ((nelx + 1) * (nely + 1) * (nelz + 1), nele)
    np.testing.assert_allclose(Hn.data, 1 - np.sqrt(0.75), rtol=1e-15)
    np.testing.assert_allclose(Hn @ np.ones(nele) / Hns, 1.0, rtol=1e-15)
    cnt = np.diff(Hn.indptr)
    assert set(np.unique(cnt)) <= {1, 2, 4, 8}


def test_upsample_and_windows():
    rng = np.random.default_rng(0)
    xn = rng.random((3, 4, 5))
    xg = upsample_linear(xn, 4)
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator([np.arange(s) for s in xn.shape], xn)
    q = np.stack(np.meshgrid(*[np.arange(0, s - 1 + 1e-9, 0.25) for s in xn.shape],
                             indexing="ij"), -1)
    np.testing.assert_allclose(xg, f(q), atol=1e-14)
    s = window_reduce(xg, 4, "sum")
    assert s.shape == (2, 3, 4)
    assert s[1, 2, 3] == pytest.approx(xg[4:9, 8:13, 12:17].sum(), rel=1e-14)
    assert window_reduce(xg, 4, "min")[1, 0, 2] == xg[4:9, 0:5, 8:13].min()
    assert window_reduce(xg, 4, "max")[0, 1, 1] == xg[0:5, 4:9, 4:9].max()


def test_smoothedge_uniform():
    nelx, nely, nelz = 4, 3, 5
    nele = nelx * nely * nelz
    Hn, Hns = HnHns3D(nelx, nely, nelz, 1)
    v = np.full(nele, 0.4)
    vx, xg, ls, top, tol = smoothedge3D(v, Hn, Hns, nelx, nely, nelz, nele, nele, 0.5)
    np.testing.assert_allclose(xg, 0.4, rtol=1e-14)
    assert tol == 1.0
    assert abs(vx.mean() - 0.4) < 1e-4


# ----------------------------------------------------------------------------
# loads
# ----------------------------------------------------------------------------
def test_forcevec_semantics():
    Fn = [np.array([5, 9, 2]), np.array([2, 7])]
    ndof = 30
    F, nf = force_vectors(Fn, ndof, [1000.0], [0.0, -500.0], [0.0], "distributed")
    assert nf == 2 and F.shape == (30, 2)
    # scalar Fmagx goes to column 0 for every region (region 2 overwrites node 2)
    assert F[3 * 5, 0] == pytest.approx(1000 / 3)
    assert F[3 * 7, 0] == pytest.approx(1000 / 2)
    assert F[3 * 2, 0] == pytest.approx(1000 / 2)
    assert F[3 * 7 + 1, 1] == pytest.approx(-250)
    Fp, _ = force_vectors(Fn, ndof, [0.0], [0.0], [1.0, 2.0], "point")
    assert np.flatnonzero(Fp[:, 0]).tolist() == [3 * 5 + 2]
    assert np.flatnonzero(Fp[:, 1]).tolist() == [3 * 2 + 2]


def test_edof_matches_nodes():
    ed = element_dofs(3, 2, 2, [0, 5])
    nd = element_nodes(3, 2, 2, [0, 5])
    assert np.array_equal(ed[:, 0::3] // 3, nd)
    # MATLAB 1-based first row for nelx=3,nely=2 (nodenr(1,1,1)=1)
    first = (ed[0] + 1).tolist()
    assert first[:6] == [4, 5, 6, 4 + 9, 5 + 9, 6 + 9]
    assert first[9:12] == [1, 2, 3]


# ----------------------------------------------------------------------------
# optimisers
# ----------------------------------------------------------------------------
def test_mma_analytic():
    # min sum c_j/x_j  s.t. sum x_j <= V ; optimum x_j = V sqrt(c_j)/sum sqrt(c)
    rng = np.random.default_rng(3)
    n = 20
    c = rng.uniform(0.5, 3.0, n)
    V = 6.0
    xopt = V * np.sqrt(c) / np.sqrt(c).sum()
    mma = MMA(n, 1, 0.01, 1.0)
    x = np.full(n, 0.2)
    for it in range(1, 80):
        f0, df0 = np.sum(c / x), -c / x ** 2
        fval = np.array([x.sum() / V - 1])
        dfdx = np.ones((1, n)) / V
        x = mma.update(it, x, f0, df0, fval, dfdx)
    np.testing.assert_allclose(x, xopt, rtol=2e-3)


def test_mma_svanberg_toy_problem():
    # Svanberg's classic 3-variable, 2-constraint test problem
    # min x.x  s.t. |x-(5,2,1)|^2 <= 9, |x-(3,4,3)|^2 <= 9, 0 <= x <= 5
    # published optimum (2.0175, 1.7800, 1.2375), both constraints active
    def g(x):
        return np.array([(x[0] - 5) ** 2 + (x[1] - 2) ** 2 + (x[2] - 1) ** 2 - 9,
                         (x[0] - 3) ** 2 + (x[1] - 4) ** 2 + (x[2] - 3) ** 2 - 9])

    def dg(x):
        return np.array([[2 * (x[0] - 5), 2 * (x[1] - 2), 2 * (x[2] - 1)],
                         [2 * (x[0] - 3), 2 * (x[1] - 4), 2 * (x[2] - 3)]])
    mma = MMA(3, 2, 0.0, 5.0, a0=1.0, a=np.zeros(2), c=1000 * np.ones(2),
              d=np.ones(2))
    x = np.array([4.0, 3.0, 2.0])
    for it in range(1, 40):
        x = mma.update(it, x, x @ x, 2 * x, g(x), dg(x))
    np.testing.assert_allclose(x, [2.017519, 1.780011, 1.237507], atol=1e-5)
    assert np.all(g(x) < 1e-5)


def test_oc_update_volume():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.3, 0.5, 500)
    dc = -rng.uniform(0.1, 1.0, 500)
    dv = np.ones(500)
    xn = oc_update(x, dc, dv, 0.4, 500)
    assert abs(xn.sum() - 0.4 * 500) / 200 < 2e-3
    assert np.all(np.abs(xn - x) <= 0.1 + 1e-12)


# ----------------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------------
def _sphere_field(n=24, r=0.35):
    g = np.linspace(0, 1, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    top = r - np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
    return FieldSnapshot(top, (10.0, 20.0, 30.0), (1 / (n - 1),) * 3)


def test_surface_closed_and_outward():
    f = _sphere_field()
    V, F = surface_from_field(f, smooth=False)
    assert V.dtype == np.float32 and F.dtype == np.int32
    assert edge_manifold(F)
    vol = signed_volume(V, F)
    assert vol == pytest.approx(4 / 3 * np.pi * 0.35 ** 3, rel=0.03)
    assert np.allclose(V.mean(0), [10.5, 20.5, 30.5], atol=0.01)


def test_surface_capped_at_boundary():
    # solid touching the domain boundary must be closed by flat caps
    n = 20
    top = np.full((n, n, n), 0.5)
    top[:, :, 10:] = -0.5
    f = FieldSnapshot(top, (0, 0, 0), (1, 1, 1))
    V, F = surface_from_field(f, smooth=True)
    assert edge_manifold(F)
    assert abs(V.min(0)[0]) < 2e-3 and abs(V.max(0)[0] - (n - 1)) < 2e-3
    assert signed_volume(V, F) > 0


def test_symmetry_placement():
    top = -np.ones((5, 6, 7))
    top[0, 1, 2] = 1          # blob near the low-x, low-y, low-z corner
    f = FieldSnapshot(top, (1.0, 2.0, 3.0), (0.5, 0.5, 0.5))
    r = apply_symmetry(f, "y-z", "right")          # copy at +x
    assert r.top.shape == (10, 6, 7) and np.allclose(r.origin, f.origin)
    assert r.top[9, 1, 2] == 1 and r.top[4, 0, 0] == r.top[5, 0, 0]
    l = apply_symmetry(f, "y-z", "left")           # copy at -x
    assert np.allclose(l.origin, (1.0 - 2.5, 2.0, 3.0))
    assert l.top[5, 1, 2] == 1                     # original stays in place
    zx = apply_symmetry(f, "z-x", "right")         # copy at -y
    assert np.allclose(zx.origin, (1.0, 2.0 - 3.0, 3.0)) and zx.top[0, 7, 2] == 1
    zx2 = apply_symmetry(f, "z-x", "left")         # copy at +y
    assert np.allclose(zx2.origin, f.origin) and zx2.top[0, 10, 2] == 1
    xy = apply_symmetry(f, "x-y", "right")         # copy at +z
    assert xy.top[0, 1, 11] == 1


# ----------------------------------------------------------------------------
# end-to-end
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cantilever(tmp_path_factory):
    d = tmp_path_factory.mktemp("cant")
    dom = write_box(d / "dom.stl", (0, 0, 0), (10, 4, 4))
    fix = write_box(d / "fix.stl", (-1, -1, -1), (0.5, 5, 5))
    frc = write_box(d / "frc.stl", (9.5, -1, -1), (11, 5, 5))
    return dom, fix, frc


def _cfg(cantilever, **kw):
    dom, fix, frc = cantilever
    base = dict(domain=dom, forces=[frc], fixed=fix, mesh_control=11,
                volfrac=0.4, fmagz=[-1.0], youngs_modulus=1.0, max_iter=12,
                solver="superlu")
    base.update(kw)
    return FreeTOConfig(**base)


def test_cantilever_end_to_end(cantilever, tmp_path):
    infos = []
    res = run_freeto(_cfg(cantilever), callback=infos.append, log=None)
    setup = infos[0]
    assert setup["stage"] == "setup"
    assert (setup["nelx"], setup["nely"], setup["nelz"]) == (10, 4, 4)
    assert setup["nnele"] == 160 and setup["ndof"] == 825 and setup["nfree"] == 750
    its = [i for i in infos if i["stage"] == "iter"]
    assert len(its) == res.iterations and 3 <= res.iterations <= 12
    c = res.history["compliance"]
    assert all(np.isfinite(c)) and all(v > 0 for v in c)
    assert c[-1] < c[0]
    assert abs(res.finalvol - 0.4) < 0.05
    fld = its[-1]["field"]
    assert fld.top.shape == (41, 17, 17)
    assert not fld.top.flags.writeable
    assert np.allclose(fld.origin, [0.01, 0.004, 0.004])
    V, F = res.surface()
    assert edge_manifold(F) and signed_volume(V, F) > 0
    # the design lives inside the domain box (physical coordinates)
    assert V.min() >= -1e-6 and V[:, 0].max() <= 10 + 1e-6
    p = tmp_path / "out.stl"
    res.write_stl(p)
    V2, F2 = read_stl(p)
    assert F2.shape[0] == F.shape[0]
    res.save_npz(tmp_path / "out.npz")
    z = np.load(tmp_path / "out.npz")
    assert z["eleden"].shape == (4, 10, 4)


def test_solvers_agree(cantilever):
    ref = run_freeto(_cfg(cantilever, max_iter=3), log=None)
    for s in available_solvers():
        if s == "superlu":
            continue
        r = run_freeto(_cfg(cantilever, max_iter=3, solver=s), log=None)
        np.testing.assert_allclose(r.history["compliance"],
                                   ref.history["compliance"], rtol=1e-6)


def test_semdot_mma_and_symmetry(cantilever):
    r = run_freeto(_cfg(cantilever, method="SEMDOT", optimizer="MMA",
                        max_iter=6, symmetry=[("x-y", "left")]), log=None)
    assert r.iterations == 6
    assert r.history["beta"][:3] == [1.0, 1.5, 2.0]    # README MMA schedule
    assert r.field.top.shape == (41, 17, 34)
    assert r.field.origin[2] == pytest.approx(0.004 - 17 * 0.998 / 4)


def test_stop_event(cantilever):
    ev = threading.Event()

    def cb(info):
        if info["stage"] == "iter" and info["iter"] == 2:
            ev.set()
    r = run_freeto(_cfg(cantilever, max_iter=50), callback=cb, stop_event=ev,
                   log=None)
    assert r.stopped and r.iterations == 2
    ev2 = threading.Event()
    ev2.set()
    r2 = run_freeto(_cfg(cantilever), stop_event=ev2, log=None)
    assert r2.stopped and r2.iterations == 0 and r2.field is not None


def test_validate_messages(cantilever):
    dom, fix, frc = cantilever
    with pytest.raises(ValueError, match="support"):
        FreeTOConfig(domain=dom, forces=[frc], fmagz=[1.0]).validate()
    with pytest.raises(ValueError, match="non-zero"):
        FreeTOConfig(domain=dom, forces=[frc], fixed=fix).validate()
    with pytest.raises(ValueError, match="load cases"):
        FreeTOConfig(domain=dom, forces=[frc], fixed=fix, fmagz=[1, 2]).validate()
    with pytest.raises(ValueError, match="not found"):
        FreeTOConfig(domain=dom + "x", forces=[frc], fixed=fix, fmagz=[1]).validate()
    with pytest.raises(ValueError, match="symmetry"):
        FreeTOConfig(domain=dom, forces=[frc], fixed=fix, fmagz=[1],
                     symmetry=[("x-z", "left")]).validate()
    assert FreeTOConfig(domain=dom, forces=[frc], fixed=fix, fmagx=[1],
                        fmagy=[1], fmagz=[1]).validate()


def test_examples_registry():
    # the 4 original README/paper examples plus the additional continuum
    # examples (see tests/test_examples.py for their own coverage)
    assert {"GE_bracket", "air_bracket", "hand", "quadcopter"} <= set(EXAMPLES)
    for name, ex in EXAMPLES.items():
        assert {"title", "description", "config_kwargs", "files"} <= set(ex)
        cfg = example_config(name)
        assert cfg.validate()


# ----------------------------------------------------------------------------
# robustness: under-constrained supports, degenerate input, size limits
# ----------------------------------------------------------------------------
from freeto import FreeTOError                          # noqa: E402
import freeto.core as _core                             # noqa: E402


def test_underconstrained_supports_precheck(cantilever):
    dom, fix, frc = cantilever
    cfg = _cfg(cantilever, fixed=None, zfixed=fix)
    with pytest.raises(FreeTOError, match="rigid-body motion"):
        run_freeto(cfg, log=None)
    # FreeTOError is a ValueError, so generic handlers still catch it
    assert issubclass(FreeTOError, ValueError)


@pytest.mark.parametrize("solver", available_solvers())
def test_singular_system_detected_by_solver(cantilever, monkeypatch, solver):
    # bypass the pre-check: the post-solve checks must catch the singular K
    monkeypatch.setattr(_core, "_check_rigid_body", lambda *a: None)
    cfg = _cfg(cantilever, fixed=None, zfixed=cantilever[1], solver=solver,
               max_iter=3)
    with pytest.raises(FreeTOError, match="rigid-body motion"):
        run_freeto(cfg, log=None)


def test_disconnected_part_detected(tmp_path):
    V1, F1 = box_mesh((0, 0, 0), (10, 4, 4))
    V2, F2 = box_mesh((12, 0, 0), (16, 4, 4))
    dom = tmp_path / "two.stl"
    write_stl(dom, np.vstack([V1, V2]), np.vstack([F1, F2 + 8]))
    fix = write_box(tmp_path / "fix.stl", (-1, -1, -1), (0.5, 5, 5))
    frc = write_box(tmp_path / "frc.stl", (9.5, -1, -1), (10.5, 5, 5))
    cfg = FreeTOConfig(domain=str(dom), forces=[frc], fixed=fix, mesh_control=17,
                       fmagz=[-1.0], max_iter=2, solver="superlu")
    with pytest.raises(FreeTOError, match="disconnected part"):
        run_freeto(cfg, log=None)


def test_matlab_mode_undecided_is_inside():
    # an open slanted triangle: rays along z, x and y all cross it once
    V = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    F = np.array([[0, 1, 2]])
    g = np.array([0.2, 0.9])
    assert inside_grid(V, F, g, g, g, mode="matlab")[0, 0, 0] == 1   # MATLAB: logical(-1)
    assert inside_grid(V, F, g, g, g, mode="matlab", undecided=-1)[0, 0, 0] == -1
    assert inside_points(V, F, [[0.2, 0.2, 0.2]])[0] == 1
    assert inside_grid(V, F, g, g, g, mode="robust")[0, 0, 0] == 0
    assert inside_grid(V, F, g, g, g, mode="matlab")[1, 1, 1] == 0


def test_degenerate_and_empty_stl(cantilever, tmp_path):
    dom, fix, frc = cantilever
    flat = write_box(tmp_path / "flat.stl", (0, 0, 0), (10, 4, 0))
    with pytest.raises(FreeTOError, match="flat/degenerate"):
        _cfg(cantilever, domain=flat).validate()
    empty = tmp_path / "empty.stl"
    write_stl(empty, np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    with pytest.raises(FreeTOError, match="no triangles"):
        _cfg(cantilever, domain=str(empty)).validate()
    with pytest.raises(FreeTOError, match="no triangles"):
        _cfg(cantilever, forces=[str(empty)]).validate()
    thin = write_box(tmp_path / "thin.stl", (0, 0, 0), (10, 4, 0.2))
    with pytest.raises(FreeTOError, match="no elements along z"):
        _cfg(cantilever, domain=thin, mesh_control=5).validate()


def test_mesh_control_limits(cantilever):
    with pytest.raises(FreeTOError, match=">= 4"):
        _cfg(cantilever, mesh_control=3).validate()
    assert _cfg(cantilever, mesh_control=4).validate()
    with pytest.raises(FreeTOError, match="above the limit"):
        _cfg(cantilever, mesh_control=600).validate()
    assert _cfg(cantilever, mesh_control=600, max_dofs=None).validate()


def test_open_domain_warns(cantilever, tmp_path):
    V, F = box_mesh((0, 0, 0), (10, 4, 4))
    op = tmp_path / "open.stl"
    write_stl(op, V, F[:-2])            # remove one face of the box
    msgs = []
    try:
        run_freeto(_cfg(cantilever, domain=str(op), max_iter=1), log=msgs.append)
    except FreeTOError:
        pass
    assert any("not watertight" in m for m in msgs)
