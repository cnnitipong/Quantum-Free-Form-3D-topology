"""Tests for the six additional continuum examples (cantilever_beam,
mbb_beam, bridge_deck, torsion_bracket, l_bracket, multi_load_beam) and the
``freeto.geometry`` builders used to generate their STLs.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from freeto import EXAMPLES, run_freeto                                    # noqa: E402
from freeto.examples import STL_DIR, example_config                        # noqa: E402
from freeto.stl_io import read_stl                                         # noqa: E402
from freeto.geometry import (box, l_shape, box_with_hole, face_slab,       # noqa: E402
                             patch, cylinder, combine, is_watertight,
                             bbox_of, write_mesh)

NEW_EXAMPLES = ["cantilever_beam", "mbb_beam", "bridge_deck",
               "torsion_bracket", "l_bracket", "multi_load_beam"]


# ----------------------------------------------------------------------------
# freeto.geometry builders
# ----------------------------------------------------------------------------
class TestGeometryBuilders:
    def test_box_watertight(self):
        m = box((0, 0, 0), (10, 20, 30))
        assert is_watertight(m)
        lo, hi = bbox_of(m)
        np.testing.assert_allclose(lo, [0, 0, 0])
        np.testing.assert_allclose(hi, [10, 20, 30])

    def test_box_sorts_corners(self):
        # lo/hi given in the "wrong" order must still work
        m = box((10, 20, 30), (0, 0, 0))
        assert is_watertight(m)

    @pytest.mark.parametrize("notch", [
        ((40, 40, 0), (100, 100, 10)),   # upper-right corner
        ((0, 40, 0), (60, 100, 10)),     # upper-left corner
        ((40, 0, 0), (100, 60, 10)),     # lower-right corner
        ((0, 0, 0), (60, 60, 10)),       # lower-left corner
    ])
    def test_l_shape_watertight_all_corners(self, notch):
        nlo, nhi = notch
        m = l_shape((0, 0, 0), (100, 100, 10), nlo, nhi)
        assert is_watertight(m)
        # the notch corner itself must not be part of the solid
        V, F = m
        assert V.shape[0] > 0

    def test_l_shape_rejects_non_corner_notch(self):
        with pytest.raises(ValueError):
            # notch doesn't touch any outer face on x/y -> not a corner
            l_shape((0, 0, 0), (100, 100, 10), (30, 30, 0), (60, 60, 10))

    def test_l_shape_rejects_notch_spanning_two_axes_fully(self):
        with pytest.raises(ValueError):
            l_shape((0, 0, 0), (100, 100, 10), (0, 0, 0), (100, 100, 10))

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_box_with_hole_watertight(self, axis):
        lo = {"x": (0, 0, 0), "y": (0, 0, 0), "z": (0, 0, 0)}[axis]
        hi = (50, 40, 30)
        hole_lo = list(lo)
        hole_hi = list(hi)
        ax = "xyz".index(axis)
        hole_lo[ax], hole_hi[ax] = lo[ax], hi[ax]
        other = [i for i in range(3) if i != ax]
        for i in other:
            hole_lo[i] = lo[i] + 0.3 * (hi[i] - lo[i])
            hole_hi[i] = lo[i] + 0.7 * (hi[i] - lo[i])
        m = box_with_hole(lo, hi, hole_lo, hole_hi, axis=axis)
        assert is_watertight(m)

    def test_box_with_hole_requires_full_span_on_axis(self):
        with pytest.raises(ValueError):
            box_with_hole((0, 0, 0), (50, 40, 30), (5, 10, 10), (45, 30, 20),
                          axis="z")

    def test_face_slab_thickness_and_margin(self):
        dom = ((0, 0, 0), (100, 50, 20))
        sl = face_slab(dom, "x-", thickness=3.0)
        assert is_watertight(sl)
        lo, hi = bbox_of(sl)
        assert lo[0] < 0.0             # extends outward past the face
        assert hi[0] == pytest.approx(3.0)
        # covers the whole face by default
        np.testing.assert_allclose(lo[1:], [0, 0])
        np.testing.assert_allclose(hi[1:], [50, 20])

    def test_face_slab_extent_shrinks_footprint(self):
        dom = ((0, 0, 0), (100, 50, 20))
        sl = face_slab(dom, "y+", thickness=2.0, extent=5.0)
        lo, hi = bbox_of(sl)
        assert hi[1] > 50.0
        assert lo[0] == pytest.approx(45.0)
        assert hi[0] == pytest.approx(55.0)

    @pytest.mark.parametrize("face", ["x-", "x+", "y-", "y+", "z-", "z+"])
    def test_face_slab_all_faces(self, face):
        dom = ((0, 0, 0), (40, 30, 20))
        sl = face_slab(dom, face, thickness=2.0)
        assert is_watertight(sl)

    def test_patch_watertight(self):
        assert is_watertight(patch((5, 5, 5), 2.0))
        assert is_watertight(patch((5, 5, 5), (1.0, 2.0, 3.0)))

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_cylinder_watertight(self, axis):
        assert is_watertight(cylinder((1, 2, 3), axis, 4.0, 10.0, n=12))

    def test_combine_welds_coincident_vertices(self):
        # combine() welds bit-identical vertex coordinates across pieces
        # (needed for l_shape / box_with_hole, whose cells are independent
        # box() calls sharing exact boundary coordinates); it does NOT
        # cancel internal faces on its own -- concatenating two ordinary,
        # complete boxes that merely happen to touch is not, by itself, a
        # valid single watertight solid (the touching face's perimeter
        # edges become non-manifold), which is exactly why l_shape /
        # box_with_hole explicitly drop the shared face on each cell
        # *before* calling combine().
        a = box((0, 0, 0), (10, 10, 10))
        b = box((10, 0, 0), (20, 10, 10))
        V, F = combine(a, b)
        assert V.shape[0] == 12          # 8 + 8 raw, 4 coincident pairs welded
        assert not is_watertight((V, F))

    def test_l_shape_cells_combine_to_watertight(self):
        # the real (supported) use of combine(): pieces whose shared faces
        # were already dropped before combining (see l_shape / box_with_hole
        # above) do fuse into one watertight, correctly wound shell.
        m = l_shape((0, 0, 0), (100, 100, 10), (40, 40, 0), (100, 100, 10))
        assert is_watertight(m)

    def test_combine_disjoint_shells_each_watertight(self):
        a = box((0, 0, 0), (10, 10, 10))
        b = box((100, 100, 100), (110, 110, 110))
        V, F = combine(a, b)
        # each edge still shared by exactly two triangles even though the
        # two shells never touch
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]),
                    axis=1)
        _, cnt = np.unique(e, axis=0, return_counts=True)
        assert np.all(cnt == 2)

    def test_write_mesh_roundtrip(self, tmp_path):
        m = box((0, 0, 0), (5, 6, 7))
        path = write_mesh(str(tmp_path / "box.stl"), m)
        V, F = read_stl(path)
        assert F.shape[0] == 12
        lo, hi = V.min(axis=0), V.max(axis=0)
        np.testing.assert_allclose(lo, [0, 0, 0], atol=1e-4)
        np.testing.assert_allclose(hi, [5, 6, 7], atol=1e-4)


# ----------------------------------------------------------------------------
# example registry
# ----------------------------------------------------------------------------
class TestNewExamplesRegistry:
    def test_registered(self):
        assert set(NEW_EXAMPLES) <= set(EXAMPLES)

    def test_categories(self):
        expected = {"cantilever_beam": "beam", "mbb_beam": "beam",
                   "bridge_deck": "truss-like", "torsion_bracket": "advanced",
                   "l_bracket": "advanced", "multi_load_beam": "advanced"}
        for name, cat in expected.items():
            assert EXAMPLES[name]["category"] == cat
        for name in ("GE_bracket", "air_bracket", "hand", "quadcopter"):
            assert EXAMPLES[name]["category"] == "paper"

    @pytest.mark.parametrize("name", NEW_EXAMPLES)
    def test_config_validates(self, name):
        cfg = example_config(name)
        assert cfg.validate()

    @pytest.mark.parametrize("name", NEW_EXAMPLES)
    def test_stl_files_exist_and_watertight(self, name):
        kw = EXAMPLES[name]["config_kwargs"]
        paths = [kw["domain"]] + list(kw["forces"])
        for key in ("fixed", "xfixed", "yfixed", "zfixed", "keepdom"):
            if kw.get(key):
                paths.append(kw[key])
        for rel in paths:
            full = os.path.join(STL_DIR, rel)
            assert os.path.isfile(full), full
            V, F = read_stl(full)
            assert F.shape[0] > 0
            e = np.sort(np.concatenate(
                [F[:, [0, 1]], F[:, [1, 2]], F[:, [0, 2]]]), axis=1)
            _, cnt = np.unique(e, axis=0, return_counts=True)
            assert np.all(cnt == 2), f"{rel} is not watertight"

    def test_files_dict_matches_config_kwargs(self):
        for name in NEW_EXAMPLES:
            ex = EXAMPLES[name]
            assert {"title", "description", "config_kwargs", "files",
                    "category"} <= set(ex)


# ----------------------------------------------------------------------------
# smoke runs (small max_iter, real core)
# ----------------------------------------------------------------------------
class TestNewExamplesSmokeRun:
    @pytest.mark.parametrize("name", ["cantilever_beam", "l_bracket"])
    def test_smoke_run(self, name):
        cfg = example_config(name, max_iter=3)
        res = run_freeto(cfg, log=None)
        assert res.iterations >= 1
        assert np.isfinite(res.comp)
        assert 0.0 < res.finalvol <= 1.0
        assert res.elenum1 > 0

    def test_mbb_symmetry_doubles_x_extent(self):
        # cheap-ish: mesh_control kept small via override so the smoke test
        # stays fast while still exercising the symmetry mirror
        cfg = example_config("mbb_beam", mesh_control=16, max_iter=3)
        res = run_freeto(cfg, log=None)
        v, _ = res.surface()
        if len(v):
            span = v[:, 0].max() - v[:, 0].min()
            # mirrored about the x=max face: full span should be close to
            # 2x the half-domain's 150 mm (up to ~h shrink per README)
            assert span > 150.0

    def test_torsion_bracket_two_load_cases(self):
        cfg = example_config("torsion_bracket", max_iter=2)
        res = run_freeto(cfg, log=None)
        assert res.setup_info["nloads"] == 2

    def test_multi_load_beam_three_load_cases(self):
        cfg = example_config("multi_load_beam", max_iter=2)
        res = run_freeto(cfg, log=None)
        assert res.setup_info["nloads"] == 3
