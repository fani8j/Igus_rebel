"""D1 edge support, F4 multi-box evidence, K1 gap-anchored top and the shadow
known-size fit on a synthetic depth scene (pinhole camera 1000 px, 1920 x 1080).

Scene: a flat 390 x 320 mm top at z = 1.2 m (fronto-parallel: the plane normal
is (0, 0, -1), camera on the + side), everything else a floor at z = 1.7 m.
"""
import math

import numpy as np
import pytest

from box_seam_detection.core import box_profiles as bp
from box_seam_detection.core import edge_support as es
from box_seam_detection.core import known_size_fit as ksf
from box_seam_detection.core import top_diag, top_from_gap
from box_seam_detection.core.autolabel import depth_top

F, CX, CY = 1000.0, 960.0, 540.0
W, H = 1920, 1080
Z_TOP, Z_FLOOR = 1.2, 1.7
N = np.array([0.0, 0.0, -1.0])
D = Z_TOP
CAM = {"fx": F, "fy": F, "cx": CX, "cy": CY, "width": W, "height": H}


def _px(x, y, z=Z_TOP):
    return np.array([F * x / z + CX, F * y / z + CY])


def _cloud(rects, floor=Z_FLOOR, noise=0.0, seed=0):
    """Organized cloud: pixel (u, v) hits the nearest rect (x0, x1, y0, y1, z) or the floor."""
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = np.full(u.shape, floor, dtype=np.float64)
    for (x0, x1, y0, y1, zr) in sorted(rects, key=lambda r: -r[4]):
        x = (u - CX) / F * zr
        y = (v - CY) / F * zr
        hit = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        z[hit] = zr
    if noise:
        z = z + rng.normal(0, noise, z.shape)
    pts = np.stack([(u - CX) / F * z, (v - CY) / F * z, z], axis=-1).reshape(-1, 3)
    return pts


def _proj(xyz):
    uv = np.stack([F * xyz[:, 0] / xyz[:, 2] + CX, F * xyz[:, 1] / xyz[:, 2] + CY], axis=1)
    return np.arange(len(xyz)), uv


def _bp(n=N, d=D):
    def f(uv):
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        r = np.stack([(uv[:, 0] - CX) / F, (uv[:, 1] - CY) / F, np.ones(len(uv))], axis=1)
        t = -d / (r @ n)
        return r * t[:, None]
    return f


def _project(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    return np.stack([F * x[:, 0] / x[:, 2] + CX, F * x[:, 1] / x[:, 2] + CY], axis=1)


HALF_X, HALF_Y = 0.195, 0.160


def _quad(x0=-HALF_X, x1=HALF_X, y0=-HALF_Y, y1=HALF_Y):
    # ring order: back-left, back-right, front-right, front-left (back = small y = top of image)
    return np.array([_px(x0, y0), _px(x1, y0), _px(x1, y1), _px(x0, y1)])


@pytest.fixture(scope="module")
def scene():
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, HALF_Y, Z_TOP)])
    return xyz, _proj(xyz)


def _grid(scene, quad):
    xyz, proj = scene
    roi = [int(quad[:, 0].min()) - 120, int(quad[:, 1].min()) - 120,
           int(quad[:, 0].max()) + 120, int(quad[:, 1].max()) + 120]
    return depth_top.height_grid(xyz, proj, N, D, roi), roi


# ------------------------------------------------------------------ D1

def test_edge_support_clean_quad_has_no_unsupported_edge(scene):
    quad = _quad()
    grid, roi = _grid(scene, quad)
    res = es.compute(quad, (N, D), grid, roi, _bp(), 0.0, -D * N)
    assert set(res["edges"]) == set(es.EDGE_NAMES)
    for e in res["edges"].values():
        assert e["status"] == "ok"
        assert abs(e["cliff_mm"]) < 3.0          # the cliff sits on the edge
        assert e["continue_frac"] == 0.0
        assert e["outside_h_mm"] < -300.0         # the floor behind it
    assert es.decide(res) == []


def test_edge_support_truncated_quad_flags_the_edge_that_ends_on_the_top(scene):
    # the quad stops 60 mm short of the real front edge (y = +0.16): the top continues
    quad = _quad(y1=HALF_Y - 0.060)
    grid, roi = _grid(scene, quad)
    res = es.compute(quad, (N, D), grid, roi, _bp(), 0.0, -D * N)
    assert es.decide(res) == ["front"]
    front = res["edges"]["front"]
    assert front["continue_frac"] == 1.0 and front["extent_mm"] >= 50.0
    assert res["edges"]["back"]["continue_frac"] == 0.0


def test_edge_support_no_depth_is_not_a_flag(scene):
    quad = _quad()
    grid, roi = _grid(scene, quad)
    grid = np.full_like(grid, np.nan)
    res = es.compute(quad, (N, D), grid, roi, _bp(), 0.0, -D * N)
    assert all(e["status"] == "insufficient_depth" for e in res["edges"].values())
    assert es.decide(res) == []


# ------------------------------------------------------------------ F4 + diagnostics

def _run_diag(xyz, quad=None):
    quad = _quad() if quad is None else quad
    c3 = _bp()(quad)
    return top_diag.run(quad, c3, (N, D), 0.0, xyz, _proj(xyz), CAM, _bp(), (W, H),
                        (390.0, 320.0), "large_392")


def test_diag_single_box_is_not_suspect(scene):
    out = _run_diag(scene[0])
    assert "error" not in out
    assert out["multi_box"]["suspect"] is False
    assert out["multi_box"]["other_box_sized"] == 0
    assert out["edge_support"]["unsupported"] == []
    fit = out["known_size_fit"]
    assert fit["score"] > 0.9 and max(fit["corner_disagreement_mm"]) < 6.0


def test_diag_second_coplanar_box_is_suspect():
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, HALF_Y, Z_TOP),
                  (0.215, 0.215 + 0.390, -HALF_Y, HALF_Y, Z_TOP)])   # 20 mm away
    out = _run_diag(xyz)
    assert out["multi_box"]["other_box_sized"] == 1
    assert out["multi_box"]["suspect"] is True
    assert "other_box_sized_coplanar_component" in out["multi_box"]["evidence"]


def test_diag_object_on_top_is_a_depth_step():
    # a 120 x 100 mm object 60 mm above the top, inside the quad
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, HALF_Y, Z_TOP),
                  (-0.06, 0.06, -0.05, 0.05, Z_TOP - 0.060)])
    out = _run_diag(xyz)
    assert out["multi_box"]["step_blobs"] and out["multi_box"]["suspect"] is True
    assert out["multi_box"]["step_blobs"][0]["above_plane"] is True


def test_diag_thin_flap_gap_is_not_a_step():
    # a 3 mm wide, 30 mm deep groove across the middle (the flap gap): no depth step
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, -0.0015, Z_TOP),
                  (-HALF_X, HALF_X, 0.0015, HALF_Y, Z_TOP),
                  (-HALF_X, HALF_X, -0.0015, 0.0015, Z_TOP + 0.030)])
    out = _run_diag(xyz)
    assert out["multi_box"]["step_blobs"] == []
    assert out["multi_box"]["suspect"] is False


# ------------------------------------------------------------------ known-size fit

def test_known_size_fit_scores_the_right_rectangle():
    rng = np.random.default_rng(1)
    pts = np.stack([rng.uniform(-195, 195, 4000), rng.uniform(-160, 160, 4000)], axis=1)
    r = ksf.fit(pts, 390.0, 320.0, 4.0)
    assert r["score"] > 0.95
    wrong = ksf.fit(pts, 250.0, 150.0, 4.0)
    assert wrong["score"] < r["score"]
    dis = ksf.corner_disagreement_mm(r["corners"], np.array([[-195, -160], [195, -160], [195, 160], [-195, 160]]))
    assert max(dis) < 3.0


# ------------------------------------------------------------------ K1

def test_k1_places_parallel_edges_at_gap_plus_minus_half_depth():
    # RGB quad: front edge 6 mm too far forward, back edge exact; the gap is the true middle
    quad = _quad(y1=HALF_Y + 0.006)
    # seam endpoints on the quad's left/right edges, at the true middle y = 0 of the BOX
    left = _px(-HALF_X, 0.0)
    right = _px(HALF_X, 0.0)
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, HALF_Y, Z_TOP)])
    res = top_from_gap.compute(quad, {"seam_left": left.tolist(), "seam_right": right.tolist()}, "lr",
                               xyz, _proj(xyz), (N, D), lambda n, d: _bp(n, d), _project,
                               390.0, 320.0, (W, H))
    assert res["applied"], res
    assert res["deltas_mm"]["front"] == pytest.approx(-6.0, abs=0.6)
    assert res["deltas_mm"]["back"] == pytest.approx(0.0, abs=0.6)
    assert res["deltas_mm"]["left"] == 0.0 and res["deltas_mm"]["right"] == 0.0
    assert res["recentred_ends"] is False and res["seam_endpoints"] is None
    s = res["sides_mm"]
    assert s[0] == pytest.approx(390.0, abs=1.0) and s[1] == pytest.approx(320.0, abs=0.5)
    # corners are the plane intersections of the new px rays
    back = _bp()(res["quad_px"])
    assert np.allclose(back, res["corners_xyz"], atol=1e-6)


def test_k1_recentres_ends_when_the_length_deviates():
    quad = _quad(x0=-HALF_X - 0.008, x1=HALF_X + 0.010)      # 18 mm too long along the seam
    left, right = _px(-HALF_X - 0.008, 0.0), _px(HALF_X + 0.010, 0.0)
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, HALF_Y, Z_TOP)])
    res = top_from_gap.compute(quad, {"seam_left": left.tolist(), "seam_right": right.tolist()}, "lr",
                               xyz, _proj(xyz), (N, D), lambda n, d: _bp(n, d), _project,
                               390.0, 320.0, (W, H))
    assert res["applied"] and res["recentred_ends"] is True
    assert res["sides_mm"][0] == pytest.approx(390.0, abs=1.0)
    assert res["seam_endpoints"] is not None
    sl, sr = res["seam_endpoints"]["seam_left"], res["seam_endpoints"]["seam_right"]
    assert (sr[0] - sl[0]) / F * Z_TOP * 1000.0 == pytest.approx(390.0, abs=1.0)


def test_k1_declines_a_truncated_quad():
    # the quad covers only the back half of the top: across distance 160 mm, not 320
    quad = _quad(y1=0.0)
    left, right = _px(-HALF_X, -0.08), _px(HALF_X, -0.08)
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, HALF_Y, Z_TOP)])
    res = top_from_gap.compute(quad, {"seam_left": left.tolist(), "seam_right": right.tolist()}, "lr",
                               xyz, _proj(xyz), (N, D), lambda n, d: _bp(n, d), _project,
                               390.0, 320.0, (W, H))
    assert not res["applied"] and res["reason"] == "across_mismatch"


def test_k1_declines_unknown_axis_and_bad_seam():
    quad = _quad()
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, HALF_Y, Z_TOP)])
    eps = {"seam_left": _px(-HALF_X, 0.0).tolist(), "seam_right": _px(HALF_X, 0.0).tolist()}
    kw = dict(xyz=xyz, proj=_proj(xyz), plane_nd=(N, D), backproject_factory=lambda n, d: _bp(n, d),
              project=_project, along_mm=390.0, across_mm=320.0, image_wh=(W, H))
    assert top_from_gap.compute(quad, eps, None, **kw)["reason"] == "axis_unknown"
    # a seam that does not end on the seam-end edges (wrong axis)
    assert top_from_gap.compute(quad, eps, "bf", **kw)["reason"] == "seam_not_on_end_edges"


def test_profile_flags():
    p = bp.PROFILES["large_392"]
    assert p.top_from_gap and p.edge_support_cut_rule and p.multi_box_reject
    for name in ("small", "big", "generic"):
        q = bp.PROFILES[name]
        assert not (q.top_from_gap or q.edge_support_cut_rule or q.multi_box_reject)
