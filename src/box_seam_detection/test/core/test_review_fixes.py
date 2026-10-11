# Provenance: ported from rgbd_captures/tests/test_review_fixes.py (HEAD 6df5994); imports made package-relative.
"""Fresh-review fixes: no rescue without a height, 400 mm plane limit for known
boxes, loud PROFILES_EXCLUDE errors, strict SAM3 gate overrides."""
import sys
from pathlib import Path

import numpy as np
import pytest


from box_seam_detection.core import box_profiles as bp  # noqa: E402
from box_seam_detection.core import pipeline as P  # noqa: E402
from box_seam_detection.core import sam3_worker as w  # noqa: E402


# 1 ---------------------------------------------------------------------------
def test_no_height_no_rescue():
    # dims 14 mm off the small top, no height: generic, not small
    prof, info = bp.resolve(0.124, 0.098, None)
    assert prof.is_generic and info["reason"] == "no_profile_within_tolerance"
    assert bp.resolve(0.101, 0.0338, None)[0].is_generic
    # exact dims match with no height stays known (classify unchanged)
    assert bp.resolve(0.1117, 0.0992, None)[0].name == "small"
    # with a measured height the rescue still works
    assert bp.resolve(0.124, 0.098, 87.0)[0].name == "small"
    assert bp.resolve(0.101, 0.0338, 85.9)[0].name == "small"


# 2 ---------------------------------------------------------------------------
def _ctx_two_planes(rng):
    """Camera-frame cloud: table plane + a parallel plane 0.55 m above it
    (a hand / arm) + a 0.10 m box top."""
    n = np.array([0.0, -0.7, -0.714]); n /= np.linalg.norm(n)
    e1 = np.cross(n, [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    d = 0.9

    def plane(h, half, cnt, c=(0.0, 0.0)):
        a = rng.uniform(-half, half, cnt) + c[0]
        b = rng.uniform(-half, half, cnt) + c[1]
        return (a[:, None] * e1 + b[:, None] * e2
                - (d - h) * n[None, :])
    pts = np.vstack([plane(0.0, 0.5, 60000), plane(0.55, 0.12, 6000)])
    return n, d, pts


def test_known_box_plane_limit_400_generic_700():
    from box_seam_detection.core.config import default_config
    rng = np.random.default_rng(0)
    n, d, pts = _ctx_two_planes(rng)
    ctx = {"xyz": pts, "cam": {"fx": 1000, "fy": 1000, "cx": 640, "cy": 360},
           "dcfg": default_config()["depth_top"]}
    ctx["_table"] = {"n": n, "d": d}
    sel = pts[pts @ n + d > 0.4]            # the 0.55 m plane only
    _, _, ch = P.choose_top_plane(sel, None, ctx)
    assert ch["method"] != "table_parallel"            # 400 mm limit
    with P._h_max_scope(ctx, True):
        _, _, ch2 = P.choose_top_plane(sel, None, ctx)
        assert ch2["method"] == "table_parallel"
        assert 520 < ch2["height_mm"] < 580
    assert "box_h_max_mm" not in ctx                   # scope restored
    _, _, ch3 = P.choose_top_plane(sel, None, ctx)
    assert ch3["method"] != "table_parallel"
    with P._h_max_scope(ctx, False):                   # disabled = no-op
        assert "box_h_max_mm" not in ctx


def test_tall_quad_decision_uses_own_height(monkeypatch):
    monkeypatch.setattr(P, "quad_height_mm", lambda *a, **k: 86.0)
    assert not P._tall_quad(None, {}, 10, 10)
    monkeypatch.setattr(P, "quad_height_mm", lambda *a, **k: 480.0)
    assert P._tall_quad(None, {}, 10, 10)
    monkeypatch.setattr(P, "quad_height_mm", lambda *a, **k: None)
    assert not P._tall_quad(None, {}, 10, 10)


# 3 ---------------------------------------------------------------------------
def test_exclude_unknown_name_raises_and_raw_recorded(monkeypatch):
    for bad in ("smal", "small,bogus", "generic"):
        monkeypatch.setenv(bp.ENV_EXCLUDE, bad)
        with pytest.raises(ValueError, match="PROFILES_EXCLUDE"):
            bp.excluded_names()
        with pytest.raises(ValueError):
            bp.classify(110, 98)
    monkeypatch.setenv(bp.ENV_EXCLUDE, " Small ")
    assert bp.excluded_names() == ("small",)
    _, rec = P.resolve_box_profile(
        np.array([[0, 0, .6], [.111, 0, .6], [.111, .098, .6], [0, .098, .6]]),
        {"measured_height_mm": 86.4})
    assert rec["profiles_excluded_raw"] == " Small "
    monkeypatch.delenv(bp.ENV_EXCLUDE)
    _, rec = P.resolve_box_profile(
        np.array([[0, 0, .6], [.111, 0, .6], [.111, .098, .6], [0, .098, .6]]),
        {"measured_height_mm": 86.4})
    assert rec["profiles_excluded_raw"] == "" and rec["profiles_excluded"] == []


# 4 ---------------------------------------------------------------------------
@pytest.mark.parametrize("g", [
    {"max_area_frac": True}, {"min_brown_frac": False},
    {"max_area_frac": float("nan")}, {"max_area_frac": float("inf")},
    {"max_area_frac": 25}, {"max_area_frac": 25.0}, {"min_brown_frac": 1.01},
    {"min_area_frac": -0.1}])
def test_gate_overrides_reject_bad_values(g):
    err = w._resolve_gates({"gates": g})[3]
    assert err, g


@pytest.mark.parametrize("g", [{"max_area_frac": 1.0}, {"min_area_frac": 0.0},
                               {"min_brown_frac": 0.0}, {"max_area_frac": 0.6}])
def test_gate_overrides_accept_fractions(g):
    assert w._resolve_gates({"gates": g})[3] is None
