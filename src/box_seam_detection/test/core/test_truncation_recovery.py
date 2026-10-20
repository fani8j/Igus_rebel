"""Truncation recovery: SAM3 outlines only one flap of the lid, D1 sees the top
continue beyond the front edge. Helpers (top_diag.edge_check, grow_coplanar_region),
the orchestration (pipeline._truncation_recovery with stubbed evaluate / rerun) and
the scene flag, on the synthetic fronto-parallel scene of test_top_depth_diag
(390 x 320 mm top at z = 1.2 m, floor at 1.7 m)."""
import math
import types

import numpy as np
import pytest

from box_seam_detection.core import pipeline as P
from box_seam_detection.core import scene as scene_mod
from box_seam_detection.core import top_diag
from box_seam_detection.core.config import default_config

from test_top_depth_diag import (CAM, CX, CY, D, F, H, HALF_X, HALF_Y, N, W, _bp, _px,
                                 _cloud, _proj, _quad)

MAN = {"camera_info": {"k": [F, 0.0, CX, 0.0, F, CY, 0.0, 0.0, 1.0]}}
PLANE = types.SimpleNamespace(normal=N, d=D)


@pytest.fixture(scope="module")
def xyz_proj():
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, HALF_Y, 1.2)], noise=0.001)
    return xyz, _proj(xyz)


def _half():
    """Back flap only: back edge -> the flap gap (y = 0)."""
    return _quad(y1=0.0)


def _corners3(quad):
    return _bp()(quad)


# ---- mode / scene flag -------------------------------------------------------

def test_mode_default_off_and_scene_flag(monkeypatch):
    monkeypatch.delenv("AUTOLABEL_V2_TRUNC_RECOVERY", raising=False)
    assert P._truncation_mode(default_config()) == "off"
    table = default_config(scene_mod.load_scene("table_cell_v1"))
    dual = default_config(scene_mod.load_scene("dual_arm_v1"))
    assert P._truncation_mode(table) == "off"
    assert P._truncation_mode(dual) == "on"
    monkeypatch.setenv("AUTOLABEL_V2_TRUNC_RECOVERY", "shadow")
    assert P._truncation_mode(dual) == "shadow"
    monkeypatch.setenv("AUTOLABEL_V2_TRUNC_RECOVERY", "off")
    assert P._truncation_mode(dual) == "off"
    monkeypatch.delenv("AUTOLABEL_V2_TRUNC_RECOVERY")
    assert P._truncation_mode({"truncation_recovery": {"mode": True}}) == "on"   # YAML `on`
    assert P._truncation_mode({"truncation_recovery": {"mode": "bogus"}}) == "off"


# ---- helpers -----------------------------------------------------------------

def test_edge_check_flags_half_top_and_not_full_top(xyz_proj):
    xyz, proj = xyz_proj
    half = top_diag.edge_check(_half(), (N, D), 1.0, xyz, proj, _bp(), (W, H))
    assert half["unsupported"] == ["front"]
    full = top_diag.edge_check(_quad(), (N, D), 1.0, xyz, proj, _bp(), (W, H))
    assert full["unsupported"] == []
    assert "error" in top_diag.edge_check(_quad(), (N, D), 1.0, None, proj, _bp(), (W, H))


def test_grow_coplanar_region_recovers_the_full_top(xyz_proj):
    xyz, proj = xyz_proj
    mask, info = top_diag.grow_coplanar_region(_half(), (N, D), 1.0, xyz, proj, CAM, (W, H))
    assert mask is not None, info
    ys, xs = np.nonzero(mask)
    full = _quad()
    assert abs(xs.min() - full[:, 0].min()) <= 4 and abs(xs.max() - full[:, 0].max()) <= 4
    assert abs(ys.min() - full[:, 1].min()) <= 4 and abs(ys.max() - full[:, 1].max()) <= 4


def test_grow_without_component_under_seed(xyz_proj):
    xyz, proj = xyz_proj
    far = _quad(-0.3, -0.25, 0.5, 0.55)       # over the floor, far from the top
    mask, info = top_diag.grow_coplanar_region(far, (N, D), 1.0, xyz, proj, CAM, (W, H))
    assert mask is None and "reason" in info


# ---- orchestration -------------------------------------------------------------

def _stubs(xyz, proj, tmp_seam="ok", profile="large_392", dims=(392.0, 327.0),
           unsupported=()):
    """(win0, depth_ctx, evaluate, rerun, calls) with stubbed refine/depth check/seam stage."""
    calls = {"evaluate": [], "rerun": []}
    depth_ctx = {"xyz": xyz, "cam": CAM, "_proj": proj}

    def win_of(quad, cand):
        return {"cand": cand, "quad": np.asarray(quad, float), "refined": None,
                "edge_rms": 0.4, "edge_support": 0.9,
                "cross": {"plane": PLANE, "plane_rms_mm": 1.0,
                          "corners_xyz": _corners3(quad), "wrong_face": False}}

    cand0 = {"source": "sam3_live", "coarse": _half(), "prov": {}}
    win0 = win_of(_half(), cand0)

    def evaluate(cand, entry):
        calls["evaluate"].append((entry, cand))
        return win_of(cand["coarse"], cand)

    def rerun(win):
        calls["rerun"].append(win)
        rec = {"box_profile": {"name": profile, "top_dims_mm": list(dims)},
               "seam": {"status": tmp_seam, "source": "observed_gap"},
               "top_face": {"corners_px": {"back_left": [0, 0]},
                            "depth_diag": {"edge_support": {"unsupported": list(unsupported)}}},
               "checks": {"reasons": []}}
        return rec, {"dbg": 1}

    return win0, depth_ctx, evaluate, rerun, calls


def _call(win0, depth_ctx, evaluate, rerun, mode="on", cands=(), attempts=None, reasons=None,
          anchored=None):
    attempts = [] if attempts is None else attempts
    reasons = [] if reasons is None else reasons
    cfg = default_config()
    if anchored is not None:
        cfg["truncation_recovery"]["anchored_rect"] = anchored
    res = P._truncation_recovery(win0, list(cands) or [win0["cand"]], mode,
                                 cfg, depth_ctx, MAN, W, H,
                                 attempts, reasons, evaluate, rerun)
    return res, attempts, reasons


def test_recovery_depth_grown_applied(xyz_proj):
    xyz, proj = xyz_proj
    win0, ctx, ev, rr, calls = _stubs(xyz, proj)
    (res, block), attempts, _ = _call(win0, ctx, ev, rr)
    assert res is not None
    rec, dbg = res
    assert dbg == {"dbg": 1}
    assert block["recovered"] and block["applied"] and block["method"] == "depth_grown"
    assert block["trigger"]["edges"] == ["front"]
    assert rec["truncation_recovery"] is block
    assert block["before"]["top_dims_mm"][1] < 200 and block["after"]["profile"] == "large_392"
    assert rec["checks"]["reasons"][-1].startswith("truncation_recovered(depth_grown")
    assert block["tried"][-1]["result"] == "verified"
    assert block["tried"][-1]["contains_original"] >= 0.9
    # the evaluated candidate is the grown quad: full top height
    coarse = calls["evaluate"][0][1]["coarse"]
    assert abs(coarse[:, 1].max() - _quad()[:, 1].max()) <= 4


def test_recovery_prefers_a_containing_sam3_alternate(xyz_proj):
    xyz, proj = xyz_proj
    win0, ctx, ev, rr, calls = _stubs(xyz, proj)
    alt = {"source": "sam3_live", "coarse": _quad(), "alternate_rank": 1,
           "concept_score": 0.7, "prov": {}}
    bad_alt = {"source": "sam3_live", "coarse": _quad(-0.1, 0.1, 0.1, 0.2),   # does not contain
               "alternate_rank": 2, "prov": {}}
    (res, block), _, _ = _call(win0, ctx, ev, rr, cands=[win0["cand"], bad_alt, alt])
    assert block["method"] == "sam3_alternate" and block["tried"][0]["alternate_rank"] == 1
    assert len(calls["evaluate"]) == 1 and res is not None


def test_recovery_shadow_does_not_apply_and_restores_lists(xyz_proj):
    xyz, proj = xyz_proj
    win0, ctx, ev, rr, _ = _stubs(xyz, proj)
    (res, block), attempts, reasons = _call(win0, ctx, ev, rr, mode="shadow",
                                            attempts=["a"], reasons=["r"])
    assert res is None and block["recovered"] and not block["applied"]
    assert attempts == ["a"] and reasons == ["r"]


@pytest.mark.parametrize("kw,why", [
    ({"tmp_seam": "ambiguous"}, "seam_not_ok"),
    ({"unsupported": ("front",)}, "final_quad_edge_support_not_clean"),
    ({"profile": "generic", "dims": (390.0, 300.0)}, "generic_profile_fails_rectangle_gate")])
def test_recovery_rejected_keeps_old_behaviour(xyz_proj, kw, why):
    xyz, proj = xyz_proj
    win0, ctx, ev, rr, _ = _stubs(xyz, proj, **kw)
    if "profile" in kw:      # generic: the 3-D rectangle gate looks at the candidate's corners
        orig = ev

        def ev(cand, entry):
            w = orig(cand, entry)
            c = w["cross"]["corners_xyz"].copy()
            c[2] += [0.05, 0, 0]
            w["cross"]["corners_xyz"] = c
            return w
    (res, block), attempts, reasons = _call(win0, ctx, ev, rr, attempts=[], reasons=[])
    assert res is None and not block["recovered"] and not block["applied"]
    assert block["tried"][-1]["result"].startswith(why)
    assert attempts == [] and reasons == []


def test_not_truncated_returns_nothing(xyz_proj):
    xyz, proj = xyz_proj
    win0, ctx, ev, rr, calls = _stubs(xyz, proj)
    full = _quad()
    win0["quad"] = full
    win0["cross"]["corners_xyz"] = _corners3(full)
    (res, block), _, _ = _call(win0, ctx, ev, rr)
    assert res is None and block is None and not calls["evaluate"]


# ---- anchored rectangle (2026-10-21) ----------------------------------------------
# A lifted flap corner leaves the depth band, so the hull 4-gon of the grown region is
# skewed and the refined back edge ended up tilted (still_truncated /
# generic_profile_fails_rectangle_gate on ~30 % of the dual-arm captures). The anchored
# rectangle keeps the supported edges of the half-top and moves only the truncated one.

def _poly_mask(corners, w=W, h=H):
    import cv2
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.rint(np.asarray(corners, float)).astype(np.int32)], 1)
    return m


def _chop(mask, tri):
    import cv2
    cv2.fillPoly(mask, [np.rint(np.asarray(tri, float)).astype(np.int32)], 0)
    return mask


def _anchored(quad, trunc, region):
    return top_diag.anchored_rect_quad(quad, trunc, (N, D), region, CAM, (W, H))


def test_anchored_rect_recovers_the_full_top_with_a_chopped_corner():
    full = _quad()
    region = _poly_mask(full)
    # the corner of the truncated side is lost (lifted flap): a hull 4-gon would skew
    _chop(region, [full[3] + [-5, 5], full[3] + [90, 5], full[3] + [-5, -70]])
    q, info = _anchored(_half(), [2], region)
    assert q is not None, info
    assert np.abs(q - full).max() <= 4.0, (q, full)
    assert abs(q[2, 1] - q[3, 1]) <= 1.0          # the new edge is parallel to the anchor
    assert info["anchor_edge"] == 0 and info["extension_mm"]["2"] > 100
    assert info["region_inside_frac"] >= 0.95 and info["region_coverage"] >= 0.8


def test_anchored_rect_ignores_tilted_short_side_edges_of_the_half_top():
    full = _quad()
    half = _half().copy()
    half[0, 0] += 6.0          # the short side edges of a half-top are noisy (several deg):
    half[1, 0] += 5.0          # their own direction must not be extrapolated
    half[2, 0] -= 5.0
    half[3, 0] += 5.0
    q, info = _anchored(half, [2], _poly_mask(full))
    assert q is not None, info
    assert np.abs(q - full).max() <= 8.0
    # square to the anchor: left/right sides are vertical in this fronto-parallel scene
    assert abs(q[0, 0] - q[3, 0]) < 1e-6 and abs(q[1, 0] - q[2, 0]) < 1e-6
    assert info["supported_tilt_deg"]["1"] < top_diag.ANCHOR_MAX_TILT_DEG


def test_anchored_rect_follows_a_yawed_box():
    a = math.radians(4.0)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def corners(y1):
        pts = np.array([[-HALF_X, -HALF_Y], [HALF_X, -HALF_Y], [HALF_X, y1], [-HALF_X, y1]])
        return np.array([_px(*(R @ p)) for p in pts])

    q, info = _anchored(corners(0.0), [2], _poly_mask(corners(HALF_Y)))
    assert q is not None, info
    assert np.abs(q - corners(HALF_Y)).max() <= 3.0


def test_anchored_rect_on_the_grown_region_of_a_noisy_cloud(xyz_proj):
    xyz, proj = xyz_proj
    hull, region, info = top_diag.grow_coplanar_region_ex(
        _half(), (N, D), 1.0, xyz, proj, CAM, (W, H))
    assert hull is not None and region is not None, info
    hull2, info2 = top_diag.grow_coplanar_region(_half(), (N, D), 1.0, xyz, proj, CAM, (W, H))
    assert np.array_equal(hull, hull2) and info == info2          # wrapper unchanged
    assert region.sum() <= hull.sum()
    q, ainfo = _anchored(_half(), [2], region)
    assert q is not None, ainfo
    assert np.abs(q - _quad()).max() <= 6.0


@pytest.mark.parametrize("trunc", [[], [0, 2], [1, 3], [0, 1, 2], [0, 1, 2, 3]])
def test_anchored_rect_unsupported_truncation_patterns_fail_closed(trunc):
    q, info = _anchored(_half(), trunc, _poly_mask(_quad()))
    assert q is None and info["reason"] == "truncation_pattern_unsupported"


def test_anchored_rect_fails_closed():
    full = _quad()
    # no region / a tiny region
    assert _anchored(_half(), [2], None)[1]["reason"] == "no_region"
    assert _anchored(_half(), [2], _poly_mask(_quad(-0.01, 0.01, -0.01, 0.01)))[1]["reason"] == "no_region"
    # the region does not extend beyond the quad: the "truncation" is not confirmed
    q, info = _anchored(_half(), [2], _poly_mask(_half()))
    assert q is None and info["reason"].startswith("region_does_not_extend_2")
    # the region is far wider than the supported side edges: another surface leaked in
    wide = _poly_mask(_quad(-0.4, 0.4, -HALF_Y, HALF_Y))
    q, info = _anchored(_half(), [2], wide)
    assert q is None and info["reason"].startswith("region_spills_out_of_rectangle")
    # the region is a thin band along the back edge plus the new end: it does not fill the rect
    sparse = _poly_mask(full)
    _chop(sparse, [full[0] + [10, 30], full[1] + [-10, 30], full[2] + [-10, -30], full[3] + [10, -30]])
    q, info = _anchored(_half(), [2], sparse)
    assert q is None and info["reason"].startswith("region_does_not_fill_rectangle")
    # supported side edge far from square to the anchor: not a rectangle
    skew = _half().copy()
    skew[0, 0] += 60.0
    skew[1, 0] -= 60.0
    q, info = _anchored(skew, [2], _poly_mask(full))
    assert q is None and info["reason"].startswith("supported_edge_not_square")


def _grown_only_stubs(xyz, proj):
    """Stubs where the hull-4-gon candidate is rejected by the (stubbed) evaluation."""
    win0, ctx, ev, rr, calls = _stubs(xyz, proj)
    log = {"attempts": None, "reasons": None, "order": []}

    def evaluate(cand, entry):
        prov = cand.get("prov") or {}
        which = ("anchored" if "truncation_anchored_rect" in prov
                 else "hull" if "truncation_grown" in prov else "other")
        log["order"].append(which)
        log["attempts"].append({"which": which})
        log["reasons"].append(f"note({which})")
        if which == "hull":
            return None
        return ev(cand, entry)

    return win0, ctx, evaluate, rr, calls, log


def _run_grown(xyz, proj, anchored, mode="on"):
    win0, ctx, evaluate, rr, calls, log = _grown_only_stubs(xyz, proj)
    attempts, reasons = ["a"], ["r"]
    log["attempts"], log["reasons"] = attempts, reasons
    res = _call(win0, ctx, evaluate, rr, mode=mode, attempts=attempts, reasons=reasons,
                anchored=anchored)
    return res, log, calls


def test_recovery_falls_back_to_the_anchored_rectangle(xyz_proj):
    xyz, proj = xyz_proj
    ((res, block), attempts, reasons), log, calls = _run_grown(xyz, proj, "fallback")
    assert log["order"] == ["hull", "anchored"]
    assert res is not None and block["applied"] and block["method"] == "depth_grown_anchored"
    assert [t["method"] for t in block["tried"]] == ["depth_grown", "depth_grown_anchored"]
    assert block["tried"][0]["result"] == "rejected_by_existing_checks"
    assert block["tried"][1]["result"] == "verified" and block["tried"][1]["contains_original"] >= 0.9
    assert block["tried"][1]["anchor_edge"] == 0 and "coarse_px" in block["tried"][1]
    assert res[0]["checks"]["reasons"][-1].startswith("truncation_recovered(depth_grown_anchored")
    # the rejected hull candidate leaves no evaluation notes behind
    assert attempts == ["a", {"which": "anchored"}] and reasons == ["r", "note(anchored)"]
    assert abs(calls["evaluate"][-1][1]["coarse"][:, 1].max() - _quad()[:, 1].max()) <= 6


def test_recovery_anchored_off_keeps_the_old_behaviour(xyz_proj):
    xyz, proj = xyz_proj
    ((res, block), attempts, reasons), log, _ = _run_grown(xyz, proj, "off")
    assert log["order"] == ["hull"] and res is None
    assert not block["recovered"] and attempts == ["a"] and reasons == ["r"]
    ((res, block), _, _), log, _ = _run_grown(xyz, proj, "bogus")      # unknown value = fallback
    assert log["order"] == ["hull", "anchored"] and res is not None


def test_recovery_anchored_first_skips_the_hull_when_it_works(xyz_proj):
    xyz, proj = xyz_proj
    ((res, block), _, _), log, _ = _run_grown(xyz, proj, "first")
    assert log["order"] == ["anchored"] and block["method"] == "depth_grown_anchored"


def test_recovery_hull_candidate_still_wins_by_default(xyz_proj):
    xyz, proj = xyz_proj
    win0, ctx, ev, rr, calls = _stubs(xyz, proj)
    (res, block), _, _ = _call(win0, ctx, ev, rr)
    assert block["method"] == "depth_grown" and len(calls["evaluate"]) == 1


# ---- saved-derived: the live half-top captures (dual_arm_v1, 2026-10-20) -------------
# test/core/data/truncation_halftop_regions.json holds, for frame_000008 and frame_000011,
# the half-top quad (refined SAM3 outline), the top plane / pinhole of the capture and the
# outline of the depth-grown coplanar region (back-left corner chopped by ~35 px). The
# six accepted captures of the same static box give the ground-truth corners.

import json
import pathlib

_GT_PX = np.array([[761.8, 708.2], [1161.3, 696.9], [1206.3, 983.9], [710.1, 997.6]])


@pytest.mark.parametrize("frame", ["h8", "h11"])
def test_anchored_rect_on_the_live_half_top_captures(frame):
    d = json.loads((pathlib.Path(__file__).parent / "data"
                    / "truncation_halftop_regions.json").read_text())[frame]
    w, h = d["wh"]
    region = _poly_mask(np.array(d["region_polygon"]), w, h)
    q, info = top_diag.anchored_rect_quad(np.array(d["quad"]), [0], (np.array(d["n"]), d["d"]),
                                          region, d["cam"], (w, h))
    assert q is not None, info
    # a coarse candidate: refine moves each edge by <= 15 px; measured <= 4 px (config refine.max_offset_px)
    assert np.abs(q - _GT_PX).max() <= 8.0, (q, _GT_PX)
    # back edge parallel to the front edge (the hull 4-gon had its back-left corner ~35 px low)
    assert abs((q[1, 1] - q[0, 1]) - (q[2, 1] - q[3, 1])) <= 8.0
    assert q[0, 1] < 725.0
    assert info["anchor_edge"] == 2 and info["extension_mm"]["0"] > 100
