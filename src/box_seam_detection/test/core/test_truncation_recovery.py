"""Truncation recovery: SAM3 outlines only one flap of the lid, D1 sees the top
continue beyond the front edge. Helpers (top_diag.edge_check, grow_coplanar_region),
the orchestration (pipeline._truncation_recovery with stubbed evaluate / rerun) and
the scene flag, on the synthetic fronto-parallel scene of test_top_depth_diag
(390 x 320 mm top at z = 1.2 m, floor at 1.7 m)."""
import types

import numpy as np
import pytest

from box_seam_detection.core import pipeline as P
from box_seam_detection.core import scene as scene_mod
from box_seam_detection.core import top_diag
from box_seam_detection.core.config import default_config

from test_top_depth_diag import (CAM, CX, CY, D, F, H, HALF_X, HALF_Y, N, W, _bp,
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


def _call(win0, depth_ctx, evaluate, rerun, mode="on", cands=(), attempts=None, reasons=None):
    attempts = [] if attempts is None else attempts
    reasons = [] if reasons is None else reasons
    res = P._truncation_recovery(win0, list(cands) or [win0["cand"]], mode,
                                 default_config(), depth_ctx, MAN, W, H,
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
