"""2026-10-24 speed / robustness changes: superset-first candidate order, main-path flap check,
per-stage clock, raised-flap retry signature (synthetic scene of test_top_depth_diag)."""
import numpy as np
import pytest

from box_seam_detection.core import pipeline as P
from box_seam_detection.live import orchestrator as orch

from test_top_depth_diag import CAM, D, HALF_X, HALF_Y, N, H, W, _cloud, _proj, _px, _quad


def _box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _live(quad, score, rank=None):
    c = {"source": "sam3_live", "coarse": quad, "prov": {}, "concept_score": score}
    if rank is not None:
        c["alternate_rank"] = rank
    return c


# ---- superset_first ---------------------------------------------------------------

def test_superset_promoted_when_it_contains_the_first_candidate():
    flap = _box(100, 100, 500, 200)                 # SAM3's first pick: one flap
    full = _box(95, 95, 505, 300)                   # the whole top (contains the flap, ~2x)
    other = _box(700, 700, 760, 760)
    cands = [_live(flap, 0.8), _live(other, 0.6, 1), _live(full, 0.5, 2)]
    moved = P.promote_superset_candidate(cands)
    assert moved is not None and moved["promoted_alternate_rank"] == 2
    assert cands[0]["coarse"] is full and cands[1]["coarse"] is flap   # order only, nothing dropped
    assert len(cands) == 3 and moved["contain"] >= 0.9 and 1.4 <= moved["area_ratio"] <= 3.5


@pytest.mark.parametrize("cand", [
    _box(300, 100, 700, 300),                       # overlaps but does not contain
    _box(0, 0, 1000, 1000),                         # contains but far too big (> 3.5x)
    _box(98, 98, 502, 210),                         # contains but about the same size (< 1.4x)
])
def test_superset_not_promoted(cand):
    first = _box(100, 100, 500, 200)
    cands = [_live(first, 0.8), _live(cand, 0.5, 1)]
    assert P.promote_superset_candidate(cands) is None
    assert cands[0]["coarse"] is first


def test_superset_ignores_unusable_candidates():
    flap = _box(100, 100, 500, 200)
    cands = [_live(flap, 0.8), {**_live(None, 0.7, 1)},
             {"source": "sam3_archived", "coarse": _box(95, 95, 505, 300), "prov": {}}]
    assert P.promote_superset_candidate(cands) is None


def test_flag_and_mode_parsing():
    assert P._flag_on("on") and P._flag_on(True) and not P._flag_on("off") and not P._flag_on(None)
    assert P._flap_check_mode({"flap_check": "shadow"}) == "shadow"
    assert P._flap_check_mode({"flap_check": True}) == "on"
    assert P._flap_check_mode({"flap_check": "bogus"}) == "off"
    assert P._flap_check_mode({}) == "off"


# ---- main-path flap check ---------------------------------------------------------

def _seam(status="ok"):
    return {"status": status, "endpoints_px": {"seam_left": list(_px(-HALF_X, 0.0)),
                                               "seam_right": list(_px(HALF_X, 0.0))}}


def _ctx(zb):
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, 0.0, 1.2), (-HALF_X, HALF_X, 0.0, HALF_Y, zb)],
                 noise=0.0005)
    return {"xyz": xyz, "cam": CAM, "_proj": _proj(xyz)}


PUB = {"normal": list(N), "offset_m": D}


def test_flap_check_coplanar_passes():
    out = P.main_flap_check(_quad(), _seam(), PUB, 1.0, _ctx(1.2), W, H, {"flap_check": "on"})
    assert out["measured"] and out["ok"] and out["step_mm"] < 1.0


def test_flap_check_raised_flap_fails_with_hint():
    out = P.main_flap_check(_quad(), _seam(), PUB, 1.0, _ctx(1.2 - 0.008), W, H,
                            {"flap_check": "shadow"})
    assert out["mode"] == "shadow" and out["measured"] and out["ok"] is False
    assert out["step_mm"] == pytest.approx(8.0, abs=0.6)
    assert out["hint"].startswith("raised_flap(step=") and out["reasons"][0].startswith("flap_step_too_large")


def test_flap_check_tightened_by_two_flap_limits():
    out = P.main_flap_check(_quad(), _seam(), PUB, 1.0, _ctx(1.2 - 0.004), W, H,
                            {"flap_check": "on", "two_flap_max_step_mm": 3.0})
    assert out["ok"] is False and out["limits"]["step_mm"] == 3.0
    out = P.main_flap_check(_quad(), _seam(), PUB, 1.0, _ctx(1.2 - 0.004), W, H,
                            {"flap_check": "on", "two_flap_max_step_mm": 9.0})   # never loosened
    assert out["ok"] is True and out["limits"]["step_mm"] == 5.0


def test_flap_check_off_and_unmeasurable():
    assert P.main_flap_check(_quad(), _seam(), PUB, 1.0, _ctx(1.2), W, H, {}) is None
    out = P.main_flap_check(_quad(), _seam("estimated"), PUB, 1.0, _ctx(1.2), W, H, {"flap_check": "on"})
    assert out["ok"] and not out["measured"] and out["skipped"] == "no_observed_seam"
    out = P.main_flap_check(_quad(), _seam(), PUB, 1.0, None, W, H, {"flap_check": "on"})
    assert out["ok"] and out["skipped"] == "no_cloud"


# ---- stage clock -----------------------------------------------------------------

def test_stage_clock_sums_and_stops():
    assert P.stage_clock_stop() is None              # off by default
    P._mark("x")                                     # no-op while off
    P.stage_clock_start()
    P._mark("a")
    P._mark("b")
    P._mark("b")
    P._stage_total("t", 1.23456)
    out = P.stage_clock_stop()
    assert set(out) == {"a", "b", "b_n", "t_incl"} and out["b_n"] == 2 and out["t_incl"] == 1.235
    assert P.stage_clock_stop() is None


# ---- orchestrator: raised-flap retry signature ------------------------------------

def test_raised_flap_signature():
    rej = {"stage": "rectangle", "reasons": ["rectangle_inconsistent(..)",
                                             "raised_flap(step=7.2mm,angle=4.9deg): press the flap flat"]}
    sig = orch.raised_flap_signature(rej)
    assert sig == {"step_mm": 7.2}
    assert orch.same_raised_flap_failure(sig, {"step_mm": 9.0})
    assert not orch.same_raised_flap_failure(sig, {"step_mm": 11.0})
    assert orch.raised_flap_signature({"reasons": ["top_size_out_of_range(x)"]}) is None
    assert not orch.same_raised_flap_failure(None, sig)


# ---- superset_first: verification of a promoted winner ------------------------------

def test_promoted_candidate_is_marked():
    flap, full = _box(100, 100, 500, 200), _box(95, 95, 505, 300)
    cands = [_live(flap, 0.8), _live(full, 0.5, 1)]
    P.promote_superset_candidate(cands)
    assert cands[0].get("superset_promoted") is True and not cands[1].get("superset_promoted")


def _win(diff_mm=0.0):
    # 390 x 280 mm rectangle (camera frame, metres); diff_mm lengthens the front edge only
    xyz = [[-0.195, -0.14, 1.2], [0.195, -0.14, 1.2], [0.195 + diff_mm / 1000, 0.14, 1.2], [-0.195, 0.14, 1.2]]
    return {"cross": {"corners_xyz": np.asarray(xyz)}}


def test_superset_verification_mirrors_recovery_checks():
    ok_seam = {"status": "ok", "source": "observed_gap_confirmed"}
    clean = {"edge_support": {"unsupported": []}}
    tc = dict(P.TRUNC_DEFAULTS)
    assert P._superset_unverified(_win(), ok_seam, clean, None, tc) is None
    assert P._superset_unverified(_win(25.0), ok_seam, clean, None, tc) == "generic_profile_fails_rectangle_gate"
    assert P._superset_unverified(_win(), {"status": "estimated"}, clean, None, tc) == "seam_not_ok(estimated)"
    assert P._superset_unverified(_win(), ok_seam, {"edge_support": {"unsupported": ["back"]}}, None, tc) \
        == "final_quad_edge_support_not_clean"
    assert P._superset_unverified(_win(), ok_seam, None, None, tc) == "final_quad_edge_support_not_clean"
