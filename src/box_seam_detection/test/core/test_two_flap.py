"""Two-flap truncation recovery: plane-pair gates, vertex-deviation gate and the orchestration on
the synthetic fronto-parallel scene of test_top_depth_diag (390 x 320 mm top at z = 1.2 m)."""
import math
import types

import numpy as np
import pytest

from box_seam_detection.core import pipeline as P
from box_seam_detection.core import two_flap as tf
from box_seam_detection.core.config import default_config

from test_top_depth_diag import (CAM, CX, CY, D, F, H, HALF_X, HALF_Y, N, W, _bp, _cloud, _px,
                                 _proj, _quad)
from test_truncation_recovery import MAN, PLANE, _corners3, _half


def _tilted(deg):
    a = math.radians(deg)
    n = np.array([0.0, math.sin(a), -math.cos(a)])
    return tf.plane_nd(n, D)


def _mid():
    return _px(0.0, 0.0)


@pytest.mark.parametrize("planeb,ok", [
    ((N, D), True),                                  # coplanar
    ((N, D - 0.002), True),                          # 2 mm step
    ((N, D - 0.011), False),                         # 11 mm step
    (_tilted(3.7), False),                           # 3.7 deg tilt
    (_tilted(20.0), False)])                         # open flap
def test_pair_gates(planeb, ok):
    met = tf.plane_pair_metrics((N, D), planeb, CAM, _mid())
    assert (tf.gate_pair(met["step_mm"], met["angle_deg"]) == []) == ok
    if not ok:
        assert tf.hint_for(met["step_mm"], met["angle_deg"]).endswith("press the flap flat")


def test_pair_reason_strings():
    assert tf.gate_pair(11.2, 0.0) == ["flap_step_too_large(11.2mm)"]
    assert tf.gate_pair(1.0, 3.7) == ["flap_angle_too_large(3.7deg)"]
    assert tf.hint_for(11.2, 0.5) == "raised_flap(step=11.2mm): press the flap flat"


def test_vertex_deviation_against_published_plane():
    poly = np.array([_px(-HALF_X, 0.0), _px(HALF_X, 0.0), _px(HALF_X, HALF_Y), _px(-HALF_X, HALF_Y)])
    own_b = (N, D - 0.004)                           # flap B 4 mm higher than A
    dev_a_pub, _ = tf.lift_deviation_mm(poly, own_b, (N, D), CAM)
    assert dev_a_pub == pytest.approx(4.0, abs=0.01)   # published on A: B deviates 4 mm -> reject (> 3)
    dev_mid, _ = tf.lift_deviation_mm(poly, own_b, (N, D - 0.002), CAM)
    assert dev_mid == pytest.approx(2.0, abs=0.01)     # published between the flaps (LSQ-like): 2 mm
    own_tilt = _tilted(2.0)                            # tilted B: the deviation grows to the far vertex
    dev_t, per = tf.lift_deviation_mm(poly, own_tilt, (N, D), CAM)
    assert dev_t == pytest.approx(max(per), abs=0.01) and dev_t > 3.0


def test_split_by_line():
    quad = _quad()
    a, b = tf.split_by_line(quad, _px(-HALF_X, 0.0), _px(HALF_X, 0.0), _quad(y1=-0.05).mean(axis=0))
    assert a[:, 1].max() == pytest.approx(_px(0, 0)[1], abs=0.01) and b[:, 1].min() == pytest.approx(_px(0, 0)[1], abs=0.01)
    assert tf.polygon_area(a) + tf.polygon_area(b) == pytest.approx(tf.polygon_area(quad), rel=1e-6)


# ---- orchestration -------------------------------------------------------------

def _scene(zb, noise=0.0005):
    xyz = _cloud([(-HALF_X, HALF_X, -HALF_Y, 0.0, 1.2), (-HALF_X, HALF_X, 0.0, HALF_Y, zb)], noise=noise)
    return xyz, _proj(xyz)


def _run(zb, pub_d=D, seam_source="observed_gap"):
    xyz, proj = _scene(zb)
    calls = []
    cand0 = {"source": "sam3_live", "coarse": _half(), "prov": {}}
    win0 = {"cand": cand0, "quad": _half(), "cross": {"plane": PLANE, "plane_rms_mm": 1.0,
                                                      "corners_xyz": _corners3(_half())}}

    def evaluate(cand, entry):
        calls.append(entry.get("truncation_recovery"))
        if entry.get("truncation_recovery") != "two_flap":
            return None
        return {"cand": cand, "quad": np.asarray(cand["coarse"], float), "refined": None,
                "edge_rms": 0.4, "edge_support": 0.9,
                "cross": {"plane": PLANE, "plane_rms_mm": 1.0, "corners_xyz": _corners3(cand["coarse"])}}

    def rerun(win):
        q = win["quad"]
        names = ("back_left", "back_right", "front_right", "front_left")
        rec = {"box_profile": {"name": "large_392", "top_dims_mm": [392.0, 320.0]},
               "seam": {"status": "ok", "source": seam_source,
                        "endpoints_px": {"seam_left": list(_px(-HALF_X, 0.0)), "seam_right": list(_px(HALF_X, 0.0))}},
               "top_face": {"corners_px": {k: list(q[i]) for i, k in enumerate(names)},
                            "plane": {"normal": list(N), "offset_m": pub_d},
                            "depth_diag": {"edge_support": {"unsupported": []}}},
               "checks": {"reasons": []}}
        return rec, {}

    cfg = default_config()
    res = P._truncation_recovery(win0, [cand0], "on", cfg, {"xyz": xyz, "cam": CAM, "_proj": proj},
                                 MAN, W, H, [], [], evaluate, rerun)
    return res, calls


@pytest.fixture(scope="module")
def small_step():
    return _run(1.2 - 0.002)


def test_small_step_recovers(small_step):
    (res, block), calls = small_step
    assert res is not None and block["method"] == "two_flap" and block["applied"]
    assert calls[-1] == "two_flap"
    two = block["two_flap"]
    assert two["applied"] and two["step_mm"] == pytest.approx(2.0, abs=1.0)
    assert max(two["dev_mm"].values()) <= 3.0
    tags = [r for r in res[0]["checks"]["reasons"] if r.startswith("two_flap(")]
    assert tags and tags[0] == two["tag"]


def test_8mm_step_rejected_with_hint():
    (res, block), _ = _run(1.2 - 0.008)
    assert res is None and not block["recovered"]
    assert block["tried"][-1]["result"].startswith("flap_step_too_large(")
    assert block["two_flap"]["hint"].startswith("raised_flap(step=") and "press the flap flat" in block["two_flap"]["hint"]


def test_4mm_step_published_on_one_flap_fails_depth_gate():
    (res, block), _ = _run(1.2 - 0.004)
    assert res is None
    assert block["tried"][-1]["result"].startswith("flap_depth_dev_too_large(")
    assert "hint" in block["two_flap"]


def test_estimated_seam_is_rejected():
    (res, block), _ = _run(1.2 - 0.002, seam_source="prior")
    assert res is None and block["tried"][-1]["result"].find("seam_not_observed") >= 0


def test_two_flap_off_and_after_others(monkeypatch):
    xyz, proj = _scene(1.2 - 0.002)
    cfg = default_config()
    cfg["truncation_recovery"]["two_flap"] = False        # YAML `off`
    cand0 = {"source": "sam3_live", "coarse": _half(), "prov": {}}
    win0 = {"cand": cand0, "quad": _half(), "cross": {"plane": PLANE, "plane_rms_mm": 1.0,
                                                      "corners_xyz": _corners3(_half())}}
    res, block = P._truncation_recovery(win0, [cand0], "on", cfg, {"xyz": xyz, "cam": CAM, "_proj": proj},
                                        MAN, W, H, [], [], lambda c, e: None, lambda w: None)
    assert res is None and "two_flap" not in block
