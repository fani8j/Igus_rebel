"""Depth-placed seam-crossed edge fix of live.detector.

Synthetic scene: a tilted 110 x 97 mm box top (plane facing the camera, the
back edge farther away), a dense on-plane cloud over the true top plus an
off-plane table, and a distorted pinhole camera. The seam (axis bf) crosses
the back and front edges; the near (front) RGB edge is displaced, the fix
must re-place it parallel to the anchor (back) edge at the depth extent.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("autolabel_v2_live_helpers_depth_edge",
                                               str(TESTS / "test_autolabel_v2_live.py"))
live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live)

det = live.det  # fixture re-export
RING = live.RING
W, H = 640, 480
K = [700.0, 0.0, 322.0, 0.0, 701.0, 238.0, 0.0, 0.0, 1.0]
D = [0.3, -0.4, 0.001, -0.0008, 0.1, 0.0, 0.0, 0.0]
LONG_MM, SHORT_MM = 110.0, 97.0
SEAM_A_MM = 4.0  # seam crosses the back/front edges at a = +4 mm

THETA = math.radians(40.0)
CENTRE = np.array([0.08, 0.03, 0.50])  # off-axis: distortion matters
E_A = np.array([1.0, 0.0, 0.0])                                # along back/front
E_B = np.array([0.0, math.cos(THETA), -math.sin(THETA)])       # back -> front (nearer)
NORMAL = -np.cross(E_A, E_B)                                   # faces the camera
OFFSET = -float(NORMAL @ CENTRE)
TRUE_AB_MM = {"back_left": (-LONG_MM / 2, -SHORT_MM / 2),
              "back_right": (LONG_MM / 2, -SHORT_MM / 2),
              "front_right": (LONG_MM / 2, SHORT_MM / 2),
              "front_left": (-LONG_MM / 2, SHORT_MM / 2)}


def _camera():
    return np.asarray(K, dtype=np.float64).reshape(3, 3), np.asarray(D[:5], dtype=np.float64)


def _xyz(a_mm, b_mm):
    return CENTRE + (a_mm / 1000.0) * E_A + (b_mm / 1000.0) * E_B


def _px(point):
    k, dist = _camera()
    uv, _ = cv2.projectPoints(np.asarray(point, dtype=np.float64).reshape(1, 1, 3),
                              np.zeros(3), np.zeros(3), k, dist)
    return [float(v) for v in uv.reshape(2)]


def _cloud(step_mm=1.0, n_keep=None):
    a = np.arange(-LONG_MM / 2, LONG_MM / 2 + 1e-9, step_mm)
    b = np.arange(-SHORT_MM / 2, SHORT_MM / 2 + 1e-9, step_mm)
    aa, bb = np.meshgrid(a, b)
    top = (CENTRE + np.outer(aa.ravel() / 1000.0, E_A) + np.outer(bb.ravel() / 1000.0, E_B))
    # A table 86 mm below the top plane (never within the 4 mm plane band).
    table = top[::7] - 0.0863 * NORMAL
    cloud = np.concatenate([top, table])
    if n_keep is not None:
        rng = np.random.default_rng(0)
        cloud = cloud[rng.choice(len(cloud), n_keep, replace=False)]
    return cloud


def _record(front_shift_mm=(0.0, 0.0), amb_back=0.05, amb_front=0.6, seam_a_mm=SEAM_A_MM,
            status="ok"):
    """Record whose RGB front corners are pulled inward along the short sides
    by (front_left, front_right) mm; back edge + short sides are true."""
    ab = dict(TRUE_AB_MM)
    ab["front_left"] = (ab["front_left"][0], ab["front_left"][1] - front_shift_mm[0])
    ab["front_right"] = (ab["front_right"][0], ab["front_right"][1] - front_shift_mm[1])
    xyz = {k: [round(float(v), 5) for v in _xyz(*ab[k])] for k in RING}
    px = {k: [round(v, 2) for v in _px(_xyz(*ab[k]))] for k in RING}

    def seam_point(edge):
        p, q = (ab["back_left"], ab["back_right"]) if edge == "back" else \
            (ab["front_left"], ab["front_right"])
        t = (seam_a_mm - p[0]) / (q[0] - p[0])
        return _px(_xyz(seam_a_mm, p[1] + t * (q[1] - p[1])))

    left, right = sorted([seam_point("back"), seam_point("front")], key=lambda p: (p[0], p[1]))
    rec = live._record(corners=px, axis="bf", endpoints=(left, right), status=status)
    rec["top_face"]["corners_xyz_camera_m"] = xyz
    rec["top_face"]["plane"] = {"normal": [float(v) for v in NORMAL], "offset_m": OFFSET,
                                "rms_mm": 0.5, "inliers": 10000}
    rec["checks"]["corner_flags"] = {"corners": {
        "back_left": {"ambiguous_frac": amb_back}, "back_right": {"ambiguous_frac": amb_back},
        "front_right": {"ambiguous_frac": amb_front},
        "front_left": {"ambiguous_frac": amb_front}}, "any_flag": False}
    return rec


def _ab_of(point):
    q = np.asarray(point, dtype=np.float64) - CENTRE
    return 1000.0 * float(q @ E_A), 1000.0 * float(q @ E_B)


def _backproject(uv):
    k, dist = _camera()
    xy = cv2.undistortPointsIter(np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2), k, dist,
                                 None, None,
                                 (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12))
    ray = np.array([*xy.reshape(2), 1.0])
    return ray * (-OFFSET / float(ray @ NORMAL))


def _fix(det, rec, cloud=None, **kw):
    seam = det.published_seam_line(rec)
    return det.depth_edge_fix(rec, seam, _cloud() if cloud is None else cloud, _camera(), **kw)


# ------------------------------------------------------------ anchor choice

def test_choose_anchor_lower_ambiguity_wins(det):
    anchor, reason = det.choose_anchor({
        "back": {"ambiguous_frac": 0.05, "distance_m": 0.40},
        "front": {"ambiguous_frac": 0.60, "distance_m": 0.45}})
    assert anchor == "back" and reason.startswith("lower_ambiguity")


def test_choose_anchor_tie_breaks_to_farther_edge(det):
    anchor, reason = det.choose_anchor({
        "left": {"ambiguous_frac": 0.20, "distance_m": 0.47},
        "right": {"ambiguous_frac": 0.25, "distance_m": 0.48}})
    assert anchor == "right" and reason.startswith("ambiguity_tie->farther")


def test_choose_anchor_both_ambiguous_tie_skips(det):
    with pytest.raises(det._DepthEdgeSkip) as skip:
        det.choose_anchor({"back": {"ambiguous_frac": 0.55, "distance_m": 0.47},
                           "front": {"ambiguous_frac": 0.60, "distance_m": 0.48}})
    assert str(skip.value).startswith("depth_edge_anchor_ambiguous")
    # Clearly different (>= 0.10) is still decidable even when both are high.
    anchor, _ = det.choose_anchor({"back": {"ambiguous_frac": 0.35, "distance_m": 0.47},
                                   "front": {"ambiguous_frac": 0.60, "distance_m": 0.48}})
    assert anchor == "back"
    # R1(b): the fix no longer skips on ambiguous anchors; it evaluates both
    # (see tests/test_r1_depth_edge.py).
    _, _, info = _fix(det, _record((3.0, 1.0), amb_back=0.5, amb_front=0.55))
    assert "anchor_ambiguous->both_anchors_evaluated" in info["reasons"]


def test_choose_anchor_unknown_ambiguity_uses_distance(det):
    anchor, reason = det.choose_anchor({
        "back": {"ambiguous_frac": None, "distance_m": 0.52},
        "front": {"ambiguous_frac": 0.1, "distance_m": 0.48}})
    assert anchor == "back" and reason.startswith("ambiguity_unknown->farther")


def test_anchor_follows_record_ambiguity_both_ways(det):
    _, _, info = _fix(det, _record((4.0, 0.0)))
    assert info["anchor"] == "back" and info["moved_edge"] == "front"
    assert info["edges"]["front"]["ambiguity_source"] == "record_corner_min"
    # Front clean, back ambiguous -> the front edge anchors, the back moves.
    _, _, info = _fix(det, _record((4.0, 0.0), amb_back=0.7, amb_front=0.05))
    assert info["anchor"] == "front" and info["moved_edge"] == "back"


# ------------------------------------------------------------ the fix

def test_displaced_near_edge_is_replaced_from_depth(det):
    shift = 3.0
    rec = _record((shift, shift))
    out, seam_out, info = _fix(det, rec)
    assert info["applied"] is True and info["anchor"] == "back"
    assert info["n_band_points"] >= 2000
    assert info["rgb_width_mm"] == pytest.approx(SHORT_MM - shift, abs=0.05)
    assert info["placement"].startswith("anchor = RGB fold edge; moved edge = depth boundary")
    # Grid spans the true top: p0.5/p99.5 sit at (or just inside) each
    # boundary; the other edge goes AT its depth boundary (delta only recorded).
    assert SHORT_MM - 1.0 <= info["depth_width_mm"] <= SHORT_MM + 1e-6
    assert info["new_width_mm"] == pytest.approx(SHORT_MM, abs=0.1)
    assert info["depth_width_mm"] + info["delta_mm"] == pytest.approx(
        info["new_width_mm"], abs=0.01)
    for name in ("front_left", "front_right"):
        assert info["moved_mm"][name] == pytest.approx(shift, abs=0.1)
        a_mm, b_mm = _ab_of(out["top_face"]["corners_xyz_camera_m"][name])
        assert b_mm == pytest.approx(SHORT_MM / 2, abs=0.1)
        assert a_mm == pytest.approx(TRUE_AB_MM[name][0], abs=0.1)  # on the short side
    # Anchor corners untouched; the input record is not mutated.
    for name in ("back_left", "back_right"):
        assert out["top_face"]["corners_px"][name] == rec["top_face"]["corners_px"][name]
    assert rec["top_face"]["corners_px"] == _record((shift, shift))["top_face"]["corners_px"]


def test_px_and_xyz_stay_consistent_through_distortion(det):
    out, _, info = _fix(det, _record((4.0, 1.0)))
    assert info["applied"] is True
    top = out["top_face"]
    for name in RING:
        back = _backproject(top["corners_px"][name])
        assert 1000.0 * float(np.linalg.norm(back - np.asarray(
            top["corners_xyz_camera_m"][name]))) < 0.05
    # The camera really distorts here: an undistorted projection is far off.
    k, _ = _camera()
    worst = 0.0
    for name in RING:
        p = np.asarray(top["corners_xyz_camera_m"][name])
        pinhole = [k[0, 0] * p[0] / p[2] + k[0, 2], k[1, 1] * p[1] / p[2] + k[1, 2]]
        worst = max(worst, math.dist(pinhole, top["corners_px"][name]))
    assert worst > 0.7


def test_skewed_quad_fails_gate_before_and_passes_after(det):
    rec = _record((4.5, 0.0))  # one near corner jumps to the other candidate line
    with pytest.raises(det.Reject) as error:
        det.rectangle_check(rec, 3.0)
    assert error.value.retryable is True
    out, _, info = _fix(det, rec)
    assert info["applied"] is True
    # The RGB edge agrees near the centre-ish but is off at the left end.
    assert info["disagreement_mm"]["front_left"] > 4.0
    result = det.rectangle_check(out, 3.0)
    assert result["pass"] is True
    assert result["diffs_mm"]["A"] < 0.2 and result["diffs_mm"]["B"] < 0.2
    assert result["lengths_mm"]["left"] == pytest.approx(SHORT_MM, abs=0.15)


def test_default_always_applies_small_box(det):
    # default min move 0: one edge convention for every small-box capture
    rec = _record((0.3, -0.2))
    out, seam_out, info = _fix(det, rec)
    assert info["applied"] is True
    assert max(info["moved_mm"].values()) < 1.0


def test_agreement_keeps_rgb(det):
    rec = _record((0.3, -0.2))
    out, seam_out, info = _fix(det, rec, min_move_mm=1.0)
    assert info["applied"] is False
    assert info["reason"].startswith("rgb_agrees_with_depth")
    assert max(info["disagreement_mm"].values()) <= 1.0
    assert out["top_face"] == rec["top_face"]
    assert seam_out["seam_left"] == det.published_seam_line(rec)["seam_left"]


def test_too_few_points_skips_with_reason(det):
    rec = _record((4.0, 4.0))
    seam = det.published_seam_line(rec)
    with pytest.raises(det._DepthEdgeSkip) as skip:
        det.depth_edge_fix(rec, seam, _cloud(n_keep=1500), _camera())
    assert str(skip.value).startswith("depth_edge_insufficient_points")
    assert skip.value.info["anchor"] == "back"


def test_apply_depth_edge_skips_without_plane_or_camera(det, tmp_path):
    rec = live._record(axis="bf")  # plane None
    seam = det.published_seam_line(rec)
    out, seam_out, info = det.apply_depth_edge(rec, seam, {}, tmp_path / "missing.pcd")
    assert out is rec and seam_out is seam
    assert info["applied"] is False and info["reason"] == "depth_edge_no_plane"
    rec = _record((4.0, 4.0))
    out, _, info = det.apply_depth_edge(rec, det.published_seam_line(rec), {},
                                        tmp_path / "missing.pcd")
    assert info["reason"] == "depth_edge_no_camera" and out is rec


def test_seam_line_recomputed_as_midline_of_fixed_quad(det):
    rec = _record((4.0, 2.0))  # old (confirmed) line at a = +4 mm
    seam = det.published_seam_line(rec)
    out, seam_out, info = det.depth_edge_fix(rec, seam, _cloud(), _camera())
    assert info["applied"] is True and info["seam_line_recomputed"] is True
    assert info["seam_endpoints_before"] == {k: seam[k] for k in ("seam_left", "seam_right")}
    for key in ("seam_left", "seam_right"):
        assert seam_out[key] == info["seam_endpoints_after"][key] != seam[key]
        assert info["seam_edge_distance_px"][key] < 0.5  # px curve of a 3-D edge
        # The fixed quad is the true rectangle: its midline is a = 0.
        a_new, b_new = _ab_of(_backproject(seam_out[key]))
        assert a_new == pytest.approx(0.0, abs=0.1)
        assert abs(b_new) == pytest.approx(SHORT_MM / 2, abs=0.1)
        assert info["seam_lateral_shift_mm"][key] == pytest.approx(SEAM_A_MM, abs=0.15)
    moved = info["seam_on_moved_edge"]
    assert _ab_of(_backproject(seam_out[moved]))[1] > 0  # the front edge
    # 4 mm > 1.5 mm off an ok line: published as not observed, status kept.
    assert info["seam_status_downgraded"] is True
    assert seam_out["observed"] is False and seam_out["status"] == "ok"
    # The record copy carries the new endpoints in the published field.
    assert out["seam"]["status"] == "ok"
    for key in ("seam_left", "seam_right"):
        assert det.published_seam_line(out)[key] == seam_out[key]
    relabel = det.contract_relabel(out["top_face"]["corners_px"], seam_out["seam_left"],
                                   seam_out["seam_right"], tol_px=2.0)
    assert relabel["geometric_axis"] == "bf"


def test_small_seam_shift_keeps_observed(det):
    rec = _record((4.0, 2.0), seam_a_mm=0.8)
    _, seam_out, info = _fix(det, rec)
    assert info["applied"] is True
    assert info["seam_lateral_shift_mm"]["max"] == pytest.approx(0.8, abs=0.15)
    assert info["seam_status_downgraded"] is False and seam_out["observed"] is True


def test_estimated_seam_is_never_downgraded(det):
    rec = _record((4.0, 2.0), status="estimated")
    _, seam_out, info = _fix(det, rec)
    assert info["applied"] is True and info["seam_lateral_shift_mm"]["max"] > 1.5
    assert info["seam_status_downgraded"] is False and seam_out["observed"] is False
    for key in ("seam_left", "seam_right"):
        assert abs(_ab_of(_backproject(seam_out[key]))[0]) < 0.1


def test_agreeing_quad_keeps_published_seam(det):
    rec = _record((0.3, -0.2))
    _, seam_out, info = _fix(det, rec, min_move_mm=1.0)
    assert info["applied"] is False and "seam_line_recomputed" not in info
    assert seam_out["observed"] is True


# ------------------------------------------------------------ CLI wiring

def _write_pcd(path, xyz):
    lines = ["# .PCD v0.7", "VERSION 0.7", "FIELDS x y z", "SIZE 4 4 4", "TYPE F F F",
             "COUNT 1 1 1", f"WIDTH {len(xyz)}", "HEIGHT 1", "VIEWPOINT 0 0 0 1 0 0 0",
             f"POINTS {len(xyz)}", "DATA ascii"]
    body = "\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in xyz)
    path.write_text("\n".join(lines) + "\n" + body + "\n", encoding="utf-8")


def _cli_capture(tmp_path):
    cap = live._capture(tmp_path)
    _write_pcd(cap["pcd"], _cloud(step_mm=1.5))
    manifest = cap["manifest_dict"]
    manifest["camera_info"].update({"k": K, "d": D, "distortion_model": "plumb_bob"})
    cap["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    return cap


def test_cli_applies_fix_before_rectangle_gate(det, tmp_path, monkeypatch):
    cap = _cli_capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, _record((4.5, 0.0)))
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path)) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    edge = doc["autolabel_v2"]["depth_edge"]
    assert edge["applied"] is True and edge["anchor"] == "back"
    assert set(edge["moved_mm"]) == {"front_left", "front_right"}
    assert doc["autolabel_v2"]["rect_check"]["pass"] is True
    # Contract corners/mids come from the moved record.
    rule = doc["autolabel_v2"]["rule_corner_names"]
    record = json.loads(Path(doc["autolabel_v2"]["record_path"]).read_text(encoding="utf-8"))
    assert record["depth_edge"]["applied"] is True
    assert rule == record["top_face"]["corners_px"]
    for contract, rule_name in doc["autolabel_v2"]["contract_from_rule"].items():
        assert doc[contract] == rule[rule_name]
        assert doc["autolabel_v2"]["contract_corners_xyz_camera_m"][contract] == \
            record["top_face"]["corners_xyz_camera_m"][rule_name]
    assert edge["seam_status_downgraded"] is True
    assert doc["seam_observed"] is False and doc["seam_status"] == "ok"
    assert doc["autolabel_v2"]["seam_status_downgraded"] is True
    mids = sorted([doc["left_mid"], doc["right_mid"]])
    assert mids == sorted([record["seam"]["endpoints_px"]["seam_left"],
                           record["seam"]["endpoints_px"]["seam_right"]])


def test_cli_no_depth_edge_keeps_gate_rejection(det, tmp_path, monkeypatch):
    cap = _cli_capture(tmp_path)
    # R9 changed this test: a 4.5 mm skew is now rescued by the joint rectangle
    # repair, so the rejection needs a skew the repair refuses (residual > 3 deg).
    live._install_fake_autolabel(monkeypatch, det, _record((12.0, 0.0)))
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path, "--no-depth-edge")) == 1
    assert not out.exists()
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    assert reject["stage"] == "rectangle" and reject["retryable"] is True
    assert reject["depth_edge"]["reason"] == "disabled(--no-depth-edge)"


def test_cli_seam_rejection_still_after_rectangle_gate(det, tmp_path, monkeypatch):
    # Ambiguous seam + skewed quad. R1(a): the quad-geometry depth fix now
    # repairs the rectangle, so the (unchanged) seam rejection is what is left.
    rec = _record((4.5, 0.0))
    # R10 changed this test: an AMBIGUOUS seam on a small-box rectangle now gets its
    # axis from the geometry, so the deferred seam rejection uses status invisible.
    rec["seam"] = {"status": "invisible", "axis": None}
    cap = _cli_capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, rec)
    assert det.main(live._argv(cap, tmp_path)) == 1
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    assert reject["stage"] == "seam"
    assert reject["depth_edge"]["mode"] == "quad_geometry"
    assert reject["depth_edge"]["applied"] is True


# ------------------------------------------------------------ guards

@pytest.mark.parametrize("shift, what", [((5.0, 5.0), "centre"), ((10.5, -4.0), "corner")])
def test_move_too_large_skips_and_gate_judges_raw_quad(det, shift, what):
    rec = _record(shift)
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, rec)
    assert str(skip.value).startswith(f"depth_edge_move_too_large({what}")
    if shift[0] != shift[1]:
        with pytest.raises(det.Reject) as error:  # raw skewed quad -> retryable
            det.rectangle_check(rec, 3.0)
        assert error.value.retryable is True


def test_boundary_gap_to_neighbouring_strip_skips(det):
    # A same-height strip 5 mm beyond the front (near) edge, 10 mm wide: the
    # p99.5 boundary would jump onto it across an empty 5 mm gap.
    a = np.arange(-LONG_MM / 2, LONG_MM / 2 + 1e-9, 1.0)
    b = np.arange(SHORT_MM / 2 + 5.0, SHORT_MM / 2 + 15.0 + 1e-9, 1.0)
    aa, bb = np.meshgrid(a, b)
    strip = CENTRE + np.outer(aa.ravel() / 1000.0, E_A) + np.outer(bb.ravel() / 1000.0, E_B)
    cloud = np.concatenate([_cloud(), strip])
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, _record((0.0, 0.0)), cloud=cloud)
    assert str(skip.value).startswith("depth_edge_boundary_gap(other side")
    gap = skip.value.info["boundary_gap"]
    assert gap["side"] == "other" and gap["start_mm"] == pytest.approx(-(SHORT_MM / 2 + 0.5)
                                                                       - 1.0, abs=1.6)
    # Without the strip the same record passes the contiguity check.
    _, _, info = _fix(det, _record((0.0, 0.0)))
    assert "boundary_gap" not in info


def test_big_box_class_not_validated(det):
    rec = _record((3.0, 1.0))
    xyz = rec["top_face"]["corners_xyz_camera_m"]
    for name in RING:  # stretch along the edges: long side 110 -> 165 mm
        p = np.asarray(xyz[name])
        xyz[name] = list(p + 0.5 * float((p - CENTRE) @ E_A) * E_A)
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, rec)
    assert str(skip.value).startswith("depth_edge_class_not_validated")
    assert skip.value.info["top_long_side_mm"] == pytest.approx(165.0, abs=0.2)


def test_lr_axis_small_box(det):
    # Same scene, rule names shifted by one along the ring so the seam crosses
    # the rule LEFT (old back) and RIGHT (old front) edges.
    new_from_old = {"back_left": "back_right", "back_right": "front_right",
                    "front_right": "front_left", "front_left": "back_left"}
    old = _record((4.5, 0.0))
    rec = json.loads(json.dumps(old))
    for field in ("corners_px", "corners_xyz_camera_m"):
        rec["top_face"][field] = {new: old["top_face"][field][o]
                                  for new, o in new_from_old.items()}
    rec["checks"]["corner_flags"]["corners"] = {
        new: old["checks"]["corner_flags"]["corners"][o] for new, o in new_from_old.items()}
    rec["seam"]["axis"] = rec["seam"]["estimate"]["axis"] = "lr"
    with pytest.raises(det.Reject):
        det.rectangle_check(rec, 3.0)
    seam = det.published_seam_line(rec)
    out, seam_out, info = det.depth_edge_fix(rec, seam, _cloud(), _camera())
    assert info["applied"] is True and info["seam_axis"] == "lr"
    assert info["anchor"] == "left" and info["moved_edge"] == "right"
    assert set(info["moved_mm"]) == {"back_right", "front_right"}
    assert info["new_width_mm"] == pytest.approx(SHORT_MM, abs=0.1)
    assert det.rectangle_check(out, 3.0)["pass"] is True
    relabel = det.contract_relabel(out["top_face"]["corners_px"], seam_out["seam_left"],
                                   seam_out["seam_right"], tol_px=2.0)
    assert relabel["geometric_axis"] == "lr" and relabel["shift"] == 0
