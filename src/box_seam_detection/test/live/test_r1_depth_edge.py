"""R1 (depth-edge fix reliability) and R6 (seam-direction soft prior) of
live.detector, on the synthetic scene of test_autolabel_v2_depth_edge.py
(tilted 110 x 97 mm top, seam axis bf, back edge = anchor)."""

import importlib.util
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("r1_base_depth_edge_helpers",
                                               str(TESTS / "test_autolabel_v2_depth_edge.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

det = base.live.det
RING = base.RING


def _fix(det, rec, seam="published", cloud=None, **kw):
    if seam == "published":
        seam = det.published_seam_line(rec)
    return det.depth_edge_fix(rec, seam, base._cloud() if cloud is None else cloud,
                              base._camera(), **kw)


def _rect_max(det, rec):
    return max(det._rect_diffs_mm({k: np.asarray(v) for k, v in
                                   rec["top_face"]["corners_xyz_camera_m"].items()}).values())


def _with_box(rec, shrink_bottom_px=None):
    px = np.asarray([rec["top_face"]["corners_px"][k] for k in RING])
    true_px = np.asarray([base._px(base._xyz(*base.TRUE_AB_MM[k])) for k in RING])
    pts = np.concatenate([px, true_px])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    if shrink_bottom_px is not None:
        y1 = y0 + shrink_bottom_px
    rec["provenance"] = {"sam3": {"box_xyxy": [float(x0), float(y0), float(x1), float(y1)]}}
    return rec


# ---------------------------------------------------------------- (a)

def test_a_no_published_seam_runs_from_quad_geometry(det):
    rec = base._record((4.5, 0.0))
    assert _rect_max(det, rec) > 3.0
    out, seam_out, info = _fix(det, rec, seam=None)
    assert seam_out is None
    assert info["applied"] is True and info["mode"] == "quad_geometry"
    assert "no_published_seam->quad_geometry" in info["reasons"]
    assert info["geometry_axes"] == ["bf"]            # back/front = the long sides
    assert _rect_max(det, out) <= 3.0
    det.rectangle_check(out, 3.0)                     # gate passes


def test_a_axisless_seam_also_uses_quad_geometry(det):
    rec = base._record((4.5, 0.0))
    seam = det.published_seam_line(rec)
    seam["axis"] = None
    _, seam_out, info = _fix(det, rec, seam=seam)
    assert info["mode"] == "quad_geometry" and info["applied"] is True
    assert seam_out["seam_left"] == seam["seam_left"]  # no seam recompute without an axis


def test_a_no_seam_and_good_quad_is_left_alone(det):
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, base._record((0.0, 0.0)), seam=None)
    assert str(skip.value) == "depth_edge_no_published_seam"


def test_a_square_ish_quad_tries_both_pairs(det):
    xyz = {k: np.asarray(base._xyz(*ab)) for k, ab in base.TRUE_AB_MM.items()}
    assert det._geometry_axes(xyz) == ["bf"]
    sq = {"back_left": (-50, -50), "back_right": (50, -50),
          "front_right": (50, 50), "front_left": (-50, 50)}
    xyz = {k: np.asarray(base._xyz(*ab)) for k, ab in sq.items()}
    assert det._geometry_axes(xyz) == ["bf", "lr"]


# ---------------------------------------------------------------- (b)

def test_b_ambiguous_anchors_evaluate_both_and_record_both(det):
    rec = base._record((3.0, 1.0), amb_back=0.5, amb_front=0.55)
    out, _, info = _fix(det, rec)
    assert info["applied"] is True
    assert "anchor_ambiguous->both_anchors_evaluated" in info["reasons"]
    cands = [c for c in info["candidates"] if "support_frac" in c]
    assert {c["anchor"] for c in cands} == {"back", "front"}
    assert all("resid_mm" in c for c in cands)
    assert _rect_max(det, out) <= 3.0


def test_b_chosen_anchor_has_the_smaller_residual_or_better_support(det):
    rec = base._record((4.0, 1.0), amb_back=0.5, amb_front=0.55)
    _, _, info = _fix(det, rec)
    by = {c["anchor"]: c for c in info["candidates"] if "resid_mm" in c}
    chosen, rest = by[info["anchor"]], [c for k, c in by.items() if k != info["anchor"]][0]
    assert chosen["resid_mm"] <= rest["resid_mm"] + 0.3
    assert chosen["resid_mm"] < rest["resid_mm"] or chosen["support_frac"] >= rest["support_frac"]


# ---------------------------------------------------------------- (c)

def test_c_supported_move_up_to_10mm_with_sam_box(det):
    # R8(d) changed this test: the RGB front edge now sits 6 mm OUTSIDE the true
    # top (shift -6) so the supported move is inward; an outward 6 mm move is
    # refused (see test_r7_r9_live.py::test_r8d_outward_guard...).
    rec = _with_box(base._record((-6.0, -6.0)))
    out, _, info = _fix(det, rec)
    assert info["applied"] is True
    ext = info["extended_move"]
    assert ext["centre_mm"] > 3.5 and ext["support_frac"] >= 0.8 and ext["inside_sam_box"]
    assert "move_cap_extended_by_support" in info["reasons"]
    assert _rect_max(det, out) <= 1.0


def test_c_without_sam_box_the_old_cap_stands(det):
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, base._record((6.0, 6.0)))
    assert str(skip.value).startswith("depth_edge_move_too_large(centre")
    assert "no_sam_box" in str(skip.value)


def test_c_moved_edge_outside_dilated_sam_box_skips(det):
    rec = _with_box(base._record((6.0, 6.0)), shrink_bottom_px=20.0)
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, rec)
    assert "outside_sam_box" in str(skip.value)


def test_c_poorly_supported_boundary_skips(det):
    # Half of the near strip is missing: only the left half supports the line.
    cloud = base._cloud()
    q = (cloud - base.CENTRE)
    a, b = q @ base.E_A * 1000.0, q @ base.E_B * 1000.0
    cloud = cloud[~((b > base.SHORT_MM / 2 - 7.0) & (a > 0) & (b < base.SHORT_MM))]
    rec = _with_box(base._record((6.0, 6.0)))
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, rec, cloud=cloud)
    assert str(skip.value).startswith("depth_edge_move_too_large(centre")
    assert "support" in str(skip.value)


def test_c_beyond_10mm_never_moves(det):
    rec = _with_box(base._record((12.0, 12.0)))
    with pytest.raises(det._DepthEdgeSkip):
        _fix(det, rec)


def test_c_corner_cap_stays_10mm(det):
    rec = _with_box(base._record((10.5, 8.0)))
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, rec)
    assert "corner" in str(skip.value) or "implausible" in str(skip.value)


# ---------------------------------------------------------------- (d)

def _set_box(monkeypatch, long_mm, short_mm):
    monkeypatch.setattr(base, "LONG_MM", long_mm)
    monkeypatch.setattr(base, "SHORT_MM", short_mm)
    monkeypatch.setattr(base, "TRUE_AB_MM", {
        "back_left": (-long_mm / 2, -short_mm / 2), "back_right": (long_mm / 2, -short_mm / 2),
        "front_right": (long_mm / 2, short_mm / 2), "front_left": (-long_mm / 2, short_mm / 2)})


def test_d_skewed_big_box_is_fixed_when_dims_are_plausible(det, monkeypatch):
    _set_box(monkeypatch, 193.0, 105.0)
    rec = base._record((5.0, 0.0))
    assert _rect_max(det, rec) > 3.0
    out, _, info = _fix(det, rec)
    assert info["applied"] is True and info["class_match"] == "big"
    assert any(r.startswith("class_gate_overridden_skewed") for r in info["reasons"])
    assert _rect_max(det, out) <= 3.0


def test_d_unskewed_big_box_is_still_not_validated(det, monkeypatch):
    _set_box(monkeypatch, 193.0, 105.0)
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, base._record((0.0, 0.0)))
    assert str(skip.value).startswith("depth_edge_class_not_validated")


def test_d_skewed_but_implausible_dims_skip(det, monkeypatch):
    _set_box(monkeypatch, 165.0, 97.0)
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, base._record((5.0, 0.0)))
    assert str(skip.value).startswith("depth_edge_class_not_validated")
    assert "no known box class" in str(skip.value)


# ---------------------------------------------------------------- (e)

def _stub_scene(det, monkeypatch, lr_move_mm):
    """R1(e) orchestration. The real first (bf) placement is followed by a 6 mm
    skew of front_left along the front edge (pair A off); the second (lr/left-anchored) placement
    moves front_left by `lr_move_mm` along the front edge (-6 undoes the skew). Real placements cannot
    leave B skewed on the synthetic parallelogram scene, hence the stub."""
    real_apply, real_place = det._apply_moved, det._place_edge
    shift = 0.006 * base.E_A

    def apply_moved(xyz, px, res):
        out = real_apply(xyz, px, res)
        if res["axis"] == "bf":
            out[0]["front_left"] = out[0]["front_left"] + shift
        return out

    def place(ctx, axis, anchor, xyz, px, extended_ok):
        if axis != "lr":
            return real_place(ctx, axis, anchor, xyz, px, extended_ok)
        other = "right" if anchor == "left" else "left"
        res = {"info": {"axis": axis, "anchor": anchor, "moved_edge": other}, "axis": axis,
               "anchor": anchor, "other": other, "support": 1.0, "noop": True, "moved": {}}
        if anchor == "left":
            new = xyz["front_left"] + (lr_move_mm / 1000.0) * base.E_A
            res.update(noop=False, moved={"front_left": (new, np.asarray(base._px(new)))})
        return res

    monkeypatch.setattr(det, "_apply_moved", apply_moved)
    monkeypatch.setattr(det, "_place_edge", place)


def test_e_second_pair_is_placed_from_depth_when_it_improves(det, monkeypatch):
    _stub_scene(det, monkeypatch, lr_move_mm=-6.0)
    out, _, info = _fix(det, base._record((3.0, 3.0)))
    assert info["second_pair"]["applied"] is True and info["second_pair"]["axis"] == "lr"
    assert any(r.startswith("second_pair_from_depth(lr/left)") for r in info["reasons"])
    assert len(info["steps"]) == 2
    assert "front_left" in info["moved_mm"]
    assert info["rect_after_mm"]["A"] < 1.0
    assert info["applied"] is True and info["reason"].endswith("+second_pair")


def test_e_second_pair_rejected_when_it_does_not_improve(det, monkeypatch):
    _stub_scene(det, monkeypatch, lr_move_mm=9.0)
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, base._record((3.0, 3.0)))
    # A stays 6 mm off after the (rejected) second pair; the original had A=0
    # (good) so the whole fix is dropped by the never-worse guard.
    assert str(skip.value).startswith("depth_edge_residual_worse")


def test_e_rejected_second_pair_keeps_the_first_placement(det, monkeypatch):
    _stub_scene(det, monkeypatch, lr_move_mm=-9.0)
    rec = base._record((0.0, 0.0))
    back_left = (-55.0 + 12.0, -48.5)      # back edge 12 mm short: A 12 mm off before
    rec["top_face"]["corners_xyz_camera_m"]["back_left"] = [
        round(float(v), 5) for v in base._xyz(*back_left)]
    rec["top_face"]["corners_px"]["back_left"] = [
        round(v, 2) for v in base._px(base._xyz(*back_left))]
    out, _, info = _fix(det, rec)
    assert info["applied"] is True and len(info["steps"]) == 1
    assert info["second_pair"]["applied"] is False
    assert info["second_pair"]["why"] == "residual_not_improved"
    assert info["rect_after_mm"]["A"] < info["rect_before_mm"]["A"]


def test_e_second_pair_not_attempted_when_other_pair_is_fine(det):
    _, _, info = _fix(det, base._record((4.5, 0.0)))
    assert "second_pair" not in info


# ------------------------------------------------- never turn good into bad

def test_never_worse_keeps_original(det, monkeypatch):
    # Force a placement that increases the residual: the original is kept.
    rec = base._record((0.5, 0.5))                # passes the gate
    real = det._rect_diffs_mm
    calls = {"n": 0}

    def fake(xyz):
        calls["n"] += 1
        out = real(xyz)
        return {"A": out["A"] + (5.0 if calls["n"] > 3 else 0.0), "B": out["B"]}

    monkeypatch.setattr(det, "_rect_diffs_mm", fake)
    with pytest.raises(det._DepthEdgeSkip) as skip:
        _fix(det, rec)
    assert str(skip.value).startswith("depth_edge_residual_worse")
    # apply_depth_edge turns any skip into 'record unchanged'.
    # (the public path never raises)


def test_good_quad_still_applies_with_small_change(det):
    rec = base._record((0.0, 0.0))
    out, _, info = _fix(det, rec)
    assert max(info.get("moved_mm", {0: 0}).values()) < 1.0
    assert _rect_max(det, out) <= 3.0


# ---------------------------------------------------------------- R6

def _rz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


@pytest.mark.parametrize("rot, outside", [(-7.0, False), (-50.0, False), (60.0, True),
                                          (-100.0, True)])
def test_r6_flags_outside_soft_range_but_never_rejects(det, rot, outside):
    rec = base._record((0.0, 0.0))
    seam = det.published_seam_line(rec)
    manifest = {"camera_info": {"k": base.K, "d": base.D, "distortion_model": "plumb_bob"}}
    out = det.compute_seam_direction(rec, seam, manifest, camera_to_base=_rz(rot))
    assert out["available"] is True
    # The synthetic seam runs along E_B = (0, cos40, -sin40): 90 deg in x-y.
    assert out["base_link_angle_deg"] == pytest.approx((90.0 + rot) % 180.0, abs=1.0)
    assert out["seam_direction_outside_soft_range"] is outside
    assert 0.0 <= out["plane_frame_angle_deg"] < 180.0
    assert out["prior_deg"] == 83.0 and out["soft_range_deg"] == 45.0


def test_r6_plane_frame_angle_without_calibration(det, tmp_path, capsys):
    rec = base._record((0.0, 0.0))
    seam = det.published_seam_line(rec)
    manifest = {"camera_info": {"k": base.K, "d": base.D, "distortion_model": "plumb_bob"}}
    missing = tmp_path / "absent.calib"
    out = det.compute_seam_direction(rec, seam, manifest, calib_file=str(missing))
    assert out["available"] is True and out["base_link_angle_deg"] is None
    assert out["reason"].startswith("calibration_unavailable")
    assert out["seam_direction_outside_soft_range"] is False
    err = capsys.readouterr().err
    assert "seam-direction calibration unavailable" in err and str(missing) in err and "UNKNOWN" in err


def test_r6_uses_the_configured_calibration_file(det, tmp_path):
    """The soft prior uses the SAME calib file as the snapshot transform (not a hard-coded path)."""
    import math
    rec = base._record((0.0, 0.0))
    seam = det.published_seam_line(rec)
    manifest = {"camera_info": {"k": base.K, "d": base.D, "distortion_model": "plumb_bob"}}

    def calib(name, qz, qw):
        path = tmp_path / name
        path.write_text(
            "parameters:\n  name: t\n  calibration_type: eye_on_base\n  robot_base_frame: base_link\n"
            "  tracking_base_frame: camera_link\ntransform:\n  translation: {x: 0.0, y: 0.0, z: 0.5}\n"
            f"  rotation: {{x: 0.0, y: 0.0, z: {qz!r}, w: {qw!r}}}\n", encoding="utf-8")
        return str(path)

    s = math.sqrt(0.5)
    a = det.compute_seam_direction(rec, seam, manifest, calib_file=calib("a.calib", 0.0, 1.0))
    b = det.compute_seam_direction(rec, seam, manifest, calib_file=calib("b.calib", s, s))   # +90 deg about z
    assert a["base_link_calib_file"].endswith("a.calib") and b["base_link_calib_file"].endswith("b.calib")
    diff = (b["base_link_angle_deg"] - a["base_link_angle_deg"]) % 180.0
    assert abs(diff - 90.0) < 1e-6


def test_r6_no_seam_is_unavailable_not_an_error(det):
    out = det.compute_seam_direction(base._record((0.0, 0.0)), None, {})
    assert out["available"] is False and out["seam_direction_outside_soft_range"] is False


@pytest.mark.skipif(not Path(
    "/home/farhan/.ros2/easy_handeye2/calibrations/rebel_handeye_h_v2.calib").is_file(),
    reason="hand-eye calibration file not present")
def test_r6_real_calibration_matrix_loads(det):
    m = det._camera_to_base_matrix()
    assert m is not None and np.asarray(m).shape == (4, 4)
    assert abs(np.linalg.det(np.asarray(m)[:3, :3]) - 1.0) < 1e-6
