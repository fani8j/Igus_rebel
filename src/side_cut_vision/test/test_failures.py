"""Every failure case gives ok=False plus a specific reason and never an exception."""
import numpy as np
import pytest

import synth
from side_cut_vision.geom import make_T
from side_cut_vision.measure import check_pair, measure


@pytest.fixture(scope="module")
def base():
    cam = synth.load_camera()
    model = synth.load_arm("left")
    T, tip, _ = synth.approach_pose(model)
    color, depth, _ = synth.render(cam, T, model, synth.SceneSpec())
    return cam, model, T, tip, color, depth


def test_baseline_is_fine(base):
    cam, model, T, tip, color, depth = base
    m = measure(color, depth, cam, T, model, tip_world=tip)
    assert m.ok and not m.reasons


def test_no_carton_in_view(base):
    cam, model, T, tip, color, depth = base
    # camera turned to look away (rotate 180 deg about the camera y axis)
    R = T[:3, :3] @ np.diag([-1.0, 1.0, -1.0])
    T2 = make_T(R, T[:3, 3])
    m = measure(color, depth, cam, T2, model, tip_world=tip)
    assert not m.ok and "edge_not_in_view" in m.reasons
    # far away: the edge is in the image but beyond 600 mm
    T3 = make_T(T[:3, :3], T[:3, 3] - T[:3, 2] * 0.8)
    m = measure(color, depth, cam, T3, model, tip_world=tip)
    assert not m.ok and m.reasons and m.reasons[0] in ("carton_too_far", "edge_not_in_view")
    # too close (< 70 mm)
    T4 = make_T(T[:3, :3], T[:3, 3] + T[:3, 2] * 0.10)
    m = measure(color, depth, cam, T4, model, tip_world=tip)
    assert not m.ok and m.reasons[0] in ("carton_too_close", "edge_not_in_view")


def test_depth_all_zero_and_garbage(base):
    cam, model, T, tip, color, depth = base
    m = measure(color, np.zeros_like(depth), cam, T, model, tip_world=tip)
    assert not m.ok and "no_depth" in m.reasons
    rng = np.random.default_rng(3)
    garbage = rng.integers(0, 3000, size=depth.shape).astype(np.uint16)
    m = measure(color, garbage, cam, T, model, tip_world=tip)
    assert not m.ok and any(r.startswith("top_plane") for r in m.reasons)
    # depth present only in a sliver: too few plane points
    sliver = np.zeros_like(depth)
    sliver[100:110, 100:400] = depth[100:110, 100:400]
    m = measure(color, sliver, cam, T, model, tip_world=tip)
    assert not m.ok and "top_plane_insufficient_points" in m.reasons


@pytest.mark.parametrize("shift_mm", [60.0, -60.0])
def test_offset_beyond_search_window(base, shift_mm):
    cam, model, T, tip, _, _ = base
    spec = synth.SceneSpec(d_across_mm=shift_mm, d_along_mm=0.0, d_normal_mm=0.0, yaw_deg=0.0)
    color, depth, _ = synth.render(cam, T, model, spec)
    m = measure(color, depth, cam, T, model, tip_world=tip)
    assert not m.ok
    assert any(r.startswith("edge_") for r in m.reasons), m.reasons


def test_tape_glare_over_seam_keeps_plane_and_edge(base):
    cam, model, T, tip, _, _ = base
    color, depth, _ = synth.render(cam, T, model, synth.SceneSpec(glare=True))
    m = measure(color, depth, cam, T, model, tip_world=tip)
    assert m.ok                                  # plane + edge valid
    assert not m.seam_ok and "seam_not_found" in m.reasons
    assert m.delta_mm[0] is None                 # no along without the seam
    assert abs(m.delta_mm[1] + 6.1) < 1.5 and abs(m.delta_mm[2] - 3.0) < 1.0   # across/normal still valid


def test_non_finite_and_invalid_transform(base):
    cam, model, T, tip, color, depth = base
    for bad in (np.full((4, 4), np.nan), np.where(np.eye(4) > 0, np.inf, 0.0), np.eye(4) * 2,
                np.zeros((3, 3)), None, "nope"):
        m = measure(color, depth, cam, bad, model, tip_world=tip)
        assert not m.ok and "non_finite_transform" in m.reasons
    Tn = T.copy()
    Tn[0, 3] = np.nan
    assert "non_finite_transform" in measure(color, depth, cam, Tn, model).reasons


def test_non_finite_tip_is_flagged_not_fatal(base):
    cam, model, T, tip, color, depth = base
    m = measure(color, depth, cam, T, model, tip_world=[np.nan, 0, 0])
    assert m.ok and "non_finite_tip" in m.reasons and m.tip_vs_seam is None


def test_mismatched_image_sizes(base):
    cam, model, T, tip, color, depth = base
    for c, d in ((color[::2, ::2], depth), (color, depth[:, :1000]), (color[:, :, 0], depth),
                 (color.astype(np.float32), depth), (color, depth.astype(np.float32).tolist())):
        m = measure(c, d, cam, T, model, tip_world=tip)
        assert not m.ok and "size_mismatch" in m.reasons


def test_stale_pair(base):
    cam, model, T, tip, color, depth = base
    assert check_pair(10.0, 10.039) == [] and check_pair(10.0, 10.05) == ["stale_pair"]
    m = measure(color, depth, cam, T, model, tip_world=tip, color_stamp_s=5.0, depth_stamp_s=5.2)
    assert not m.ok and "stale_pair" in m.reasons
    m = measure(color, depth, cam, T, model, tip_world=tip, color_stamp_s=float("nan"), depth_stamp_s=1.0)
    assert not m.ok and "stale_pair" in m.reasons
    m = measure(color, depth, cam, T, model, tip_world=tip, color_stamp_s=5.0, depth_stamp_s=5.03)
    assert m.ok


def test_measurement_dict_is_strict_json(base):
    import json
    cam, model, T, tip, color, depth = base
    for m in (measure(color, depth, cam, T, model, tip_world=tip),
              measure(color, np.zeros_like(depth), cam, T, model)):
        json.dumps(m.to_dict(), allow_nan=False)
