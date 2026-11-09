"""Synthetic recovery of a hidden carton offset (spec: 1 mm, along 2 mm, yaw 0.5 deg)."""
import numpy as np
import pytest

import synth
from side_cut_vision.measure import agree, measure

CASES = [
    # (along, across, normal) mm, yaw deg
    ((8.0, -6.0, 3.0), 2.0),        # the spec's example
    ((-12.0, 5.0, -4.0), -1.5),
    ((0.0, 0.0, 0.0), 0.0),
    ((15.0, 9.0, 5.0), 3.0),
]


def _run(spec, arm="left", **pose_kw):
    cam = synth.load_camera()
    model = synth.load_arm(arm)
    T, tip, _ = synth.approach_pose(model, **pose_kw)
    color, depth, truth = synth.render(cam, T, model, spec)
    return measure(color, depth, cam, T, model, tip_world=tip), truth, (cam, model, T, tip, color, depth)


@pytest.mark.parametrize("delta,yaw", CASES)
def test_recovers_hidden_offset_left(delta, yaw):
    spec = synth.SceneSpec(d_along_mm=delta[0], d_across_mm=delta[1], d_normal_mm=delta[2], yaw_deg=yaw)
    m, truth, _ = _run(spec)
    assert m.ok and m.seam_ok, (m.reasons, m.to_dict()["quality"])
    da, dc, dn = m.delta_mm
    print(f"hidden {delta} yaw {yaw}  recovered ({da:.2f}, {dc:.2f}, {dn:.2f}) yaw {m.delta_yaw_deg:.3f}")
    assert abs(da - delta[0]) <= 2.0
    assert abs(dc - delta[1]) <= 1.0
    assert abs(dn - delta[2]) <= 1.0
    assert abs(m.delta_yaw_deg - yaw) <= 0.5


def test_tip_vs_seam_numbers():
    spec = synth.SceneSpec()                         # hidden (8, -6, 3), yaw 2
    m, _, (cam, model, T, tip, _, _) = _run(spec, tip_along_m=0.030, standoff_m=0.0165, below_top_m=0.003)
    t = m.tip_vs_seam
    # tip is 30 mm beyond the model crossing; the carton moved +8 along -> 22 mm from the seam
    assert abs(t.along_mm - 22.0) < 2.0
    # tip 16.5 mm outside the model edge; the carton moved 6 mm inward -> 22.5 mm outside (negative inward)
    assert abs(t.inward_from_edge_mm + 22.5) < 1.0
    # tip 3 mm below the model top; the top moved +3 mm up -> 6 mm below the top
    assert abs(t.below_top_mm - 6.0) < 1.0


def test_two_frames_agree_within_1mm_and_repeatability():
    spec_a = synth.SceneSpec(seed=11)
    spec_b = synth.SceneSpec(seed=22)
    ma, _, _ = _run(spec_a)
    mb, _, _ = _run(spec_b)
    assert agree(ma, mb, 1.0)
    # a displaced second frame must NOT agree
    mc, _, _ = _run(synth.SceneSpec(seed=22, d_across_mm=-3.0))
    assert not agree(ma, mc, 1.0)


def test_right_arm_oblique_view_and_side_plane():
    # Right face seen from a more oblique pose (the right 305 shows top and side face).
    spec = synth.SceneSpec(d_along_mm=-5.0, d_across_mm=4.0, d_normal_mm=-2.0, yaw_deg=-1.0)
    m, _, _ = _run(spec, arm="right", tilt_deg=35.0, tip_along_m=0.02)
    assert m.ok, (m.reasons, m.to_dict()["quality"])
    assert abs(m.delta_mm[1] - 4.0) <= 1.0 and abs(m.delta_mm[2] + 2.0) <= 1.0
    assert abs(m.delta_yaw_deg + 1.0) <= 0.5
    # the side face is visible enough at 35 deg tilt for the optional side plane fit (log only)
    assert m.side is not None
    if m.side.ok:
        assert m.side.n @ np.array([0, -1.0, 0]) > 0.95


def test_harsh_noise_and_lighting_gradient_still_recovers():
    spec = synth.SceneSpec(depth_noise_mm_at_14cm=1.2, color_noise=9.0, light_gradient=0.35,
                           speckle_frac=0.30, hole_frac=0.25, seed=99)
    m, _, _ = _run(spec)
    assert m.ok and m.seam_ok, (m.reasons, m.to_dict()["quality"])
    da, dc, dn = m.delta_mm
    assert abs(da - 8.0) <= 2.0 and abs(dc + 6.0) <= 1.0 and abs(dn - 3.0) <= 1.0
    assert abs(m.delta_yaw_deg - 2.0) <= 0.5


def test_noise_seeds_repeatability():
    res = []
    for seed in range(5):
        m, _, _ = _run(synth.SceneSpec(seed=100 + seed))
        assert m.ok and m.seam_ok
        res.append(m.delta_mm + [m.delta_yaw_deg])
    sd = np.std(np.array(res), axis=0)
    assert sd[0] < 0.5 and sd[1] < 0.3 and sd[2] < 0.3 and sd[3] < 0.1, sd
