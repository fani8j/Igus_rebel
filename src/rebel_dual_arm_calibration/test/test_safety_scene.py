"""Safety scene geometry (safety_scene.py): horizontal slab, board stand, pole / head pads, fail-closed rules."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from rebel_dual_arm_calibration import safety_scene as ss
from rebel_dual_arm_calibration.transforms import make_T, rotvec_to_R, xyz_rpy_to_T

BOARD = (0.199, 0.160)


def board_at(centre, tilt_deg=0.0, yaw_deg=0.0):
    """Board frame with z INTO the table; tilted about base x (the end toward -x... any horizontal axis)."""
    R = rotvec_to_R([math.radians(tilt_deg), 0, 0]) @ rotvec_to_R([math.pi, 0, 0]) @ rotvec_to_R([0, 0, math.radians(yaw_deg)])
    return make_T(R, centre)


def obj(oid, size, pos, frame="base_link", yaw=0.0, kind=1):
    q = [0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]
    return NS(id=oid, header=NS(frame_id=frame), pose=NS(position=NS(x=0, y=0, z=0), orientation=NS(x=0, y=0, z=0, w=1)),
              primitives=[NS(type=kind, dimensions=list(size))], meshes=[], planes=[],
              primitive_poses=[NS(position=NS(x=pos[0], y=pos[1], z=pos[2]),
                                  orientation=NS(x=q[0], y=q[1], z=q[2], w=q[3]))])


POLE = ss.box_from_object(obj("camera_pole", (0.03, 0.03, 1.0), (-0.1, -0.3, 0.5), yaw=0.4))
HEAD = ss.box_from_object(obj("camera_head", (0.1, 0.1, 0.1), (-0.05, -0.21, 0.87), yaw=0.4))


def test_slab_top_is_highest_corner_plus_50mm_for_a_tilted_board_and_is_horizontal():
    B = board_at([0.59, -0.154, -0.186], tilt_deg=15.0, yaw_deg=20.0)
    corners = ss.board_corners(B, BOARD)
    slab = ss.build_slab(B, BOARD)
    assert slab.top_z() == pytest.approx(corners[:, 2].max() + 0.05, abs=1e-9)
    assert np.allclose(slab.T[:3, :3], np.eye(3))                            # horizontal, axis aligned in base_link
    assert corners[:, 2].max() - corners[:, 2].min() > 0.03                  # the tilt is real
    assert slab.size[0] == pytest.approx(1.2) and slab.size[1] == pytest.approx(1.2) and slab.size[2] == pytest.approx(0.1)
    assert slab.centre[0] == pytest.approx(0.59) and slab.centre[1] == pytest.approx(-0.154)


def test_lead_flat_board_reading_gives_expected_slab():
    B = board_at([0.514, -0.148, -0.251], tilt_deg=1.5)
    slab = ss.build_slab(B, BOARD)
    assert slab.top_z() == pytest.approx(ss.board_corners(B, BOARD)[:, 2].max() + 0.05)
    assert -0.26 < slab.top_z() < -0.15


def test_slab_never_intersects_the_robot_base():
    keep = ss.ROBOT_BASE_RADIUS_M + ss.MARGIN_M
    # (a) far below the base bottom: fine, the footprint may cover the base area
    B = board_at([0.5, -0.15, -0.25])
    slab = ss.build_slab(B, BOARD)
    assert slab.top_z() <= -0.05
    assert not ss.boxes_intersect_disc(slab, ss.ROBOT_BASE_RADIUS_M + 0.0, 0.0, 0.4)
    # (b) a high board: slab top within 50 mm of the base bottom -> the footprint is clipped away from the base footprint
    for centre in ([0.60, -0.10, -0.03], [0.55, 0.30, -0.02], [-0.60, 0.10, -0.03], [0.1, -0.7, -0.03]):
        B = board_at(centre)
        slab = ss.build_slab(B, BOARD)
        assert slab.top_z() > -0.05
        assert not ss.boxes_intersect_disc(slab, keep - 1e-6, -0.2, 0.4), centre
        corners = ss.board_corners(B, BOARD)
        lo = slab.centre[:2] - np.array(slab.size[:2]) / 2
        hi = slab.centre[:2] + np.array(slab.size[:2]) / 2
        assert np.all(corners[:, :2] >= lo) and np.all(corners[:, :2] <= hi)       # the board is still covered
        assert slab.size[0] < 1.2 or slab.size[1] < 1.2                              # it was clipped
    # (c) a board right next to the base cannot be clipped away from it: refuse
    with pytest.raises(ss.SceneRefused):
        ss.build_slab(board_at([0.12, 0.0, -0.02]), BOARD)
    # (d) absurd height / tilt: refuse
    with pytest.raises(ss.SceneRefused):
        ss.build_slab(board_at([0.5, 0.0, 0.5]), BOARD)
    with pytest.raises(ss.SceneRefused):
        ss.build_slab(board_at([0.5, 0.0, -0.2], tilt_deg=70.0), BOARD)


def test_board_stand_encloses_the_corners_with_padding():
    B = board_at([0.59, -0.154, -0.186], tilt_deg=15.0, yaw_deg=20.0)
    slab = ss.build_slab(B, BOARD)
    stand = ss.build_stand(B, BOARD, slab)
    corners = ss.board_corners(B, BOARD)
    zmax = corners[:, 2].max()
    # z: from the slab top up to (highest corner + 0.03) + 0.05 padding
    assert stand.centre[2] - stand.size[2] / 2 == pytest.approx(slab.top_z(), abs=1e-9)
    assert stand.top_z() == pytest.approx(zmax + 0.03 + 0.05, abs=1e-9)
    # footprint (rotated by the board yaw) contains every corner with >= 0.10 m beyond it in plane
    R = stand.T[:3, :3]
    for c in corners:
        local = R.T @ (c - stand.centre)
        assert abs(local[0]) <= stand.size[0] / 2 - 0.10 + 1e-9 and abs(local[1]) <= stand.size[1] / 2 - 0.10 + 1e-9


def test_pads_are_pole_and_head_plus_50mm_on_every_side():
    B = board_at([0.5, -0.15, -0.25])
    boxes = {b.id: b for b in ss.build_scene(B, BOARD, POLE, HEAD)}
    assert set(boxes) == set(ss.OUR_IDS)
    for pad, src in ((boxes[ss.POLE_PAD_ID], POLE), (boxes[ss.HEAD_PAD_ID], HEAD)):
        assert np.allclose(np.array(pad.size), np.array(src.size) + 0.1)
        assert np.allclose(pad.T, src.T)
    assert all(b.frame == "base_link" for b in boxes.values())


def test_refuse_on_missing_pole_or_head_fail_closed():
    B = board_at([0.5, -0.15, -0.25])
    with pytest.raises(ss.SceneRefused, match="camera_pole"):
        ss.build_scene(B, BOARD, None, HEAD)
    with pytest.raises(ss.SceneRefused, match="camera_head"):
        ss.build_scene(B, BOARD, POLE, None)


def test_object_parsing_rules():
    assert ss.box_from_object(obj("camera_pole", (0.03, 0.03, 1.0), (0, 0, 0.5), frame="world")).frame == "world"
    with pytest.raises(ss.SceneRefused):
        ss.box_from_object(obj("camera_pole", (0.03, 0.03, 1.0), (0, 0, 0.5), frame="camera_link"))
    with pytest.raises(ss.SceneRefused):
        ss.box_from_object(obj("camera_pole", (0.03, 0.03, 1.0), (0, 0, 0.5), kind=3))          # a cylinder
    with pytest.raises(ss.SceneRefused):
        ss.box_from_object(obj("camera_pole", (0.03, -1, 1.0), (0, 0, 0.5)))
    b = ss.box_from_object(obj("x", (1, 2, 3), (4, 5, 6), yaw=math.pi / 2))
    assert np.allclose(b.T[:3, 3], [4, 5, 6]) and np.allclose(b.T[:3, :3], rotvec_to_R([0, 0, math.pi / 2]), atol=1e-9)


def test_non_finite_board_is_refused():
    B = board_at([0.5, 0, -0.25])
    B[0, 3] = float("nan")
    with pytest.raises(ss.SceneRefused):
        ss.build_slab(B, BOARD)
