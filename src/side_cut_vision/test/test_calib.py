import numpy as np
import pytest

import synth
from side_cut_vision.calib import CALIB_OK, CALIB_SEED, CalibError, load_hand_eye
from side_cut_vision.detect import tool_mask_full
from side_cut_vision.geom import inv_T, is_valid_T


def test_real_handeye_loads_with_sha_and_status():
    he = load_hand_eye(synth.HANDEYE)
    assert is_valid_T(he.T_link6_optical)
    assert np.allclose(he.T_link6_optical[:3, 3], [0.032346, 0.09846, 0.004704])
    assert len(he.sha256) == 64 and he.calib_status == CALIB_OK
    seed = load_hand_eye(synth.HANDEYE, seed_from_left=True)
    assert seed.calib_status == CALIB_SEED and seed.sha256 == he.sha256


def test_handeye_plus_tcp_puts_tip_where_the_real_s0_frames_show_it():
    # The real S0 frames (2026-10-20) show the blade tip at about (940, 410) px, inside the knife
    # body region. Hand-eye + pivot TCP must reproduce that independently of any robot pose.
    he = load_hand_eye(synth.HANDEYE)
    cam = synth.load_camera()
    tip_cam = inv_T(he.T_link6_optical) @ np.append(synth.tip_in_link6(), 1.0)
    uv = cam.project(tip_cam[None, :3])[0]
    assert abs(uv[0] - 940) < 25 and abs(uv[1] - 415) < 25, uv
    assert 0.12 < tip_cam[2] < 0.15
    tool = tool_mask_full(1280, 800)
    assert tool[int(round(uv[1])), int(round(uv[0]))]


def test_bad_calib_files(tmp_path):
    with pytest.raises(CalibError):
        load_hand_eye(str(tmp_path / "missing.yaml"))
    p = tmp_path / "x.yaml"
    p.write_text("schema: something/else\n")
    with pytest.raises(CalibError):
        load_hand_eye(str(p))
    p.write_text("schema: wrist_handeye/v1\nxyz: [0,0,.nan]\nquat_xyzw: [0,0,0,1]\n")
    with pytest.raises(CalibError):
        load_hand_eye(str(p))
