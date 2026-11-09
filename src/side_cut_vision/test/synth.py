"""Test-side synthetic scene helpers: real data files + the library renderer (side_cut_vision.synthcam)."""
from __future__ import annotations

import os

from side_cut_vision import synthcam
from side_cut_vision.calib import load_hand_eye
from side_cut_vision.camera import CameraModel
from side_cut_vision.model import ArmModel, build_arm_model, load_snapshot
from side_cut_vision.synthcam import (SceneSpec, Carton, TCP_IN_TOOL0, LINK6_TO_TOOL0_Z, render,  # noqa: F401
                                      tip_in_link6)

DATA = os.path.join(os.path.dirname(__file__), "data")
CAMERA_INFO = os.path.join(DATA, "camera_info_left_305.json")
SNAPSHOT = os.path.join(DATA, "snapshot_20261020_122458.json")
HANDEYE = os.path.join(DATA, "handeye_left_20261013.yaml")


def load_camera() -> CameraModel:
    return CameraModel.from_json(CAMERA_INFO)


def load_arm(arm: str = "left") -> ArmModel:
    return build_arm_model(load_snapshot(SNAPSHOT), arm)


def approach_pose(arm: ArmModel, **kw):
    return synthcam.approach_pose(arm, load_hand_eye(HANDEYE), **kw)
