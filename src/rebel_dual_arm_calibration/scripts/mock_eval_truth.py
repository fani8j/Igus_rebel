#!/usr/bin/env python3
"""MOCK ONLY - compare a solved dual_arm_calibration.yaml with the hidden ground truth of the mock cell.

    ros2 run rebel_dual_arm_calibration mock_eval_truth.py <dual_arm_calibration.yaml> [--truth-file mock_truth.yaml]

Prints the error of arm_right_base and camera_link (mm, deg) in world = arm_left_base_link, and of the camera in the
right base frame.  Reads the truth that only the mock tools know; a real calibration has no such reference.
"""
import argparse

import numpy as np
import yaml

from rebel_dual_arm_calibration.mock_common import load_truth
from rebel_dual_arm_calibration.transforms import inv_T, rotation_angle_deg, xyz_rpy_to_T


def compare(name, T_est, T_true):
    d = inv_T(T_true) @ T_est
    dp = T_est[:3, 3] - T_true[:3, 3]
    print(f"{name:28s} error {np.linalg.norm(dp) * 1e3:7.3f} mm  (dx,dy,dz = {dp[0] * 1e3:+.3f} {dp[1] * 1e3:+.3f} "
          f"{dp[2] * 1e3:+.3f} mm)   rotation {rotation_angle_deg(np.eye(3), d[:3, :3]):.4f} deg")
    return float(np.linalg.norm(dp) * 1e3), rotation_angle_deg(np.eye(3), d[:3, :3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("calibration_yaml")
    ap.add_argument("--truth-file", default="")
    a = ap.parse_args()
    truth = load_truth(a.truth_file or None)
    with open(a.calibration_yaml, encoding="utf-8") as f:
        c = yaml.safe_load(f)["dual_arm_calibration"]
    T_rb = xyz_rpy_to_T(c["arm_right_base"]["xyz"], c["arm_right_base"]["rpy"])
    T_cam = xyz_rpy_to_T(c["camera_link"]["xyz"], c["camera_link"]["rpy"])
    print(f"source: {c.get('source')}")
    compare("arm_right_base (world)", T_rb, truth.T_w_rb)
    compare("camera_link (world)", T_cam, truth.T_w_cam)
    compare("camera in right base frame", inv_T(T_rb) @ T_cam, inv_T(truth.T_w_rb) @ truth.T_w_cam)
    print("nominal seed (config/dual_arm_calibration.yaml) for comparison: right base y -0.4474 -> truth is "
          f"{(truth.T_w_rb[:3, 3] * 1e3).round(2).tolist()} mm")


if __name__ == "__main__":
    main()
