"""Cross-check helper: ONLY the box_h_cut_executor of move_group.launch.py, in plan_export_only mode
with the executor's own snapshot_frame_mode fixed_transform + snapshot_relabel rotate_180 (style of
src/dual_cut_bringup/config/arm_right_executor_overrides.yaml), the base pose read from the
calibration file. Start next to a stack launched with `right_arm.launch.py run_executor:=false`
(move_group, camera pole, parked left arm) on a separate ROS domain. Not for operator use.

  ros2 launch crosscheck_executor.launch.py hardware_protocol:=mock_hardware run_box_h_cut_executor:=true \
      run_camera_pole:=false use_gui:=false
"""
import importlib.util
import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import LogInfo
from launch_ros.actions import Node as RosNode

HERE = Path(os.path.realpath(__file__)).parent
sys.path.insert(0, str(HERE))
import right_snapshot as rs  # noqa: E402


def generate_launch_description():
    spec = importlib.util.spec_from_file_location(
        "mg_launch", Path(get_package_share_directory("igus_rebel_moveit_config")) / "launch" / "move_group.launch.py")
    mg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mg)
    base = rs.load_calibration(rs.DEFAULT_CALIBRATION)["arm_right_base"]
    overrides = {
        "plan_export_only": True,
        "snapshot_frame_mode": "fixed_transform",
        "snapshot_frame_id": "base_link",
        "snapshot_relabel": "rotate_180",
        "snapshot_planning_base_xyz": base["xyz"],
        "snapshot_planning_base_rpy": base["rpy"],
        "cut_pattern": "left_only",
        "middle_seam_center_overlap_m": 0.0075,
    }

    if os.environ.get("XC_MODE") == "planning_frame":
        # isolation experiment: export-only executor, NO transform/relabel (feed it the converted file)
        overrides = {"plan_export_only": True, "cut_pattern": "left_only",
                     "middle_seam_center_overlap_m": 0.0075}

    def patched_node(*args, **kwargs):
        if kwargs.get("executable") != "box_h_cut_executor":
            return LogInfo(msg=f"[crosscheck] skipping {kwargs.get('executable')}")
        kwargs["parameters"] = list(kwargs["parameters"]) + [overrides]
        return RosNode(*args, **kwargs)

    mg.Node = patched_node
    return mg.generate_launch_description()
