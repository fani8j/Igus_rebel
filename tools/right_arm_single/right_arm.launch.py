"""RIGHT ReBeL alone: move_group + box_h_cut_executor + camera pole + parked left arm.

Run by path (no colcon package needed):

  source /opt/ros/humble/setup.bash
  source install/setup.bash && source install_right/setup.bash   # install_right LAST (newer executor)
  export ROS_DOMAIN_ID=<id> ROS_LOCALHOST_ONLY=<1 for mock>
  ros2 launch tools/right_arm_single/right_arm.launch.py hardware_protocol:=mock_hardware

The driver is NOT started here: run `ros2 launch igus_rebel rebel.launch.py hardware_protocol:=...
robot_ip:=192.168.3.12` in another terminal (mock: hardware_protocol:=mock_hardware).

Adds to the normal single-arm move_group.launch.py:
  - static TF base_link -> camera_link = B_right^-1 * T_world_camera_link (dual_arm_calibration.yaml);
    on the real cell the camera driver supplies camera_link -> camera_color_optical_frame; in mock
    this launch supplies it (camera_optical_tf:=auto). Do NOT run box_perception with
    publish_calibration on the real cell: it publishes the camera TF in the LEFT base frame.
  - the `left_arm_parked` collision box (left_arm_parked.py).
"""

import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(os.path.realpath(__file__)).parent
sys.path.insert(0, str(HERE))
import right_snapshot as rs  # noqa: E402

PACKAGE = "igus_rebel_moveit_config"


def _setup(context):
    calibration_file = LaunchConfiguration("calibration_file").perform(context)
    hardware = LaunchConfiguration("hardware_protocol").perform(context)
    tool_variant = LaunchConfiguration("tool_variant").perform(context)

    # The executor must be the build of the CURRENT source (middle_seam_stop, snapshot_frame_mode).
    executor = Path(get_package_prefix(PACKAGE)) / "lib" / PACKAGE / "box_h_cut_executor"
    real = executor.resolve()
    if b"middle_seam_stop" not in real.read_bytes():
        raise RuntimeError(
            f"box_h_cut_executor {real} is too old (no middle_seam_stop). Source install_right "
            "AFTER install: source install/setup.bash && source install_right/setup.bash")

    calibration = rs.load_calibration(calibration_file)
    xyz, q = rs.camera_in_right_base(calibration)
    actions = [
        LogInfo(msg=f"[right_arm] executor binary: {real}"),
        LogInfo(msg=f"[right_arm] calibration {calibration_file} (source: {calibration['source']}); "
                    f"camera_link in right base_link xyz={[round(v, 4) for v in xyz]} "
                    f"quat={[round(v, 5) for v in q]}"),
    ]
    publish_camera = LaunchConfiguration("publish_camera_tf").perform(context).lower() == "true"
    optical = LaunchConfiguration("camera_optical_tf").perform(context).lower()
    publish_optical = optical == "true" or (optical == "auto" and hardware == "mock_hardware")
    if publish_camera:
        actions.append(Node(
            package="tf2_ros", executable="static_transform_publisher", name="right_camera_link_tf",
            arguments=["--x", repr(xyz[0]), "--y", repr(xyz[1]), "--z", repr(xyz[2]),
                       "--qx", repr(q[0]), "--qy", repr(q[1]), "--qz", repr(q[2]), "--qw", repr(q[3]),
                       "--frame-id", "base_link", "--child-frame-id", "camera_link"]))
    if publish_optical:
        actions.append(Node(
            package="tf2_ros", executable="static_transform_publisher",
            name="right_camera_optical_tf",
            arguments=["--x", "0", "--y", "0", "--z", "0",
                       "--qx", "-0.5", "--qy", "0.5", "--qz", "-0.5", "--qw", "0.5",
                       "--frame-id", "camera_link", "--child-frame-id", "camera_color_optical_frame"]))
    actions.append(ExecuteProcess(
        cmd=[sys.executable, str(HERE / "left_arm_parked.py"),
             "--calibration-file", calibration_file, "--tool-variant", tool_variant,
             "--padding", LaunchConfiguration("left_arm_padding").perform(context)],
        output="screen", name="left_arm_parked"))
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(PACKAGE), "launch", "move_group.launch.py")),
        launch_arguments={
            "hardware_protocol": hardware,
            "tool_variant": tool_variant,
            "use_gui": LaunchConfiguration("use_gui").perform(context),
            "run_teleop": "false",
            "run_gamepad": "false",
            "run_camera_pole": "true",
            "run_box_h_cut_executor": LaunchConfiguration("run_executor").perform(context),
        }.items()))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("hardware_protocol", default_value="mock_hardware",
                              choices=["mock_hardware", "rebel"],
                              description="Must match the driver launch (rebel.launch.py)"),
        DeclareLaunchArgument("tool_variant", default_value="knife_c", choices=["knife_c", "xeg32_knife_v1"]),
        DeclareLaunchArgument("use_gui", default_value="true"),
        DeclareLaunchArgument("calibration_file", default_value=str(rs.DEFAULT_CALIBRATION)),
        DeclareLaunchArgument("publish_camera_tf", default_value="true",
                              description="publish base_link -> camera_link (right frame)"),
        DeclareLaunchArgument("camera_optical_tf", default_value="auto",
                              description="camera_link -> camera_color_optical_frame: auto = only in "
                                          "mock (the camera driver supplies it on the real cell)"),
        DeclareLaunchArgument("left_arm_padding", default_value="0.05"),
        DeclareLaunchArgument("run_executor", default_value="true",
                              description="false = no box_h_cut_executor (cross-check stacks only)"),
        OpaqueFunction(function=_setup),
    ])
