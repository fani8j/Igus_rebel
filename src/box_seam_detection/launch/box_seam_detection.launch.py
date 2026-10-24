"""Launch the box_seam_detection DetectBox node (never the camera driver: start rebel_cameras separately)."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_cfg = str(Path(get_package_share_directory("box_seam_detection")) / "config" / "box_seam_detection.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_cfg, description="parameter YAML"),
        DeclareLaunchArgument("capture_source", default_value="node", description="node | recorder"),
        DeclareLaunchArgument("snapshot_path", default_value="/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json"),
        DeclareLaunchArgument("output_root", default_value="/home/farhan/Igus_rebel/outputs/rgbd_captures"),
        DeclareLaunchArgument("warm_daemon", default_value="true", description="ensure the SAM3 daemon at start"),
        Node(
            package="box_seam_detection",
            executable="box_seam_detector_node",
            name="box_seam_detector",
            output="screen",
            additional_env={"OPENBLAS_NUM_THREADS": "1", "AUTOLABEL_V2_BLAS_PINNED": "1"},
            parameters=[
                LaunchConfiguration("params_file"),
                {
                    "capture_source": LaunchConfiguration("capture_source"),
                    "snapshot_path": LaunchConfiguration("snapshot_path"),
                    "output_root": LaunchConfiguration("output_root"),
                    "warm_daemon": LaunchConfiguration("warm_daemon"),
                },
            ],
        ),
    ])
