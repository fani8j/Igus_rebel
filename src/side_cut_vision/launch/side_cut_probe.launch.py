"""Launch the passive side-cut probe. It adds ONE process (nice 10) and changes nothing else.

  ros2 launch side_cut_vision side_cut_probe.launch.py camera_domain_id:=11

camera_domain_id is REQUIRED (no default): the real-cell cameras run in domain 11 while the probe runs in
the robot environment (ROS_DOMAIN_ID of the shell, 12 on the real cell). Use -1 only in a mock where
robot and cameras share one domain.
"""
import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration as LC

LEFT_CALIB = '/home/farhan/igus_rebel_ros2/scan_results/wrist_handeye_20261013_112818/result.yaml'
SNAPSHOT = '/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json'


def generate_launch_description():
    exe = os.path.join(get_package_prefix('side_cut_vision'), 'lib', 'side_cut_vision', 'side_cut_probe')
    args = [
        DeclareLaunchArgument('camera_domain_id', description='ROS domain of the cameras (REQUIRED; 11 on the real cell)'),
        DeclareLaunchArgument('arms', default_value='left,right'),
        DeclareLaunchArgument('left_calib', default_value=LEFT_CALIB),
        DeclareLaunchArgument('right_calib', default_value='',
                              description='empty = seed from the left calibration (UNCALIBRATED, flagged everywhere)'),
        DeclareLaunchArgument('snapshot_file', default_value=SNAPSHOT),
        DeclareLaunchArgument('out_dir', default_value='', description="empty = cut_recorder's default out dir"),
        DeclareLaunchArgument('live_rate_hz', default_value='3.0'),
        DeclareLaunchArgument('overlay_context', default_value='robot'),
    ]
    cmd = ['nice', '-n', '10', exe,
           '--camera-domain-id', LC('camera_domain_id'), '--arms', LC('arms'),
           '--left-calib', LC('left_calib'), '--right-calib', LC('right_calib'),
           '--snapshot-file', LC('snapshot_file'), '--live-rate-hz', LC('live_rate_hz'),
           '--overlay-context', LC('overlay_context')]
    return LaunchDescription(args + [
        ExecuteProcess(cmd=cmd + ['--out-dir', LC('out_dir')], output='screen', name='side_cut_probe'),
    ])
