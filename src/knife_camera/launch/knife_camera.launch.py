"""Knife-view camera (Logitech C930e) with a fixed manual focus.

Publishes /knife_camera/image_raw (+ /compressed, /camera_info) via usb_cam.
Focus is applied with v4l2-ctl from config/knife_camera_controls.yaml, because
usb_cam 0.8.1 only knows the pre-kernel-6 control names (see knife_camera.yaml).

    ros2 launch knife_camera knife_camera.launch.py
    ros2 launch knife_camera knife_camera.launch.py focus:=45   # try a value
"""

import os
import subprocess

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, LogInfo,
                            OpaqueFunction, RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Controls are re-applied this long after usb_cam starts, in case opening the
# stream reset any of them.
REAPPLY_DELAY_S = 3.0


def _v4l2_commands(device, controls):
    # One v4l2-ctl call per control so the order is kept: focus_absolute is
    # rejected while focus_automatic_continuous is still on.
    return [['v4l2-ctl', '-d', device, '-c', f'{name}={value}']
            for name, value in controls.items()]


def _launch_setup(context):
    params_file = LaunchConfiguration('params_file').perform(context)
    controls_file = LaunchConfiguration('controls_file').perform(context)
    focus = LaunchConfiguration('focus').perform(context)

    with open(params_file) as f:
        params = yaml.safe_load(f)['/**']['ros__parameters']
    device = params['video_device']
    with open(controls_file) as f:
        controls = dict(yaml.safe_load(f)['controls'])
    if focus:
        controls['focus_absolute'] = int(focus)

    if not os.path.exists(device):
        raise RuntimeError(f'knife camera not found at {device}')

    commands = _v4l2_commands(device, controls)
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed: {result.stderr.strip()}")

    camera = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='knife_camera',
        namespace='knife_camera',
        # usb_cam mangles the by-id symlink into /dev/../../videoN and then
        # rejects it, so pass the resolved device node.
        parameters=[params_file, {'video_device': os.path.realpath(device)}],
        output='screen',
    )
    reapply = RegisterEventHandler(OnProcessStart(
        target_action=camera,
        on_start=[TimerAction(
            period=REAPPLY_DELAY_S,
            actions=[ExecuteProcess(cmd=cmd, output='log') for cmd in commands],
        )],
    ))

    summary = ', '.join(f'{k}={v}' for k, v in controls.items())
    return [LogInfo(msg=f'knife_camera controls: {summary}'), camera, reapply]


def generate_launch_description():
    share = get_package_share_directory('knife_camera')
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(share, 'config', 'knife_camera.yaml'),
            description='usb_cam parameters'),
        DeclareLaunchArgument(
            'controls_file',
            default_value=os.path.join(share, 'config', 'knife_camera_controls.yaml'),
            description='V4L2 controls applied with v4l2-ctl'),
        DeclareLaunchArgument(
            'focus', default_value='',
            description='Override focus_absolute (0..255, step 5); empty = use controls_file'),
        OpaqueFunction(function=_launch_setup),
    ])
