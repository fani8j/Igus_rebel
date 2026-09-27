"""All cell cameras in one launch.

    /camera/...        Gemini 336, static scene camera   (config/scene_camera.yaml)
    /wrist_camera/...  Gemini 305, wrist top view        (config/wrist_camera.yaml)
    /knife_camera/...  Logitech C930e, knife view        (knife_camera package)

    ros2 launch rebel_cameras cameras.launch.py
    ros2 launch rebel_cameras cameras.launch.py wrist:=false knife:=false

All three stamp frames with the host clock (Orbbec time_domain=global,
usb_cam at capture), so consumers pair them with message_filters
ApproximateTimeSynchronizer; check with `ros2 run rebel_cameras camera_sync_check.py`.
There is no hardware trigger between them.
"""

import os

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# Orbbec recommends staggering multiple devices; the wrist camera also waits
# for its USB reset.
WRIST_START_DELAY_S = 2.0


def _orbbec(launch_file, config_file):
    with open(config_file) as f:
        camera_name = yaml.safe_load(f)['camera_name']
    orbbec_launch = os.path.join(
        get_package_share_directory('orbbec_camera'), 'launch', launch_file)
    # camera_name is also passed as an argument: orbbec_camera builds the
    # namespace from the launch argument, not from config_file_path.
    return GroupAction([IncludeLaunchDescription(
        PythonLaunchDescriptionSource(orbbec_launch),
        launch_arguments={
            'camera_name': camera_name,
            'config_file_path': config_file,
        }.items(),
    )])


def _launch_setup(context):
    share = get_package_share_directory('rebel_cameras')

    def enabled(name):
        return LaunchConfiguration(name).perform(context).lower() in ('true', '1')

    actions = []
    if enabled('scene'):
        actions.append(_orbbec('gemini_330_series.launch.py',
                               os.path.join(share, 'config', 'scene_camera.yaml')))

    if enabled('wrist'):
        wrist_config = os.path.join(share, 'config', 'wrist_camera.yaml')
        wrist = TimerAction(period=WRIST_START_DELAY_S, actions=[
            _orbbec('gemini_301_series.launch.py', wrist_config)])
        if enabled('reset_wrist_usb'):
            with open(wrist_config) as f:
                serial = str(yaml.safe_load(f)['serial_number'])
            reset = ExecuteProcess(
                cmd=[os.path.join(get_package_prefix('rebel_cameras'), 'lib',
                                  'rebel_cameras', 'usb_reset.py'), serial],
                name='wrist_usb_reset', output='screen')
            actions += [reset, RegisterEventHandler(OnProcessExit(
                target_action=reset,
                on_exit=lambda event, _: [wrist] if event.returncode == 0 else []))]
        else:
            actions.append(wrist)

    if enabled('knife'):
        actions.append(GroupAction([IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory('knife_camera'), 'launch',
                'knife_camera.launch.py')))]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene', default_value='true',
                              description='Gemini 336 scene camera (/camera)'),
        DeclareLaunchArgument('wrist', default_value='true',
                              description='Gemini 305 wrist camera (/wrist_camera)'),
        DeclareLaunchArgument('knife', default_value='true',
                              description='C930e knife camera (/knife_camera)'),
        DeclareLaunchArgument(
            'reset_wrist_usb', default_value='true',
            description='USB-reset the Gemini 305 before starting it; without it the '
                        '305 only streams on the first start after being plugged in'),
        OpaqueFunction(function=_launch_setup),
    ])
