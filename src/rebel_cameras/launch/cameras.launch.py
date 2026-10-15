"""All cell cameras in one launch.

    /camera/...              Gemini 336, static scene camera  (config/scene_camera.yaml)
    /wrist_camera/...        Gemini 305, left arm knife       (config/wrist_camera.yaml)
    /right_wrist_camera/...  Gemini 305, right arm knife      (config/right_wrist_camera.yaml)
    /knife_camera/...        Logitech C930e, knife view       (knife_camera package)

    ros2 launch rebel_cameras cameras.launch.py
    ros2 launch rebel_cameras cameras.launch.py wrist:=false right_wrist:=false knife:=false

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
                            IncludeLaunchDescription, LogInfo, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

# Orbbec recommends staggering multiple devices.
WRIST_START_DELAY_S = 2.0
RIGHT_WRIST_START_DELAY_S = 4.0


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


def _after_usb_reset(name, config_file, start, reset):
    """Start `start` after a USB port reset of the camera in config_file.

    Both Orbbecs (orbbec_camera 2.9.3) can hang after "Initialize device cost"
    with no frames when the driver is restarted without a USB reconnect: the
    305 on every restart, the 336 intermittently. A reset avoids it.
    """
    if not reset:
        return [start]
    with open(config_file) as f:
        serial = str(yaml.safe_load(f)['serial_number'])
    reset_process = ExecuteProcess(
        cmd=[os.path.join(get_package_prefix('rebel_cameras'), 'lib',
                          'rebel_cameras', 'usb_reset.py'), serial],
        name=f'{name}_usb_reset', output='screen')
    return [reset_process, RegisterEventHandler(OnProcessExit(
        target_action=reset_process,
        on_exit=lambda event, _: [start] if event.returncode == 0 else []))]


def _launch_setup(context):
    share = get_package_share_directory('rebel_cameras')

    def enabled(name):
        return LaunchConfiguration(name).perform(context).lower() in ('true', '1')

    actions = []
    if enabled('scene'):
        scene_config = os.path.join(share, 'config', 'scene_camera.yaml')
        actions += _after_usb_reset(
            'scene', scene_config,
            _orbbec('gemini_330_series.launch.py', scene_config),
            enabled('reset_scene_usb'))

    if enabled('wrist'):
        wrist_config = os.path.join(share, 'config', 'wrist_camera.yaml')
        actions += _after_usb_reset(
            'wrist', wrist_config,
            TimerAction(period=WRIST_START_DELAY_S, actions=[
                _orbbec('gemini_301_series.launch.py', wrist_config)]),
            enabled('reset_wrist_usb'))

    if enabled('right_wrist'):
        right_wrist_config = os.path.join(share, 'config', 'right_wrist_camera.yaml')
        actions += _after_usb_reset(
            'right_wrist', right_wrist_config,
            TimerAction(period=RIGHT_WRIST_START_DELAY_S, actions=[
                _orbbec('gemini_301_series.launch.py', right_wrist_config)]),
            enabled('reset_wrist_usb'))

    if enabled('knife'):
        knife_share = get_package_share_directory('knife_camera')
        with open(os.path.join(knife_share, 'config', 'knife_camera.yaml')) as f:
            knife_device = yaml.safe_load(f)['/**']['ros__parameters']['video_device']
        if os.path.exists(knife_device):
            actions.append(GroupAction([IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(
                    knife_share, 'launch', 'knife_camera.launch.py')))]))
        else:
            # knife_camera.launch.py raises when the webcam is missing, which
            # would abort the whole launch; skip it so the Orbbecs still start.
            actions.append(LogInfo(msg=f'WARNING: knife camera not found at {knife_device}; '
                                       'starting without it (knife:=false hides this)'))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene', default_value='true',
                              description='Gemini 336 scene camera (/camera)'),
        DeclareLaunchArgument('wrist', default_value='true',
                              description='Gemini 305 on the left arm (/wrist_camera)'),
        DeclareLaunchArgument('right_wrist', default_value='true',
                              description='Gemini 305 on the right arm (/right_wrist_camera)'),
        DeclareLaunchArgument('knife', default_value='true',
                              description='C930e knife camera (/knife_camera); skipped '
                                          'with a warning when not plugged in'),
        DeclareLaunchArgument(
            'reset_scene_usb', default_value='true',
            description='USB-reset the Gemini 336 before starting it; restarts '
                        'without a reconnect can hang with no frames'),
        DeclareLaunchArgument(
            'reset_wrist_usb', default_value='true',
            description='USB-reset the Gemini 305s before starting them; without it a '
                        '305 only streams on the first start after being plugged in'),
        OpaqueFunction(function=_launch_setup),
    ])
