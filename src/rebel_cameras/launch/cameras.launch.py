"""All cell cameras in one launch.

    /camera/...                    Gemini 336, static scene camera (config/scene_camera.yaml)
    /arm_left/wrist_camera/...     Gemini 305, left arm knife      (config/wrist_camera.yaml)
    /arm_left/gripper_camera/...   USB camera, left gripper        (config/left_gripper_camera.yaml)
    /arm_right/wrist_camera/...    Gemini 305, right arm knife     (config/right_wrist_camera.yaml)
    /arm_right/gripper_camera/...  USB camera, right gripper       (config/right_gripper_camera.yaml)

    ros2 launch rebel_cameras cameras.launch.py                       # scene + grippers
    ros2 launch rebel_cameras cameras.launch.py wrist:=true right_wrist:=true

The wrist 305s are off by default (2026-10-26: not needed now, and with the
336, a 305 and both gripper cameras on one hub the 336 dropped off USB every
~20 s; scene + grippers alone ran stable).

The Logitech C930e knife webcam is no longer started here (removed 2026-10-20);
run `ros2 launch knife_camera knife_camera.launch.py` separately if needed.

Topics are grouped per arm, but TF frame names still come from each config's
camera_name (wrist_camera_*, right_wrist_camera_*), so the left-arm hand-eye
(link6 -> wrist_camera_link) is unchanged. The gripper cameras publish only
image_raw/compressed (camera JPEG passthrough) and camera_info. Each runs under
scripts/gripper_camera_supervisor.py, which waits for an unplugged camera and
restarts usb_cam after a crash, a stall or a replug.

All cameras stamp frames with the host clock (Orbbec time_domain=global,
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
from launch_ros.actions import PushRosNamespace

# Orbbec recommends staggering multiple devices.
WRIST_START_DELAY_S = 2.0
RIGHT_WRIST_START_DELAY_S = 4.0
# usb_cam scans every /dev/video* node when it starts, and aborts if one
# disappears mid-scan; Orbbec USB resets and driver starts remove and recreate
# theirs. So the gripper cameras start first and the Orbbecs wait this long.
GRIPPER_SCAN_DELAY_S = 2.0
# Both gripper cameras ("HD USB Camera"). The supervisor refuses any other
# device at the configured by-path (e.g. a Gemini 305 on the same hub port).
GRIPPER_CAMERA_USB_ID = '05a3:9230'


def _orbbec(launch_file, config_file, namespace='', topic_name=None):
    """Orbbec driver publishing under /<namespace>/<topic_name>/...

    orbbec_camera takes its topic namespace and node name from the
    camera_name launch argument, but the TF frame prefix from the camera_name
    parameter, which the config file overrides. So topic_name can differ from
    the config's camera_name without renaming any frames.
    """
    with open(config_file) as f:
        camera_name = yaml.safe_load(f)['camera_name']
    orbbec_launch = os.path.join(
        get_package_share_directory('orbbec_camera'), 'launch', launch_file)
    actions = [PushRosNamespace(namespace)] if namespace else []
    return GroupAction(actions + [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(orbbec_launch),
        launch_arguments={
            'camera_name': topic_name or camera_name,
            'config_file_path': config_file,
        }.items(),
    )])


def _gripper_camera_paths():
    """by-path names of the plugged-in gripper cameras (USB id GRIPPER_CAMERA_USB_ID)."""
    by_path = '/dev/v4l/by-path'
    paths = []
    for link in sorted(os.listdir(by_path)) if os.path.isdir(by_path) else []:
        if not link.endswith(':1.0-video-index0'):
            continue
        usb_dev = f'/sys/class/video4linux/{os.path.basename(os.path.realpath(os.path.join(by_path, link)))}/device/..'
        try:
            with open(f'{usb_dev}/idVendor') as v, open(f'{usb_dev}/idProduct') as p:
                if f'{v.read().strip()}:{p.read().strip()}' == GRIPPER_CAMERA_USB_ID:
                    paths.append(os.path.join(by_path, link))
        except OSError:
            pass
    return paths


def _gripper_camera(name, config_file, namespace):
    """Supervisor that runs usb_cam for a gripper camera and restarts it on failure.

    Always started: when the camera is unplugged (now or later) the supervisor
    waits for it and restarts usb_cam after a crash, a stall or a replug.
    """
    with open(config_file) as f:
        device = yaml.safe_load(f)['/**']['ros__parameters']['video_device']
    actions = []
    if not os.path.exists(device):
        found = ', '.join(_gripper_camera_paths()) or 'none'
        actions.append(LogInfo(msg=f'WARNING: {name} camera not found at {device}; the '
                                   f'supervisor waits for it ({name}:=false hides this). '
                                   f'Gripper cameras plugged in now: {found} '
                                   f'(set video_device in {os.path.basename(config_file)})'))
    actions.append(ExecuteProcess(
        cmd=[os.path.join(get_package_prefix('rebel_cameras'), 'lib', 'rebel_cameras',
                          'gripper_camera_supervisor.py'),
             '--params-file', config_file,
             '--namespace', f'/{namespace}/gripper_camera',
             '--node-name', 'gripper_camera',
             '--usb-id', GRIPPER_CAMERA_USB_ID],
        name=f'{name}_supervisor', output='screen',
        # After SIGINT the supervisor stops its usb_cam (up to 5 s) before exiting.
        sigterm_timeout='10', sigkill_timeout='5',
        # No launch respawn: Ctrl+C signals the whole process group, usb_cam
        # exits before launch starts shutting down, and launch respawns it
        # forever. The supervisor runs usb_cam in its own session instead.
    ))
    return actions


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
    gripper_cameras = False
    for side in ('left', 'right'):
        if enabled(f'{side}_gripper'):
            gripper_cameras = True
            actions += _gripper_camera(
                f'{side}_gripper',
                os.path.join(share, 'config', f'{side}_gripper_camera.yaml'),
                f'arm_{side}')
    orbbec_delay = GRIPPER_SCAN_DELAY_S if gripper_cameras else 0.0

    orbbecs = []
    if enabled('scene'):
        scene_config = os.path.join(share, 'config', 'scene_camera.yaml')
        orbbecs += _after_usb_reset(
            'scene', scene_config,
            _orbbec('gemini_330_series.launch.py', scene_config),
            enabled('reset_scene_usb'))

    if enabled('wrist'):
        wrist_config = os.path.join(share, 'config', 'wrist_camera.yaml')
        orbbecs += _after_usb_reset(
            'wrist', wrist_config,
            TimerAction(period=WRIST_START_DELAY_S, actions=[
                _orbbec('gemini_301_series.launch.py', wrist_config,
                        'arm_left', 'wrist_camera')]),
            enabled('reset_wrist_usb'))

    if enabled('right_wrist'):
        right_wrist_config = os.path.join(share, 'config', 'right_wrist_camera.yaml')
        orbbecs += _after_usb_reset(
            'right_wrist', right_wrist_config,
            TimerAction(period=RIGHT_WRIST_START_DELAY_S, actions=[
                _orbbec('gemini_301_series.launch.py', right_wrist_config,
                        'arm_right', 'wrist_camera')]),
            enabled('reset_wrist_usb'))

    if orbbecs and orbbec_delay:
        orbbecs = [TimerAction(period=orbbec_delay, actions=orbbecs)]
    return actions + orbbecs


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('scene', default_value='true',
                              description='Gemini 336 scene camera (/camera)'),
        DeclareLaunchArgument('wrist', default_value='false',
                              description='Gemini 305 on the left arm (/arm_left/wrist_camera)'),
        DeclareLaunchArgument('right_wrist', default_value='false',
                              description='Gemini 305 on the right arm (/arm_right/wrist_camera)'),
        DeclareLaunchArgument('left_gripper', default_value='true',
                              description='USB camera on the left gripper '
                                          '(/arm_left/gripper_camera); a supervisor waits for '
                                          'it when not plugged in and restarts it on failure'),
        DeclareLaunchArgument('right_gripper', default_value='true',
                              description='USB camera on the right gripper '
                                          '(/arm_right/gripper_camera); a supervisor waits for '
                                          'it when not plugged in and restarts it on failure'),
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
