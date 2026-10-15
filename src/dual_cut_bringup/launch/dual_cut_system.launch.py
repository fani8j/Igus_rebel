"""The whole dual-arm cutting stack in one command (cameras are NOT included).

    ROS_DOMAIN_ID=<unused> ROS_LOCALHOST_ONLY=1 \\
      ros2 launch dual_cut_bringup dual_cut_system.launch.py hardware_protocol:=mock_hardware
    ros2 launch dual_cut_bringup dual_cut_system.launch.py hardware_protocol:=rebel \\
      left_robot_ip:=192.168.3.11 right_robot_ip:=192.168.3.12     # real arms, operator approval only

`hardware_protocol` has NO default (the single-arm rebel.launch.py defaults to the real robot; this launch
refuses to guess). Start-up order, with delays so that each part finds the previous one:

  t = 0                  dual driver: rebel_dual_table_description dual_table.launch.py (one controller_manager,
                         arm_left_hw / arm_right_hw in fault group dual_arm_table, JSB, both trajectory
                         controllers; with `rebel` the controllers stay INACTIVE unless activate_controllers:=true)
  t = move_group_delay   dual move_group (planning scene of the cell, camera pole; RViz only with use_rviz:=true)
  t = planning_delay     both per-arm plan-only stacks (move_group + executor + camera pole + joint-state relay)
  t = coordinator_delay  dual_cut_coordinator, execution_enabled and the per-phase enables all false

Safety defaults: nothing here enables execution. The coordinator's enables are runtime parameters that the
operator sets on purpose (see README.md); a coordinator_config that sets any of them to true is refused.
"""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

DRIVER_PACKAGE = "rebel_dual_table_description"
MOVEIT_PACKAGE = "rebel_dual_table_moveit_config"
COORDINATOR_PACKAGE = "dual_cut_coordinator"
THIS = "dual_cut_bringup"
ENABLE_PARAMS = ("execution_enabled", "execute_phase_a_enabled", "execute_phase_b_enabled")
REAL_ROS_DOMAIN = "11"


def include(package, launch_file, arguments):
    path = os.path.join(get_package_share_directory(package), "launch", launch_file)
    return IncludeLaunchDescription(PythonLaunchDescriptionSource(path), launch_arguments=arguments.items())


def check_coordinator_config(path):
    """Refuse a coordinator parameter file that enables execution (the enables are set at runtime)."""
    with open(path) as handle:
        document = yaml.safe_load(handle)
    for key, body in (document or {}).items():
        params = body.get("ros__parameters", {}) if isinstance(body, dict) else {}
        for name in ENABLE_PARAMS:
            if params.get(name) not in (None, False):
                raise RuntimeError(
                    f"coordinator_config {path}: {name} is {params.get(name)!r} under {key!r}. Execution "
                    f"enables must stay false at launch; the operator sets them at runtime with "
                    f"'ros2 param set /dual_cut_coordinator {name} true' (see the dual_cut_bringup README).")


def operator_checklist(hw, left_ip, right_ip, activate):
    rule = "=" * 78
    if hw == "rebel":
        controllers = ("ACTIVE at start-up (activate_controllers:=true was given: NO identity check will be "
                       "asked)" if activate == "true" else
                       "loaded INACTIVE (activate_controllers:=false)")
        return [
            rule,
            "REAL HARDWARE (hardware_protocol:=rebel). Execution stays disabled; the arms do not move.",
            f"  arm_left  -> {left_ip}    arm_right -> {right_ip}",
            f"  Trajectory controllers: {controllers}.",
            "OPERATOR CHECKLIST",
            "  1. Verify arm identity: jog ONE arm from its pendant and watch /joint_states; the joints of",
            "     arm_left_* must be the ones that move for the left-IP arm (and the same for the right).",
            "     A swapped IP pair sends each arm the other arm's trajectories.",
            "  2. Only then activate the controllers, with the identity prompt:",
            "       ros2 run dual_cut_bringup dual_arm_recover.py activate",
            "  3. Check the state any time:  ros2 run dual_cut_bringup dual_arm_recover.py status",
            "  4. A fault on one arm latches both (fault group dual_arm_table): see the README, recover with",
            "       ros2 run dual_cut_bringup dual_arm_recover.py recover --root left|right|auto",
            "     then rearm the coordinator yourself (ros2 service call /dual_cut_coordinator/rearm ...).",
            rule,
        ]
    return [
        rule,
        "MOCK (hardware_protocol:=mock_hardware): no robot is involved. Execution stays disabled.",
        "  Controllers start active (activate_controllers:=auto). Do NOT use ROS domain 11 (real robot).",
        rule,
    ]


def launch_setup(context):
    hw = LaunchConfiguration("hardware_protocol").perform(context).strip()
    if hw not in ("mock_hardware", "rebel"):
        raise RuntimeError(f"hardware_protocol must be mock_hardware or rebel, got {hw!r}")
    domain = os.environ.get("ROS_DOMAIN_ID", "")
    if hw == "mock_hardware" and domain == REAL_ROS_DOMAIN:
        raise RuntimeError(
            f"hardware_protocol:=mock_hardware refused: ROS_DOMAIN_ID={REAL_ROS_DOMAIN} is the real robot's "
            f"domain. Use an unused domain with ROS_LOCALHOST_ONLY=1.")
    if hw == "mock_hardware" and os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError(
            "hardware_protocol:=mock_hardware refused: set ROS_LOCALHOST_ONLY=1 so the mock stack cannot be "
            "seen from (or reach) other machines.")

    left_ip = LaunchConfiguration("left_robot_ip").perform(context).strip()
    right_ip = LaunchConfiguration("right_robot_ip").perform(context).strip()
    activate = LaunchConfiguration("activate_controllers").perform(context).strip()
    calibration = os.path.expanduser(LaunchConfiguration("calibration_file").perform(context).strip())
    if not calibration:
        calibration = os.path.join(get_package_share_directory(DRIVER_PACKAGE), "config",
                                   "dual_arm_calibration.yaml")
    coordinator_config = os.path.expanduser(LaunchConfiguration("coordinator_config").perform(context).strip())
    if not coordinator_config:
        coordinator_config = os.path.join(get_package_share_directory(COORDINATOR_PACKAGE), "config",
                                          "coordinator.yaml")
    check_coordinator_config(coordinator_config)

    driver_args = {"hardware_protocol": hw, "calibration_file": calibration, "activate_controllers": activate}
    if left_ip:
        driver_args["left_robot_ip"] = left_ip
    if right_ip:
        driver_args["right_robot_ip"] = right_ip
    # Shown in the checklist; the driver's own defaults when not given.
    shown_left = left_ip or "192.168.3.11 (dual_table.launch.py default)"
    shown_right = right_ip or "192.168.3.12 (dual_table.launch.py default)"

    move_group_args = {"hardware_protocol": hw, "calibration_file": calibration,
                       "use_gui": LaunchConfiguration("use_rviz"),
                       "run_camera_pole": LaunchConfiguration("run_camera_pole"),
                       "use_sim_time": LaunchConfiguration("use_sim_time")}
    planning_args = {name: LaunchConfiguration(name) for name in (
        "use_sim_time", "executor_config", "executor_overrides_left", "executor_overrides_right",
        "box_h_cut_start_delay", "blade_collision")}
    planning_args["calibration_file"] = calibration   # the SAME file as the driver and the dual move_group

    def after(delay, action):
        return TimerAction(period=float(LaunchConfiguration(delay).perform(context)), actions=[action])

    return [
        *[LogInfo(msg=line) for line in operator_checklist(hw, shown_left, shown_right, activate)],
        include(DRIVER_PACKAGE, "dual_table.launch.py", driver_args),
        after("move_group_delay", include(MOVEIT_PACKAGE, "move_group.launch.py", move_group_args)),
        after("planning_delay", include(THIS, "both_arms_planning.launch.py", planning_args)),
        after("coordinator_delay", include(COORDINATOR_PACKAGE, "coordinator.launch.py",
                                           {"config": coordinator_config})),
    ]


def generate_launch_description():
    return LaunchDescription([
        # No default on purpose: DeclareLaunchArgument without a default aborts the launch when it is missing.
        DeclareLaunchArgument(
            "hardware_protocol", choices=["mock_hardware", "rebel"],
            description="REQUIRED. mock_hardware, or rebel (real CRI drivers; operator approval only)"),
        DeclareLaunchArgument("left_robot_ip", default_value="",
                              description="CRI controller of arm_left (rebel); empty = dual_table.launch.py default"),
        DeclareLaunchArgument("right_robot_ip", default_value="",
                              description="CRI controller of arm_right (rebel); empty = dual_table.launch.py default"),
        DeclareLaunchArgument("calibration_file", default_value="",
                              description="dual_arm_calibration YAML (empty = rebel_dual_table_description default); "
                                          "passed to the driver AND the dual move_group"),
        DeclareLaunchArgument("activate_controllers", default_value="auto", choices=["auto", "true", "false"],
                              description="auto: active in mock, INACTIVE with rebel (activate with dual_arm_recover.py)"),
        DeclareLaunchArgument("use_rviz", default_value="false", description="RViz of the dual move_group"),
        DeclareLaunchArgument("run_camera_pole", default_value="true",
                              description="camera_pole / camera_head boxes in the dual scene (needs camera TF)"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("executor_config", default_value="",
                              description="base executor YAML for both per-arm planners"),
        DeclareLaunchArgument("executor_overrides_left", default_value=""),
        DeclareLaunchArgument("executor_overrides_right", default_value=""),
        DeclareLaunchArgument("box_h_cut_start_delay", default_value="3.0"),
        DeclareLaunchArgument("blade_collision", default_value="true", choices=["true", "false"],
                              description="per-arm planners: knife_c_blade collision box + carton_contact_links "
                                          "(see per_arm_planning.launch.py)"),
        DeclareLaunchArgument("coordinator_config", default_value="",
                              description="coordinator parameter YAML (default: dual_cut_coordinator "
                                          "config/coordinator.yaml); the execution enables must be false in it"),
        DeclareLaunchArgument("move_group_delay", default_value="2.0",
                              description="seconds after the driver before the dual move_group starts"),
        DeclareLaunchArgument("planning_delay", default_value="4.0",
                              description="seconds after the driver before the per-arm planners start"),
        DeclareLaunchArgument("coordinator_delay", default_value="10.0",
                              description="seconds after the driver before the coordinator starts"),
        OpaqueFunction(function=launch_setup),
    ])
