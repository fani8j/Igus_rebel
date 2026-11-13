"""Launch one JXC61 gripper node. `arm:=` selects the driver node it talks to.

arm:=''     single arm: driver node /igus_rebel, service /set_digital_outputs, node /jxc_gripper_node
arm:=left   dual arm: driver node /igus_rebel_left, service /igus_rebel_left/set_digital_outputs,
            node /arm_left/jxc_gripper_node
arm:=right  likewise with right
(the dual-arm driver node names igus_rebel_left / igus_rebel_right come from the node_name
parameter in rebel_dual_table_description/urdf/rebel_dual_table.urdf.xacro)
Pins (operator confirmed 2026-11-13): DOut21 IN0, 22 IN1, 23 SETUP, 24 DRIVE, 25 RESET, 26 SVON;
DIn21 SETON, 22 INP, 23 SVRE, 24 *ALARM.
Pin overrides: params_file:=<yaml> with the usual ROS parameter file for jxc_gripper_node.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _nodes(context):
    arm = LaunchConfiguration("arm").perform(context).strip()
    params_file = LaunchConfiguration("params_file").perform(context).strip()
    if arm not in ("", "left", "right"):
        raise RuntimeError(f"arm must be '', 'left' or 'right', got '{arm}'")
    driver = f"igus_rebel_{arm}" if arm else "igus_rebel"
    service = f"/{driver}/set_digital_outputs" if arm else "/set_digital_outputs"
    parameters = []
    if params_file:
        parameters.append(params_file)
    parameters.append({
        "digital_io_topic": f"/{driver}/digital_io",
        "set_outputs_service": service,
    })
    return [Node(
        package="rebel_gripper_io",
        executable="jxc_gripper_node",
        name="jxc_gripper_node",
        namespace=f"arm_{arm}" if arm else "",
        parameters=parameters,
        output="screen",
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("arm", default_value="", description="'', 'left' or 'right'"),
        DeclareLaunchArgument("params_file", default_value="",
                              description="optional ROS parameter YAML (pins/timeouts)"),
        OpaqueFunction(function=_nodes),
    ])
