"""Planning-only MoveIt 2 (Humble) stack for the dual-arm cell: robot_state_publisher + move_group + neutral joint states.

NO ros2_control, NO controllers, NO trajectory execution (allow_trajectory_execution false) -- nothing can move hardware.

  export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
  nice -n 10 ros2 launch /home/farhan/igus_rebel_ros2/tools/cell_planner_check/cell_planning.launch.py \
      build_dir:=/home/farhan/igus_rebel_ros2/tools/cell_planner_check/build/h000n.0_s0355

build_dir is written by gen_cell.py (dual_arm.urdf, dual_arm.srdf, kinematics.yaml, joint_limits.yaml, ompl_planning.yaml, pilz_planning.yaml).
The config is assembled with moveit_configs_utils.MoveItConfigsBuilder using explicit absolute file paths (the builder needs *a* package name
for its default lookups; `moveit_configs_utils` itself is used for that and no file is read from it except by absolute path we give).
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_param_builder import load_yaml
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    b = Path(LaunchConfiguration('build_dir').perform(context))
    pipelines = LaunchConfiguration('pipelines').perform(context).split(',')
    mc = (MoveItConfigsBuilder('dual_arm_cell', package_name='moveit_configs_utils')
          .robot_description(file_path=str(b / 'dual_arm.urdf'))
          .robot_description_semantic(file_path=str(b / 'dual_arm.srdf'))
          .robot_description_kinematics(file_path=str(b / 'kinematics.yaml'))
          .joint_limits(file_path=str(b / 'joint_limits.yaml'))
          .pilz_cartesian_limits(file_path=str(b / 'pilz_cartesian_limits.yaml'))
          .planning_pipelines(pipelines=['ompl'])           # placeholder (replaced below by our own files)
          .trajectory_execution(file_path=str(b / 'no_controllers.yaml'), moveit_manage_controllers=False)
          .to_moveit_configs())
    # planning pipelines from our own files (the builder's pipeline loader only reads <package>/config/<name>_planning.yaml)
    pp = {'planning_pipelines': pipelines, 'default_planning_pipeline': 'ompl'}
    if 'ompl' in pipelines:
        pp['ompl'] = load_yaml(b / 'ompl_planning.yaml')
    if 'pilz_industrial_motion_planner' in pipelines:
        pp['pilz_industrial_motion_planner'] = load_yaml(b / 'pilz_planning.yaml')
    mc.planning_pipelines = pp
    # Pilz also reads its Cartesian limits from robot_description_planning (already part of joint_limits.yaml)
    traj = {'allow_trajectory_execution': False, 'moveit_manage_controllers': False}
    mc.trajectory_execution = traj
    params = mc.to_dict()
    params.update({
        'publish_planning_scene': True, 'publish_geometry_updates': True, 'publish_state_updates': True,
        'publish_transforms_updates': True, 'use_sim_time': False,
        'planning_scene_monitor': {'publish_planning_scene': True},
        'allow_trajectory_execution': False,
    })
    env = {'FASTDDS_BUILTIN_TRANSPORTS': 'UDPv4'} if os.environ.get('CELL_FASTDDS_UDP') else None
    nodes = [
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='log',
             parameters=[mc.robot_description, {'use_sim_time': False}]),
        # neutral (all-zero) joint states so move_group has a current state; no hardware is involved
        Node(package='joint_state_publisher', executable='joint_state_publisher', output='log',
             parameters=[mc.robot_description, {'rate': 10}]),
        Node(package='moveit_ros_move_group', executable='move_group', output='screen', parameters=[params],
             additional_env=env),
        Node(package='rviz2', executable='rviz2', output='log', condition=IfCondition(LaunchConfiguration('rviz')),
             parameters=[mc.robot_description, mc.robot_description_semantic, mc.robot_description_kinematics, mc.planning_pipelines]),
    ]
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('build_dir', description='directory written by gen_cell.py'),
        DeclareLaunchArgument('pipelines', default_value='ompl,pilz_industrial_motion_planner'),
        DeclareLaunchArgument('rviz', default_value='false'),
        OpaqueFunction(function=launch_setup),
    ])
