from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    use_sim_time_arg = DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true')
        
    # Include the empty world launch from Gazebo
    gz_sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments=[('gz_args', [' -r empty.sdf'])],
    )

    gz_bridge_config_file = os.path.join(
        get_package_share_directory('igus_rebel_moveit_config'), 'config', 'gz_bridge.yaml'
    )

    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name="param_bridge",
        output='screen',
        arguments=['--ros-args', '-p', ['config_file:=', gz_bridge_config_file]],
        parameters=[
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }
        ],
    )

    # Spawn the robot model in Gazebo
    spawn_node = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        parameters=[{'topic': 'robot_description',
                     'name': 'igus_rebel',
                     'use_sim_time': True,
                     }],
    )

    # Include the ROS controllers launch file
    ros_controllers_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('igus_rebel_moveit_config'), 'launch', 'ros_controllers.launch.py')
        ),
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        gz_bridge,
        gz_sim_launch,
        spawn_node,
        TimerAction(period=10.0, actions=[ros_controllers_launch]),
    ])
