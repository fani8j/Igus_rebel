from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command, FindExecutable

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    debug_arg = DeclareLaunchArgument(
        'debug', default_value='false', description='')
    load_robot_description_arg = DeclareLaunchArgument(
        'load_robot_description', default_value='false', description='')
    use_gui_arg = DeclareLaunchArgument(
        'use_gui', default_value='true', description='')
    gazebo_gui_arg = DeclareLaunchArgument(
        'gazebo_gui', default_value='false', description='')
    paused_arg = DeclareLaunchArgument(
        'paused', default_value='false', description='')
    hardware_protocol_arg = DeclareLaunchArgument(
        "hardware_protocol",
        default_value="gazebo",
        choices=["mock_hardware", "gazebo", "rebel"],
        description="Which hardware protocol or mock hardware should be used",)
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', 
        default_value='true', 
        description='Use sim time if true')
    
    hardware_protocol = LaunchConfiguration('hardware_protocol')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_gui = LaunchConfiguration('use_gui')
    
    robot_description_file = os.path.join(
        get_package_share_directory('igus_rebel_description'),
        'urdf',
        'igus_rebel_robot2.urdf.xacro'
    )

    robot_description = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            robot_description_file,
            " hardware_protocol:=",
            hardware_protocol,
        ]
    )
    
    # Launch gazebo simulator and spwan the robot
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('igus_rebel_moveit_config'), 'launch', 'gazebo.launch.py')
        ),
    )
    
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': use_sim_time}],
        remappings=[],
        output="both",
    )
    
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('igus_rebel_moveit_config'), 'launch', 'move_group.launch.py')
        ),
        launch_arguments={
            'hardware_protocol': hardware_protocol,
            'use_sim_time': use_sim_time,
            'use_gui': use_gui,
        }.items(),
    )
    
    return LaunchDescription([
        debug_arg,
        load_robot_description_arg,
        use_gui_arg,
        gazebo_gui_arg,
        paused_arg,
        hardware_protocol_arg,
        use_sim_time_arg,
        gazebo_launch,
        robot_state_pub_node,
        moveit_launch,
    ])
    