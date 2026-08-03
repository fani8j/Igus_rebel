from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, Command

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    debug_arg = DeclareLaunchArgument(
        'debug', default_value='false', description='')
    load_robot_description_arg = DeclareLaunchArgument(
        'load_robot_description', default_value='false', description='')
    use_gui_arg = DeclareLaunchArgument(
        'use_gui', default_value='false', description='')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation clock')
    run_mtc_program_arg = DeclareLaunchArgument(
        'run_mtc_program', default_value='false',
        description='Plan the configured PTP then LIN task')
    execute_mtc_arg = DeclareLaunchArgument(
        'execute_mtc', default_value='false',
        description='Execute the MTC solution after planning')
    
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('igus_rebel_moveit_config'), 'launch', 'move_group.launch.py')
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'use_gui': LaunchConfiguration('use_gui'),
            'run_mtc_program': LaunchConfiguration('run_mtc_program'),
            'execute_mtc': LaunchConfiguration('execute_mtc'),
        }.items(),
    )

    
    return LaunchDescription([
        debug_arg,
        load_robot_description_arg,
        use_gui_arg,
        use_sim_time_arg,
        run_mtc_program_arg,
        execute_mtc_arg,
        moveit_launch,
    ])
    