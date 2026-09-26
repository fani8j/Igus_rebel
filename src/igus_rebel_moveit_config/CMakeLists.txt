cmake_minimum_required(VERSION 3.5)
project(igus_rebel_moveit_config)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(igus_rebel_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(moveit_task_constructor_core REQUIRED)
find_package(moveit_task_constructor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(rebel_servo_teleop_keyboard src/rebel_servo_teleop_keyboard.cpp)
ament_target_dependencies(rebel_servo_teleop_keyboard 
  rclcpp 
  control_msgs
  geometry_msgs
  moveit_msgs
  igus_rebel_msgs
)

add_executable(rebel_servo_teleop_gamepad src/rebel_servo_teleop_gamepad.cpp)
ament_target_dependencies(rebel_servo_teleop_gamepad 
  rclcpp 
  control_msgs
  sensor_msgs
  geometry_msgs
  moveit_msgs
)

add_executable(rebel_mtc_ptp_lin src/rebel_mtc_ptp_lin.cpp)
ament_target_dependencies(rebel_mtc_ptp_lin
  rclcpp
  rclcpp_action
  std_srvs
  control_msgs
  moveit_task_constructor_msgs
  geometry_msgs
  moveit_task_constructor_core
)

add_executable(box_motion_executor src/box_motion_executor.cpp)
ament_target_dependencies(box_motion_executor
  rclcpp
  std_srvs
  diagnostic_msgs
  geometry_msgs
  visualization_msgs
  moveit_task_constructor_core
)

add_executable(box_h_cut_executor src/box_h_cut_executor.cpp)
ament_target_dependencies(box_h_cut_executor
  rclcpp
  sensor_msgs
  std_srvs
  geometry_msgs
  moveit_msgs
  shape_msgs
  visualization_msgs
  moveit_task_constructor_core
)
target_compile_features(box_h_cut_executor PUBLIC cxx_std_17)

add_executable(calibration_verification src/calibration_verification.cpp)
ament_target_dependencies(calibration_verification
  rclcpp rclcpp_action std_srvs sensor_msgs geometry_msgs visualization_msgs
  control_msgs moveit_core moveit_msgs moveit_ros_planning shape_msgs
  moveit_task_constructor_core moveit_task_constructor_msgs
  moveit_ros_planning_interface
)
target_compile_features(calibration_verification PUBLIC cxx_std_17)
install(TARGETS
  rebel_servo_teleop_keyboard
  rebel_servo_teleop_gamepad
  rebel_mtc_ptp_lin
  box_motion_executor
  box_h_cut_executor
  calibration_verification
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/camera_pole_collision.py
  DESTINATION lib/${PROJECT_NAME}
)


install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Declare ament package
ament_package()
