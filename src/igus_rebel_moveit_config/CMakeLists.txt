cmake_minimum_required(VERSION 3.8)
project(igus_rebel_moveit_config)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Boost REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(igus_rebel_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
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
  std_msgs
  moveit_msgs
  igus_rebel_msgs
  std_srvs
)

add_executable(rebel_servo_teleop_gamepad src/rebel_servo_teleop_gamepad.cpp)
ament_target_dependencies(rebel_servo_teleop_gamepad
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  moveit_msgs
  std_srvs
)

add_executable(box_h_cut_executor src/box_h_cut_executor.cpp)
ament_target_dependencies(box_h_cut_executor
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
  control_msgs
  geometry_msgs
  moveit_core
  moveit_msgs
  shape_msgs
  visualization_msgs
  moveit_task_constructor_core
  moveit_task_constructor_msgs
)
# boost::property_tree (header-only) parses the external snapshot JSON.
target_link_libraries(box_h_cut_executor Boost::boost)
target_compile_features(box_h_cut_executor PUBLIC cxx_std_17)

add_executable(calibration_verification src/calibration_verification.cpp)
ament_target_dependencies(calibration_verification
  rclcpp rclcpp_action std_srvs sensor_msgs geometry_msgs visualization_msgs
  control_msgs moveit_core moveit_msgs moveit_ros_planning shape_msgs
  moveit_task_constructor_core moveit_task_constructor_msgs
)
target_compile_features(calibration_verification PUBLIC cxx_std_17)
install(TARGETS
  rebel_servo_teleop_keyboard
  rebel_servo_teleop_gamepad
  box_h_cut_executor
  calibration_verification
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/camera_pole_collision.py
  scripts/start_servo.py
  DESTINATION lib/${PROJECT_NAME}
)


install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Declare ament package
ament_package()
