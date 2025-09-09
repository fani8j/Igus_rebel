cmake_minimum_required(VERSION 3.5)
project(igus_rebel_moveit_config)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(control_msgs)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(rebel_servo_teleop_keyboard src/rebel_servo_teleop_keyboard.cpp)
ament_target_dependencies(rebel_servo_teleop_keyboard 
  rclcpp 
  control_msgs
  geometry_msgs
  moveit_msgs
)

add_executable(rebel_servo_teleop_gamepad src/rebel_servo_teleop_gamepad.cpp)
ament_target_dependencies(rebel_servo_teleop_gamepad 
  rclcpp 
  control_msgs
  sensor_msgs
  geometry_msgs
  moveit_msgs
)

install(TARGETS
  rebel_servo_teleop_keyboard
  rebel_servo_teleop_gamepad
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# Declare ament package
ament_package()
