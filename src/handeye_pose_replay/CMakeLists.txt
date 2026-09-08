cmake_minimum_required(VERSION 3.8)
project(handeye_pose_replay)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rcl_interfaces REQUIRED)

add_executable(replay_handeye src/replay_handeye.cpp)

ament_target_dependencies(
  replay_handeye
  rclcpp
  rcl_interfaces
  moveit_ros_planning_interface
)

install(
  TARGETS replay_handeye
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
