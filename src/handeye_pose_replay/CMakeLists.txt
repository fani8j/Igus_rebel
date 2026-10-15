cmake_minimum_required(VERSION 3.8)
project(handeye_pose_replay)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(nlohmann_json REQUIRED)

# Pure pose parsing/mapping/validation; no ROS dependency, unit tested below.
add_library(handeye_pose_loader STATIC src/pose_loader.cpp)
target_include_directories(
  handeye_pose_loader PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(handeye_pose_loader PUBLIC nlohmann_json::nlohmann_json)

add_executable(replay_handeye src/replay_handeye.cpp)
target_link_libraries(replay_handeye handeye_pose_loader)

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

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_pose_loader test/test_pose_loader.cpp)
  target_link_libraries(test_pose_loader handeye_pose_loader)
endif()

ament_package()
