cmake_minimum_required(VERSION 3.8)
project(rebel_servo_relay)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(moveit_core REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(urdf REQUIRED)

# Pure relay logic (no ROS), unit-tested.
add_library(rebel_servo_relay_core src/relay_core.cpp)
target_include_directories(rebel_servo_relay_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# MoveIt Servo smoothing plugin that does nothing (Servo 2.5.10 always loads one).
add_library(rebel_servo_relay_filter SHARED src/pass_through_filter_plugin.cpp)
ament_target_dependencies(rebel_servo_relay_filter rclcpp pluginlib moveit_core)
pluginlib_export_plugin_description_file(moveit_core pass_through_plugin.xml)

add_executable(servo_trajectory_relay src/servo_trajectory_relay.cpp)
target_link_libraries(servo_trajectory_relay rebel_servo_relay_core)
ament_target_dependencies(servo_trajectory_relay rclcpp sensor_msgs trajectory_msgs rcl_interfaces urdf)

install(TARGETS rebel_servo_relay_core rebel_servo_relay_filter
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS servo_trajectory_relay DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_relay_core test/test_relay_core.cpp)
  target_link_libraries(test_relay_core rebel_servo_relay_core)
endif()

ament_package()
