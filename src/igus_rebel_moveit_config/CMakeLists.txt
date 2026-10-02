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
find_package(Eigen3 REQUIRED)

include_directories(include)

# Closed-form (Pieper-style) IK for the ReBeL arm + xeg32_tool_tip TCP. No ROS/MoveIt
# dependency: Eigen only, so it is unit-testable in isolation (see test/).
add_library(rebel_analytic_ik src/rebel_analytic_ik.cpp)
target_include_directories(rebel_analytic_ik PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(rebel_analytic_ik Eigen3)
target_compile_features(rebel_analytic_ik PUBLIC cxx_std_17)

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
target_link_libraries(box_h_cut_executor Boost::boost rebel_analytic_ik)
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
install(TARGETS
  rebel_analytic_ik
  EXPORT rebel_analytic_ik_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(PROGRAMS
  scripts/camera_pole_collision.py
  scripts/start_servo.py
  DESTINATION lib/${PROJECT_NAME}
)


install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(kdl_parser QUIET)
  find_package(orocos_kdl QUIET)
  find_package(urdf QUIET)
  find_package(ament_index_cpp REQUIRED)
  find_package(igus_rebel_description QUIET)

  ament_add_gtest(test_rebel_analytic_ik test/test_rebel_analytic_ik.cpp)
  target_link_libraries(test_rebel_analytic_ik rebel_analytic_ik)
  ament_target_dependencies(test_rebel_analytic_ik ament_index_cpp)

  if(kdl_parser_FOUND AND orocos_kdl_FOUND AND urdf_FOUND AND igus_rebel_description_FOUND)
    message(STATUS "igus_rebel_moveit_config: KDL comparison IK test enabled")
    target_compile_definitions(test_rebel_analytic_ik PRIVATE REBEL_ANALYTIC_IK_HAVE_KDL=1)
    ament_target_dependencies(test_rebel_analytic_ik kdl_parser urdf)
    target_link_libraries(test_rebel_analytic_ik ${orocos_kdl_LIBRARIES})
    target_include_directories(test_rebel_analytic_ik PRIVATE ${orocos_kdl_INCLUDE_DIRS})
  else()
    message(WARNING
      "igus_rebel_moveit_config: kdl_parser/orocos_kdl/urdf/igus_rebel_description not all "
      "found; skipping the analytic-vs-KDL IK comparison gtest (self-consistency and "
      "unreachable-pose tests still run)")
    target_compile_definitions(test_rebel_analytic_ik PRIVATE REBEL_ANALYTIC_IK_HAVE_KDL=0)
  endif()
endif()

# Declare ament package
ament_package()
