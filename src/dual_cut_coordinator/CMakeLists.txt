cmake_minimum_required(VERSION 3.8)
project(dual_cut_coordinator)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The swept-volume prefilter is Eigen-heavy and ~50x slower unoptimised (colcon passes no build type
# by default), so default to an optimised build for this package only.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(dual_cut_msgs REQUIRED)
find_package(geometric_shapes REQUIRED)
find_package(moveit_core REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(control_msgs REQUIRED)
find_package(controller_manager_msgs REQUIRED)
find_package(lifecycle_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(srdfdom REQUIRED)
find_package(urdf REQUIRED)

# Pure logic (WP5): trajectory sampling / concatenation / prefixing, swept-volume validator,
# phase bookkeeping. No node, no executor dependency.
add_library(dual_cut_coordinator_lib SHARED
  src/trajectory_tools.cpp
  src/swept_volume.cpp
  src/phase_plan.cpp
  src/gates.cpp
  src/phase_validator.cpp
)
target_include_directories(dual_cut_coordinator_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(dual_cut_coordinator_lib PUBLIC Eigen3::Eigen)
ament_target_dependencies(dual_cut_coordinator_lib PUBLIC
  dual_cut_msgs geometric_shapes moveit_core sensor_msgs trajectory_msgs)

# Stub node: starts, creates nothing, spins. WP6 replaces it (coordinator_node.cpp, execution.cpp,
# monitor.cpp).
add_executable(dual_cut_coordinator_node src/coordinator_node.cpp src/execution.cpp src/monitor.cpp)
target_link_libraries(dual_cut_coordinator_node dual_cut_coordinator_lib)
ament_target_dependencies(dual_cut_coordinator_node
  rclcpp rclcpp_action control_msgs controller_manager_msgs lifecycle_msgs moveit_msgs std_srvs std_msgs shape_msgs rcl_interfaces geometric_shapes
  sensor_msgs trajectory_msgs geometry_msgs dual_cut_msgs)

# Offline measurement tool (WP5b): FCL cross-arm distance vs a brute-force triangle reference.
add_executable(fcl_accuracy tools/fcl_accuracy.cpp)
target_link_libraries(fcl_accuracy dual_cut_coordinator_lib)
ament_target_dependencies(fcl_accuracy ament_index_cpp srdfdom urdf)
install(TARGETS fcl_accuracy DESTINATION lib/${PROJECT_NAME})
add_executable(scenario_probe tools/scenario_probe.cpp)
target_link_libraries(scenario_probe dual_cut_coordinator_lib)
ament_target_dependencies(scenario_probe ament_index_cpp srdfdom urdf)

# Fake per-arm executor export (WP6 mock tests): serves GetPlannedSolution / PlanFromState.
find_package(yaml-cpp REQUIRED)
add_executable(fake_plan_provider tools/fake_plan_provider.cpp)
target_link_libraries(fake_plan_provider dual_cut_coordinator_lib yaml-cpp)
ament_target_dependencies(fake_plan_provider rclcpp srdfdom urdf)
add_executable(blade_probe tools/blade_probe.cpp)
target_link_libraries(blade_probe dual_cut_coordinator_lib yaml-cpp)
ament_target_dependencies(blade_probe ament_index_cpp srdfdom urdf)
install(TARGETS fake_plan_provider scenario_probe blade_probe DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include)
install(TARGETS dual_cut_coordinator_lib
  EXPORT export_dual_cut_coordinator_lib
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS dual_cut_coordinator_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_index_cpp REQUIRED)
  find_package(srdfdom REQUIRED)
  find_package(urdf REQUIRED)
  find_package(joint_trajectory_controller QUIET)

  set(DUAL_CUT_TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test/data")

  foreach(t test_trajectory_tools test_swept_volume test_phase_plan test_real_dual_model test_gates test_phase_validator)
    ament_add_gtest(${t} test/${t}.cpp)
    target_link_libraries(${t} dual_cut_coordinator_lib)
    ament_target_dependencies(${t} ament_index_cpp srdfdom urdf)
    target_compile_definitions(${t} PRIVATE DUAL_CUT_TEST_DATA_DIR="${DUAL_CUT_TEST_DATA_DIR}")
  endforeach()
  # Compare our sampler with the installed Humble JTC sampler when it can be linked.
  if(joint_trajectory_controller_FOUND)
    target_compile_definitions(test_trajectory_tools PRIVATE DUAL_CUT_HAVE_JTC)
    ament_target_dependencies(test_trajectory_tools joint_trajectory_controller rclcpp)
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(dual_cut_coordinator_lib)
ament_export_dependencies(dual_cut_msgs geometric_shapes moveit_core sensor_msgs trajectory_msgs Eigen3)
ament_package()
