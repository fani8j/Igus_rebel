cmake_minimum_required(VERSION 3.8)
project(dual_cut_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/joint_state_relay.py scripts/dual_arm_recover.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_joint_state_relay test/test_joint_state_relay.py)
  ament_add_pytest_test(test_dual_arm_recover test/test_dual_arm_recover.py)
  ament_add_pytest_test(test_dual_cut_system_launch test/test_dual_cut_system_launch.py)
endif()

ament_package()
