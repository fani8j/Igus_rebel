cmake_minimum_required(VERSION 3.8)
project(rebel_dual_table_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/make_srdf.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_dual_table_srdf test/test_dual_table_srdf.py)
endif()

ament_package()
