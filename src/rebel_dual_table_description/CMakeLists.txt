cmake_minimum_required(VERSION 3.8)
project(rebel_dual_table_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config launch meshes DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/check_fk_equivalence.py
  scripts/make_arm_macro.py
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_dual_table_description test/test_dual_table_description.py)
endif()

ament_package()
