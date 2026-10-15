cmake_minimum_required(VERSION 3.10)
project(rebel_dual_arm_stand_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf config meshes isaac scripts DESTINATION share/${PROJECT_NAME})
install(FILES NOTICE THIRD_PARTY.md README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # a user-local anyio plugin breaks Ubuntu 22.04's pytest 6.2.5 (also ament_has_pytest's probe), so disable it for configure and test
  set(ENV{PYTEST_ADDOPTS} "-p no:anyio")
  # kinematics equivalence of urdf/rebel_arm_macro.xacro with OUR single-arm URDF (needs `xacro`; -p no:anyio: a user-local anyio breaks pytest)
  ament_add_pytest_test(check_arm_fk test/check_arm_fk.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ENV PYTEST_ADDOPTS=-p\ no:anyio
    TIMEOUT 120)
  ament_add_pytest_test(check_tool_frames test/check_tool_frames.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ENV PYTEST_ADDOPTS=-p\ no:anyio
    TIMEOUT 120)
endif()

ament_package()
