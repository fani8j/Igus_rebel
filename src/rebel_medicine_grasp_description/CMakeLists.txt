cmake_minimum_required(VERSION 3.10)
project(rebel_medicine_grasp_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY urdf meshes DESTINATION share/${PROJECT_NAME})
install(FILES NOTICE README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # a user-local anyio plugin breaks Ubuntu 22.04's pytest 6.2.5, so disable it for configure and test
  set(ENV{PYTEST_ADDOPTS} "-p no:anyio")
  ament_add_pytest_test(check_stand_kinematics test/check_stand_kinematics.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ENV PYTEST_ADDOPTS=-p\ no:anyio
    TIMEOUT 180)
  ament_add_pytest_test(check_amr_clearance test/check_amr_clearance.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ENV PYTEST_ADDOPTS=-p\ no:anyio
    TIMEOUT 180)
  ament_add_pytest_test(check_tool_frames test/check_tool_frames.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ENV PYTEST_ADDOPTS=-p\ no:anyio
    TIMEOUT 180)
endif()

ament_package()
