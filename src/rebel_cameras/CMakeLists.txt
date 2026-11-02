cmake_minimum_required(VERSION 3.8)
project(rebel_cameras)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/usb_reset.py scripts/camera_sync_check.py
  scripts/gripper_camera_supervisor.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  # a user-local anyio plugin breaks Ubuntu 22.04's pytest 6.2.5 (also ament_has_pytest's probe), so disable it for configure and test
  set(ENV{PYTEST_ADDOPTS} "-p no:anyio")
  # restart/shutdown logic of the gripper camera supervisor, fake child, no hardware
  ament_add_pytest_test(test_gripper_camera_supervisor test/test_gripper_camera_supervisor.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ENV PYTEST_ADDOPTS=-p\ no:anyio
    TIMEOUT 180)
endif()

ament_package()
