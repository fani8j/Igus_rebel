cmake_minimum_required(VERSION 3.8)
project(rebel_cameras)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/usb_reset.py scripts/camera_sync_check.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
