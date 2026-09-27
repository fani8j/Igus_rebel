cmake_minimum_required(VERSION 3.8)
project(knife_camera)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
install(
  PROGRAMS scripts/focus_tune.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
