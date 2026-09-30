cmake_minimum_required(VERSION 3.8)
project(side_tape_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Observation.msg"
  "msg/SupervisorStatus.msg"
  "srv/StartCut.srv"
  "srv/SetSeed.srv"
  "srv/StartInsert.srv"
  "srv/StartLateral.srv"
  DEPENDENCIES std_msgs builtin_interfaces
)

ament_package()
