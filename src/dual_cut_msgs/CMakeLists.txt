cmake_minimum_required(VERSION 3.8)
project(dual_cut_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_task_constructor_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PlannedSegment.msg"
  "msg/CoordinatorStatus.msg"
  "srv/GetPlannedSolution.srv"
  "srv/PlanFromState.srv"
  DEPENDENCIES
    builtin_interfaces
    geometry_msgs
    moveit_msgs
    moveit_task_constructor_msgs
    sensor_msgs
    std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
