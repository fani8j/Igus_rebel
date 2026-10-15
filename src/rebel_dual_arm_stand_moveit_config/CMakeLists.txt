cmake_minimum_required(VERSION 3.10)
project(rebel_dual_arm_stand_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/add_cell_scene.py DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/make_srdf.py scripts/verify_plan_only.py DESTINATION share/${PROJECT_NAME}/scripts)
install(FILES NOTICE README.md DESTINATION share/${PROJECT_NAME})

ament_package()
