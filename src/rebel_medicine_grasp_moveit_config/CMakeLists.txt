cmake_minimum_required(VERSION 3.10)
project(rebel_medicine_grasp_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/add_medicine_shelf.py DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/make_srdf.py scripts/verify_plan_only.py scripts/add_medicine_shelf.py scripts/add_medicine_shelf.py DESTINATION share/${PROJECT_NAME}/scripts)
install(FILES NOTICE README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  set(ENV{PYTEST_ADDOPTS} "-p no:anyio")
  ament_add_pytest_test(check_scene_pose test/check_scene_pose.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ENV PYTEST_ADDOPTS=-p\ no:anyio
    TIMEOUT 120)
endif()

ament_package()
