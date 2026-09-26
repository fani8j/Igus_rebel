cmake_minimum_required(VERSION 3.8)
project(igus_rebel)

# Set the required C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(igus_rebel_msgs REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)

# CRI protocol, framing, socket and safety helpers. Linked into the plugin and the tests;
# everything except RebelSocket is free of ROS and I/O.
add_library(${PROJECT_NAME}_cri STATIC
  src/CriMessages.cpp
  src/CriFraming.cpp
  src/RebelSafety.cpp
  src/RebelSocket.cpp
)
set_target_properties(${PROJECT_NAME}_cri PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PROJECT_NAME}_cri PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(${PROJECT_NAME}_cri rclcpp)

# Add hardware interface shared library
add_library(${PROJECT_NAME} SHARED
  src/Rebel.cpp
)
target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_cri)
ament_target_dependencies(${PROJECT_NAME}
  rclcpp rclcpp_lifecycle pluginlib hardware_interface igus_rebel_msgs
)

pluginlib_export_plugin_description_file(hardware_interface rebel.xml)

# Install the shared library
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install the plugin description xml file
install(FILES
  rebel.xml
  DESTINATION share/${PROJECT_NAME}
)

# Install package folders
install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_cri_status test/test_cri_status.cpp)
  target_link_libraries(test_cri_status ${PROJECT_NAME}_cri)

  ament_add_gtest(test_cri_framing test/test_cri_framing.cpp)
  target_link_libraries(test_cri_framing ${PROJECT_NAME}_cri)

  ament_add_gtest(test_rebel_safety test/test_rebel_safety.cpp)
  target_link_libraries(test_rebel_safety ${PROJECT_NAME}_cri)

  # Socket and full driver against a fake CRI server on 127.0.0.1 (never the real robot).
  ament_add_gtest(test_rebel_driver test/test_rebel_driver.cpp
    TIMEOUT 180 SKIP_LINKING_MAIN_LIBRARIES)
  target_link_libraries(test_rebel_driver ${PROJECT_NAME})
  ament_target_dependencies(test_rebel_driver
    rclcpp rclcpp_lifecycle hardware_interface igus_rebel_msgs
  )

  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
