cmake_minimum_required(VERSION 3.8)
project(triton_navigation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(ompl REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

include_directories(
  include
)

add_executable(path_planning src/path_planning_node.cpp src/path_planning.cpp)
target_include_directories(path_planning PRIVATE ${OMPL_INCLUDE_DIRS})
target_link_libraries(path_planning ${OMPL_LIBRARIES})
ament_target_dependencies(path_planning
  rclcpp
  Eigen3
  ompl
  nav_msgs
  geometry_msgs
  visualization_msgs
  std_srvs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(path_following_controller src/path_following_controller.cpp)
ament_target_dependencies(path_following_controller
  rclcpp
  nav_msgs
  geometry_msgs
  std_msgs
  std_srvs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

install(TARGETS path_planning path_following_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY maps launch rviz config
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/orienteering.py
  scripts/op_solver_bridge.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
