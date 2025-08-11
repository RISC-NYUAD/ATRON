cmake_minimum_required(VERSION 3.8)
project(triton_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

include_directories(
  include
)

install(
  DIRECTORY
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/thruster_controller.py
  scripts/triton_controller.py
  scripts/PID.py
  scripts/get_gyro.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
