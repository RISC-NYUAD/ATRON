cmake_minimum_required(VERSION 3.8)
project(triton_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(imu_tools REQUIRED)

include_directories(
  include
)

install(
  DIRECTORY config launch urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/usb_cam.py
  scripts/arduino.py
  scripts/yolo.py
  scripts/cubemap.py
  scripts/offline_cubemap.py
  scripts/offline_bottom.py
  scripts/bottom_view.py
  scripts/bbox.py
  scripts/bag_to_cubemap.py
  scripts/bag_to_cubemap_gpu.py
  scripts/undistort.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
