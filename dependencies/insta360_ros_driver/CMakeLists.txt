cmake_minimum_required(VERSION 3.5)
project(insta360_ros_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(imu_tools REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_search_module(SWSCALE REQUIRED libswscale)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${AVCODEC_INCLUDE_DIRS}
  ${AVFORMAT_INCLUDE_DIRS}
  ${AVUTIL_INCLUDE_DIRS}
  ${SWSCALE_INCLUDE_DIRS}
)

link_directories(
  ${PROJECT_SOURCE_DIR}/lib
  ${PROJECT_SOURCE_DIR}/include
)

install(
  DIRECTORY config launch lib
  DESTINATION share/${PROJECT_NAME}
)

add_executable(${PROJECT_NAME} src/main.cpp)
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  camera_info_manager
  cv_bridge
  image_transport
  std_msgs
  std_srvs
  OpenCV
)
target_link_libraries(${PROJECT_NAME} CameraSDK  ${OpenCV_LIBRARIES} ${AVCODEC_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVUTIL_LIBRARIES} ${SWSCALE_LIBRARIES})

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(decoder src/decoder.cpp)
ament_target_dependencies(decoder
  rclcpp
  camera_info_manager
  cv_bridge
  image_transport
  std_msgs
  std_srvs
  OpenCV
)
target_link_libraries(decoder CameraSDK  ${OpenCV_LIBRARIES} ${AVCODEC_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVUTIL_LIBRARIES} ${SWSCALE_LIBRARIES})

install(TARGETS decoder
  DESTINATION lib/${PROJECT_NAME}
)

# Add C++ equirectangular node
add_executable(equirectangular_cpp src/equirectangular.cpp)
ament_target_dependencies(equirectangular_cpp
  rclcpp
  cv_bridge
  OpenCV
)
target_link_libraries(equirectangular_cpp ${OpenCV_LIBRARIES})

install(TARGETS equirectangular_cpp
  DESTINATION lib/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/equirectangular.py
  scripts/calibrate.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
