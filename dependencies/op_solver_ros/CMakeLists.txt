cmake_minimum_required(VERSION 3.10)
project(op_solver_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
# Find system dependencies
find_library(GMP_LIBRARY gmp)
if(NOT GMP_LIBRARY)
  message(FATAL_ERROR "GMP library not found. Please install libgmp-dev")
endif()

# Configure and build op-solver as external project
include(ExternalProject)
set(OPSOLVER_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op-solver)
set(OPSOLVER_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/op-solver-build)
set(OPSOLVER_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/op-solver-install)

# Create build directory
file(MAKE_DIRECTORY ${OPSOLVER_BUILD_DIR})

# Run autogen.sh and configure for op-solver
execute_process(
  COMMAND ${OPSOLVER_SOURCE_DIR}/autogen.sh
  WORKING_DIRECTORY ${OPSOLVER_SOURCE_DIR}
  RESULT_VARIABLE AUTOGEN_RESULT
)

if(NOT AUTOGEN_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to run autogen.sh for op-solver")
endif()

# Configure op-solver (heuristic only, no CPLEX)
execute_process(
  COMMAND ${OPSOLVER_SOURCE_DIR}/configure --prefix=${OPSOLVER_INSTALL_DIR}
  WORKING_DIRECTORY ${OPSOLVER_BUILD_DIR}
  RESULT_VARIABLE CONFIGURE_RESULT
)

if(NOT CONFIGURE_RESULT EQUAL 0)
  message(FATAL_ERROR "Failed to configure op-solver")
endif()

# Build op-solver
add_custom_target(build_op_solver ALL
  COMMAND make -j$(nproc)
  COMMAND make install
  WORKING_DIRECTORY ${OPSOLVER_BUILD_DIR}
  COMMENT "Building op-solver"
)

# Include directories
include_directories(
  include
  ${OPSOLVER_SOURCE_DIR}/src
  ${OPSOLVER_INSTALL_DIR}/include
)

# Link directories
link_directories(${OPSOLVER_INSTALL_DIR}/lib)

# Create ROS2 node executable
add_executable(op_solver_node src/op_solver_node.cpp)

# Ensure op-solver is built before our node
add_dependencies(op_solver_node build_op_solver)

# Link libraries
target_link_libraries(op_solver_node
  opsolver
  ${GMP_LIBRARY}
  m
)

# ROS2 dependencies
ament_target_dependencies(op_solver_node
  rclcpp
  std_msgs
  std_srvs
)

# Create wrapper library
add_library(op_solver_wrapper STATIC src/op_solver_wrapper.c)
target_include_directories(op_solver_wrapper PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OPSOLVER_SOURCE_DIR}/src
  ${OPSOLVER_INSTALL_DIR}/include
)
add_dependencies(op_solver_wrapper build_op_solver)
target_link_libraries(op_solver_wrapper
  opsolver
  ${GMP_LIBRARY}
  m
)

# Note: The marker-based and service nodes have been removed.
# Use the Python op_solver_bridge in triton_navigation package instead,
# which calls the external op-solver executable.

# Install executables
install(TARGETS
  op_solver_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/launch)
  install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}/
  )
endif()

# Install config files if they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
  install(DIRECTORY
    config
    DESTINATION share/${PROJECT_NAME}/
  )
endif()

if(BUILD_TESTING)
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