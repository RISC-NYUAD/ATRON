# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

Build this package:
```bash
cd /home/triton/ros2_ws
colcon build --packages-select triton_navigation --symlink-install
source install/setup.bash
```

Clean build:
```bash
cd /home/triton/ros2_ws
rm -rf build/triton_navigation install/triton_navigation
colcon build --packages-select triton_navigation --symlink-install
source install/setup.bash
```

## Package Architecture

This package implements navigation capabilities for the Triton water surface vehicle, consisting of two main components:

### Path Planning Node (`path_planning`)
- **Core Class**: `Planner2D` (defined in `include/triton_navigation/path_planning.hpp`)
- **Algorithm**: OMPL-based planners (RRT-Connect default, also supports RRT*, EST, PRM)
- **Services**:
  - `/get_path` (nav_msgs/srv/GetPlan) - Plans path between two poses
  - `/record_waypoints` (std_srvs/srv/Empty) - Starts recording clicked points
  - `/save_waypoints` (std_srvs/srv/Empty) - Stops recording and publishes waypoints
  - `/generate_path` (std_srvs/srv/Empty) - Creates combined path through all waypoints
  - `/navigate` (std_srvs/srv/Empty) - Triggers navigation (placeholder)
- **Topics Published**:
  - `/planned_path` - Single segment paths
  - `/combined_path` - Multi-waypoint combined paths
  - `/waypoints` - Recorded waypoint array
- **Topics Subscribed**:
  - `/map` - Occupancy grid
  - `/clicked_point` - RViz clicked points (when recording)
  - `/waypoints` - Waypoint array input
- **Key Features**:
  - Multi-waypoint path planning with smooth transitions
  - Interactive waypoint recording via RViz
  - Robot footprint collision checking (3.6m x 2.4m rectangle)
  - Natural orientation handling between waypoints

### Path Following Controller (`path_following_controller`)
- **Algorithm**: Pure Pursuit with dynamic adaptations
- **Input**: Paths from `/planned_path` or `/combined_path`, robot pose from TF
- **Output**: Velocity commands to `/cmd_vel`
- **Key Features**:
  - Subscribes to both `/planned_path` and `/combined_path` for compatibility
  - Dynamic lookahead distance adjustment (1.0m default)
  - Velocity scaling for turns and near goals
  - Turn-in-place for large heading errors (>90°)
  - Path validation to handle dynamic map updates

## Launch System

Three launch configurations available:
1. **Path planning only**: `ros2 launch triton_navigation navigation.launch.xml`
2. **Full navigation**: `ros2 launch triton_navigation navigation_with_controller.launch.xml`
3. **Controller only**: `ros2 launch triton_navigation path_following.launch.xml`

All launch files expect:
- Map file in `maps/` directory (currently uses `map_home.pgm`)
- Configuration files in `config/` directory
- Nav2 lifecycle manager for proper startup sequencing

## Configuration Management

### Robot Footprint (`config/footprint.yaml`)
```yaml
footprint: [[-1.8, -1.2], [1.8, -1.2], [1.8, 1.2], [-1.8, 1.2]]  # 3.6m x 2.4m rectangle
planning_time: 10.0
planner_type: "RRTConnect"  # Options: RRT, RRTConnect, RRTstar, EST, PRM
treat_unknown_as_free: true
```

### Path Following (`config/path_following.yaml`)
```yaml
lookahead_distance: 1.0
max_linear_velocity: 1.0
max_angular_velocity: 2.5
min_linear_velocity: 0.1
goal_tolerance: 0.5
control_frequency: 10.0
```

## Key Implementation Details

### Path Planning (`src/path_planning.cpp`)
- Uses OMPL's `RealVectorStateSpace` for 2D planning
- Collision checking via `GridCollisionChecker` with robot footprint
- Simplifies paths before publishing to reduce waypoints
- Thread-safe map updates with mutex protection

### Path Following (`src/path_following_controller.cpp`)
- Pure Pursuit with adaptive lookahead based on path curvature
- Velocity command calculation:
  ```cpp
  angular_vel = 2 * linear_vel * sin(steering_angle) / lookahead_distance
  ```
- Path validation checks:
  - Path age (rejects if >10 seconds old)
  - Path-map consistency (rejects if path goes through obstacles)
  - Minimum path length (needs at least 2 points)

## Common Development Tasks

### Multi-Waypoint Navigation Workflow
```bash
# 1. Launch navigation stack
ros2 launch triton_navigation navigation_with_controller.launch.xml

# 2. Start recording waypoints
ros2 service call /record_waypoints std_srvs/srv/Empty

# 3. Click points in RViz using "Publish Point" tool

# 4. Save the waypoints
ros2 service call /save_waypoints std_srvs/srv/Empty

# 5. Generate combined path through all waypoints
ros2 service call /generate_path std_srvs/srv/Empty

# 6. Start navigation (automatic when path is published)
ros2 service call /navigate std_srvs/srv/Empty
```

### Testing Single Path Planning
```bash
# Launch with visualization
ros2 launch triton_navigation navigation.launch.xml
# Use the GetPlan service directly
ros2 service call /get_path nav_msgs/srv/GetPlan "{start: {pose: {position: {x: 0.0, y: 0.0}}}, goal: {pose: {position: {x: 5.0, y: 5.0}}}}"
```

### Testing Path Following
```bash
# Launch full navigation
ros2 launch triton_navigation navigation_with_controller.launch.xml
# Monitor velocity commands
ros2 topic echo /cmd_vel
# Check navigation status
ros2 topic echo /navigation_active
```

### Debugging Tips
- Enable debug logging: `ros2 run triton_navigation path_planning --ros-args --log-level debug`
- Visualize footprint: Check `/footprint_marker` in RViz
- Monitor planning time: Look for "Path found in X seconds" messages
- Path validation failures: Check for "Path validation failed" messages

## Integration Points

This package integrates with:
- **Nav2**: Uses map_server for occupancy grids, follows Nav2 conventions
- **SLAM packages**: Works with any SLAM that publishes `/map` (gmapping, slam_toolbox)
- **triton_bringup**: Launched as part of the navigation stack
- **Robot hardware**: Outputs standard `/cmd_vel` messages for motor control

The path planner and controller are designed to work with dynamic maps from SLAM, handling map updates gracefully without causing navigation failures.