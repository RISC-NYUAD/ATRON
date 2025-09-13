#include <memory>
#include <chrono>
#include <functional>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <algorithm>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/srv/get_plan.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_srvs/srv/empty.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "triton_navigation/path_planning.hpp"

using std::placeholders::_1;
using namespace std::chrono_literals;

class NavigationNode : public rclcpp::Node
{
private:
    // Subscriptions
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pose_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr clicked_point_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr waypoints_subscription_;
    rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr combined_markers_subscription_;
    
    // Publishers
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr footprint_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr footprint_array_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr waypoints_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr combined_path_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr obstacle_footprint_publisher_;
    
    // Services
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr generate_path_service_;
    rclcpp::Service<nav_msgs::srv::GetPlan>::SharedPtr get_path_service_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr record_waypoints_service_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr save_waypoints_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr filter_waypoints_service_;
    // Navigate service moved to path_following_controller
    // rclcpp::Service<std_srvs::srv::Empty>::SharedPtr navigate_service_;
    
    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
    
    // TF2
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // Data members
    nav_msgs::msg::OccupancyGrid global_map_;
    path_planning::Planner2D planner_;
    bool map_received_;
    bool should_plan_path_;
    
    // Waypoint recording state
    bool recording_waypoints_;
    std::vector<geometry_msgs::msg::Point> recorded_points_;
    geometry_msgs::msg::PoseArray current_waypoints_;
    
    // Footprint parameters
    std::vector<double> footprint_vertices_;
    geometry_msgs::msg::Polygon footprint_polygon_;
    
    // Store current start position and orientation for footprint visualization
    double start_x_, start_y_, start_theta_;
    bool start_position_set_;
    
    // Store current goal position and orientation
    double goal_x_, goal_y_, goal_theta_;
    bool goal_position_set_;
    
    // Track last published footprint array size for cleanup
    size_t last_footprint_array_size_;
    
    // Control automatic path planning
    bool enable_automatic_planning_;
    
    // Storage for cylinder obstacles from markers
    visualization_msgs::msg::MarkerArray latest_combined_markers_;
    std::mutex markers_mutex_;
    
    void createFootprintPolygon() {
        footprint_polygon_.points.clear();
        
        // Create polygon from vertices (x1, y1, x2, y2, ...)
        for (size_t i = 0; i < footprint_vertices_.size(); i += 2) {
            if (i + 1 < footprint_vertices_.size()) {
                geometry_msgs::msg::Point32 point;
                point.x = footprint_vertices_[i];
                point.y = footprint_vertices_[i + 1];
                point.z = 0.0;
                footprint_polygon_.points.push_back(point);
            }
        }
    }
    
    geometry_msgs::msg::PolygonStamped createFootprintAtPosition(double x, double y, double theta = 0.0) {
        geometry_msgs::msg::PolygonStamped footprint_stamped;
        footprint_stamped.header.frame_id = "map";
        footprint_stamped.header.stamp = this->get_clock()->now();
        
        // Create a copy of the base footprint and transform it to the desired position and orientation
        footprint_stamped.polygon.points.clear();
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        
        for (const auto& point : footprint_polygon_.points) {
            geometry_msgs::msg::Point32 transformed_point;
            // Rotate then translate
            transformed_point.x = (point.x * cos_theta - point.y * sin_theta) + x;
            transformed_point.y = (point.x * sin_theta + point.y * cos_theta) + y;
            transformed_point.z = point.z;
            footprint_stamped.polygon.points.push_back(transformed_point);
        }
        
        return footprint_stamped;
    }
    
    visualization_msgs::msg::Marker createFootprintMarker(double x, double y, double theta, int id) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "footprint_array";
        marker.id = id;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        
        marker.pose.position.x = 0;
        marker.pose.position.y = 0;
        marker.pose.position.z = 0.01;
        marker.pose.orientation.w = 1.0;
        
        marker.scale.x = 0.05; // Line width
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 0.8;
        
        // Add footprint points with rotation
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        
        for (const auto& point : footprint_polygon_.points) {
            geometry_msgs::msg::Point p;
            // Rotate then translate
            p.x = (point.x * cos_theta - point.y * sin_theta) + x;
            p.y = (point.x * sin_theta + point.y * cos_theta) + y;
            p.z = 0.01;
            marker.points.push_back(p);
        }
        
        // Close the polygon by adding the first point again
        if (!footprint_polygon_.points.empty()) {
            geometry_msgs::msg::Point p;
            p.x = (footprint_polygon_.points[0].x * cos_theta - footprint_polygon_.points[0].y * sin_theta) + x;
            p.y = (footprint_polygon_.points[0].x * sin_theta + footprint_polygon_.points[0].y * cos_theta) + y;
            p.z = 0.01;
            marker.points.push_back(p);
        }
        
        return marker;
    }
    
    void clearFootprintArray() {
        if (last_footprint_array_size_ > 0) {
            visualization_msgs::msg::MarkerArray clear_array;
            
            // Create DELETE markers for all previously published markers
            for (size_t i = 0; i < last_footprint_array_size_; ++i) {
                visualization_msgs::msg::Marker delete_marker;
                delete_marker.header.frame_id = "map";
                delete_marker.header.stamp = this->get_clock()->now();
                delete_marker.ns = "footprint_array";
                delete_marker.id = static_cast<int>(i);
                delete_marker.action = visualization_msgs::msg::Marker::DELETE;
                clear_array.markers.push_back(delete_marker);
            }
            
            footprint_array_publisher_->publish(clear_array);
            last_footprint_array_size_ = 0;
            RCLCPP_INFO(this->get_logger(), "Cleared previous footprint markers");
        }
    }
    
    void publishFootprintArray(const nav_msgs::msg::Path& planned_path) {
        // Clear previous markers first
        clearFootprintArray();
        
        if (!planned_path.poses.empty()) {
            visualization_msgs::msg::MarkerArray footprint_array;
            
            // Create footprint markers at each pose in the path
            for (size_t i = 0; i < planned_path.poses.size(); ++i) {
                double x = planned_path.poses[i].pose.position.x;
                double y = planned_path.poses[i].pose.position.y;
                
                // Extract orientation from quaternion
                double qx = planned_path.poses[i].pose.orientation.x;
                double qy = planned_path.poses[i].pose.orientation.y;
                double qz = planned_path.poses[i].pose.orientation.z;
                double qw = planned_path.poses[i].pose.orientation.w;
                double theta = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
                
                // Debug: Log every 10th pose to avoid spam
                if (i % 10 == 0 || i < 3 || i == planned_path.poses.size() - 1) {
                    RCLCPP_INFO(this->get_logger(), "Footprint %zu: pos=(%.2f, %.2f), theta=%.2f rad (%.1f deg), quat=(%.3f,%.3f,%.3f,%.3f)", 
                                i, x, y, theta, theta * 180.0 / M_PI, qx, qy, qz, qw);
                }
                
                visualization_msgs::msg::Marker footprint_marker = createFootprintMarker(x, y, theta, static_cast<int>(i));
                footprint_array.markers.push_back(footprint_marker);
            }
            
            footprint_array_publisher_->publish(footprint_array);
            last_footprint_array_size_ = footprint_array.markers.size();
            RCLCPP_INFO(this->get_logger(), "Published footprint array with %zu markers", footprint_array.markers.size());
        }
    }
    
    void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        global_map_ = *msg;
        map_received_ = true;
        should_plan_path_ = true;
        RCLCPP_INFO(this->get_logger(), "Map received");
    }
    
    void initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;
        
        // Extract orientation from quaternion
        double qx = msg->pose.pose.orientation.x;
        double qy = msg->pose.pose.orientation.y;
        double qz = msg->pose.pose.orientation.z;
        double qw = msg->pose.pose.orientation.w;
        double theta = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
        
        start_x_ = x;
        start_y_ = y;
        start_theta_ = theta;
        start_position_set_ = true;
        planner_.setStartPosition(x, y, theta);
        // Removed automatic planning trigger - now only via services
        RCLCPP_INFO(this->get_logger(), "Initial pose set: (%.2f, %.2f, %.2f)", x, y, theta);
    }
    
    void goal_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        double x = msg->pose.position.x;
        double y = msg->pose.position.y;
        
        // Extract orientation from quaternion
        double qx = msg->pose.orientation.x;
        double qy = msg->pose.orientation.y;
        double qz = msg->pose.orientation.z;
        double qw = msg->pose.orientation.w;
        double theta = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
        
        goal_x_ = x;
        goal_y_ = y;
        goal_theta_ = theta;
        goal_position_set_ = true;
        planner_.setGoalPosition(x, y, theta);
        // Removed automatic planning trigger - now only via services
        RCLCPP_INFO(this->get_logger(), "Goal pose set: (%.2f, %.2f, %.2f)", x, y, theta);
    }
    
    void timer_callback() {
        // Always try to publish footprint at robot's current position
        try {
            geometry_msgs::msg::TransformStamped transform = 
                tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
            
            double robot_x = transform.transform.translation.x;
            double robot_y = transform.transform.translation.y;
            
            // Extract yaw from quaternion
            double qx = transform.transform.rotation.x;
            double qy = transform.transform.rotation.y;
            double qz = transform.transform.rotation.z;
            double qw = transform.transform.rotation.w;
            double robot_theta = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
            
            geometry_msgs::msg::PolygonStamped footprint_at_robot = createFootprintAtPosition(robot_x, robot_y, robot_theta);
            footprint_publisher_->publish(footprint_at_robot);
            
        } catch (tf2::TransformException& ex) {
            // If we can't get robot position from TF, fall back to start position if available
            if (start_position_set_) {
                geometry_msgs::msg::PolygonStamped footprint_at_start = createFootprintAtPosition(start_x_, start_y_, start_theta_);
                footprint_publisher_->publish(footprint_at_start);
            }
        }
        
        // Plan and publish path if conditions are met and automatic planning is enabled
        if (enable_automatic_planning_ && map_received_ && planner_.isReadyToPlan() && should_plan_path_) {
            RCLCPP_INFO(this->get_logger(), "Planning path...");
            
            nav_msgs::msg::Path planned_path = planner_.planPath(global_map_);
            planned_path.header.stamp = this->get_clock()->now();
            path_publisher_->publish(planned_path);
            
            // Create and publish footprint array along the path
            publishFootprintArray(planned_path);
            
            should_plan_path_ = false; // Don't plan again until something changes
            RCLCPP_INFO(this->get_logger(), "Path published with %zu poses", planned_path.poses.size());
        }
    }
    
    void getPathCallback(const std::shared_ptr<nav_msgs::srv::GetPlan::Request> request,
                        std::shared_ptr<nav_msgs::srv::GetPlan::Response> response) {
        if (!map_received_) {
            RCLCPP_WARN(this->get_logger(), "GetPlan: No map received");
            return;
        }
        
        // Set start and goal in planner
        double start_yaw = std::atan2(2.0 * (request->start.pose.orientation.w * request->start.pose.orientation.z + 
                                             request->start.pose.orientation.x * request->start.pose.orientation.y),
                                      1.0 - 2.0 * (request->start.pose.orientation.y * request->start.pose.orientation.y + 
                                                   request->start.pose.orientation.z * request->start.pose.orientation.z));
        
        double goal_yaw = std::atan2(2.0 * (request->goal.pose.orientation.w * request->goal.pose.orientation.z + 
                                            request->goal.pose.orientation.x * request->goal.pose.orientation.y),
                                     1.0 - 2.0 * (request->goal.pose.orientation.y * request->goal.pose.orientation.y + 
                                                  request->goal.pose.orientation.z * request->goal.pose.orientation.z));
        
        planner_.setStartPosition(request->start.pose.position.x, 
                                 request->start.pose.position.y, 
                                 start_yaw);
        planner_.setGoalPosition(request->goal.pose.position.x, 
                                request->goal.pose.position.y, 
                                goal_yaw);
        
        // Plan path
        if (planner_.isReadyToPlan()) {
            response->plan = planner_.planPath(global_map_);
            response->plan.header.stamp = this->get_clock()->now();
            response->plan.header.frame_id = "map";
            RCLCPP_INFO(this->get_logger(), "GetPlan: Path found with %zu poses", response->plan.poses.size());
        } else {
            RCLCPP_ERROR(this->get_logger(), "GetPlan: Planner not ready");
        }
    }
    
    void recordWaypointsCallback(const std::shared_ptr<std_srvs::srv::Empty::Request> request,
                                std::shared_ptr<std_srvs::srv::Empty::Response> response) {
        (void)request;
        (void)response;
        
        recording_waypoints_ = true;
        recorded_points_.clear();
        
        // Create subscription to clicked points if not already created
        if (!clicked_point_subscription_) {
            clicked_point_subscription_ = this->create_subscription<geometry_msgs::msg::PointStamped>(
                "/clicked_point", 10, 
                std::bind(&NavigationNode::clickedPointCallback, this, _1));
        }
        
        RCLCPP_INFO(this->get_logger(), "Started recording waypoints. Click points in RViz.");
    }
    
    void saveWaypointsCallback(const std::shared_ptr<std_srvs::srv::Empty::Request> request,
                              std::shared_ptr<std_srvs::srv::Empty::Response> response) {
        (void)request;
        (void)response;
        
        recording_waypoints_ = false;
        
        // Convert recorded points to PoseArray
        geometry_msgs::msg::PoseArray waypoints;
        waypoints.header.stamp = this->get_clock()->now();
        waypoints.header.frame_id = "map";
        
        for (const auto& point : recorded_points_) {
            geometry_msgs::msg::Pose pose;
            pose.position = point;
            pose.orientation.w = 1.0;  // Identity quaternion
            waypoints.poses.push_back(pose);
        }
        
        // Publish and store waypoints
        waypoints_publisher_->publish(waypoints);
        current_waypoints_ = waypoints;
        
        RCLCPP_INFO(this->get_logger(), "Saved %zu waypoints", waypoints.poses.size());
    }
    
    void clickedPointCallback(const geometry_msgs::msg::PointStamped::SharedPtr msg) {
        if (!recording_waypoints_) {
            return;
        }
        
        recorded_points_.push_back(msg->point);
        RCLCPP_INFO(this->get_logger(), "Recorded waypoint %zu at (%.2f, %.2f)", 
                   recorded_points_.size(), msg->point.x, msg->point.y);
    }
    
    void waypointsCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg) {
        current_waypoints_ = *msg;
        RCLCPP_INFO(this->get_logger(), "Received %zu waypoints", current_waypoints_.poses.size());
    }
    
    geometry_msgs::msg::PoseStamped getCurrentRobotPose() {
        geometry_msgs::msg::PoseStamped robot_pose;
        robot_pose.header.frame_id = "map";
        robot_pose.header.stamp = this->get_clock()->now();
        
        try {
            geometry_msgs::msg::TransformStamped transform = 
                tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
            
            robot_pose.pose.position.x = transform.transform.translation.x;
            robot_pose.pose.position.y = transform.transform.translation.y;
            robot_pose.pose.position.z = transform.transform.translation.z;
            robot_pose.pose.orientation = transform.transform.rotation;
            
        } catch (tf2::TransformException& ex) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get robot pose: %s", ex.what());
            // Return identity pose if transform fails
            robot_pose.pose.orientation.w = 1.0;
        }
        
        return robot_pose;
    }
    
    double getYawFromQuaternion(const geometry_msgs::msg::Quaternion& q) {
        return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    }
    
    void generatePathCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                            std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        (void)request;
        
        if (!map_received_) {
            response->success = false;
            response->message = "Cannot generate path: No map received";
            RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
            return;
        }
        
        if (current_waypoints_.poses.empty()) {
            response->success = false;
            response->message = "Cannot generate path: No waypoints set";
            RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
            return;
        }
        
        // Extract cylinder obstacles from the latest combined markers
        auto node_obstacles = extractCylinderObstacles();
        std::vector<path_planning::CylinderObstacle> planner_obstacles;
        {
            for (const auto& obs : node_obstacles) {
                path_planning::CylinderObstacle planner_obs;
                planner_obs.x = obs.x;
                planner_obs.y = obs.y;
                planner_obs.radius = obs.radius;
                planner_obstacles.push_back(planner_obs);
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Planning with %zu cylinder obstacles from combined markers", 
                    planner_obstacles.size());
        
        // Publish obstacle footprints for visualization
        publishObstacleFootprints(node_obstacles);
        
        // Get robot's current pose
        geometry_msgs::msg::PoseStamped current_start = getCurrentRobotPose();
        
        nav_msgs::msg::Path combined_path;
        combined_path.header.stamp = this->get_clock()->now();
        combined_path.header.frame_id = "map";
        
        std::vector<size_t> skipped_waypoints;
        std::vector<geometry_msgs::msg::Point> selected_waypoints;
        size_t successful_segments = 0;
        
        // Keep track of successfully reached positions for backtracking
        std::vector<geometry_msgs::msg::PoseStamped> reached_positions;
        reached_positions.push_back(current_start);  // Add robot's starting position
        
        // Plan path through all waypoints
        size_t i = 0;
        while (i < current_waypoints_.poses.size()) {
            bool segment_planned = false;
            size_t attempts_from_current = 0;
            
            // First, validate the current start position
            double start_x = current_start.pose.position.x;
            double start_y = current_start.pose.position.y;
            double start_theta = getYawFromQuaternion(current_start.pose.orientation);
            
            // Update planner with current map and obstacles before validation
            planner_.setMapAndObstacles(global_map_, planner_obstacles);
            
            if (!planner_.isStateValid(start_x, start_y, start_theta)) {
                RCLCPP_ERROR(this->get_logger(), 
                            "Current start position (%.2f, %.2f) is invalid (in collision). Attempting to backtrack.", 
                            start_x, start_y);
                
                // Try to backtrack to a previous valid position
                bool found_valid_start = false;
                for (int back_idx = reached_positions.size() - 2; back_idx >= 0; --back_idx) {
                    current_start = reached_positions[back_idx];
                    double back_x = current_start.pose.position.x;
                    double back_y = current_start.pose.position.y;
                    double back_theta = getYawFromQuaternion(current_start.pose.orientation);
                    
                    if (planner_.isStateValid(back_x, back_y, back_theta)) {
                        RCLCPP_INFO(this->get_logger(), 
                                   "Backtracked to valid position (%.2f, %.2f) from reached position %d", 
                                   back_x, back_y, back_idx);
                        found_valid_start = true;
                        // Remove positions after the backtrack point
                        reached_positions.erase(reached_positions.begin() + back_idx + 1, reached_positions.end());
                        break;
                    }
                }
                
                if (!found_valid_start) {
                    RCLCPP_ERROR(this->get_logger(), "No valid start position found even after backtracking. Stopping.");
                    break;
                }
                
                start_x = current_start.pose.position.x;
                start_y = current_start.pose.position.y;
                start_theta = getYawFromQuaternion(current_start.pose.orientation);
            }
            
            // Try to plan to waypoint i and beyond, skipping intermediate waypoints if necessary
            for (size_t target = i; target < current_waypoints_.poses.size() && !segment_planned; ++target) {
                attempts_from_current++;
                
                // First check if the target waypoint itself is valid
                double target_x = current_waypoints_.poses[target].position.x;
                double target_y = current_waypoints_.poses[target].position.y;
                
                // Check multiple orientations for the target waypoint
                bool target_valid = false;
                const int num_orientations = 8;
                for (int orient = 0; orient < num_orientations; ++orient) {
                    double test_theta = (2.0 * M_PI * orient) / num_orientations;
                    if (planner_.isStateValid(target_x, target_y, test_theta)) {
                        target_valid = true;
                        break;
                    }
                }
                
                if (!target_valid) {
                    RCLCPP_WARN(this->get_logger(), 
                               "Waypoint %zu at (%.2f, %.2f) is invalid (in collision), skipping to next waypoint", 
                               target, target_x, target_y);
                    if (std::find(skipped_waypoints.begin(), skipped_waypoints.end(), target) == skipped_waypoints.end()) {
                        skipped_waypoints.push_back(target);
                    }
                    continue;
                }
                
                // Create goal pose with waypoint position
                geometry_msgs::msg::PoseStamped goal;
                goal.header = current_waypoints_.header;
                goal.pose.position = current_waypoints_.poses[target].position;
                goal.pose.orientation.w = 1.0;  // Will be determined by planner
                
                // Set planner start and goal
                planner_.setStartPosition(start_x, start_y, start_theta);
                planner_.setGoalPosition(target_x, target_y);
                
                // Plan segment
                if (!planner_.isReadyToPlan()) {
                    RCLCPP_WARN(this->get_logger(), "Planner not ready for waypoint %zu", target);
                    continue;
                }
                
                // Publish a visualization of what we're attempting (start to goal as straight line)
                nav_msgs::msg::Path attempt_visualization;
                attempt_visualization.header.stamp = this->get_clock()->now();
                attempt_visualization.header.frame_id = "map";
                
                // Add start point
                geometry_msgs::msg::PoseStamped start_pose;
                start_pose.header = attempt_visualization.header;
                start_pose.pose.position.x = start_x;
                start_pose.pose.position.y = start_y;
                start_pose.pose.position.z = 0.01;
                start_pose.pose.orientation.w = 1.0;
                attempt_visualization.poses.push_back(start_pose);
                
                // Add goal point
                geometry_msgs::msg::PoseStamped goal_pose;
                goal_pose.header = attempt_visualization.header;
                goal_pose.pose.position.x = target_x;
                goal_pose.pose.position.y = target_y;
                goal_pose.pose.position.z = 0.01;
                goal_pose.pose.orientation.w = 1.0;
                attempt_visualization.poses.push_back(goal_pose);
                
                // Publish the attempt visualization briefly
                path_publisher_->publish(attempt_visualization);
                rclcpp::sleep_for(std::chrono::milliseconds(50));
                
                nav_msgs::msg::Path segment = planner_.planPath(global_map_, planner_obstacles);
                
                if (!segment.poses.empty()) {
                    // Successfully planned to waypoint target
                    RCLCPP_INFO(this->get_logger(), "Successfully planned segment to waypoint %zu at (%.2f, %.2f)", 
                               target, target_x, target_y);
                    
                    // Publish this segment to /planned_path for real-time visualization
                    segment.header.stamp = this->get_clock()->now();
                    segment.header.frame_id = "map";
                    path_publisher_->publish(segment);
                    
                    // Add selected waypoint to list
                    selected_waypoints.push_back(current_waypoints_.poses[target].position);
                    
                    // Mark any skipped waypoints (those between i and target, excluding target)
                    for (size_t j = i; j < target; ++j) {
                        if (std::find(skipped_waypoints.begin(), skipped_waypoints.end(), j) == skipped_waypoints.end()) {
                            skipped_waypoints.push_back(j);
                            RCLCPP_WARN(this->get_logger(), "Skipped intermediate waypoint %zu", j);
                        }
                    }
                    
                    // Combine paths
                    if (combined_path.poses.empty()) {
                        // First segment - include all points
                        combined_path.poses = segment.poses;
                    } else {
                        // Subsequent segments - skip first point to avoid duplicate
                        for (size_t j = 1; j < segment.poses.size(); ++j) {
                            combined_path.poses.push_back(segment.poses[j]);
                        }
                    }
                    
                    // Next segment starts where this one ends
                    current_start = segment.poses.back();
                    reached_positions.push_back(current_start);  // Save this position for potential backtracking
                    segment_planned = true;
                    successful_segments++;
                    i = target + 1;  // Move to the next waypoint after the one we reached
                    
                    // Small delay for visualization (100ms)
                    rclcpp::sleep_for(std::chrono::milliseconds(100));
                } else {
                    RCLCPP_WARN(this->get_logger(), 
                               "Failed to plan segment to waypoint %zu at (%.2f, %.2f)", 
                               target, target_x, target_y);
                }
            }
            
            // If we couldn't plan to any remaining waypoint from current position
            if (!segment_planned) {
                RCLCPP_WARN(this->get_logger(), 
                           "Could not plan from position (%.2f, %.2f) to any waypoints starting from index %zu (tried %zu waypoints)", 
                           start_x, start_y, i, attempts_from_current);
                
                // Mark all remaining waypoints as skipped
                for (size_t j = i; j < current_waypoints_.poses.size(); ++j) {
                    if (std::find(skipped_waypoints.begin(), skipped_waypoints.end(), j) == skipped_waypoints.end()) {
                        skipped_waypoints.push_back(j);
                        RCLCPP_INFO(this->get_logger(), 
                                   "Marking remaining waypoint %zu at (%.2f, %.2f) as unreachable", 
                                   j, current_waypoints_.poses[j].position.x, 
                                   current_waypoints_.poses[j].position.y);
                    }
                }
                break;  // No more waypoints can be reached
            }
        }
        
        // Only publish if we have at least one successful segment
        if (!combined_path.poses.empty()) {
            combined_path_publisher_->publish(combined_path);
            
            size_t reached_waypoints = current_waypoints_.poses.size() - skipped_waypoints.size();
            
            // Build response message with selected waypoint coordinates and skip reasons
            std::stringstream ss;
            ss << "Generated path through " << selected_waypoints.size() << "/" 
               << current_waypoints_.poses.size() << " waypoints.";
            
            if (!selected_waypoints.empty()) {
                ss << " Selected: [";
                for (size_t i = 0; i < selected_waypoints.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << "(" << std::fixed << std::setprecision(2) 
                       << selected_waypoints[i].x << ", " 
                       << selected_waypoints[i].y << ")";
                }
                ss << "]";
            }
            
            if (!skipped_waypoints.empty()) {
                ss << " Skipped " << skipped_waypoints.size() << " waypoints due to collisions or unreachability.";
            }
            
            response->success = true;
            response->message = ss.str();
            
            RCLCPP_INFO(this->get_logger(), 
                       "Generated combined path with %zu poses through %zu/%zu waypoints (%zu skipped)",
                       combined_path.poses.size(), reached_waypoints, 
                       current_waypoints_.poses.size(), skipped_waypoints.size());
            
            // Log detailed skip information
            if (!skipped_waypoints.empty()) {
                std::stringstream skip_details;
                skip_details << "Skipped waypoints: ";
                for (size_t idx : skipped_waypoints) {
                    if (idx < current_waypoints_.poses.size()) {
                        skip_details << "[" << idx << "](" 
                                    << std::fixed << std::setprecision(2)
                                    << current_waypoints_.poses[idx].position.x << ", "
                                    << current_waypoints_.poses[idx].position.y << ") ";
                    }
                }
                RCLCPP_WARN(this->get_logger(), "%s", skip_details.str().c_str());
            }
            
            // Publish footprint array along the combined path
            publishFootprintArray(combined_path);
        } else {
            response->success = false;
            response->message = "Failed to generate any valid path segments";
            RCLCPP_ERROR(this->get_logger(), "%s", response->message.c_str());
        }
    }
    
    void navigateCallback(const std::shared_ptr<std_srvs::srv::Empty::Request> request,
                         std::shared_ptr<std_srvs::srv::Empty::Response> response) {
        (void)request;
        (void)response;
        
        // The path following controller will automatically start following
        // the combined path when it's published
        RCLCPP_INFO(this->get_logger(), "Navigate service called - path following should start automatically");
    }
    
    void filterWaypointsCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        (void)request;
        
        if (!map_received_) {
            response->success = false;
            response->message = "Cannot filter waypoints: No map received";
            RCLCPP_WARN(this->get_logger(), "%s", response->message.c_str());
            return;
        }
        
        if (current_waypoints_.poses.empty()) {
            response->success = false;
            response->message = "No waypoints to filter";
            RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
            return;
        }
        
        // Extract cylinder obstacles
        auto node_obstacles = extractCylinderObstacles();
        std::vector<path_planning::CylinderObstacle> planner_obstacles;
        for (const auto& obs : node_obstacles) {
            path_planning::CylinderObstacle planner_obs;
            planner_obs.x = obs.x;
            planner_obs.y = obs.y;
            planner_obs.radius = obs.radius;
            planner_obstacles.push_back(planner_obs);
        }
        
        // Update planner with current map and obstacles
        planner_.setMapAndObstacles(global_map_, planner_obstacles);
        
        RCLCPP_INFO(this->get_logger(), "Filtering waypoints with %zu cylinder obstacles", 
                    planner_obstacles.size());
        
        // Filter waypoints
        geometry_msgs::msg::PoseArray filtered_waypoints;
        filtered_waypoints.header = current_waypoints_.header;
        filtered_waypoints.header.stamp = this->get_clock()->now();
        
        size_t removed_count = 0;
        const int num_orientations = 8;  // Check 8 different orientations (every 45 degrees)
        
        for (size_t i = 0; i < current_waypoints_.poses.size(); ++i) {
            double waypoint_x = current_waypoints_.poses[i].position.x;
            double waypoint_y = current_waypoints_.poses[i].position.y;
            
            bool waypoint_valid = false;
            
            // Check multiple orientations to see if any is valid
            for (int j = 0; j < num_orientations; ++j) {
                double test_theta = (2.0 * M_PI * j) / num_orientations;
                if (planner_.isStateValid(waypoint_x, waypoint_y, test_theta)) {
                    waypoint_valid = true;
                    break;
                }
            }
            
            if (waypoint_valid) {
                // Keep this waypoint
                filtered_waypoints.poses.push_back(current_waypoints_.poses[i]);
                RCLCPP_DEBUG(this->get_logger(), "Waypoint %zu at (%.2f, %.2f) is valid", 
                            i, waypoint_x, waypoint_y);
            } else {
                // Remove this waypoint
                removed_count++;
                RCLCPP_WARN(this->get_logger(), "Removing waypoint %zu at (%.2f, %.2f) - collision detected", 
                           i, waypoint_x, waypoint_y);
            }
        }
        
        // Update and publish filtered waypoints
        current_waypoints_ = filtered_waypoints;
        waypoints_publisher_->publish(filtered_waypoints);
        
        // Prepare response
        std::stringstream ss;
        ss << "Filtered waypoints: " << removed_count << " removed, " 
           << filtered_waypoints.poses.size() << " remaining";
        
        response->success = true;
        response->message = ss.str();
        
        RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
    }
    
    void combinedMarkersCallback(const visualization_msgs::msg::MarkerArray::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(markers_mutex_);
        latest_combined_markers_ = *msg;
        RCLCPP_DEBUG(this->get_logger(), "Received %zu combined markers", msg->markers.size());
    }
    
    // Structure to hold cylinder obstacle data
    struct CylinderObstacle {
        double x, y;     // Center position
        double radius;   // Radius of the cylinder
    };
    
    std::vector<CylinderObstacle> extractCylinderObstacles() {
        std::lock_guard<std::mutex> lock(markers_mutex_);
        std::vector<CylinderObstacle> obstacles;
        
        for (const auto& marker : latest_combined_markers_.markers) {
            // Check if marker is a cylinder (type 3)
            if (marker.type == visualization_msgs::msg::Marker::CYLINDER) {
                CylinderObstacle obstacle;
                obstacle.x = marker.pose.position.x;
                obstacle.y = marker.pose.position.y;
                // Use the larger of x or y scale as radius (cylinders should have equal x,y scale)
                obstacle.radius = std::max(marker.scale.x, marker.scale.y) / 2.0;
                obstacles.push_back(obstacle);
                
                RCLCPP_DEBUG(this->get_logger(), "Extracted cylinder obstacle at (%.2f, %.2f) with radius %.2f",
                            obstacle.x, obstacle.y, obstacle.radius);
            }
        }
        
        return obstacles;
    }
    
    void publishObstacleFootprints(const std::vector<CylinderObstacle>& obstacles) {
        visualization_msgs::msg::MarkerArray obstacle_markers;
        
        // Clear previous markers
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.header.frame_id = "map";
        clear_marker.header.stamp = this->get_clock()->now();
        clear_marker.ns = "obstacle_footprints";
        clear_marker.id = 0;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        obstacle_markers.markers.push_back(clear_marker);
        
        int marker_id = 0;
        for (const auto& obstacle : obstacles) {
            // Create cylinder footprint as a polygon
            visualization_msgs::msg::Marker cylinder_marker;
            cylinder_marker.header.frame_id = "map";
            cylinder_marker.header.stamp = this->get_clock()->now();
            cylinder_marker.ns = "obstacle_footprints";
            cylinder_marker.id = marker_id++;
            cylinder_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            cylinder_marker.action = visualization_msgs::msg::Marker::ADD;
            
            cylinder_marker.pose.position.x = 0;
            cylinder_marker.pose.position.y = 0;
            cylinder_marker.pose.position.z = 0.02;  // Slightly above ground
            cylinder_marker.pose.orientation.w = 1.0;
            
            cylinder_marker.scale.x = 0.05;  // Line width
            cylinder_marker.color.r = 1.0;   // Red color for obstacles
            cylinder_marker.color.g = 0.0;
            cylinder_marker.color.b = 0.0;
            cylinder_marker.color.a = 0.8;
            
            // Generate circular polygon points
            const int num_points = 20;  // Number of points to approximate circle
            for (int i = 0; i <= num_points; ++i) {
                double angle = 2.0 * M_PI * i / num_points;
                geometry_msgs::msg::Point p;
                p.x = obstacle.x + obstacle.radius * std::cos(angle);
                p.y = obstacle.y + obstacle.radius * std::sin(angle);
                p.z = 0.02;
                cylinder_marker.points.push_back(p);
            }
            
            obstacle_markers.markers.push_back(cylinder_marker);
            
            // Add safety margin visualization
            visualization_msgs::msg::Marker margin_marker;
            margin_marker.header = cylinder_marker.header;
            margin_marker.ns = "obstacle_margins";
            margin_marker.id = marker_id++;
            margin_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
            margin_marker.action = visualization_msgs::msg::Marker::ADD;
            
            margin_marker.pose = cylinder_marker.pose;
            margin_marker.scale.x = 0.03;  // Thinner line
            margin_marker.color.r = 1.0;   // Purple color for safety margin
            margin_marker.color.g = 0.0;
            margin_marker.color.b = 1.0;
            margin_marker.color.a = 0.5;
            
            // Generate safety margin polygon
            double margin_radius = obstacle.radius + planner_.getSafetyMargin();
            for (int i = 0; i <= num_points; ++i) {
                double angle = 2.0 * M_PI * i / num_points;
                geometry_msgs::msg::Point p;
                p.x = obstacle.x + margin_radius * std::cos(angle);
                p.y = obstacle.y + margin_radius * std::sin(angle);
                p.z = 0.02;
                margin_marker.points.push_back(p);
            }
            
            obstacle_markers.markers.push_back(margin_marker);
        }
        
        obstacle_footprint_publisher_->publish(obstacle_markers);
        RCLCPP_INFO(this->get_logger(), "Published %zu obstacle footprints with safety margins", obstacles.size());
    }

public:
    NavigationNode() : Node("path_planning_node"), map_received_(false), should_plan_path_(false), 
                      start_position_set_(false), goal_position_set_(false), last_footprint_array_size_(0), 
                      enable_automatic_planning_(false), recording_waypoints_(false) {
        // Declare parameters
        this->declare_parameter("footprint.vertices", std::vector<double>{});
        this->declare_parameter("enable_automatic_planning", false);
        this->declare_parameter("treat_unknown_as_free", true);
        this->declare_parameter("planning_time", 1.0);
        this->declare_parameter("planner_type", "RRTstar");
        this->declare_parameter("safety_margin", 0.1);
        this->declare_parameter("occupancy_threshold", 50);
        
        // Get parameters
        footprint_vertices_ = this->get_parameter("footprint.vertices").as_double_array();
        enable_automatic_planning_ = this->get_parameter("enable_automatic_planning").as_bool();
        bool treat_unknown_as_free = this->get_parameter("treat_unknown_as_free").as_bool();
        double planning_time = this->get_parameter("planning_time").as_double();
        std::string planner_type = this->get_parameter("planner_type").as_string();
        double safety_margin = this->get_parameter("safety_margin").as_double();
        int occupancy_threshold = this->get_parameter("occupancy_threshold").as_int();
        
        // Create footprint polygon
        createFootprintPolygon();
        
        // Pass footprint to planner
        planner_.setFootprint(footprint_polygon_);
        
        // Pass planning parameters to planner
        planner_.setTreatUnknownAsFree(treat_unknown_as_free);
        planner_.setPlanningTime(planning_time);
        planner_.setPlannerType(planner_type);
        planner_.setSafetyMargin(safety_margin);
        planner_.setOccupancyThreshold(occupancy_threshold);
        
        RCLCPP_INFO(this->get_logger(), "Footprint configured with %zu vertices (%zu coordinate pairs)", 
                    footprint_polygon_.points.size(), footprint_vertices_.size() / 2);
        
        RCLCPP_INFO(this->get_logger(), "Automatic path planning: %s", 
                    enable_automatic_planning_ ? "ENABLED" : "DISABLED");
        
        RCLCPP_INFO(this->get_logger(), "Treat unknown as free: %s", 
                    treat_unknown_as_free ? "ENABLED" : "DISABLED");
        
        RCLCPP_INFO(this->get_logger(), "Planning time: %.1f seconds", planning_time);
        
        RCLCPP_INFO(this->get_logger(), "Planner type: %s", planner_type.c_str());
        
        RCLCPP_INFO(this->get_logger(), "Safety margin: %.3f meters", safety_margin);
        
        RCLCPP_INFO(this->get_logger(), "Occupancy threshold: %d", occupancy_threshold);
        
        // Initialize TF2
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        
        // Create subscriptions
        map_subscription_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "map", 10, std::bind(&NavigationNode::map_callback, this, _1));
            
        initial_pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "initialpose", 10, std::bind(&NavigationNode::initial_pose_callback, this, _1));
            
        goal_pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "goal_pose", 10, std::bind(&NavigationNode::goal_pose_callback, this, _1));
            
        waypoints_subscription_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/waypoints", 10, std::bind(&NavigationNode::waypointsCallback, this, _1));
            
        combined_markers_subscription_ = this->create_subscription<visualization_msgs::msg::MarkerArray>(
            "/combined_markers", 10, std::bind(&NavigationNode::combinedMarkersCallback, this, _1));
        
        // Create publishers
        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/planned_path", 10);
        footprint_publisher_ = this->create_publisher<geometry_msgs::msg::PolygonStamped>("/footprint", 10);
        footprint_array_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/footprint_array", 10);
        waypoints_publisher_ = this->create_publisher<geometry_msgs::msg::PoseArray>("/waypoints", 10);
        combined_path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/combined_path", 10);
        obstacle_footprint_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/obstacle_footprint", 10);
        
        // Create timer
        timer_ = this->create_wall_timer(500ms, std::bind(&NavigationNode::timer_callback, this));
        
        // Create services
        get_path_service_ = this->create_service<nav_msgs::srv::GetPlan>(
            "/get_path",
            std::bind(&NavigationNode::getPathCallback, this, std::placeholders::_1, std::placeholders::_2));
            
        record_waypoints_service_ = this->create_service<std_srvs::srv::Empty>(
            "/record_waypoints",
            std::bind(&NavigationNode::recordWaypointsCallback, this, std::placeholders::_1, std::placeholders::_2));
            
        save_waypoints_service_ = this->create_service<std_srvs::srv::Empty>(
            "/save_waypoints",
            std::bind(&NavigationNode::saveWaypointsCallback, this, std::placeholders::_1, std::placeholders::_2));
            
        generate_path_service_ = this->create_service<std_srvs::srv::Trigger>(
            "/generate_path", 
            std::bind(&NavigationNode::generatePathCallback, this, std::placeholders::_1, std::placeholders::_2));
            
        filter_waypoints_service_ = this->create_service<std_srvs::srv::Trigger>(
            "/filter_waypoints",
            std::bind(&NavigationNode::filterWaypointsCallback, this, std::placeholders::_1, std::placeholders::_2));
            
        // Navigate service is now handled by path_following_controller
        // navigate_service_ = this->create_service<std_srvs::srv::Empty>(
        //     "/navigate",
        //     std::bind(&NavigationNode::navigateCallback, this, std::placeholders::_1, std::placeholders::_2));
        
        RCLCPP_INFO(this->get_logger(), "Navigation node started with waypoint recording and multi-path planning capabilities.");
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<NavigationNode>());
    rclcpp::shutdown();
    return 0;
}
