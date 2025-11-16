#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_srvs/srv/empty.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <algorithm>
#include <string>

class PathFollowingController : public rclcpp::Node
{
public:
    PathFollowingController() : Node("path_following_controller")
    {
        // Declare parameters
        this->declare_parameter("lookahead_distance", 0.5);
        this->declare_parameter("max_linear_velocity", 0.5);
        this->declare_parameter("max_angular_velocity", 1.0);
        this->declare_parameter("min_linear_velocity", 0.1);
        this->declare_parameter("goal_tolerance", 0.1);
        this->declare_parameter("control_frequency", 10.0);
        this->declare_parameter("angular_velocity_gain", 2.0);
        this->declare_parameter("slowdown_radius", 0.5);
        this->declare_parameter("max_path_start_distance", 2.0);
        this->declare_parameter("min_path_completion", 0.8);
        this->declare_parameter("smooth_path", false);
        this->declare_parameter("controller_type", std::string("pure_pursuit"));

        // Get parameters
        lookahead_distance_ = this->get_parameter("lookahead_distance").as_double();
        max_linear_velocity_ = this->get_parameter("max_linear_velocity").as_double();
        max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
        min_linear_velocity_ = this->get_parameter("min_linear_velocity").as_double();
        goal_tolerance_ = this->get_parameter("goal_tolerance").as_double();
        double control_frequency = this->get_parameter("control_frequency").as_double();
        angular_velocity_gain_ = this->get_parameter("angular_velocity_gain").as_double();
        slowdown_radius_ = this->get_parameter("slowdown_radius").as_double();
        max_path_start_distance_ = this->get_parameter("max_path_start_distance").as_double();
        min_path_completion_ = this->get_parameter("min_path_completion").as_double();
        smooth_path_ = this->get_parameter("smooth_path").as_bool();
        controller_type_ = this->get_parameter("controller_type").as_string();

        // Backward compatibility: treat legacy \"bezier\" controller_type as
        // the new yaw_control controller with path smoothing enabled.
        if (controller_type_ == "bezier") {
            RCLCPP_WARN(this->get_logger(),
                        "controller_type 'bezier' is deprecated. "
                        "Using 'yaw_control' with smooth_path=true instead.");
            controller_type_ = "yaw_control";
            smooth_path_ = true;
        }

        if (controller_type_ != "pure_pursuit" &&
            controller_type_ != "point_turn" &&
            controller_type_ != "yaw_control") {
            RCLCPP_WARN(this->get_logger(),
                        "Unknown controller_type '%s', defaulting to 'pure_pursuit'",
                        controller_type_.c_str());
            controller_type_ = "pure_pursuit";
        }

        // Initialize TF2
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publishers and Subscribers
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        navigation_active_pub_ = this->create_publisher<std_msgs::msg::Bool>("/navigation_active", 10);
        bezier_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/bezier_path", 10);
        
        // Subscribe to both planned_path and combined_path for compatibility
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/planned_path", 10,
            std::bind(&PathFollowingController::pathCallback, this, std::placeholders::_1));
            
        combined_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/combined_path", 10,
            std::bind(&PathFollowingController::pathCallback, this, std::placeholders::_1));

        // Services for navigation control
        navigate_service_ = this->create_service<std_srvs::srv::Empty>(
            "/navigate",
            std::bind(&PathFollowingController::navigateCallback, this, std::placeholders::_1, std::placeholders::_2));
            
        stop_navigation_service_ = this->create_service<std_srvs::srv::Empty>(
            "/stop_navigation",
            std::bind(&PathFollowingController::stopNavigationCallback, this, std::placeholders::_1, std::placeholders::_2));

        // Timer for control loop
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / control_frequency)),
            std::bind(&PathFollowingController::controlLoop, this));

        // Initialize navigation as disabled
        navigation_enabled_ = false;

        RCLCPP_INFO(this->get_logger(), "Path Following Controller started");
        RCLCPP_INFO(this->get_logger(), "Lookahead distance: %.2f m", lookahead_distance_);
        RCLCPP_INFO(this->get_logger(), "Max linear velocity: %.2f m/s", max_linear_velocity_);
        RCLCPP_INFO(this->get_logger(), "Max angular velocity: %.2f rad/s", max_angular_velocity_);
        RCLCPP_INFO(this->get_logger(), "Min path completion: %.0f%% (for cyclic paths)", min_path_completion_ * 100.0);
        RCLCPP_INFO(this->get_logger(), "Smooth path: %s", smooth_path_ ? "true" : "false");
        RCLCPP_INFO(this->get_logger(), "Controller type: %s", controller_type_.c_str());
        RCLCPP_INFO(this->get_logger(), "Navigation is initially DISABLED. Call /navigate service to start.");
    }

private:
    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg)
    {
        if (msg->poses.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty path");
            stopRobot();
            current_path_.poses.clear();
            interpolated_path_.poses.clear();
            publishNavigationStatus(false);
            return;
        }
        
        // Get current robot pose to validate the new path
        geometry_msgs::msg::TransformStamped transform;
        try {
            transform = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "Failed to get robot pose for path validation: %s", ex.what());
            // If we can't get robot pose, we can't validate the path
            current_path_ = *msg;
            interpolatePath();
            path_index_ = 0;
            return;
        }
        
        double robot_x = transform.transform.translation.x;
        double robot_y = transform.transform.translation.y;
        
        // Check if the new path starts reasonably close to the robot
        double dx = msg->poses[0].pose.position.x - robot_x;
        double dy = msg->poses[0].pose.position.y - robot_y;
        double distance_to_path_start = std::hypot(dx, dy);
        
        // If path starts too far from robot, it might be a stale path
        if (distance_to_path_start > max_path_start_distance_) {
            RCLCPP_WARN(this->get_logger(), 
                       "New path starts %.2f m from robot position (max: %.2f m), ignoring", 
                       distance_to_path_start, max_path_start_distance_);
            return;
        }
        
        current_path_ = *msg;
        interpolatePath();
        path_index_ = 0;
        
        RCLCPP_INFO(this->get_logger(), "Received new path with %zu waypoints, interpolated to %zu", 
                   current_path_.poses.size(), interpolated_path_.poses.size());
        
        // Don't automatically start navigation - wait for /navigate service call
        if (!navigation_enabled_) {
            RCLCPP_INFO(this->get_logger(), "Path stored. Call /navigate service to start following it.");
            publishNavigationStatus(false);
        } else {
            publishNavigationStatus(true);
        }
    }

    void controlLoop()
    {
        // Check if navigation is enabled
        if (!navigation_enabled_) {
            return;
        }
        
        if (interpolated_path_.poses.empty()) {
            return;
        }

        // Get current robot pose
        geometry_msgs::msg::TransformStamped transform;
        try {
            transform = tf_buffer_->lookupTransform("map", "base_link", tf2::TimePointZero);
        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "Failed to get robot pose: %s", ex.what());
            return;
        }

        // Extract robot position and orientation
        double robot_x = transform.transform.translation.x;
        double robot_y = transform.transform.translation.y;
        
        // Convert quaternion to yaw
        double qx = transform.transform.rotation.x;
        double qy = transform.transform.rotation.y;
        double qz = transform.transform.rotation.z;
        double qw = transform.transform.rotation.w;
        double robot_yaw = std::atan2(2.0 * (qw * qz + qx * qy), 
                                      1.0 - 2.0 * (qy * qy + qz * qz));

        // Check if we've reached the goal (use original path's last point)
        auto& goal_pose = current_path_.poses.back();
        double goal_x = goal_pose.pose.position.x;
        double goal_y = goal_pose.pose.position.y;
        double goal_distance = std::hypot(goal_x - robot_x, goal_y - robot_y);

        // Calculate path progress percentage
        size_t active_path_size = interpolated_path_.poses.size();
        if (smooth_path_ && !bezier_path_.poses.empty()) {
            active_path_size = bezier_path_.poses.size();
        }
        if (active_path_size == 0) {
            RCLCPP_WARN(this->get_logger(), "Active path is empty while navigation is enabled");
            stopRobot();
            navigation_enabled_ = false;
            publishNavigationStatus(false);
            return;
        }
        double progress_percentage = static_cast<double>(path_index_) /
                                     static_cast<double>(active_path_size);
        
        // For cyclic paths, ensure we've traversed at least min_path_completion_ before checking goal
        if (goal_distance < goal_tolerance_ && progress_percentage > min_path_completion_) {
            RCLCPP_INFO(this->get_logger(), "Goal reached! (Progress: %.1f%%)", progress_percentage * 100.0);
            stopRobot();
            current_path_.poses.clear();
            interpolated_path_.poses.clear();
            navigation_enabled_ = false;
            publishNavigationStatus(false);
            return;
        }

        // Find the lookahead point
        geometry_msgs::msg::Point lookahead_point;
        if (!findLookaheadPoint(robot_x, robot_y, lookahead_point)) {
            RCLCPP_WARN(this->get_logger(), "Failed to find lookahead point");
            stopRobot();
            return;
        }

        // Calculate control commands using selected controller
        geometry_msgs::msg::Twist cmd_vel;
        calculateVelocityCommand(robot_x, robot_y, robot_yaw, lookahead_point, 
                                goal_distance, cmd_vel);

        // Publish velocity command
        cmd_vel_pub_->publish(cmd_vel);
    }

    bool findLookaheadPoint(double robot_x, double robot_y, 
                            geometry_msgs::msg::Point& lookahead_point)
    {
        const nav_msgs::msg::Path* path = &interpolated_path_;
        if (smooth_path_ && !bezier_path_.poses.empty()) {
            path = &bezier_path_;
        }

        if (path->poses.empty()) {
            return false;
        }

        // Find closest point on interpolated path
        double min_distance = std::numeric_limits<double>::max();
        size_t closest_index = 0;
        
        for (size_t i = 0; i < path->poses.size(); ++i) {
            double dx = path->poses[i].pose.position.x - robot_x;
            double dy = path->poses[i].pose.position.y - robot_y;
            double distance = std::hypot(dx, dy);
            
            if (distance < min_distance) {
                min_distance = distance;
                closest_index = i;
            }
        }
        
        // Only update path_index_ if we're moving forward
        if (closest_index >= path_index_) {
            path_index_ = closest_index;
        }
        
        // If we're too far from the path, something is wrong
        if (min_distance > 1.0) {  // 1 meter threshold
            RCLCPP_WARN(this->get_logger(), 
                       "Robot is %.2f m from nearest path point", min_distance);
        }

        // Find lookahead point starting from closest point
        for (size_t i = closest_index; i < path->poses.size(); ++i) {
            double dx = path->poses[i].pose.position.x - robot_x;
            double dy = path->poses[i].pose.position.y - robot_y;
            double distance = std::hypot(dx, dy);
            
            if (distance >= lookahead_distance_) {
                lookahead_point = path->poses[i].pose.position;
                return true;
            }
        }

        // If no point is far enough, use the last point of original path
        if (!current_path_.poses.empty()) {
            lookahead_point = current_path_.poses.back().pose.position;
            return true;
        }

        return false;
    }

    void calculateVelocityCommand(double robot_x, double robot_y, double robot_yaw,
                                 const geometry_msgs::msg::Point& lookahead_point,
                                 double goal_distance,
                                 geometry_msgs::msg::Twist& cmd_vel)
    {
        // Calculate direction to lookahead point
        double dx = lookahead_point.x - robot_x;
        double dy = lookahead_point.y - robot_y;
        
        // Calculate desired heading
        double desired_yaw = std::atan2(dy, dx);
        
        // Calculate heading error
        double yaw_error = desired_yaw - robot_yaw;
        
        // Normalize angle to [-pi, pi]
        while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI;
        while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI;

        if (controller_type_ == "point_turn") {
            const double yaw_threshold = 0.1;

            cmd_vel.linear.x = 0.0;
            cmd_vel.angular.z = 0.0;

            if (std::abs(yaw_error) > yaw_threshold) {
                double angular_velocity = angular_velocity_gain_ * yaw_error;
                angular_velocity = std::clamp(angular_velocity,
                                              -max_angular_velocity_,
                                              max_angular_velocity_);
                cmd_vel.angular.z = angular_velocity;
            } else {
                double linear_velocity = max_linear_velocity_;

                if (goal_distance < slowdown_radius_) {
                    double slowdown_factor = goal_distance / slowdown_radius_;
                    linear_velocity *= slowdown_factor;
                }

                linear_velocity = std::max(linear_velocity, min_linear_velocity_);

                cmd_vel.linear.x = linear_velocity;
                cmd_vel.angular.z = 0.0;
            }
        } else if (controller_type_ == "yaw_control") {
            // Smooth yaw-correcting controller without large in-place turns.
            double angular_velocity = angular_velocity_gain_ * yaw_error;
            angular_velocity = std::clamp(angular_velocity,
                                          -max_angular_velocity_,
                                          max_angular_velocity_);

            double angular_factor = 1.0 - std::min(std::abs(angular_velocity) / max_angular_velocity_, 1.0);
            double linear_velocity = min_linear_velocity_ +
                                     (max_linear_velocity_ - min_linear_velocity_) * angular_factor;

            if (goal_distance < slowdown_radius_) {
                double slowdown_factor = goal_distance / slowdown_radius_;
                linear_velocity *= slowdown_factor;
                linear_velocity = std::max(linear_velocity, min_linear_velocity_);
            }

            cmd_vel.linear.x = linear_velocity;
            cmd_vel.angular.z = angular_velocity;
        } else {
            // Pure Pursuit controller (CMU-RI-TR-92-01):
            // 1) Transform goal point into vehicle coordinates.
            // 2) Compute curvature kappa = 2 * y / Ld^2.
            // 3) Command angular velocity omega = kappa * v.

            // Transform lookahead point into vehicle frame (x forward, y left)
            double cos_yaw = std::cos(robot_yaw);
            double sin_yaw = std::sin(robot_yaw);
            double x_v =  cos_yaw * dx + sin_yaw * dy;
            double y_v = -sin_yaw * dx + cos_yaw * dy;

            double Ld = std::hypot(x_v, y_v);
            if (Ld < 1e-6) {
                cmd_vel.linear.x = 0.0;
                cmd_vel.angular.z = 0.0;
                return;
            }

            // Curvature that drives an arc from rear axle to goal point
            double curvature = 2.0 * y_v / (Ld * Ld);

            // Linear speed is chosen independently of curvature (pure pursuit
            // only defines the steering geometry).
            double linear_velocity = max_linear_velocity_;

            // Optional slowdown near the goal
            if (goal_distance < slowdown_radius_) {
                double slowdown_factor = goal_distance / slowdown_radius_;
                linear_velocity *= slowdown_factor;
                linear_velocity = std::max(linear_velocity, min_linear_velocity_);
            }

            double angular_velocity = curvature * linear_velocity;

            // Enforce angular velocity limits by saturating omega only.
            angular_velocity = std::clamp(angular_velocity,
                                          -max_angular_velocity_,
                                          max_angular_velocity_);

            cmd_vel.linear.x = linear_velocity;
            cmd_vel.angular.z = angular_velocity;
        }
    }

    void stopRobot()
    {
        geometry_msgs::msg::Twist stop_cmd;
        stop_cmd.linear.x = 0.0;
        stop_cmd.angular.z = 0.0;
        cmd_vel_pub_->publish(stop_cmd);
    }
    
    void publishNavigationStatus(bool is_active)
    {
        std_msgs::msg::Bool status_msg;
        status_msg.data = is_active;
        navigation_active_pub_->publish(status_msg);
    }
    
    void navigateCallback(const std::shared_ptr<std_srvs::srv::Empty::Request> request,
                         std::shared_ptr<std_srvs::srv::Empty::Response> response)
    {
        (void)request;
        (void)response;
        
        if (interpolated_path_.poses.empty()) {
            RCLCPP_WARN(this->get_logger(), "Cannot start navigation - no path available");
            return;
        }
        
        navigation_enabled_ = true;
        path_index_ = 0;  // Reset to start of path
        if (smooth_path_) {
            generateBezierPath();
            if (!bezier_path_.poses.empty()) {
                bezier_path_pub_->publish(bezier_path_);
                RCLCPP_INFO(this->get_logger(), "Published Bezier path with %zu poses",
                            bezier_path_.poses.size());
            } else {
                RCLCPP_WARN(this->get_logger(), "Bezier controller selected but Bezier path is empty");
            }
        }
        publishNavigationStatus(true);
        RCLCPP_INFO(this->get_logger(), "Navigation ENABLED - starting to follow path");
    }
    
    void stopNavigationCallback(const std::shared_ptr<std_srvs::srv::Empty::Request> request,
                               std::shared_ptr<std_srvs::srv::Empty::Response> response)
    {
        (void)request;
        (void)response;
        
        navigation_enabled_ = false;
        stopRobot();
        publishNavigationStatus(false);
        RCLCPP_INFO(this->get_logger(), "Navigation DISABLED - robot stopped");
    }
    
    void interpolatePath()
    {
        interpolated_path_.poses.clear();
        interpolated_path_.header = current_path_.header;
        
        if (current_path_.poses.size() < 2) {
            // If path has 0 or 1 points, just copy it
            interpolated_path_ = current_path_;
            return;
        }
        
        // Interpolation step size (meters)
        const double interpolation_step = 0.1;  // 10cm between points
        
        for (size_t i = 0; i < current_path_.poses.size() - 1; ++i) {
            const auto& start_pose = current_path_.poses[i];
            const auto& end_pose = current_path_.poses[i + 1];
            
            double dx = end_pose.pose.position.x - start_pose.pose.position.x;
            double dy = end_pose.pose.position.y - start_pose.pose.position.y;
            double segment_length = std::hypot(dx, dy);
            
            // Add the start point
            interpolated_path_.poses.push_back(start_pose);
            
            // Calculate number of intermediate points
            int num_intermediate = static_cast<int>(segment_length / interpolation_step);
            
            // Add intermediate points
            for (int j = 1; j <= num_intermediate; ++j) {
                double t = static_cast<double>(j) / (num_intermediate + 1);
                
                geometry_msgs::msg::PoseStamped intermediate_pose;
                intermediate_pose.header = start_pose.header;
                
                // Linear interpolation of position
                intermediate_pose.pose.position.x = start_pose.pose.position.x + t * dx;
                intermediate_pose.pose.position.y = start_pose.pose.position.y + t * dy;
                intermediate_pose.pose.position.z = start_pose.pose.position.z;
                
                // Set orientation to point along the path
                double path_yaw = std::atan2(dy, dx);
                intermediate_pose.pose.orientation.w = std::cos(path_yaw / 2.0);
                intermediate_pose.pose.orientation.x = 0.0;
                intermediate_pose.pose.orientation.y = 0.0;
                intermediate_pose.pose.orientation.z = std::sin(path_yaw / 2.0);
                
                interpolated_path_.poses.push_back(intermediate_pose);
            }
        }
        
        // Add the last point
        interpolated_path_.poses.push_back(current_path_.poses.back());
    }

    void generateBezierPath()
    {
        bezier_path_.poses.clear();
        bezier_path_.header = current_path_.header;

        const auto& input_path = current_path_;
        size_t n = input_path.poses.size();

        if (n == 0) {
            return;
        }

        if (n == 1) {
            bezier_path_ = input_path;
            return;
        }

        const int samples_per_segment = 10;
        double last_yaw = 0.0;
        bool has_last_yaw = false;

        auto getPoint = [&](int index) -> geometry_msgs::msg::Point
        {
            if (index < 0) {
                index = 0;
            } else if (static_cast<size_t>(index) >= n) {
                index = static_cast<int>(n - 1);
            }
            return input_path.poses[static_cast<size_t>(index)].pose.position;
        };

        for (size_t i = 0; i < n - 1; ++i) {
            geometry_msgs::msg::Point Pm1 = getPoint(static_cast<int>(i) - 1);
            geometry_msgs::msg::Point P0 = getPoint(static_cast<int>(i));
            geometry_msgs::msg::Point P1 = getPoint(static_cast<int>(i) + 1);
            geometry_msgs::msg::Point P2 = getPoint(static_cast<int>(i) + 2);

            geometry_msgs::msg::Point C0 = P0;
            geometry_msgs::msg::Point C3 = P1;

            geometry_msgs::msg::Point C1;
            C1.x = P0.x + (P1.x - Pm1.x) / 6.0;
            C1.y = P0.y + (P1.y - Pm1.y) / 6.0;
            C1.z = P0.z;

            geometry_msgs::msg::Point C2;
            C2.x = P1.x - (P2.x - P0.x) / 6.0;
            C2.y = P1.y - (P2.y - P0.y) / 6.0;
            C2.z = P1.z;

            for (int s = 0; s < samples_per_segment; ++s) {
                double t = static_cast<double>(s) / static_cast<double>(samples_per_segment);
                double one_minus_t = 1.0 - t;

                double b0 = one_minus_t * one_minus_t * one_minus_t;
                double b1 = 3.0 * one_minus_t * one_minus_t * t;
                double b2 = 3.0 * one_minus_t * t * t;
                double b3 = t * t * t;

                double x = b0 * C0.x + b1 * C1.x + b2 * C2.x + b3 * C3.x;
                double y = b0 * C0.y + b1 * C1.y + b2 * C2.y + b3 * C3.y;
                double z = b0 * C0.z + b1 * C1.z + b2 * C2.z + b3 * C3.z;

                double dx = 3.0 * one_minus_t * one_minus_t * (C1.x - C0.x) +
                            6.0 * one_minus_t * t * (C2.x - C1.x) +
                            3.0 * t * t * (C3.x - C2.x);
                double dy = 3.0 * one_minus_t * one_minus_t * (C1.y - C0.y) +
                            6.0 * one_minus_t * t * (C2.y - C1.y) +
                            3.0 * t * t * (C3.y - C2.y);

                double yaw;
                if (std::abs(dx) > 1e-6 || std::abs(dy) > 1e-6) {
                    yaw = std::atan2(dy, dx);
                    last_yaw = yaw;
                    has_last_yaw = true;
                } else if (has_last_yaw) {
                    yaw = last_yaw;
                } else {
                    yaw = 0.0;
                }

                geometry_msgs::msg::PoseStamped pose;
                pose.header = input_path.header;
                pose.pose.position.x = x;
                pose.pose.position.y = y;
                pose.pose.position.z = z;
                pose.pose.orientation.w = std::cos(yaw / 2.0);
                pose.pose.orientation.x = 0.0;
                pose.pose.orientation.y = 0.0;
                pose.pose.orientation.z = std::sin(yaw / 2.0);

                bezier_path_.poses.push_back(pose);
            }
        }

        // Ensure the final goal point is included exactly
        const auto& goal_pose = input_path.poses.back();
        geometry_msgs::msg::PoseStamped final_pose = goal_pose;
        double final_yaw = last_yaw;
        final_pose.pose.orientation.w = std::cos(final_yaw / 2.0);
        final_pose.pose.orientation.x = 0.0;
        final_pose.pose.orientation.y = 0.0;
        final_pose.pose.orientation.z = std::sin(final_yaw / 2.0);
        bezier_path_.poses.push_back(final_pose);
    }

    // ROS2 interfaces
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr navigation_active_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr bezier_path_pub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr combined_path_sub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr navigate_service_;
    rclcpp::Service<std_srvs::srv::Empty>::SharedPtr stop_navigation_service_;

    // TF2
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Path following state
    nav_msgs::msg::Path current_path_;
    nav_msgs::msg::Path interpolated_path_;  // Dense interpolated path
    nav_msgs::msg::Path bezier_path_;        // Bezier-smoothed path
    size_t path_index_ = 0;
    bool navigation_enabled_ = false;

    // Parameters
    double lookahead_distance_;
    double max_linear_velocity_;
    double max_angular_velocity_;
    double min_linear_velocity_;
    double goal_tolerance_;
    double angular_velocity_gain_;
    double slowdown_radius_;
    double max_path_start_distance_;
    double min_path_completion_;
    bool smooth_path_;
    std::string controller_type_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PathFollowingController>());
    rclcpp::shutdown();
    return 0;
}
