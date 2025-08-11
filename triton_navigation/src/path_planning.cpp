#include "triton_navigation/path_planning.hpp"
#include <rclcpp/rclcpp.hpp>
#include <cmath>
#include <chrono>
#include <limits>

namespace ob = ompl::base;
namespace og = ompl::geometric;

namespace path_planning{

nav_msgs::msg::OccupancyGrid occupancyMap;
geometry_msgs::msg::Polygon robotFootprint;
bool g_treat_unknown_as_free = true;
double g_safety_margin = 0.1; // Safety margin in meters (10cm default)
int g_occupancy_threshold = 50; // Occupancy threshold for obstacles
std::vector<CylinderObstacle> g_cylinder_obstacles; // Global storage for cylinder obstacles

Planner2D::Planner2D(void)
{
    startSet = false;
    goalSet = false;
    planning_time_ = 5.0;  // Default planning time
    treat_unknown_as_free_ = true;  // Default
    planner_type_ = "RRTstar";  // Default to optimal planner
    configure();
}

Planner2D::~Planner2D(){};

bool isStateValid(const ob::State *state){
    // get SE(2) state (x, y, theta)
    const auto *se2state = state->as<ob::SE2StateSpace::StateType>();

    // Get robot position and orientation in world coordinates
    double robot_x = se2state->getX();
    double robot_y = se2state->getY();
    double robot_theta = se2state->getYaw();
    
    // First check collision with cylinder obstacles
    for (const auto& cylinder : g_cylinder_obstacles) {
        // Check if any part of the robot footprint intersects with the cylinder
        // We'll check each vertex and edge of the footprint
        
        // Precompute trigonometric values for rotation
        double cos_theta = std::cos(robot_theta);
        double sin_theta = std::sin(robot_theta);
        
        // Check each vertex of the footprint
        for (const auto& vertex : robotFootprint.points) {
            // Transform footprint vertex to world coordinates with rotation
            double vertex_world_x = robot_x + (vertex.x * cos_theta - vertex.y * sin_theta);
            double vertex_world_y = robot_y + (vertex.x * sin_theta + vertex.y * cos_theta);
            
            // Check distance to cylinder center
            double dx = vertex_world_x - cylinder.x;
            double dy = vertex_world_y - cylinder.y;
            double distance = std::sqrt(dx * dx + dy * dy);
            
            // Add safety margin to cylinder radius
            if (distance < cylinder.radius + g_safety_margin) {
                return false; // Collision with cylinder
            }
        }
        
        // Check edges of the footprint against the cylinder
        // This is important for large robots where vertices might miss the obstacle
        for (size_t i = 0; i < robotFootprint.points.size(); ++i) {
            size_t j = (i + 1) % robotFootprint.points.size();
            
            // Get edge endpoints in world coordinates
            double x1 = robot_x + (robotFootprint.points[i].x * cos_theta - robotFootprint.points[i].y * sin_theta);
            double y1 = robot_y + (robotFootprint.points[i].x * sin_theta + robotFootprint.points[i].y * cos_theta);
            double x2 = robot_x + (robotFootprint.points[j].x * cos_theta - robotFootprint.points[j].y * sin_theta);
            double y2 = robot_y + (robotFootprint.points[j].x * sin_theta + robotFootprint.points[j].y * cos_theta);
            
            // Check closest point on edge to cylinder center
            double edge_dx = x2 - x1;
            double edge_dy = y2 - y1;
            double edge_len_sq = edge_dx * edge_dx + edge_dy * edge_dy;
            
            if (edge_len_sq > 0.0001) {  // Avoid division by zero
                double t = std::max(0.0, std::min(1.0, 
                    ((cylinder.x - x1) * edge_dx + (cylinder.y - y1) * edge_dy) / edge_len_sq));
                
                double closest_x = x1 + t * edge_dx;
                double closest_y = y1 + t * edge_dy;
                
                double dist = std::sqrt((closest_x - cylinder.x) * (closest_x - cylinder.x) + 
                                       (closest_y - cylinder.y) * (closest_y - cylinder.y));
                
                if (dist < cylinder.radius + g_safety_margin) {
                    return false; // Edge too close to cylinder
                }
            }
        }
        
        // Also check if cylinder center is inside the robot footprint
        // Transform cylinder center to robot's local coordinates
        double local_cx = (cylinder.x - robot_x) * cos_theta + (cylinder.y - robot_y) * sin_theta;
        double local_cy = -(cylinder.x - robot_x) * sin_theta + (cylinder.y - robot_y) * cos_theta;
        
        // Simple point-in-polygon test for convex footprint
        bool inside = true;
        for (size_t i = 0; i < robotFootprint.points.size(); ++i) {
            size_t j = (i + 1) % robotFootprint.points.size();
            double edge_x = robotFootprint.points[j].x - robotFootprint.points[i].x;
            double edge_y = robotFootprint.points[j].y - robotFootprint.points[i].y;
            double to_point_x = local_cx - robotFootprint.points[i].x;
            double to_point_y = local_cy - robotFootprint.points[i].y;
            double cross = edge_x * to_point_y - edge_y * to_point_x;
            if (cross > 0) {
                inside = false;
                break;
            }
        }
        
        if (inside) {
            return false; // Cylinder center is inside robot footprint
        }
    }
    
    // Check if occupancy map is available
    if (occupancyMap.data.empty()) {
        return true; // If no map is available, assume state is valid
    }
    
    // Precompute trigonometric values for rotation
    double cos_theta = std::cos(robot_theta);
    double sin_theta = std::sin(robot_theta);
    
    // Check collision for each vertex of the footprint
    for (const auto& vertex : robotFootprint.points) {
        // Transform footprint vertex to world coordinates with rotation
        double vertex_world_x = robot_x + (vertex.x * cos_theta - vertex.y * sin_theta);
        double vertex_world_y = robot_y + (vertex.x * sin_theta + vertex.y * cos_theta);
        
        // Convert world coordinates to map coordinates
        double map_x = (vertex_world_x - occupancyMap.info.origin.position.x) / occupancyMap.info.resolution;
        double map_y = (vertex_world_y - occupancyMap.info.origin.position.y) / occupancyMap.info.resolution;
        
        // Convert to grid indices using proper rounding
        int grid_x = static_cast<int>(std::round(map_x));
        int grid_y = static_cast<int>(std::round(map_y));
        
        // Check bounds
        if (grid_x < 0 || grid_x >= static_cast<int>(occupancyMap.info.width) ||
            grid_y < 0 || grid_y >= static_cast<int>(occupancyMap.info.height)) {
            return false; // Outside map bounds
        }
        
        // Calculate array index (row-major order)
        int index = grid_y * occupancyMap.info.width + grid_x;
        
        // Check if index is within data array bounds
        if (index < 0 || index >= static_cast<int>(occupancyMap.data.size())) {
            return false;
        }
        
        // Get occupancy value (0 = free, 100 = occupied, -1 = unknown)
        int8_t occupancy_value = occupancyMap.data[index];
        
        // Consider occupied cells as obstacles based on threshold
        if (occupancy_value >= g_occupancy_threshold) {
            return false; // Occupied space
        }
        
        // Handle unknown cells based on configuration
        if (occupancy_value == -1 && !g_treat_unknown_as_free) {
            return false; // Unknown space treated as obstacle
        }
        
        // Check safety margin around this point
        if (g_safety_margin > 0) {
            int margin_cells = static_cast<int>(std::ceil(g_safety_margin / occupancyMap.info.resolution));
            
            // Check cells in a square around the current cell
            for (int dx = -margin_cells; dx <= margin_cells; ++dx) {
                for (int dy = -margin_cells; dy <= margin_cells; ++dy) {
                    int check_x = grid_x + dx;
                    int check_y = grid_y + dy;
                    
                    // Skip if outside map bounds
                    if (check_x < 0 || check_x >= static_cast<int>(occupancyMap.info.width) ||
                        check_y < 0 || check_y >= static_cast<int>(occupancyMap.info.height)) {
                        continue;
                    }
                    
                    // Check if within circular radius
                    double dist = std::sqrt(dx * dx + dy * dy) * occupancyMap.info.resolution;
                    if (dist <= g_safety_margin) {
                        int check_index = check_y * occupancyMap.info.width + check_x;
                        if (check_index >= 0 && check_index < static_cast<int>(occupancyMap.data.size())) {
                            int8_t check_value = occupancyMap.data[check_index];
                            if (check_value >= g_occupancy_threshold) {
                                return false; // Obstacle within safety margin
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Also check multiple points along the edges of the footprint for better collision detection
    if (robotFootprint.points.size() >= 3) {
        for (size_t i = 0; i < robotFootprint.points.size(); ++i) {
            size_t j = (i + 1) % robotFootprint.points.size();
            
            // Sample points along each edge
            const int samples_per_edge = 5;
            for (int k = 1; k < samples_per_edge; ++k) {
                double t = static_cast<double>(k) / samples_per_edge;
                
                // Interpolate between vertices
                double edge_x = robotFootprint.points[i].x + t * (robotFootprint.points[j].x - robotFootprint.points[i].x);
                double edge_y = robotFootprint.points[i].y + t * (robotFootprint.points[j].y - robotFootprint.points[i].y);
                
                // Transform to world coordinates with rotation
                double edge_world_x = robot_x + (edge_x * cos_theta - edge_y * sin_theta);
                double edge_world_y = robot_y + (edge_x * sin_theta + edge_y * cos_theta);
                
                // Convert to map coordinates
                double map_x = (edge_world_x - occupancyMap.info.origin.position.x) / occupancyMap.info.resolution;
                double map_y = (edge_world_y - occupancyMap.info.origin.position.y) / occupancyMap.info.resolution;
                
                // Convert to grid indices using proper rounding
                int grid_x = static_cast<int>(std::round(map_x));
                int grid_y = static_cast<int>(std::round(map_y));
                
                // Check bounds
                if (grid_x < 0 || grid_x >= static_cast<int>(occupancyMap.info.width) ||
                    grid_y < 0 || grid_y >= static_cast<int>(occupancyMap.info.height)) {
                    return false;
                }
                
                // Calculate array index
                int index = grid_y * occupancyMap.info.width + grid_x;
                
                // Check occupancy
                if (index >= 0 && index < static_cast<int>(occupancyMap.data.size())) {
                    int8_t occupancy_value = occupancyMap.data[index];
                    if (occupancy_value >= g_occupancy_threshold) {
                        return false;
                    }
                    // Handle unknown cells based on configuration
                    if (occupancy_value == -1 && !g_treat_unknown_as_free) {
                        return false;
                    }
                    
                    // Check safety margin for edge points too
                    if (g_safety_margin > 0) {
                        int margin_cells = static_cast<int>(std::ceil(g_safety_margin / occupancyMap.info.resolution));
                        
                        for (int dx = -margin_cells; dx <= margin_cells; ++dx) {
                            for (int dy = -margin_cells; dy <= margin_cells; ++dy) {
                                int check_x = grid_x + dx;
                                int check_y = grid_y + dy;
                                
                                if (check_x < 0 || check_x >= static_cast<int>(occupancyMap.info.width) ||
                                    check_y < 0 || check_y >= static_cast<int>(occupancyMap.info.height)) {
                                    continue;
                                }
                                
                                double dist = std::sqrt(dx * dx + dy * dy) * occupancyMap.info.resolution;
                                if (dist <= g_safety_margin) {
                                    int check_index = check_y * occupancyMap.info.width + check_x;
                                    if (check_index >= 0 && check_index < static_cast<int>(occupancyMap.data.size())) {
                                        int8_t check_value = occupancyMap.data[check_index];
                                        if (check_value >= g_occupancy_threshold) {
                                            return false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return true; // Free space
}

nav_msgs::msg::Path Planner2D::extractPath(ob::ProblemDefinition* pdef){
    nav_msgs::msg::Path plannedPath;
    plannedPath.header.frame_id = "/map";
    
    // get the obtained path
    ob::PathPtr path = pdef->getSolutionPath();
    // convert to geometric path
    auto *path_ = path.get()->as<og::PathGeometric>();
    
    if (path_->getStateCount() == 0) {
        return plannedPath;
    }
    
    // Simply extract the path without adding extra waypoints
    for(unsigned int i = 0; i < path_->getStateCount(); ++i){
        const auto *state = path_->getState(i)->as<ob::SE2StateSpace::StateType>();
        
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.pose.position.x = state->getX();
        pose_stamped.pose.position.y = state->getY();
        pose_stamped.pose.position.z = 0.01;
        
        // Calculate orientation for each waypoint based on direction to next waypoint
        double yaw = state->getYaw(); // Get the planned orientation from OMPL
        
        // Only override orientation if we're not using the planner's orientation
        // For now, always use direction to next waypoint for better path following
        if (i < path_->getStateCount() - 1) {
            // For all but the last waypoint, orient towards the next waypoint
            const auto *nextState = path_->getState(i+1)->as<ob::SE2StateSpace::StateType>();
            double dx = nextState->getX() - state->getX();
            double dy = nextState->getY() - state->getY();
            if (std::sqrt(dx*dx + dy*dy) > 0.01) {  // Only if there's significant movement
                yaw = std::atan2(dy, dx);
            }
        } else if (i > 0) {
            // For the last waypoint, maintain the orientation from the previous segment
            const auto *prevState = path_->getState(i-1)->as<ob::SE2StateSpace::StateType>();
            double dx = state->getX() - prevState->getX();
            double dy = state->getY() - prevState->getY();
            if (std::sqrt(dx*dx + dy*dy) > 0.01) {
                yaw = std::atan2(dy, dx);
            }
        }
        
        pose_stamped.pose.orientation.w = std::cos(yaw / 2.0);
        pose_stamped.pose.orientation.x = 0.0;
        pose_stamped.pose.orientation.y = 0.0;
        pose_stamped.pose.orientation.z = std::sin(yaw / 2.0);
        
        // Debug logging for first few poses
        if (i < 5 || i == path_->getStateCount() - 1) {
            RCLCPP_DEBUG(rclcpp::get_logger("planner_2d"), 
                        "Path pose %u: (%.2f, %.2f) yaw=%.2f rad (%.1f deg), OMPL yaw=%.2f rad", 
                        i, state->getX(), state->getY(), yaw, yaw * 180.0 / M_PI, state->getYaw());
        }
        pose_stamped.header.frame_id = "/map";
        pose_stamped.header.stamp = rclcpp::Clock().now();
        
        plannedPath.poses.push_back(pose_stamped);
    }
    
    return plannedPath;
}

void Planner2D::configure(void){
    maxStepLength = 0.3;  // Increased for better exploration and faster convergence

    // construct the SE(2) state space
    space = std::make_shared<ob::SE2StateSpace>();

    // create bounds for position
    bounds.reset(new ob::RealVectorBounds(2));
    bounds->setLow(0, -1.0);
    bounds->setHigh(0, 13.0);
    bounds->setLow(1, -5.0);
    bounds->setHigh(1, 5.0);

    // set bounds on the space
    space->setBounds(*bounds.get());

    // initialize the start and goal positions
    start.reset(new ob::ScopedState<>(space));
    goal.reset(new ob::ScopedState<>(space));
}

nav_msgs::msg::Path Planner2D::planPath(const nav_msgs::msg::OccupancyGrid& globalMap){
    // Clear cylinder obstacles when using the basic planPath
    g_cylinder_obstacles.clear();
    return planPath(globalMap, g_cylinder_obstacles);
}

nav_msgs::msg::Path Planner2D::planPath(const nav_msgs::msg::OccupancyGrid& globalMap, 
                                        const std::vector<CylinderObstacle>& cylinderObstacles){
    occupancyMap = globalMap;
    g_cylinder_obstacles = cylinderObstacles;
    
    RCLCPP_INFO(rclcpp::get_logger("planner_2d"), 
                "Planning with %zu cylinder obstacles", cylinderObstacles.size());

    // Update bounds based on the map if available
    if (!globalMap.data.empty()) {
        double min_x = globalMap.info.origin.position.x;
        double max_x = min_x + globalMap.info.width * globalMap.info.resolution;
        double min_y = globalMap.info.origin.position.y;
        double max_y = min_y + globalMap.info.height * globalMap.info.resolution;
        
        bounds->setLow(0, min_x - 1.0);   // Add 1m margin
        bounds->setHigh(0, max_x + 1.0);
        bounds->setLow(1, min_y - 1.0);
        bounds->setHigh(1, max_y + 1.0);
        
        space->setBounds(*bounds.get());
    }

    // search space information
    auto si(std::make_shared<ompl::base::SpaceInformation>(space));
    si->setStateValidityChecker(path_planning::isStateValid);
    si->setStateValidityCheckingResolution(0.001);
    si->setup();

    // problem definition
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));
    pdef->setStartAndGoalStates(*start.get(), *goal.get());
    
    // Set optimization objective for optimal planners
    if (planner_type_ == "RRTstar") {
        // Create a simple optimization objective that minimizes path length
        class PathLengthObjective : public ob::OptimizationObjective
        {
        public:
            PathLengthObjective(const ob::SpaceInformationPtr &si) : OptimizationObjective(si) 
            {
                // Set a description
                description_ = "Path Length";
            }
            
            ob::Cost stateCost(const ob::State *) const override
            {
                return ob::Cost(0.0);
            }
            
            ob::Cost motionCost(const ob::State *s1, const ob::State *s2) const override
            {
                return ob::Cost(si_->distance(s1, s2));
            }
            
            // Provide a heuristic for the cost-to-go (straight line distance to goal)
            ob::Cost motionCostHeuristic(const ob::State *s1, const ob::State *s2) const override
            {
                return ob::Cost(si_->distance(s1, s2));
            }
            
            // Combine costs by addition (for path length)
            ob::Cost combineCosts(ob::Cost c1, ob::Cost c2) const override
            {
                return ob::Cost(c1.value() + c2.value());
            }
            
            // Identity cost is zero
            ob::Cost identityCost() const override
            {
                return ob::Cost(0.0);
            }
            
            // Set whether the objective is symmetric
            bool isSymmetric() const override
            {
                return true;
            }
        };
        
        auto objective = std::make_shared<PathLengthObjective>(si);
        
        // Set up cost-to-go heuristic for informed sampling
        objective->setCostToGoHeuristic([](const ob::State *state, const ob::Goal *goal) -> ob::Cost {
            // Cast to proper state type
            const auto *se2state = state->as<ob::SE2StateSpace::StateType>();
            
            // Get goal state (assuming single goal state)
            if (goal->hasType(ob::GoalType::GOAL_STATE)) {
                const auto *goalState = goal->as<ob::GoalState>()->getState()->as<ob::SE2StateSpace::StateType>();
                
                // Compute Euclidean distance as admissible heuristic
                double dx = se2state->getX() - goalState->getX();
                double dy = se2state->getY() - goalState->getY();
                return ob::Cost(std::sqrt(dx * dx + dy * dy));
            }
            
            return ob::Cost(0.0);
        });
        
        pdef->setOptimizationObjective(objective);
        
        RCLCPP_INFO(rclcpp::get_logger("planner_2d"), 
                    "Using RRTstar with path length optimization objective and informed sampling");
    }

    // create planner based on type
    std::shared_ptr<ob::Planner> planner;
    if (planner_type_ == "RRTstar") {
        // RRT* for asymptotically optimal planning
        auto rrtstar = std::make_shared<og::RRTstar>(si);
        rrtstar->setRange(maxStepLength);
        // Enable informed sampling for better performance after initial solution
        rrtstar->setInformedSampling(true);
        // Set goal bias to occasionally sample the goal (helps find initial solution faster)
        rrtstar->setGoalBias(0.1);
        // Increase rewiring factor for better path optimization (default is 1.1)
        rrtstar->setRewireFactor(2.0);  // Increased for more aggressive rewiring
        // Use radius-based rewiring for better consistency
        rrtstar->setKNearest(false);
        // Delay optimization focus until we have a solution
        rrtstar->setDelayCC(true);
        // Focus new samples in the informed subset after solution is found
        rrtstar->setFocusSearch(true);
        // Set a reasonable pruning threshold
        rrtstar->setPruneThreshold(0.1);
        planner = rrtstar;
    } else {
        // RRTConnect for fast planning
        auto rrtconnect = std::make_shared<og::RRTConnect>(si);
        rrtconnect->setRange(maxStepLength);
        planner = rrtconnect;
    }
    
    planner->setProblemDefinition(pdef);
    planner->setup();

    // attempt to solve the planning problem in the given time
    // RRT* will continue to improve the solution until time runs out
    ob::PlannerStatus solved;
    
    if (planner_type_ == "RRTstar") {
        // For RRTstar, use a custom termination condition that stops when:
        // 1. Solution cost improvement is minimal over time
        // 2. Maximum planning time is reached
        double last_cost = std::numeric_limits<double>::infinity();
        double cost_improvement_threshold = 0.01; // 1% improvement threshold
        int no_improvement_iterations = 0;
        const int max_no_improvement = 10;
        
        // Run planner with intermediate checks
        auto start_time = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count() < planning_time_) {
            // Run planner for a short burst
            solved = planner->ob::Planner::solve(0.5);
            
            if (solved && pdef->hasExactSolution()) {
                double current_cost = pdef->getSolutionPath()->cost(pdef->getOptimizationObjective()).value();
                double improvement = (last_cost - current_cost) / last_cost;
                
                if (improvement < cost_improvement_threshold) {
                    no_improvement_iterations++;
                    if (no_improvement_iterations >= max_no_improvement) {
                        RCLCPP_INFO(rclcpp::get_logger("planner_2d"), 
                                   "RRTstar early termination: Cost converged at %.3f after %.2fs", 
                                   current_cost, 
                                   std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count());
                        break;
                    }
                } else {
                    no_improvement_iterations = 0;
                }
                
                last_cost = current_cost;
            }
        }
    } else {
        // For other planners, use standard solve
        solved = planner->ob::Planner::solve(planning_time_);
    }

    nav_msgs::msg::Path plannedPath;
    if (solved) {
        // Log the final solution cost
        if (planner_type_ == "RRTstar") {
            auto cost = pdef->getSolutionPath()->cost(pdef->getOptimizationObjective());
            
            // Get planning data for additional info
            ob::PlannerData plannerData(si);
            planner->getPlannerData(plannerData);
            
            RCLCPP_INFO(rclcpp::get_logger("planner_2d"), 
                        "RRTstar planning complete: Final cost: %.3f, States in tree: %u, Planning time: %.2fs", 
                        cost.value(), plannerData.numVertices(), planning_time_);
        }
        
        // get the goal representation from the problem definition (not the same as the goal state)
        // and inquire about the found path
        ob::PathPtr path = pdef->getSolutionPath();
        og::PathGeometric* pth = pdef->getSolutionPath()->as<og::PathGeometric>();
        
        // Optimize the path using OMPL's path simplifier
        og::PathSimplifier pathSimplifier(si);
        
        // For RRTstar, apply minimal simplification to preserve optimality
        if (planner_type_ == "RRTstar") {
            // Only remove redundant vertices that don't affect path quality
            pathSimplifier.collapseCloseVertices(*pth, 0.1);
            // Light smoothing to improve path following without losing optimality
            pathSimplifier.smoothBSpline(*pth, 2);
        } else {
            // For non-optimal planners, apply more aggressive simplification
            pathSimplifier.simplifyMax(*pth);
            pathSimplifier.collapseCloseVertices(*pth);
            pathSimplifier.reduceVertices(*pth);
            pathSimplifier.smoothBSpline(*pth, 5);
            pathSimplifier.simplify(*pth, 2.0);
        }
        
        plannedPath = extractPath(pdef.get());
    } else {
        // Return empty path if no solution found
        plannedPath.header.frame_id = "/map";
    }
    return plannedPath;
}

void Planner2D::setStartPosition(double x, double y, double theta) {
    auto *se2state = start->get()->as<ob::SE2StateSpace::StateType>();
    se2state->setX(x);
    se2state->setY(y);
    se2state->setYaw(theta);
    startSet = true;
}

void Planner2D::setGoalPosition(double x, double y, double theta) {
    auto *se2state = goal->get()->as<ob::SE2StateSpace::StateType>();
    se2state->setX(x);
    se2state->setY(y);
    se2state->setYaw(theta);
    goalSet = true;
}

bool Planner2D::isReadyToPlan() const {
    return startSet && goalSet;
}

void Planner2D::setFootprint(const geometry_msgs::msg::Polygon& footprint) {
    footprint_ = footprint;
    robotFootprint = footprint;
}

void Planner2D::setTreatUnknownAsFree(bool treat_unknown_as_free) {
    treat_unknown_as_free_ = treat_unknown_as_free;
    g_treat_unknown_as_free = treat_unknown_as_free;
}

void Planner2D::setPlanningTime(double planning_time) {
    planning_time_ = planning_time;
}

void Planner2D::setPlannerType(const std::string& planner_type) {
    planner_type_ = planner_type;
}

void Planner2D::setSafetyMargin(double safety_margin) {
    g_safety_margin = safety_margin;
    RCLCPP_INFO(rclcpp::get_logger("planner_2d"), "Set safety margin to: %.3f meters", g_safety_margin);
}

void Planner2D::setOccupancyThreshold(int occupancy_threshold) {
    g_occupancy_threshold = occupancy_threshold;
    RCLCPP_INFO(rclcpp::get_logger("planner_2d"), "Set occupancy threshold to: %d", g_occupancy_threshold);
}

double Planner2D::getSafetyMargin() const {
    return g_safety_margin;
}

bool Planner2D::isStateValid(double x, double y, double theta) const {
    // Create a temporary state for validation
    auto temp_space = std::make_shared<ob::SE2StateSpace>();
    ob::ScopedState<> state(temp_space);
    auto* se2state = state->as<ob::SE2StateSpace::StateType>();
    se2state->setX(x);
    se2state->setY(y);
    se2state->setYaw(theta);
    
    // Use the global isStateValid function
    return path_planning::isStateValid(state.get());
}

void Planner2D::setMapAndObstacles(const nav_msgs::msg::OccupancyGrid& map, 
                                  const std::vector<CylinderObstacle>& obstacles) {
    occupancyMap = map;
    g_cylinder_obstacles = obstacles;
}

}
