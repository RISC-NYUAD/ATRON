#pragma once

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/base/SpaceInformation.h>
#include <ompl/geometric/PathSimplifier.h>
#include <ompl/base/OptimizationObjective.h>
#include <ompl/base/Goal.h>
#include <ompl/base/goals/GoalState.h>

#include <iostream>

namespace path_planning
{

// Structure to represent cylinder obstacles
struct CylinderObstacle {
    double x, y;     // Center position
    double radius;   // Radius of the cylinder
};

class Planner2D
{
private:
    double maxStepLength;
    std::shared_ptr<ompl::base::RealVectorBounds> bounds;

    std::shared_ptr<ompl::base::ScopedState<>> start;
    std::shared_ptr<ompl::base::ScopedState<>> goal;
    std::shared_ptr<ompl::base::SE2StateSpace> space;
    
    bool startSet;
    bool goalSet;
    
    // Robot footprint
    geometry_msgs::msg::Polygon footprint_;
    
    // Planning parameters
    bool treat_unknown_as_free_;
    double planning_time_;
    std::string planner_type_;

    void configure(void);

    nav_msgs::msg::Path extractPath(ompl::base::ProblemDefinition* pdef);

public:
    Planner2D(void);

    virtual ~Planner2D();

    nav_msgs::msg::Path planPath(const nav_msgs::msg::OccupancyGrid& globalMap);
    nav_msgs::msg::Path planPath(const nav_msgs::msg::OccupancyGrid& globalMap, 
                                const std::vector<CylinderObstacle>& cylinderObstacles);
    
    void setStartPosition(double x, double y, double theta = 0.0);
    void setGoalPosition(double x, double y, double theta = 0.0);
    bool isReadyToPlan() const;
    void setFootprint(const geometry_msgs::msg::Polygon& footprint);
    void setTreatUnknownAsFree(bool treat_unknown_as_free);
    void setPlanningTime(double planning_time);
    void setPlannerType(const std::string& planner_type);
    void setSafetyMargin(double safety_margin);
    void setOccupancyThreshold(int occupancy_threshold);
    double getSafetyMargin() const;
    
    // Public collision checking methods
    bool isStateValid(double x, double y, double theta) const;
    void setMapAndObstacles(const nav_msgs::msg::OccupancyGrid& map, 
                           const std::vector<CylinderObstacle>& obstacles);
};
}
