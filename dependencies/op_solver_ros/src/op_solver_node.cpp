#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <filesystem>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

class OpSolverNode : public rclcpp::Node
{
public:
    OpSolverNode() : Node("op_solver_node")
    {
        this->declare_parameter<std::string>("input_file", "");
        this->declare_parameter<std::string>("output_file", "/home/john/git/op_ws/stats.json");
        this->declare_parameter<int>("time_limit", 18000000); // Note: currently not used by op-solver
        this->declare_parameter<bool>("exact", false);
        
        solve_service_ = this->create_service<std_srvs::srv::Trigger>(
            "solve_problem", 
            std::bind(&OpSolverNode::solveProblem, this, std::placeholders::_1, std::placeholders::_2));
            
        RCLCPP_INFO(this->get_logger(), "Op Solver Node initialized");
    }

private:
    void solveProblem(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        std::string input_file = this->get_parameter("input_file").as_string();
        std::string output_file = this->get_parameter("output_file").as_string();
        bool exact = this->get_parameter("exact").as_bool();
        
        if (input_file.empty()) {
            response->success = false;
            response->message = "No input file specified. Set the 'input_file' parameter.";
            return;
        }
        
        // Check if input file exists
        std::ifstream infile(input_file);
        if (!infile.good()) {
            response->success = false;
            response->message = "Input file does not exist: " + input_file;
            return;
        }
        infile.close();
        
        // Build command to run op-solver
        std::stringstream cmd;
        // Get the path to the op-solver executable
        std::filesystem::path current_path = std::filesystem::current_path();
        std::filesystem::path op_solver_path = current_path / "build" / "op_solver_ros" / "op-solver-build" / "src" / "op-solver";
        
        cmd << op_solver_path.string() << " opt ";
        cmd << "--op-exact " << (exact ? "1" : "0") << " ";
        cmd << "--stats " << output_file << " ";
        cmd << input_file;
        
        RCLCPP_INFO(this->get_logger(), "Running command: %s", cmd.str().c_str());
        
        // Execute the command
        int result = std::system(cmd.str().c_str());
        
        if (result == 0) {
            // Read the output file
            std::ifstream output_stream(output_file);
            if (output_stream.is_open()) {
                std::stringstream buffer;
                buffer << output_stream.rdbuf();
                response->success = true;
                response->message = buffer.str();
                output_stream.close();
            } else {
                response->success = false;
                response->message = "Failed to read output file: " + output_file;
            }
        } else {
            response->success = false;
            response->message = "Op-solver execution failed with code: " + std::to_string(result);
        }
    }
    
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr solve_service_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OpSolverNode>());
    rclcpp::shutdown();
    return 0;
}