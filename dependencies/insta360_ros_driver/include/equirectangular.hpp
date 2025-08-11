#ifndef EQUIRECTANGULAR_HPP
#define EQUIRECTANGULAR_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <mutex>
#include <atomic>

class EquirectangularNode : public rclcpp::Node
{
public:
    explicit EquirectangularNode();
    ~EquirectangularNode();

private:
    // Callback functions
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters);
    
    // Initialization functions
    void loadParameters();
    void updateCameraParameters();
    void initMapping(int img_height, int img_width);
    
    // Processing functions
    cv::Mat createEquirectangular(const cv::Mat& front_img, const cv::Mat& back_img);
    void initCubemapMapping(int equirect_width, int equirect_height);
    void createCubemapFaces(const cv::Mat& equirect_img);
    
    // ROS2 communication
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr dual_fisheye_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr equirect_pub_;
    
    // Cubemap publishers
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cubemap_front_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cubemap_back_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cubemap_left_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr cubemap_right_pub_;
    
    // Parameters
    double cx_offset_;
    double cy_offset_;
    int crop_size_;
    double tx_, ty_, tz_;
    double roll_, pitch_, yaw_;
    bool gpu_enabled_;
    int out_width_;
    int out_height_;
    
    // Cubemap parameters
    bool publish_cubemap_;
    int cubemap_face_size_;
    
    // Camera parameters
    double cx_, cy_;
    cv::Mat back_to_front_rotation_;
    cv::Vec3d back_to_front_translation_;
    
    // Mapping matrices
    cv::Mat front_map_x_, front_map_y_;
    cv::Mat back_map_x_, back_map_y_;
    cv::Mat front_mask_, back_mask_;
    
    // Cubemap mapping matrices
    cv::Mat cubemap_front_map_x_, cubemap_front_map_y_;
    cv::Mat cubemap_back_map_x_, cubemap_back_map_y_;
    cv::Mat cubemap_left_map_x_, cubemap_left_map_y_;
    cv::Mat cubemap_right_map_x_, cubemap_right_map_y_;
    bool cubemap_maps_initialized_;
    
    // State management
    std::atomic<bool> maps_initialized_;
    std::atomic<bool> params_changed_;
    int img_height_;
    int img_width_;
    
    // Thread safety
    std::mutex processing_mutex_;
};

#endif // EQUIRECTANGULAR_HPP