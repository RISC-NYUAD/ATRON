#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class UndistortNode(Node):
    def __init__(self):
        super().__init__('undistort_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Declare parameter for output image size
        self.declare_parameter('output_size', 960)
        self.output_size = self.get_parameter('output_size').value
        
        # Camera parameters
        self.fx = 305.85 * (1920/1152)
        self.fy = 305.85 * (1920/1152)
        self.cx = 576.0  * (1920/1152)
        self.cy = 576.0  * (1920/1152)
        
        # Create camera matrix
        self.K = np.array([[self.fx, 0, self.cx],
                          [0, self.fy, self.cy],
                          [0, 0, 1]], dtype=np.float32)
        
        # Fisheye distortion coefficients
        self.D = np.array([0.0829955798117611,
                          -0.027906274475464777,
                          0.0076202648985968895,
                          -0.0010836351255689319], dtype=np.float32)
        
        # Subscribe to dual fisheye image
        self.subscription = self.create_subscription(
            Image,
            '/dual_fisheye/image',
            self.image_callback,
            10
        )
        
        # Publishers for undistorted images
        self.front_pub = self.create_publisher(Image, '/cubemap_front/image', 10)
        self.back_pub = self.create_publisher(Image, '/cubemap_back/image', 10)
        
        self.get_logger().info('Undistort node initialized')
        
        # Initialize undistortion maps (will be created on first image)
        self.map1_front = None
        self.map2_front = None
        self.map1_back = None
        self.map2_back = None
        self.image_size = None
    
    def image_callback(self, msg):
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Get image dimensions
            height, width = cv_image.shape[:2]
            single_width = width // 2
            
            # Initialize undistortion maps if not done yet
            if self.map1_front is None:
                self.image_size = (single_width, height)
                self.initialize_undistortion_maps(single_width, height)
            
            # Split the image into front and back
            front_image = cv_image[:, single_width:]
            front_image = cv2.rotate(front_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            back_image = cv_image[:, :single_width]
            back_image = cv2.rotate(back_image, cv2.ROTATE_90_CLOCKWISE)
            
            # Undistort both images
            front_undistorted = cv2.remap(front_image, self.map1_front, self.map2_front, 
                                         interpolation=cv2.INTER_LINEAR, 
                                         borderMode=cv2.BORDER_CONSTANT)
            
            back_undistorted = cv2.remap(back_image, self.map1_back, self.map2_back, 
                                        interpolation=cv2.INTER_LINEAR, 
                                        borderMode=cv2.BORDER_CONSTANT)
            
            # Resize the undistorted images to the desired output size
            front_undistorted = cv2.resize(front_undistorted, 
                                         (self.output_size, self.output_size), 
                                         interpolation=cv2.INTER_LINEAR)
            
            back_undistorted = cv2.resize(back_undistorted, 
                                        (self.output_size, self.output_size), 
                                        interpolation=cv2.INTER_LINEAR)
            
            # Convert back to ROS messages and publish
            front_msg = self.bridge.cv2_to_imgmsg(front_undistorted, encoding='bgr8')
            back_msg = self.bridge.cv2_to_imgmsg(back_undistorted, encoding='bgr8')
            
            # Copy header from original message
            front_msg.header.stamp = msg.header.stamp
            front_msg.header.frame_id = 'front_camera_frame'
            back_msg.header.stamp = msg.header.stamp
            back_msg.header.frame_id = 'back_camera_frame'
            
            # Publish
            self.front_pub.publish(front_msg)
            self.back_pub.publish(back_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in image callback: {str(e)}')
    
    def initialize_undistortion_maps(self, width, height):
        """Initialize the undistortion maps for fisheye cameras"""
        # Create new camera matrix for undistorted image
        new_K = self.K.copy()
        
        # Create undistortion maps
        self.map1_front, self.map2_front = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), new_K, (width, height), cv2.CV_16SC2
        )
        
        # For back camera, use the same maps (assuming identical cameras)
        self.map1_back = self.map1_front
        self.map2_back = self.map2_front
        
        self.get_logger().info(f'Initialized undistortion maps for {width}x{height} images')


def main(args=None):
    rclpy.init(args=args)
    node = UndistortNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()