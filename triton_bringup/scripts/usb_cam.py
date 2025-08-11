#!/usr/bin/env python3
# filepath: /home/abanesjo/ros2_ws/src/Triton/triton_bringup/scripts/usb_cam.py
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class UsbCameraNode(Node):
    def __init__(self):
        super().__init__('usb_camera_node')
        
        # Initialize the publisher
        self.publisher = self.create_publisher(
            Image,
            '/camera',
            10
        )
        
        # Initialize camera
        self.cam = cv2.VideoCapture(4)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1472)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 736)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create timer for camera capture
        self.timer = self.create_timer(0.033, self.timer_callback)  # ~30fps
        
        self.get_logger().info('USB camera node started')
    
    def timer_callback(self):
        ret, frame = self.cam.read()
        if not ret:
            self.get_logger().error('Failed to grab frame')
            return
        
        # Get image dimensions
        height, width, _ = frame.shape
        
        # Split the image vertically - take only left half
        left_half = frame[:, 0:width//2]
        
        # Convert to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(left_half, encoding='bgr8')
        
        # Publish
        self.publisher.publish(ros_image)
    
    def __del__(self):
        if hasattr(self, 'cam'):
            self.cam.release()

def main(args=None):
    rclpy.init(args=args)
    
    usb_cam_node = UsbCameraNode()
    
    try:
        rclpy.spin(usb_cam_node)
    except KeyboardInterrupt:
        usb_cam_node.get_logger().info('Node stopped by keyboard interrupt')
    finally:
        usb_cam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()