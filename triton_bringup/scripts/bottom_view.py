#!/usr/bin/env python3

import numpy as np
from triton_bringup.omnicv import fisheyeImgConv

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class BottomView(Node):
    def __init__(self):
        super().__init__('bottom_view_node')
        self.bridge = CvBridge()
        self.mapper = fisheyeImgConv()
        
        # Configure QoS for reliable communication with buffer size 1
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE
        )
        
        self.image_sub = self.create_subscription(
            Image,
            '/equirectangular/image',
            self.bottom_view,
            qos
        )

        self.bottom_view_pub = self.create_publisher(
            Image,
            '/bottom_view/image',
            qos
        )

    def bottom_view(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        bottom_view = self.mapper.eqruirect2persp(frame, 150, 5, -90, 640, 640)
        bottom_msg = self.bridge.cv2_to_imgmsg(bottom_view, encoding='rgb8')
        bottom_msg.header = msg.header
        self.bottom_view_pub.publish(bottom_msg)


def main():
    rclpy.init()
    node = BottomView()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
