#!/usr/bin/env python3
import os
import cv2
import numpy as np
import math
import time
import sys
from triton_bringup.omnicv import fisheyeImgConv

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageProcessing(Node):
    def __init__(self):
        super().__init__('image_processing_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/equirectangular/image',
            self.cubemap, 10
        )
        self.cubemap_pub = self.create_publisher(Image, '/cubemap/image', 10)
        self.front_pub = self.create_publisher(Image, '/cubemap_front/image', 10)
        self.back_pub = self.create_publisher(Image, '/cubemap_back/image', 10)
        self.left_pub = self.create_publisher(Image, '/cubemap_left/image', 10)
        self.right_pub = self.create_publisher(Image, '/cubemap_right/image', 10)
        # self.down_pub = self.create_publisher(Image, '/cubemap_down/image', 10)
        self.mapper = fisheyeImgConv()

    def cubemap(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        height, width = cv_image.shape[:2]
        side = int(width/4)
        cubemap = self.mapper.equirect2cubemap(cv_image, side=side, dice=True)
        
        cubemap_msg = self.bridge.cv2_to_imgmsg(cubemap, encoding='rgb8')
        cubemap_msg.header = msg.header
        self.cubemap_pub.publish(cubemap_msg)
        self.get_logger().info('Published cubemap image')

        front = cubemap[side:2*side, side:2*side]
        front_msg = self.bridge.cv2_to_imgmsg(front, encoding='rgb8')
        front_msg.header = msg.header
        front_msg.header.frame_id = 'front_camera_frame'
        self.front_pub.publish(front_msg)

        back = cubemap[side:2*side, 3*side:4*side]
        back_msg = self.bridge.cv2_to_imgmsg(back, encoding='rgb8')
        back_msg.header = msg.header
        back_msg.header.frame_id = 'back_camera_frame'
        self.back_pub.publish(back_msg)

        left = cubemap[side:2*side, 0:side]
        left_msg = self.bridge.cv2_to_imgmsg(left, encoding='rgb8')
        left_msg.header = msg.header
        left_msg.header.frame_id = 'left_camera_frame'
        self.left_pub.publish(left_msg)

        right = cubemap[side:2*side, 2*side:3*side]
        right_msg = self.bridge.cv2_to_imgmsg(right, encoding='rgb8')
        right_msg.header = msg.header
        right_msg.header.frame_id = 'right_camera_frame'
        self.right_pub.publish(right_msg)

        # down = cubemap[2*side:3*side, side:2*side]
        # down_msg = self.bridge.cv2_to_imgmsg(down, encoding='rgb8')
        # down_msg.header = msg.header
        # self.down_pub.publish(down_msg)


def main(args=None):
    rclpy.init()
    node = ImageProcessing()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

