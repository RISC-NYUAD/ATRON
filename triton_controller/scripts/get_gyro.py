#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math


class IMUDataPrinter(Node):
    def __init__(self):
        super().__init__('get_gyro')
        
        # Subscribe to IMU data
        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)
        
        self.get_logger().info('IMU Data Printer node started. Waiting for IMU data...')
    
    def imu_callback(self, msg):
        """Process and print IMU data"""
        
        # Extract quaternion orientation and convert to Euler angles
        
        # Extract angular velocity (rad/s)
        angular_vel = msg.angular_velocity

        yaw_vel = angular_vel.x

        self.get_logger().info(f'Yaw Vel: {yaw_vel} rad/s')


def main(args=None):
    rclpy.init(args=args)
    
    imu_printer = IMUDataPrinter()
    
    try:
        rclpy.spin(imu_printer)
    except KeyboardInterrupt:
        pass
    finally:
        imu_printer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()