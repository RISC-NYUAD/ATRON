#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import csv
import math
from datetime import datetime


class OdometryComparator(Node):
    def __init__(self):
        super().__init__('odometry_comparator')
        
        # Initialize data storage
        self.odom_data = None
        self.filtered_odom_data = None
        self.imu_data = None
        
        # Create subscribers
        self.odom_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        self.filtered_odom_subscriber = self.create_subscription(
            Odometry,
            '/odometry/filtered',
            self.filtered_odom_callback,
            10
        )
        
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        # Create CSV file
        self.csv_filename = f'odom_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write CSV header
        self.csv_writer.writerow(['timestamp', 'odom_roll', 'odom_pitch', 'odom_yaw', 'odom_vx', 'odom_vy', 'odom_vz', 'imu_roll', 'imu_pitch', 'imu_yaw', 'filtered_roll', 'filtered_pitch', 'filtered_yaw', 'filtered_vx', 'filtered_vy', 'filtered_vz'])
        
        self.get_logger().info(f'Odometry comparator started. Saving data to {self.csv_filename}')

    def quaternion_to_euler(self, x, y, z, w):
        """
        Convert quaternion to euler angles (roll, pitch, yaw)
        Returns yaw in radians
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def odom_callback(self, msg):
        """Callback for /odom topic"""
        orientation = msg.pose.pose.orientation
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        
        # Extract linear velocities (already in base_link frame)
        linear_vel = msg.twist.twist.linear
        
        self.odom_data = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'vx': linear_vel.x,
            'vy': linear_vel.y,
            'vz': linear_vel.z
        }
        
        # Try to write data if we have all measurements
        self.try_write_data()

    def filtered_odom_callback(self, msg):
        """Callback for /odometry/filtered topic"""
        orientation = msg.pose.pose.orientation
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        
        # Extract linear velocities (already in base_link frame)
        linear_vel = msg.twist.twist.linear
        
        self.filtered_odom_data = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'vx': linear_vel.x,
            'vy': linear_vel.y,
            'vz': linear_vel.z
        }
        
        # Try to write data if we have all measurements
        self.try_write_data()

    def imu_callback(self, msg):
        """Callback for /imu/data topic"""
        orientation = msg.orientation
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )
        
        self.imu_data = {
            'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        }
        
        # Try to write data if we have all measurements
        self.try_write_data()

    def try_write_data(self):
        """Write data to CSV if we have all three data sources"""
        if self.odom_data is not None and self.filtered_odom_data is not None and self.imu_data is not None:
            # Use the most recent timestamp
            timestamp = max(self.odom_data['timestamp'], self.filtered_odom_data['timestamp'], self.imu_data['timestamp'])
            
            # Extract all orientation and velocity data
            odom_roll = self.odom_data['roll']
            odom_pitch = self.odom_data['pitch']
            odom_yaw = self.odom_data['yaw']
            odom_vx = self.odom_data['vx']
            odom_vy = self.odom_data['vy']
            odom_vz = self.odom_data['vz']
            
            imu_roll = self.imu_data['roll']
            imu_pitch = self.imu_data['pitch']
            imu_yaw = self.imu_data['yaw']
            
            filtered_roll = self.filtered_odom_data['roll']
            filtered_pitch = self.filtered_odom_data['pitch']
            filtered_yaw = self.filtered_odom_data['yaw']
            filtered_vx = self.filtered_odom_data['vx']
            filtered_vy = self.filtered_odom_data['vy']
            filtered_vz = self.filtered_odom_data['vz']
            
            # Write to CSV
            self.csv_writer.writerow([timestamp, odom_roll, odom_pitch, odom_yaw, odom_vx, odom_vy, odom_vz, imu_roll, imu_pitch, imu_yaw, filtered_roll, filtered_pitch, filtered_yaw, filtered_vx, filtered_vy, filtered_vz])
            self.csv_file.flush()
            
            # Reset data for next comparison
            self.odom_data = None
            self.filtered_odom_data = None
            self.imu_data = None

    def destroy_node(self):
        """Clean up resources when node is destroyed"""
        if hasattr(self, 'csv_file'):
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    odometry_comparator = OdometryComparator()
    
    try:
        rclpy.spin(odometry_comparator)
    except KeyboardInterrupt:
        pass
    finally:
        odometry_comparator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()