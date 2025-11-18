#!/usr/bin/env python3
import math
import copy
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_yaw(yaw):
    from geometry_msgs.msg import Quaternion
    q = Quaternion()
    q.w = math.cos(0.5 * yaw)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(0.5 * yaw)
    return q


class OdomThrottle(Node):
    def __init__(self):
        super().__init__('odom_throttle')

        self.last_msg = None

        self.pub = self.create_publisher(Odometry, '/odom', 10)
        self.sub = self.create_subscription(
            Odometry,
            '/odom_true',
            self.cb,
            50,
        )

        self.timer = self.create_timer(0.25, self.on_timer)   # 10 Hz

        # std devs matching the covariances below
        self.sigma_pos = 0.8      # [m]     x,y
        self.sigma_yaw = 0.02       # [rad]
        self.sigma_v = 0.05         # [m/s]   vx,vy
        self.sigma_yaw_rate = 0.02  # [rad/s]

    def cb(self, msg: Odometry):
        self.last_msg = msg

    def on_timer(self):
        if self.last_msg is None:
            return

        # work on a copy so we don't accumulate noise
        msg = copy.deepcopy(self.last_msg)

        # ---- sample noise ----
        dx = np.random.normal(0.0, self.sigma_pos)
        dy = np.random.normal(0.0, self.sigma_pos)
        dyaw = np.random.normal(0.0, self.sigma_yaw)

        dvx = np.random.normal(0.0, self.sigma_v)
        dvy = np.random.normal(0.0, self.sigma_v)
        dyaw_rate = np.random.normal(0.0, self.sigma_yaw_rate)

        # ---- apply to pose ----
        msg.pose.pose.position.x += dx
        msg.pose.pose.position.y += dy

        q = msg.pose.pose.orientation
        yaw = yaw_from_quat(q)
        yaw_noisy = yaw + dyaw
        msg.pose.pose.orientation = quat_from_yaw(yaw_noisy)

        # ---- apply to twist ----
        msg.twist.twist.linear.x += dvx
        msg.twist.twist.linear.y += dvy
        msg.twist.twist.angular.z += dyaw_rate

        # --- Pose covariance (6x6, row-major) ---
        pose_cov = [0.0] * 36
        pose_cov[0]  = self.sigma_pos**2    # var(x)
        pose_cov[7]  = self.sigma_pos**2    # var(y)
        pose_cov[35] = self.sigma_yaw**2    # var(yaw)
        msg.pose.covariance = pose_cov

        # --- Twist covariance (6x6, row-major) ---
        twist_cov = [0.0] * 36
        twist_cov[0]  = self.sigma_v**2           # var(vx)
        twist_cov[7]  = self.sigma_v**2           # var(vy)
        twist_cov[35] = self.sigma_yaw_rate**2    # var(yaw_rate)
        msg.twist.covariance = twist_cov

        self.pub.publish(msg)


def main():
    rclpy.init()
    rclpy.spin(OdomThrottle())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
