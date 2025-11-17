#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose


class CustomPath(Node):
    def __init__(self) -> None:
        super().__init__('custom_path')

        # Publisher for waypoints used by triton_navigation/path_planning_node
        self.waypoints_pub = self.create_publisher(PoseArray, '/waypoints', 10)

        # Publish at 1 Hz so late-starting consumers still receive waypoints
        self.timer = self.create_timer(1.0, self.timer_callback)

        # Waypoints in map frame (x, y) as recorded:
        #  (9.21,  0.07)
        #  (9.16, -8.01)
        #  (5.96, -8.02)
        #  (6.01, -3.88)
        #  (3.91, -4.05)
        #  (3.99, -7.98)
        #  (2.06, -8.07)
        #  (2.02, -4.99)
        #  (-0.73, -5.06)
        self._waypoints_map = [
            (9.21, 0.07),
            (9.16, -8.01),
            (5.96, -8.02),
            (6.01, -3.88),
            (3.91, -4.05),
            (3.99, -7.98),
            (2.06, -8.07),
            (2.02, -4.99),
            (-0.73, -5.06),
        ]

    def timer_callback(self) -> None:
        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'
        pose_array.header.stamp = self.get_clock().now().to_msg()

        for x_m, y_m in self._waypoints_map:
            pose = Pose()
            pose.position.x = x_m
            pose.position.y = y_m
            pose.position.z = 0.0
            # Orientation will be determined by the planner; use identity here
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.waypoints_pub.publish(pose_array)
        self.get_logger().info(
            f'Published {len(pose_array.poses)} custom waypoints on /waypoints in map frame')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CustomPath()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
