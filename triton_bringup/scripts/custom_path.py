#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose


class CustomPath(Node):
    def __init__(self) -> None:
        super().__init__('custom_path')

        # Publisher for waypoints used by triton_navigation/path_planning_node
        self.waypoints_pub = self.create_publisher(PoseArray, '/waypoints', 10)

        # Timer used to publish waypoints once after startup
        self._published_once = False
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
        # self._waypoints_map = [
        #     (9.21, 0.07),
        #     (9.16, -8.01),
        #     (5.96, -8.02),
        #     (6.01, -3.88),
        #     (3.91, -4.05),
        #     (3.99, -7.98),
        #     (2.06, -8.07),
        #     (2.02, -4.99),
        #     (-0.73, -5.06),
        # ]

        self._waypoints_map = [
            (0.0, 0.0),
            (10.0, 0.0),      # P1
            (10.0, -10.0),     # P2
            (0.0, -10.0),     # P3
            (0.0, -2.0),
        ]

    @property
    def published_once(self) -> bool:
        return self._published_once

    def timer_callback(self) -> None:
        # Ensure we only publish once, then stop the timer
        if self._published_once:
            # Defensive: cancel timer if it somehow fires again
            if self.timer is not None:
                self.timer.cancel()
            return

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
        self._published_once = True
        # Stop future publications
        self.timer.cancel()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CustomPath()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    try:
        # Spin until the node has published once, then exit
        while rclpy.ok() and not node.published_once:
            executor.spin_once(timeout_sec=0.1)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
