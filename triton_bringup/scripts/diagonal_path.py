#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from tf2_ros import Buffer, TransformListener, TransformException


class DiagonalPath(Node):
    def __init__(self) -> None:
        super().__init__('diagonal_path')

        # Publisher for waypoints used by triton_navigation/path_planning_node
        self.waypoints_pub = self.create_publisher(PoseArray, '/waypoints', 10)

        # TF buffer/listener for map -> base_link transform
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publish at 1 Hz so late-starting consumers still receive waypoints
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self) -> None:
        try:
            # Transform that converts base_link-frame coordinates into map-frame coordinates
            transform = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
        except TransformException as ex:
            # Wait until TF is available
            self.get_logger().warn(f'Failed to get transform map->base_link: {ex}')
            return

        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        q = transform.transform.rotation

        # Extract yaw from quaternion (same convention as in path_planning_node.cpp)
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        def to_map_frame(x_b: float, y_b: float) -> tuple[float, float]:
            """
            Convert a 2D point from base_link frame (x forward, y left)
            into the map frame using the current transform.
            """
            x_m = cos_yaw * x_b - sin_yaw * y_b + tx
            y_m = sin_yaw * x_b + cos_yaw * y_b + ty
            return x_m, y_m

        # Define a 10 m x 10 m square in base_link frame:
        #   A (start) = (0, 0)
        #   B = (10, 0)      -> 10 m in front
        #   C = (10, -10)    -> 10 m in front, then 10 m to the right
        #   D = (0, -10)     -> 10 m to the right
        #
        # A standard perimeter would use [B, C, D]. Here we swap B and C
        # to force the vehicle to traverse a diagonal twice:
        #   A -> C (diagonal), C -> B (side), B -> D (diagonal).
        waypoints_base = [
            (10.0, -10.0),   # C: forward 10, right 10
            (10.0, 0.0),     # B: forward 10
            (0.0, -10.0),    # D: right 10
            (0.0, -5.0),     # 2 m before origin along D->A
        ]

        pose_array = PoseArray()
        pose_array.header.frame_id = 'map'
        pose_array.header.stamp = self.get_clock().now().to_msg()

        for x_b, y_b in waypoints_base:
            x_m, y_m = to_map_frame(x_b, y_b)
            pose = Pose()
            pose.position.x = x_m
            pose.position.y = y_m
            pose.position.z = 0.0
            # Orientation will be determined by the planner; use identity here
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.waypoints_pub.publish(pose_array)
        self.get_logger().info(
            f'Published {len(pose_array.poses)} diagonal waypoints on /waypoints in map frame')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DiagonalPath()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
