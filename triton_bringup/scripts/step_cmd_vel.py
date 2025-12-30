#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class StepCmdVel(Node):
    def __init__(self) -> None:
        super().__init__('step_cmd_vel')

        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        self.step_period = 4.0
        self.publish_rate = 30.0  # Hz
        # self.step_values = [0.0, 0.3, 0.0, -0.3]
        self.step_values = [0.0, 0.6, 0.0, -0.6]
        self.current_index = 0
        self.last_toggle_time = self.get_clock().now()

        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

    def timer_callback(self) -> None:
        now = self.get_clock().now()
        elapsed = (now - self.last_toggle_time).nanoseconds / 1e9

        if elapsed >= self.step_period:
            self.current_index = (self.current_index + 1) % len(self.step_values)
            self.last_toggle_time = now

        msg = Twist()
        # msg.angular.z = self.step_values[self.current_index]
        msg.linear.x = self.step_values[self.current_index]

        self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing cmd_vel linear.x = {msg.linear.x}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StepCmdVel()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
