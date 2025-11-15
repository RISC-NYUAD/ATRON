#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import sys
import os

# Add the scripts directory to path to import PID
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PID import PID


class USVPIDController(Node):
    def __init__(self):
        super().__init__('usv_pid_controller')
        
        # Hardcoded parameter defaults
        # Linear velocity PID gains
        self.Kp_linear = 15.0
        self.Ki_linear = 2.0
        self.Kd_linear = 5.0
        
        # Angular velocity PID gains
        self.Kp_angular = 25.0
        self.Ki_angular = 3.0
        self.Kd_angular = 8.0
        
        # Thrust limits (in Newtons)
        self.max_thrust = 40.0
        self.min_thrust = -20.0
        
        # Declare parameters (with hardcoded defaults)
        self.declare_parameter('Kp_linear', self.Kp_linear)
        self.declare_parameter('Ki_linear', self.Ki_linear)
        self.declare_parameter('Kd_linear', self.Kd_linear)
        self.declare_parameter('Kp_angular', self.Kp_angular)
        self.declare_parameter('Ki_angular', self.Ki_angular)
        self.declare_parameter('Kd_angular', self.Kd_angular)
        self.declare_parameter('max_thrust', self.max_thrust)
        self.declare_parameter('min_thrust', self.min_thrust)
        
        # Get parameters
        self.Kp_linear = self.get_parameter('Kp_linear').get_parameter_value().double_value
        self.Ki_linear = self.get_parameter('Ki_linear').get_parameter_value().double_value
        self.Kd_linear = self.get_parameter('Kd_linear').get_parameter_value().double_value
        self.Kp_angular = self.get_parameter('Kp_angular').get_parameter_value().double_value
        self.Ki_angular = self.get_parameter('Ki_angular').get_parameter_value().double_value
        self.Kd_angular = self.get_parameter('Kd_angular').get_parameter_value().double_value
        self.max_thrust = self.get_parameter('max_thrust').get_parameter_value().double_value
        self.min_thrust = self.get_parameter('min_thrust').get_parameter_value().double_value
        
        # Initialize PID controllers
        self.linear_pid = PID(
            Kp=self.Kp_linear,
            Ki=self.Ki_linear,
            Kd=self.Kd_linear,
            setpoint=0.0
        )
        
        self.angular_pid = PID(
            Kp=self.Kp_angular,
            Ki=self.Ki_angular,
            Kd=self.Kd_angular,
            setpoint=0.0
        )
        
        # State variables
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.desired_linear_velocity = 0.0
        self.desired_angular_velocity = 0.0
        
        # Create subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
        # Create publishers for thruster forces
        self.thruster_left_pub = self.create_publisher(
            Float32,
            '/thruster_left',
            10
        )
        
        self.thruster_right_pub = self.create_publisher(
            Float32,
            '/thruster_right',
            10
        )
        
        # Control loop timer (50 Hz)
        self.control_timer = self.create_timer(0.02, self.control_loop)
        
        self.get_logger().info('USV PID Controller initialized')
        self.get_logger().info(f'Linear PID: Kp={self.Kp_linear}, Ki={self.Ki_linear}, Kd={self.Kd_linear}')
        self.get_logger().info(f'Angular PID: Kp={self.Kp_angular}, Ki={self.Ki_angular}, Kd={self.Kd_angular}')
        self.get_logger().info(f'Thrust limits: [{self.min_thrust}, {self.max_thrust}] N')
    
    def cmd_vel_callback(self, msg: Twist):
        """Handle desired velocity commands"""
        self.desired_linear_velocity = msg.linear.x
        self.desired_angular_velocity = msg.angular.z
        
        # Update PID setpoints
        self.linear_pid.set_setpoint(self.desired_linear_velocity)
        self.angular_pid.set_setpoint(self.desired_angular_velocity)
    
    def odom_callback(self, msg: Odometry):
        """Extract current velocities from odometry"""
        # Linear velocity in robot frame (x-axis)
        self.current_linear_velocity = msg.twist.twist.linear.x
        # Angular velocity around z-axis
        self.current_angular_velocity = msg.twist.twist.angular.z
    
    def control_loop(self):
        """Main control loop - compute PID and publish thrust commands"""
        # Compute PID outputs (these are effort/force values)
        linear_effort = self.linear_pid.update(self.current_linear_velocity)
        angular_effort = self.angular_pid.update(self.current_angular_velocity)
        
        # Differential drive mixing
        # For a USV with two thrusters (left and right):
        # - Both thrusters pushing forward creates linear motion
        # - Differential thrust creates rotation
        # 
        # Linear effort contributes equally to both thrusters
        # Angular effort creates differential (opposite on each side)
        
        left_thrust = linear_effort - angular_effort
        right_thrust = linear_effort + angular_effort
        
        # Apply thrust limits
        left_thrust = self.clamp_thrust(left_thrust)
        right_thrust = self.clamp_thrust(right_thrust)
        
        # Publish thrust commands
        left_msg = Float32()
        left_msg.data = float(left_thrust)
        self.thruster_left_pub.publish(left_msg)
        
        right_msg = Float32()
        right_msg.data = float(right_thrust)
        self.thruster_right_pub.publish(right_msg)
        
        # Debug logging (reduced frequency)
        if self.get_clock().now().nanoseconds % 1000000000 < 20000000:  # Log roughly once per second
            self.get_logger().debug(
                f'Velocities - Desired: [{self.desired_linear_velocity:.3f}, {self.desired_angular_velocity:.3f}], '
                f'Current: [{self.current_linear_velocity:.3f}, {self.current_angular_velocity:.3f}]'
            )
            self.get_logger().debug(
                f'PID Efforts - Linear: {linear_effort:.2f}, Angular: {angular_effort:.2f}'
            )
            self.get_logger().debug(
                f'Thrust Commands - Left: {left_thrust:.2f} N, Right: {right_thrust:.2f} N'
            )
    
    def clamp_thrust(self, thrust: float) -> float:
        """Clamp thrust value to allowable range"""
        return max(self.min_thrust, min(self.max_thrust, thrust))
    
    def emergency_stop(self):
        """Stop all thrusters"""
        stop_msg = Float32()
        stop_msg.data = 0.0
        self.thruster_left_pub.publish(stop_msg)
        self.thruster_right_pub.publish(stop_msg)
        self.get_logger().warn('Emergency stop - all thrusters set to 0 N')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        usv_controller = USVPIDController()
        rclpy.spin(usv_controller)
    except KeyboardInterrupt:
        usv_controller.get_logger().info('Shutting down USV PID Controller')
        usv_controller.emergency_stop()
    except Exception as e:
        print(f'Error: {e}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()