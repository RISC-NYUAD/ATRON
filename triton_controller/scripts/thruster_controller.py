#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import json
import serial

class ThrusterController(Node):
    def __init__(self):
        super().__init__('thruster_controller')
        
        # Serial connection parameters
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        
        # Velocity limits
        self.declare_parameter('max_linear_velocity', 1.0)  # m/s
        self.declare_parameter('max_angular_velocity', 0.314)  # rad/s
        
        # Idle thrust to keep motors primed (1500 is neutral, > 1500 is forward)
        self.declare_parameter('idle_thrust', 1510)  # Small forward thrust
        self.declare_parameter('cmd_vel_timeout', 0.5)  # seconds
        
        serial_port = self.get_parameter('serial_port').get_parameter_value().string_value
        baud_rate = self.get_parameter('baud_rate').get_parameter_value().integer_value
        self.max_linear_vel = self.get_parameter('max_linear_velocity').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter('max_angular_velocity').get_parameter_value().double_value
        self.idle_thrust = self.get_parameter('idle_thrust').get_parameter_value().integer_value
        self.cmd_vel_timeout = self.get_parameter('cmd_vel_timeout').get_parameter_value().double_value
        
        try:
            self.serial_conn = serial.Serial(serial_port, baud_rate, timeout=1)
            self.get_logger().info(f'Connected to serial port: {serial_port} at {baud_rate} baud')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to serial port: {e}')
            self.serial_conn = None
            
        # Command template (initialized with idle thrust)
        self.command_template ={
            "GET_IMU": 0, 
            "SET_RC_MODE": 0,
            "SET_CONVEYOR_MODE": 0,
            "ESC1": self.idle_thrust,  # Idle thrust
            "ESC2": self.idle_thrust,  # Idle thrust
            "ESC3": self.idle_thrust,  # Idle thrust
            "ESC4": self.idle_thrust   # Idle thrust
        }
        
        # Subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Publisher for PWM values
        self.pwm_pub = self.create_publisher(Float32MultiArray, '/pwm', 10)
        
        # Timer to send commands at regular intervals
        self.timer = self.create_timer(0.1, self.send_command)  # 10Hz
        
        # Track last cmd_vel time
        self.last_cmd_vel_time = self.get_clock().now()
        self.is_idle = False
        
        self.get_logger().info(f'Thruster Controller node started (idle thrust: {self.idle_thrust})')
    
    def cmd_vel_callback(self, msg):
        """Convert Twist message to ESC commands"""
        # Update last command time
        self.last_cmd_vel_time = self.get_clock().now()
        self.is_idle = False
        
        # TODO: Implement proper mixing for your thruster configuration
        # This is a simple example for differential drive
        
        linear_x = msg.linear.x  # Forward/backward
        angular_z = msg.angular.z  # Rotation
        
        # Clamp input velocities to maximum values
        linear_x = max(-self.max_linear_vel, min(self.max_linear_vel, linear_x))
        angular_z = max(-self.max_angular_vel, min(self.max_angular_vel, angular_z))
        
        # Normalize velocities to [-1, 1] range
        linear_normalized = linear_x / self.max_linear_vel if self.max_linear_vel > 0 else 0
        angular_normalized = angular_z / self.max_angular_vel if self.max_angular_vel > 0 else 0
        
        # Map normalized velocities to ESC values (1000-2000 range, 1500 is neutral)
        # Linear component can use full range (±500)
        # Angular component uses partial range to allow for mixing
        linear_component = int(linear_normalized * 500)
        angular_component = int(angular_normalized * 300)
        
        # Calculate thrust for each side (using idle_thrust as the center point)
        left_thrust = self.idle_thrust + linear_component - angular_component
        right_thrust = self.idle_thrust + linear_component + angular_component
        
        # Clamp values to valid ESC range
        left_thrust = max(1000, min(2000, left_thrust))
        right_thrust = max(1000, min(2000, right_thrust))
        
        # Update ESC values (adjust mapping to your thruster layout)
        self.command_template["ESC1"] = left_thrust
        self.command_template["ESC2"] = right_thrust
        self.command_template["ESC3"] = left_thrust
        self.command_template["ESC4"] = right_thrust
        
        # Print ESC values
        self.get_logger().info(
            f'CMD received - Linear: {linear_x:.2f} m/s, Angular: {angular_z:.2f} rad/s | '
            f'ESC values - ESC1: {left_thrust}, ESC2: {right_thrust}, '
            f'ESC3: {left_thrust}, ESC4: {right_thrust}'
        )
    
    def send_command(self):
        """Send command to serial"""
        if self.serial_conn is None:
            return
        
        # Check if we've timed out on cmd_vel
        time_since_last_cmd = (self.get_clock().now() - self.last_cmd_vel_time).nanoseconds / 1e9
        
        if time_since_last_cmd > self.cmd_vel_timeout:
            if not self.is_idle:
                # Transition to idle mode
                self.is_idle = True
                self.get_logger().info(f'No cmd_vel received for {time_since_last_cmd:.1f}s, switching to idle thrust mode')
                
            # Set all ESCs to idle thrust value
            self.command_template["ESC1"] = self.idle_thrust
            self.command_template["ESC2"] = self.idle_thrust
            self.command_template["ESC3"] = self.idle_thrust
            self.command_template["ESC4"] = self.idle_thrust
            
        try:
            # Convert to JSON and send
            json_str = json.dumps(self.command_template)
            self.serial_conn.write((json_str + '\n').encode('utf-8'))
            
            # Publish PWM values
            pwm_msg = Float32MultiArray()
            
            # Set dimensions
            dim = MultiArrayDimension()
            dim.label = "ESC_values"
            dim.size = 4
            dim.stride = 4
            pwm_msg.layout.dim = [dim]
            pwm_msg.layout.data_offset = 0
            
            # Set data
            pwm_msg.data = [
                float(self.command_template["ESC1"]),
                float(self.command_template["ESC2"]),
                float(self.command_template["ESC3"]),
                float(self.command_template["ESC4"])
            ]
            self.pwm_pub.publish(pwm_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to send command: {e}')
            
    def emergency_stop(self):
        """Set all thrusters to idle thrust (keeps motors primed)"""
        self.command_template["ESC1"] = self.idle_thrust
        self.command_template["ESC2"] = self.idle_thrust
        self.command_template["ESC3"] = self.idle_thrust
        self.command_template["ESC4"] = self.idle_thrust
        self.send_command()
        self.get_logger().info(f'Emergency stop - set all ESCs to idle thrust: {self.idle_thrust}')

def main(args=None):
    rclpy.init(args=args)
    thruster_controller = ThrusterController()
    
    try:
        rclpy.spin(thruster_controller)
    except KeyboardInterrupt:
        thruster_controller.emergency_stop()
    finally:
        if thruster_controller.serial_conn:
            thruster_controller.serial_conn.close()
        thruster_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()