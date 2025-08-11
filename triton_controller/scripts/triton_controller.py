#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3, Quaternion
from sensor_msgs.msg import Imu, Joy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import json
import serial
import math
import threading
from PID import PID


class TritonControllerAutonomous(Node):
    def __init__(self):
        super().__init__('triton_controller')

        # Serial connection parameters
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)

        # Velocity limits
        self.declare_parameter('max_linear_velocity', 1.0)  # m/s
        self.declare_parameter('max_angular_velocity', 0.314)  # rad/s

        # Control parameters
        self.declare_parameter('control_frequency', 10.0)  # Hz
        self.declare_parameter('imu_frame_id', 'imu_frame')
        self.declare_parameter('imu_topic', '/imu/data')

        # Conveyor parameters
        self.declare_parameter('conveyor_duty_cycle', 0.5)  # 0.0 to 1.0

        # IMU request parameter
        # Whether to request IMU data
        self.declare_parameter('request_imu', False)

        # Idle thrust to keep motors primed
        self.declare_parameter('idle_thrust', 1510)  # Small forward thrust
        self.declare_parameter('cmd_vel_timeout', 0.5)  # seconds

        serial_port = self.get_parameter(
            'serial_port').get_parameter_value().string_value
        baud_rate = self.get_parameter(
            'baud_rate').get_parameter_value().integer_value
        self.max_linear_vel = self.get_parameter(
            'max_linear_velocity').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter(
            'max_angular_velocity').get_parameter_value().double_value
        control_frequency = self.get_parameter(
            'control_frequency').get_parameter_value().double_value
        imu_frame_id = self.get_parameter(
            'imu_frame_id').get_parameter_value().string_value
        imu_topic = self.get_parameter(
            'imu_topic').get_parameter_value().string_value
        self.conveyor_duty_cycle = self.get_parameter(
            'conveyor_duty_cycle').get_parameter_value().double_value
        self.request_imu = self.get_parameter(
            'request_imu').get_parameter_value().bool_value
        self.idle_thrust = self.get_parameter(
            'idle_thrust').get_parameter_value().integer_value
        self.cmd_vel_timeout = self.get_parameter(
            'cmd_vel_timeout').get_parameter_value().double_value

        # Initialize serial connection
        try:
            self.serial_conn = serial.Serial(
                serial_port, baud_rate, timeout=0.1)
            self.get_logger().info(
                f'Connected to serial port: {serial_port} at {baud_rate} baud')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to serial port: {e}')
            self.serial_conn = None

        # Command template (initialized with idle thrust)
        self.command_template = {
            "GET_IMU": 1 if self.request_imu else 0,  # Request IMU data based on parameter
            "SET_RC_MODE": 0,
            "SET_CONVEYOR_MODE": 0,
            "SET_CONVEYOR_DUTY_CYCLE": 0.5,
            "ESC1": self.idle_thrust,  # Idle thrust
            "ESC2": self.idle_thrust,  # Idle thrust
            "ESC3": self.idle_thrust,  # Idle thrust
            "ESC4": self.idle_thrust   # Idle thrust
        }

        # Latest cmd_vel storage
        self.latest_cmd_vel = Twist()
        self.cmd_vel_lock = threading.Lock()

        # Joystick state
        self.conveyor_enabled = False  # Toggle state for conveyor
        self.prev_y_button_state = False  # Track previous button state for edge detection
        self.joy_lock = threading.Lock()

        # Subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Subscriber for joystick commands
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        # Publisher for IMU data
        self.imu_publisher = self.create_publisher(Imu, imu_topic, 10)

        # Publisher for PWM values
        self.pwm_pub = self.create_publisher(Float32MultiArray, '/pwm', 10)

        # IMU message template
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = imu_frame_id

        # Set covariance matrices
        self.imu_msg.orientation_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]

        self.imu_msg.angular_velocity_covariance = [
            0.001, 0.0, 0.0,
            0.0, 0.001, 0.0,
            0.0, 0.0, 0.001
        ]

        self.imu_msg.linear_acceleration_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]

        # Serial buffer for partial messages
        self.serial_buffer = ""

        # Timer for main control loop at specified frequency
        timer_period = 1.0 / control_frequency
        self.timer = self.create_timer(timer_period, self.control_loop)

        # Track last cmd_vel time
        self.last_cmd_vel_time = self.get_clock().now()
        self.is_idle = False

        self.get_logger().info(
            f'Triton Controller node started at {control_frequency}Hz (idle thrust: {self.idle_thrust})')

        # Subscribe to IMU data
        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10)

        # IMU data
        self.angular_velocity_current = 0

        # PID Parameters with some overshoot
        self.linearKu = 5
        self.linearTu = 2
        self.linearKp = self.linearKu * 1 / 3
        self.linearKi = (2/3) * self.linearKu * (1/self.linearTu)
        self.linearKd = (1/9) * self.linearKu * self.linearTu

        self.angularKu = 3
        self.angularTu = 2
        self.angularKp = self.angularKu * 1 / 3
        self.angularKi = (2/3) * self.angularKu * (1/self.angularTu)
        self.angularKd = (1/9) * self.angularKu * self.angularTu

        self.linearPID = PID(self.linearKp, self.linearKi, self.linearKd, 0)
        self.angularPID = PID(
            self.angularKp, self.angularKi, self.angularKd, 0)

        self.prev_linear_target = 0
        self.prev_angular_target = 0

    def imu_callback(self, msg):
        imu_data = msg.angular_velocity

        self.angular_velocity_current = imu_data.x

    def cmd_vel_callback(self, msg):
        """Store the latest velocity command"""
        # Update last command time
        self.last_cmd_vel_time = self.get_clock().now()
        self.is_idle = False

        with self.cmd_vel_lock:
            self.latest_cmd_vel = msg

    def joy_callback(self, msg):
        """Handle joystick input for conveyor control"""
        with self.joy_lock:
            # Y button is index 3 (4th button)
            if len(msg.buttons) > 3:
                current_y_state = (msg.buttons[3] == 1)

                # Detect rising edge (button just pressed)
                if current_y_state and not self.prev_y_button_state:
                    self.conveyor_enabled = not self.conveyor_enabled
                    self.get_logger().info(
                        f'Conveyor toggled: {"ON" if self.conveyor_enabled else "OFF"}')

                self.prev_y_button_state = current_y_state

    def convert_cmd_vel_to_esc(self):
        """Convert Twist message to ESC commands"""
        with self.cmd_vel_lock:
            # Update if cmd_vel has changed
            if self.prev_linear_target != self.latest_cmd_vel.linear.x:
                target_linear_vel = self.latest_cmd_vel.linear.x
                self.prev_linear_target = self.latest_cmd_vel.linear.x
                self.linearPID.set_setpoint(target_linear_vel)
            if self.prev_angular_target != self.latest_cmd_vel.angular.z:
                target_angular_vel = self.latest_cmd_vel.angular.z
                self.prev_angular_target = self.latest_cmd_vel.angular.z
                self.angularPID.set_setpoint(target_angular_vel)

        # Update PID controller
        linear_PID_output = self.linearPID.update(
            0)  # 0 for always forward/backward
        angular_PID_output = self.angularPID.update(
            self.angular_velocity_current)

        # Normalize velocities to [-1, 1] range
        linear_PID_output = self.map(
            linear_PID_output, (-1) *
            self.max_linear_vel, self.max_linear_vel, -1, 1
        )
        angular_PID_output = self.map(
            angular_PID_output, (-1) * self.max_angular_vel, self.max_angular_vel, -1, 1)

        # Calculate thrust for each side
        max_output = max(abs(linear_PID_output - angular_PID_output),
                         abs(linear_PID_output + angular_PID_output))
        if max_output == 0:
            max_output = 1
        left_thrust_normalized = (
            linear_PID_output - angular_PID_output) / max_output
        right_thrust_normalized = (
            linear_PID_output + angular_PID_output) / max_output

        # Map normalized velocities to ESC values (centered on idle_thrust)
        # Map from -1,1 to range where 0 maps to idle_thrust
        left_thrust = int(
            self.map(left_thrust_normalized, -1, 1, 1000, 2000))
        right_thrust = int(
            self.map(right_thrust_normalized, -1, 1, 1000, 2000))

        # Adjust mapping to center on idle_thrust instead of 1500
        if left_thrust_normalized == 0:
            left_thrust = self.idle_thrust
        else:
            left_thrust = self.idle_thrust + (left_thrust - 1500)

        if right_thrust_normalized == 0:
            right_thrust = self.idle_thrust
        else:
            right_thrust = self.idle_thrust + (right_thrust - 1500)

        # Clamp values to valid ESC range
        left_thrust = max(1000, min(2000, left_thrust))
        right_thrust = max(1000, min(2000, right_thrust))

        # Update ESC values
        self.command_template["ESC1"] = left_thrust
        self.command_template["ESC2"] = right_thrust
        self.command_template["ESC3"] = left_thrust
        self.command_template["ESC4"] = right_thrust

    def map(self, x, in_min, in_max, out_min, out_max):
        if in_max == in_min:
            return out_min
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (degrees) to quaternion"""
        # Convert degrees to radians
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        # Convert to quaternion
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy

        return q

    def process_imu_data(self, imu_data):
        """Process IMU data and publish to ROS topic"""
        # Update timestamp
        self.imu_msg.header.stamp = self.get_clock().now().to_msg()

        # Set orientation from Euler angles
        orientation = self.euler_to_quaternion(
            float(imu_data['roll']),
            float(imu_data['pitch']),
            float(imu_data['heading'])
        )
        self.imu_msg.orientation = orientation

        # Set angular velocity (convert to rad/s)
        self.imu_msg.angular_velocity.x = math.radians(
            float(imu_data['gyro_x']))
        self.imu_msg.angular_velocity.y = math.radians(
            float(imu_data['gyro_y']))
        self.imu_msg.angular_velocity.z = math.radians(
            float(imu_data['gyro_z']))

        # Set linear acceleration (m/s²)
        self.imu_msg.linear_acceleration.x = float(imu_data['accel_x'])
        self.imu_msg.linear_acceleration.y = float(imu_data['accel_y'])
        self.imu_msg.linear_acceleration.z = float(imu_data['accel_z'])

        # Publish the message
        self.imu_publisher.publish(self.imu_msg)

    def control_loop(self):
        """Main control loop running at 10Hz"""
        if self.serial_conn is None:
            return

        # Check if we've timed out on cmd_vel
        time_since_last_cmd = (self.get_clock().now() -
                               self.last_cmd_vel_time).nanoseconds / 1e9

        if time_since_last_cmd > self.cmd_vel_timeout:
            if not self.is_idle:
                # Transition to idle mode
                self.is_idle = True
                self.get_logger().info(
                    f'No cmd_vel received for {time_since_last_cmd:.1f}s, switching to idle thrust mode')

            # Set all ESCs to idle thrust value
            self.command_template["ESC1"] = self.idle_thrust
            self.command_template["ESC2"] = self.idle_thrust
            self.command_template["ESC3"] = self.idle_thrust
            self.command_template["ESC4"] = self.idle_thrust
        else:
            # Update ESC values from latest cmd_vel
            self.convert_cmd_vel_to_esc()

        # Update conveyor state based on toggle
        with self.joy_lock:
            self.command_template["SET_CONVEYOR_MODE"] = 1 if self.conveyor_enabled else 0
            self.command_template["SET_CONVEYOR_DUTY_CYCLE"] = self.conveyor_duty_cycle

        # Update IMU request based on parameter
        self.command_template["GET_IMU"] = 1 if self.request_imu else 0

        try:
            # Send command to microcontroller
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

            # Print thruster and conveyor commands to terminal
            self.get_logger().info(f'Thruster commands - ESC1: {self.command_template["ESC1"]}, '
                                   f'ESC2: {self.command_template["ESC2"]}, '
                                   f'ESC3: {self.command_template["ESC3"]}, '
                                   f'ESC4: {self.command_template["ESC4"]}, '
                                   f'Conveyor: {self.command_template["SET_CONVEYOR_MODE"]} @ {self.command_template["SET_CONVEYOR_DUTY_CYCLE"]:.2f}')

            # Read response from serial (with buffering for partial messages)
            if self.serial_conn.in_waiting > 0:
                try:
                    data = self.serial_conn.read(
                        self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    self.serial_buffer += data

                    # Process complete messages (ended by newline)
                    while '\n' in self.serial_buffer:
                        line, self.serial_buffer = self.serial_buffer.split(
                            '\n', 1)
                        line = line.strip()

                        if line and line != '':
                            try:
                                data = json.loads(line)
                                if 'IMU' in data:
                                    self.process_imu_data(data['IMU'])
                            except json.JSONDecodeError as e:
                                # Only warn if line looks like it should be JSON
                                if line.startswith('{') or line.startswith('['):
                                    self.get_logger().warn(
                                        f'Failed to parse JSON: {e}')
                                    self.get_logger().debug(
                                        f'Failed to parse line: "{line}"')
                except UnicodeDecodeError as e:
                    self.get_logger().warn(f'Unicode decode error: {e}')

        except Exception as e:
            self.get_logger().error(f'Serial communication error: {e}')

    def emergency_stop(self):
        """Set all thrusters to idle thrust (keeps motors primed)"""
        self.command_template["ESC1"] = self.idle_thrust
        self.command_template["ESC2"] = self.idle_thrust
        self.command_template["ESC3"] = self.idle_thrust
        self.command_template["ESC4"] = self.idle_thrust
        self.get_logger().info(
            f'Emergency stop - set all ESCs to idle thrust: {self.idle_thrust}')

        if self.serial_conn:
            try:
                json_str = json.dumps(self.command_template)
                self.serial_conn.write((json_str + '\n').encode('utf-8'))
            except:
                pass


class TritonController(Node):
    def __init__(self):
        super().__init__('triton_controller')

        # Serial connection parameters
        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)

        # Velocity limits
        self.declare_parameter('max_linear_velocity', 1.0)  # m/s
        self.declare_parameter('max_angular_velocity', 0.314)  # rad/s

        # Control parameters
        self.declare_parameter('control_frequency', 10.0)  # Hz
        self.declare_parameter('imu_frame_id', 'imu_frame')
        self.declare_parameter('imu_topic', '/imu/data')

        # Conveyor parameters
        self.declare_parameter('conveyor_duty_cycle', 0.5)  # 0.0 to 1.0

        # IMU request parameter
        # Whether to request IMU data
        self.declare_parameter('request_imu', False)

        # Idle thrust to keep motors primed
        self.declare_parameter('idle_thrust', 1510)  # Small forward thrust
        self.declare_parameter('cmd_vel_timeout', 0.5)  # seconds

        serial_port = self.get_parameter(
            'serial_port').get_parameter_value().string_value
        baud_rate = self.get_parameter(
            'baud_rate').get_parameter_value().integer_value
        self.max_linear_vel = self.get_parameter(
            'max_linear_velocity').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter(
            'max_angular_velocity').get_parameter_value().double_value
        control_frequency = self.get_parameter(
            'control_frequency').get_parameter_value().double_value
        imu_frame_id = self.get_parameter(
            'imu_frame_id').get_parameter_value().string_value
        imu_topic = self.get_parameter(
            'imu_topic').get_parameter_value().string_value
        self.conveyor_duty_cycle = self.get_parameter(
            'conveyor_duty_cycle').get_parameter_value().double_value
        self.request_imu = self.get_parameter(
            'request_imu').get_parameter_value().bool_value
        self.idle_thrust = self.get_parameter(
            'idle_thrust').get_parameter_value().integer_value
        self.cmd_vel_timeout = self.get_parameter(
            'cmd_vel_timeout').get_parameter_value().double_value

        # Initialize serial connection
        try:
            self.serial_conn = serial.Serial(
                serial_port, baud_rate, timeout=0.1)
            self.get_logger().info(
                f'Connected to serial port: {serial_port} at {baud_rate} baud')
        except Exception as e:
            self.get_logger().error(f'Failed to connect to serial port: {e}')
            self.serial_conn = None

        # Command template (initialized with idle thrust)
        self.command_template = {
            "GET_IMU": 1 if self.request_imu else 0,  # Request IMU data based on parameter
            "SET_RC_MODE": 0,
            "SET_CONVEYOR_MODE": 0,
            "SET_CONVEYOR_DUTY_CYCLE": 0.5,
            "ESC1": self.idle_thrust,  # Idle thrust
            "ESC2": self.idle_thrust,  # Idle thrust
            "ESC3": self.idle_thrust,  # Idle thrust
            "ESC4": self.idle_thrust   # Idle thrust
        }

        # Latest cmd_vel storage
        self.latest_cmd_vel = Twist()
        self.cmd_vel_lock = threading.Lock()

        # Joystick state
        self.conveyor_enabled = False  # Toggle state for conveyor
        self.prev_y_button_state = False  # Track previous button state for edge detection
        self.joy_lock = threading.Lock()

        # Subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Subscriber for joystick commands
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10
        )

        # Publisher for IMU data
        self.imu_publisher = self.create_publisher(Imu, imu_topic, 10)

        # Publisher for PWM values
        self.pwm_pub = self.create_publisher(Float32MultiArray, '/pwm', 10)

        # IMU message template
        self.imu_msg = Imu()
        self.imu_msg.header.frame_id = imu_frame_id

        # Set covariance matrices
        self.imu_msg.orientation_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]

        self.imu_msg.angular_velocity_covariance = [
            0.001, 0.0, 0.0,
            0.0, 0.001, 0.0,
            0.0, 0.0, 0.001
        ]

        self.imu_msg.linear_acceleration_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.01
        ]

        # Serial buffer for partial messages
        self.serial_buffer = ""

        # Timer for main control loop at specified frequency
        timer_period = 1.0 / control_frequency
        self.timer = self.create_timer(timer_period, self.control_loop)

        # Track last cmd_vel time
        self.last_cmd_vel_time = self.get_clock().now()
        self.is_idle = False

        self.get_logger().info(
            f'Triton Controller node started at {control_frequency}Hz (idle thrust: {self.idle_thrust})')

    def cmd_vel_callback(self, msg):
        """Store the latest velocity command"""
        # Update last command time
        self.last_cmd_vel_time = self.get_clock().now()
        self.is_idle = False

        with self.cmd_vel_lock:
            self.latest_cmd_vel = msg

    def joy_callback(self, msg):
        """Handle joystick input for conveyor control"""
        with self.joy_lock:
            # Y button is index 3 (4th button)
            if len(msg.buttons) > 3:
                current_y_state = (msg.buttons[3] == 1)

                # Detect rising edge (button just pressed)
                if current_y_state and not self.prev_y_button_state:
                    self.conveyor_enabled = not self.conveyor_enabled
                    self.get_logger().info(
                        f'Conveyor toggled: {"ON" if self.conveyor_enabled else "OFF"}')

                self.prev_y_button_state = current_y_state

    def convert_cmd_vel_to_esc(self):
        """Convert Twist message to ESC commands"""
        with self.cmd_vel_lock:
            linear_x = self.latest_cmd_vel.linear.x
            angular_z = self.latest_cmd_vel.angular.z

        # Clamp input velocities to maximum values
        linear_x = max(-self.max_linear_vel,
                       min(self.max_linear_vel, linear_x))
        angular_z = max(-self.max_angular_vel,
                        min(self.max_angular_vel, angular_z))

        # Normalize velocities to [-1, 1] range
        linear_normalized = linear_x / self.max_linear_vel if self.max_linear_vel > 0 else 0
        angular_normalized = angular_z / \
            self.max_angular_vel if self.max_angular_vel > 0 else 0

        # Map normalized velocities to ESC values (1000-2000 range, 1500 is neutral)
        linear_component = int(linear_normalized * 500)
        angular_component = int(angular_normalized * 300)

        # Calculate thrust for each side (using idle_thrust as center)
        left_thrust = self.idle_thrust + linear_component - angular_component
        right_thrust = self.idle_thrust + linear_component + angular_component

        # Clamp values to valid ESC range
        left_thrust = max(1000, min(2000, left_thrust))
        right_thrust = max(1000, min(2000, right_thrust))

        # Update ESC values
        self.command_template["ESC1"] = left_thrust
        self.command_template["ESC2"] = right_thrust
        self.command_template["ESC3"] = left_thrust
        self.command_template["ESC4"] = right_thrust

    def euler_to_quaternion(self, roll, pitch, yaw):
        """Convert Euler angles (degrees) to quaternion"""
        # Convert degrees to radians
        roll_rad = math.radians(roll)
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)

        # Convert to quaternion
        cy = math.cos(yaw_rad * 0.5)
        sy = math.sin(yaw_rad * 0.5)
        cp = math.cos(pitch_rad * 0.5)
        sp = math.sin(pitch_rad * 0.5)
        cr = math.cos(roll_rad * 0.5)
        sr = math.sin(roll_rad * 0.5)

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy

        return q

    def process_imu_data(self, imu_data):
        """Process IMU data and publish to ROS topic"""
        # Update timestamp
        self.imu_msg.header.stamp = self.get_clock().now().to_msg()

        # Set orientation from Euler angles
        orientation = self.euler_to_quaternion(
            float(imu_data['roll']),
            float(imu_data['pitch']),
            float(imu_data['heading'])
        )
        self.imu_msg.orientation = orientation

        # Set angular velocity (convert to rad/s)
        self.imu_msg.angular_velocity.x = math.radians(
            float(imu_data['gyro_x']))
        self.imu_msg.angular_velocity.y = math.radians(
            float(imu_data['gyro_y']))
        self.imu_msg.angular_velocity.z = math.radians(
            float(imu_data['gyro_z']))

        # Set linear acceleration (m/s²)
        self.imu_msg.linear_acceleration.x = float(imu_data['accel_x'])
        self.imu_msg.linear_acceleration.y = float(imu_data['accel_y'])
        self.imu_msg.linear_acceleration.z = float(imu_data['accel_z'])

        # Publish the message
        self.imu_publisher.publish(self.imu_msg)

    def control_loop(self):
        """Main control loop running at 10Hz"""
        if self.serial_conn is None:
            return

        # Check if we've timed out on cmd_vel
        time_since_last_cmd = (self.get_clock().now() -
                               self.last_cmd_vel_time).nanoseconds / 1e9

        if time_since_last_cmd > self.cmd_vel_timeout:
            if not self.is_idle:
                # Transition to idle mode
                self.is_idle = True
                self.get_logger().info(
                    f'No cmd_vel received for {time_since_last_cmd:.1f}s, switching to idle thrust mode')

            # Set all ESCs to idle thrust value
            self.command_template["ESC1"] = self.idle_thrust
            self.command_template["ESC2"] = self.idle_thrust
            self.command_template["ESC3"] = self.idle_thrust
            self.command_template["ESC4"] = self.idle_thrust
        else:
            # Update ESC values from latest cmd_vel
            self.convert_cmd_vel_to_esc()

        # Update conveyor state based on toggle
        with self.joy_lock:
            self.command_template["SET_CONVEYOR_MODE"] = 1 if self.conveyor_enabled else 0
            self.command_template["SET_CONVEYOR_DUTY_CYCLE"] = self.conveyor_duty_cycle

        # Update IMU request based on parameter
        self.command_template["GET_IMU"] = 1 if self.request_imu else 0

        try:
            # Send command to microcontroller
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

            # Print thruster and conveyor commands to terminal
            self.get_logger().info(f'Thruster commands - ESC1: {self.command_template["ESC1"]}, '
                                   f'ESC2: {self.command_template["ESC2"]}, '
                                   f'ESC3: {self.command_template["ESC3"]}, '
                                   f'ESC4: {self.command_template["ESC4"]}, '
                                   f'Conveyor: {self.command_template["SET_CONVEYOR_MODE"]} @ {self.command_template["SET_CONVEYOR_DUTY_CYCLE"]:.2f}')

            # Read response from serial (with buffering for partial messages)
            if self.serial_conn.in_waiting > 0:
                try:
                    data = self.serial_conn.read(
                        self.serial_conn.in_waiting).decode('utf-8', errors='ignore')
                    self.serial_buffer += data

                    # Process complete messages (ended by newline)
                    while '\n' in self.serial_buffer:
                        line, self.serial_buffer = self.serial_buffer.split(
                            '\n', 1)
                        line = line.strip()

                        if line and line != '':
                            try:
                                data = json.loads(line)
                                if 'IMU' in data:
                                    self.process_imu_data(data['IMU'])
                            except json.JSONDecodeError as e:
                                # Only warn if line looks like it should be JSON
                                if line.startswith('{') or line.startswith('['):
                                    self.get_logger().warn(
                                        f'Failed to parse JSON: {e}')
                                    self.get_logger().debug(
                                        f'Failed to parse line: "{line}"')
                except UnicodeDecodeError as e:
                    self.get_logger().warn(f'Unicode decode error: {e}')

        except Exception as e:
            self.get_logger().error(f'Serial communication error: {e}')

    def emergency_stop(self):
        """Set all thrusters to idle thrust (keeps motors primed)"""
        self.command_template["ESC1"] = self.idle_thrust
        self.command_template["ESC2"] = self.idle_thrust
        self.command_template["ESC3"] = self.idle_thrust
        self.command_template["ESC4"] = self.idle_thrust
        self.get_logger().info(
            f'Emergency stop - set all ESCs to idle thrust: {self.idle_thrust}')

        if self.serial_conn:
            try:
                json_str = json.dumps(self.command_template)
                self.serial_conn.write((json_str + '\n').encode('utf-8'))
            except:
                pass


def main(args=None):
    """
    Entry point for running the Triton Controller.

    This function initializes the ROS 2 node and runs either the manual or autonomous
    version of the Triton controller based on command-line arguments.

    Usage:
    ------
    To run in **manual mode** (default):
        $ ros2 run your_package_name your_script.py

    To run in **autonomous mode**:
        $ ros2 run your_package_name your_script.py --mode=Autonomous

    Parameters:
    -----------
    args : list, optional
        List of command-line arguments. Typically provided by the ROS 2 launch system.

    Behavior:
    ---------
    - Initializes ROS 2.
    - Selects `TritonControllerAutonomous` if `--mode=Autonomous` is in args.
      Otherwise, defaults to `TritonController`.
    - Spins the selected controller node.
    - On keyboard interrupt (Ctrl+C), performs emergency stop, closes serial connection,
      destroys the node, and shuts down ROS cleanly.
    """
    import sys

    # Use sys.argv if args is None
    if args is None:
        args = sys.argv[1:]

    rclpy.init(args=args)

    # Create the appropriate controller based on command line arguments
    if '--mode=Autonomous' in args:
        triton_controller = TritonControllerAutonomous()
    else:
        triton_controller = TritonController()

    try:
        rclpy.spin(triton_controller)
    except KeyboardInterrupt:
        triton_controller.emergency_stop()
    finally:
        if triton_controller.serial_conn:
            triton_controller.serial_conn.close()
        triton_controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
