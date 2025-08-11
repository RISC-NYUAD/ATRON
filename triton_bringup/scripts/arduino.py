#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import time

class ArduinoControl(Node):
    def __init__(self):
        super().__init__('arduino_control')
        
        # Create subscription to /cmd_vel
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        # Define frequency and duty cycles
        self.frequency = 500                   # 500 Hz
        self.period_us = int((1.0 / self.frequency) * 1e6)  # period in microseconds
        
        self.duty_off = 0.75       # Neutral (motor off) at 75% duty cycle
        self.duty_forward = 1.0    # Full forward at 100% duty cycle
        self.duty_reverse = 0.5    # Full reverse at 50% duty cycle
        
        # Compute PWM values based on period and duty cycles
        self.pwm_off = int(self.period_us * self.duty_off)
        self.pwm_forward = int(self.period_us * self.duty_forward)
        self.pwm_reverse = int(self.period_us * self.duty_reverse)
        
        # Current PWM values for left and right motors
        self.current_pwm_left = self.pwm_off
        self.current_pwm_right = self.pwm_off

        # Setup Arduino serial connection
        self.arduino = serial.Serial('/dev/ttyACM0', 9600)
        time.sleep(2)  # Allow time for Arduino to reset
        
        # Timer to update PWM at specified frequency
        self.timer = self.create_timer(1.0 / self.frequency, self.timer_callback)
    
    def cmd_vel_callback(self, msg: Twist):
        linear = msg.linear.x     # Expected range: -1.5 to 1.5
        angular = msg.angular.z   # Expected range: -1.0 to 1.0

        # Deadzone thresholds
        deadzone_linear = 0.75
        deadzone_angular = 0.5
        
        # Remap linear command if above deadzone
        if abs(linear) >= deadzone_linear:
            # Remap value from [deadzone_linear, 1.5] to [0, 1]
            effective_linear = (abs(linear) - deadzone_linear) / (1.5 - deadzone_linear)
        else:
            effective_linear = 0.0

        # Remap angular command if above deadzone
        if abs(angular) >= deadzone_angular:
            # Remap value from [deadzone_angular, 1.0] to [0, 1]
            effective_angular = (abs(angular) - deadzone_angular) / (1.0 - deadzone_angular)
        else:
            effective_angular = 0.0
        
        # Default both motors to neutral PWM
        pwm_left = self.pwm_off
        pwm_right = self.pwm_off

        # Choose between linear and angular control based on the higher effective value
        if effective_linear >= effective_angular:
            # Linear motion: both motors scale the same way.
            if linear >= 0:
                # Forward: scale from pwm_off to pwm_forward
                pwm_val = self.pwm_off + int((self.pwm_forward - self.pwm_off) * effective_linear)
            else:
                # Reverse: scale from pwm_off down to pwm_reverse
                pwm_val = self.pwm_off - int((self.pwm_off - self.pwm_reverse) * effective_linear)
            pwm_left = pwm_val
            pwm_right = pwm_val
        else:
            # Angular motion: motors act oppositely.
            if angular >= 0:
                # Turning left: right motor scales upward, left motor scales downward.
                pwm_right = self.pwm_off + int((self.pwm_forward - self.pwm_off) * effective_angular)
                pwm_left = self.pwm_off - int((self.pwm_off - self.pwm_reverse) * effective_angular)
            else:
                # Turning right: left motor scales upward, right motor scales downward.
                pwm_left = self.pwm_off + int((self.pwm_forward - self.pwm_off) * effective_angular)
                pwm_right = self.pwm_off - int((self.pwm_off - self.pwm_reverse) * effective_angular)

        self.current_pwm_left = pwm_left
        self.current_pwm_right = pwm_right

    def timer_callback(self):
        # Prepare the command in the format "left_pwm right_pwm\n"
        command_str = f"{self.current_pwm_left} {self.current_pwm_right}\n"
        self.arduino.write(command_str.encode('utf-8'))
        print({"left_pwm": self.current_pwm_left, "right_pwm": self.current_pwm_right})


def main(args=None):
    rclpy.init(args=args)
    node = ArduinoControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()