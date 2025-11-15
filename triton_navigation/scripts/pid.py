#!/usr/bin/env python3

import time


class PID:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, setpoint=0.0,
                 output_limits=None, integral_limits=None):
        """
        Initialize the PID controller with specified gains and setpoint.

        Parameters:
        Kp (float): Proportional gain coefficient. Default is 1.0.
        Ki (float): Integral gain coefficient. Default is 0.0.
        Kd (float): Derivative gain coefficient. Default is 0.0.
        setpoint (float): Desired target value for the PID controller. Default is 0.0.
        output_limits (tuple(min, max) or None): Optional clamp on controller output.
        integral_limits (tuple(min, max) or None): Optional clamp on integral term.
        """
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.setpoint = setpoint

        self.output_limits = output_limits
        self.integral_limits = integral_limits

        # Use a monotonic clock to avoid issues with system time adjustments
        self._prev_time = time.monotonic()
        self._prev_error = 0.0
        self._integral = 0.0

    def update(self, measured_value, dt=None):
        """
        Updates the PID controller with a new measured value.

        Parameters
        ----------
        measured_value : float
            The current measured value.
        dt : float or None
            Optional time step (seconds). If None, wall-clock time is used.

        Returns
        -------
        float
            The output of the PID controller.
        """
        # Calculate time difference dt if not supplied
        if dt is None:
            current_time = time.monotonic()
            dt = current_time - self._prev_time
            self._prev_time = current_time

        # Guard against non-positive dt
        if dt <= 0.0:
            dt = 1e-6

        # Calculate error
        error = self.setpoint - measured_value

        # Save previous integral in case we need to roll back on saturation
        prev_integral = self._integral

        # Calculate the integral (see https://en.wikipedia.org/wiki/Trapezoidal_rule)
        self._integral += error * dt

        # Optional clamp on integral term (basic anti-windup)
        if self.integral_limits is not None:
            i_min, i_max = self.integral_limits
            if i_min is not None:
                self._integral = max(i_min, self._integral)
            if i_max is not None:
                self._integral = min(i_max, self._integral)

        # Calculate the derivative (see https://en.wikipedia.org/wiki/Finite_difference)
        derivative = (error - self._prev_error) / dt

        # Calculate the output
        output = (self.Kp * error) + (self.Ki * self._integral) + (self.Kd * derivative)

        # Apply output limits with simple anti-windup: if we saturate, roll back integral
        if self.output_limits is not None:
            out_min, out_max = self.output_limits

            # Only clamp if limits are provided (not None)
            if out_min is not None and output < out_min:
                output = out_min
                self._integral = prev_integral
            elif out_max is not None and output > out_max:
                output = out_max
                self._integral = prev_integral

        self._prev_error = error

        return output

    def set_Kp(self, Kp):
        self.Kp = Kp

    def set_Ki(self, Ki):
        self.Ki = Ki

    def set_Kd(self, Kd):
        self.Kd = Kd

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def set_output_limits(self, output_limits):
        """Update output limits at runtime."""
        self.output_limits = output_limits

    def set_integral_limits(self, integral_limits):
        """Update integral term limits at runtime."""
        self.integral_limits = integral_limits

    def reset(self, setpoint=None):
        """
        Reset the controller state (integral and derivative terms).
        Optionally update the setpoint.
        """
        if setpoint is not None:
            self.setpoint = setpoint
        self._prev_time = time.monotonic()
        self._prev_error = 0.0
        self._integral = 0.0
