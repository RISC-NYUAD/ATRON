import time


class PID:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, setpoint=0.0):
        """
        Initialize the PID controller with specified gains and setpoint.

        Parameters:
        Kp (float): Proportional gain coefficient. Default is 1.0.
        Ki (float): Integral gain coefficient. Default is 0.0.
        Kd (float): Derivative gain coefficient. Default is 0.0.
        setpoint (float): Desired target value for the PID controller. Default is 0.0.
        """
        self.Kp = Kp  # Proportional gain
        self.Ki = Ki  # Integral gain
        self.Kd = Kd  # Derivative gain
        self.setpoint = setpoint

        self._prev_time = time.time()
        self._prev_error = 0.0
        self._integral = 0.0

    def update(self, measured_value):
        """
        Updates the PID controller with a new measured value.

        Parameters
        ----------
        measured_value : float
            The current measured value.

        Returns
        -------
        float
            The output of the PID controller.
        """
        # Calculate time difference dt
        current_time = time.time()
        dt = current_time - self._prev_time
        self._prev_time = current_time

        # Calculate error
        error = self.setpoint - measured_value

        # Calculate the integral (see https://en.wikipedia.org/wiki/Trapezoidal_rule)
        self._integral += error * dt

        # Calculate the derivative (see https://en.wikipedia.org/wiki/Finite_difference)
        if dt > 0:
            derivative = (error - self._prev_error) / dt
        else:
            derivative = 0

        # Calculate the output
        output = (self.Kp * error) + (self.Ki * self._integral) + \
            (self.Kd * derivative)
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
