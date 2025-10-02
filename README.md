# triton_isaac

This repo contains instructions for how to use the Isaac SIM simulation for the Triton water surface vehicle.

## Installation

```
mkdir -p ros2_ws/src
cd src
git clone https://github.com/Abanesjo/Triton
cd ../
rosdep install --from-path src --ignore-src -r -y
colcon build --symlink-install && source install/setup.bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```
## Simulation
### Usage
1. Run the simulation in Isaac Sim using the file in <code>triton_isaac/isaac/Collected_competition_world/competition_world.usd</code>

<!-- isaac sim image -->
![isaac_sim.png](docs/isaac_sim.png)

2. Use the python library by importing the following into your script:
```
from triton_isaac.tools import IsaacHelper

ih = IsaacHelper()
```
Refer to the next section for the various implemented methods

### Python API
The `IsaacHelper` class provides methods for controlling and reading sensor data from the simulated robot.

- **`set_velocity(linear, angular)`**  
  Publishes a velocity command using a ROS 2 `Twist` message.  
  • `linear` (float): Linear velocity.  
  • `angular` (float): Angular velocity.

- **`get_position()`**  
  Returns the current position as a tuple of `(x, y, z)`. **The origin (0,0,0) position is determined by the robot's starting location**

- **`get_orientation()`**  
  Returns the current orientation as a tuple of `(roll, pitch, yaw)`.

- **`get_front_image()`**  
  Returns the latest front camera image (OpenCV `numpy` array).

- **`get_back_image()`**  
  Returns the latest back camera image (OpenCV `numpy` array).

- **`spin_once()`**  
  Processes callbacks once, allowing subscriptions to receive messages. You need to call this every time you want to update the variables.

#### Sample Code
The following code tells the robot to move forward, as well as creates an OpenCV window that displays the image feed from the front camera (remember that the simulation needs to be running for you to begin getting values)
```
#!/usr/bin/env python3

from triton_isaac.tools import IsaacHelper
import cv2

ih = IsaacHelper()
ih.set_velocity(0.5, 0.0)

while True:
    print(f"Position: {ih.get_position()}")

    front_image = ih.get_front_image()
    back_image = ih.get_back_image()

    if front_image is not None:
        cv2.imshow('image', front_image)
        cv2.waitKey(1)

    ih.spin_once()
```

## Orienteering
```
cd ~/ros2_ws/src/Triton/triton_navigation/config
ros2 service call /export_oplib std_srvs/srv/Trigger
./../../dependencies/op_solver_ros/op-solver/build/src/op-solver opt --op-exact 0 problem.oplib
ros2 service call /create_waypoints std_srvs/srv/Trigger
ros2 service call /generate_path std_srvs/srv/Trigger
ros2 service call /navigate std_srvs/srv/Empty
```
