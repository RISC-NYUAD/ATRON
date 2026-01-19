# ATRON: Autonomous Trash Retrieval for Oceanic Neatness

Please visit our website and paper for more information about this project. 

[[Website]](https://risc-nyuad.github.io/atron_website/)
[[Paper]](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1718177)

This repo contains instructions for how to use the Isaac SIM simulation for the Atron USV autonomous debris collection vehicle.

<div align="center">

<img width="1608" height="904" alt="thumbnail" src="https://github.com/user-attachments/assets/b08e13d7-e6f6-4be5-bbbc-9ff5694b01ce" />

[Youtube Link](https://youtu.be/WVKWoPhqZnQ)

</div>

## Installation

```
mkdir -p ros2_ws/src
cd src
git clone https://github.com/RISC-NYUAD/ATRON
cd ../
rosdep install --from-path src --ignore-src -r -y
colcon build --symlink-install && source install/setup.bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Usage
1. Run the simulation in Isaac Sim using the file in 
```
isaac/atron_simulation.usd
```

![isaac_sim.png](docs/isaac_sim.png)


2. Run the autonomous routine using:

```
ros2 launch triton_bringup simulation.launch.xml
```

3. In another terminal, run
```
ros2 launch triton_navigation op_solver_bridge.launch.py
```

This will launch the planning algorithms, as create a link to the [orienteering solver](https://github.com/gkobeaga/op-solver). 

![simulation.png](docs/simulation.png)

3. Run the following in another terminal to run the Orienteering -> OMPL -> Pure pursuit navigation pipeline.

```
cd ~/ros2_ws/src/Triton/triton_navigation/config
ros2 service call /export_oplib std_srvs/srv/Trigger
./../../dependencies/op_solver_ros/op-solver/build/src/op-solver opt --op-exact 0 problem.oplib
ros2 service call /create_waypoints std_srvs/srv/Trigger
ros2 service call /generate_path std_srvs/srv/Trigger
ros2 service call /navigate std_srvs/srv/Empty
```

## High-Fidelity Simulation
We also provide a high-fidelity simulation which includes improved visual and physical details.
<img width="1267" height="688" alt="image" src="https://github.com/user-attachments/assets/6832dd5b-d6cb-4469-bc66-286769209b8a" />


Buoyancy is simulated via force calculations on a height wavefunction. Movemement is implemented via thrust forces controlled via PID to execute the desired velocity.
![buoyancy](https://github.com/user-attachments/assets/1821aef3-a781-492a-9545-72286b81069a)

See the releases tab for the **atron_highfidelity** for the simulation file. The simulation can physically simulate sea states from 0 to 4 on the Beaufort Scale
<img width="786" height="431" alt="image" src="https://github.com/user-attachments/assets/51b20a06-0b1b-41e8-8251-419bbf6eeaff" />


We also implement a correction step for the debris and obstacle localization process. Below are videos showing the before and after for the correction step. 

Without the correction in a buoyancy environment (cylinders are for the obstacle class, spheres for the debris class):
![no_correction_short](https://github.com/user-attachments/assets/5c7fc663-b809-4e3b-b493-78bc8d18ac29)

With the correction:
![with_correction_short](https://github.com/user-attachments/assets/c3d0bf53-8fcc-46aa-92ed-415600fcd81f)

The photorealistic environment also allows the training of object detection models, as seen below
<img width="1047" height="688" alt="image" src="https://github.com/user-attachments/assets/18edd772-e298-4b4d-877e-59c6949549d7" />





