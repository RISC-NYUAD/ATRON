import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('triton_bringup')
    config_file = os.path.join(pkg_dir, 'config', 'teleop_config.yaml')

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'device_id': 0,
            'deadzone': 0.1,
            # 'autorepeat_rate': 20.0
            'autorepeat_rate': 0.0
        }],
        output='screen'
    )

    teleop_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        parameters=[config_file],
        output='screen'
    )

    arduino_node = Node(
        package='triton_bringup',
        executable='arduino.py',
        name='arduino_node'
    )

    return LaunchDescription([
        joy_node,
        teleop_node,
        # arduino_node
    ])

if __name__ == '__main__':
    generate_launch_description()