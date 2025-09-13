from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='triton_navigation',
            executable='op_solver_bridge.py',
            name='op_solver_bridge',
            output='screen',
            parameters=[{
                'cost_limit': 50.0,
                'oplib_file': 'config/problem.oplib',
                'solution_file': 'config/stats.json',
                'cluster_radius': 1.5
            }]
        )
    ])