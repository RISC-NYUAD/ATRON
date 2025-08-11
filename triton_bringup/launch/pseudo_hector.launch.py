import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    pkg_dir = get_package_share_directory('triton_bringup')
    slam_config = os.path.join(pkg_dir, 'config', 'slam_toolbox.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # robot_state_publisher = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[{'use_sim_time': use_sim_time}],
    #     arguments=[os.path.join(pkg_dir, 'urdf', 'catamaran.urdf')]
    # )
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[os.path.join(pkg_dir, 'urdf', 'vision_system.urdf')]
    )

    scan_matcher = Node(
        package='ros2_laser_scan_matcher',
        executable='laser_scan_matcher',
        name='scan_matcher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'base_frame': 'imu_frame',
            'odom_frame': 'map',
            'map_frame': 'map_corrected',
            'laser_frame': 'laser_frame',
            'publish_odom': 'odom',
            'publish_tf': True}],
        output='screen'
    )

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'online_sync_launch.py')
        ),
        launch_arguments={
            'slam_params_file': slam_config,
            'use_sim_time': 'true'
        }.items()
    )

    
    # --- Launch Description ---
    ld = LaunchDescription()
    ld.add_action(robot_state_publisher)
    # ld.add_action(scan_matcher)
    # ld.add_action(slam_toolbox)

    return ld