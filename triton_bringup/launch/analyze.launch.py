#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('triton_bringup')
    extrinsics = os.path.join(pkg_dir, 'config', 'extrinsics_x2.json')
    vslam_config = os.path.join(pkg_dir, 'config', 'vslam.yaml')
    vocab_file = os.path.join(pkg_dir, 'config', 'orb_vocab.fbow')

    # Declare launch arguments
    equirectangular_arg = DeclareLaunchArgument(
        'equirectangular',
        default_value='true',
        description='Enable equirectangular projection'
    )

    config_arg = DeclareLaunchArgument(
        'config',
        default_value='config.yaml',
        description='Path to the configuration file'
    )

    # Define the bringup node with parameters
    
    equirectangular_node = Node(
        package='insta360_ros_driver',
        executable='equirectangular.py',
        name='equirectangular_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('equirectangular')),
        arguments=[
            '--gpu',
            '--calibration_file', extrinsics]
    )

    imu_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('triton_bringup'),
                'config',
                'imu_filter.yaml'
            ])
        ]
    )

    stella_vslam = Node(
        package='stella_vslam_ros',
        executable='run_slam',
        name='stella_vslam_node',
        output='screen',
        arguments=[
            '-v', vocab_file,
            '-c', vslam_config,
            '--ros-args', '--remap', '/camera/image_raw:=/equirectangular/image',
        ]
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        arguments=[os.path.join(pkg_dir, 'urdf', 'catamaran.urdf')]
    )

    imu_camera_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='imu_to_camera_static_transform',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--qx', '-0.500',
            '--qy', '0.500',
            '--qz', '0.500',
            '--qw', '0.500',
            '--frame-id', 'imu_frame',
            '--child-frame-id', 'camera_frame'
        ]
    )



    ld = LaunchDescription()
    ld.add_action(equirectangular_arg)
    # ld.add_action(equirectangular_node)
    # ld.add_action(imu_node)
    # ld.add_action(stella_vslam)
    ld.add_action(robot_state_publisher)
    # ld.add_action(imu_camera_transform)

    return ld
