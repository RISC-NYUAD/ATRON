#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_dir = get_package_share_directory('triton_bringup')
    slam_config = os.path.join(pkg_dir, 'config', 'slam_toolbox.yaml')
    vslam_config = os.path.join(pkg_dir, 'config', 'vslam.yaml')
    vocab_file = os.path.join(pkg_dir, 'config', 'orb_vocab.fbow')

    camera_bringup = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('insta360_ros_driver'),
            'launch',
            'bringup.launch.py'
        ]),
        launch_arguments={
            'config': os.path.join(pkg_dir, 'config', 'insta_config.yaml'),
            'equirectangular': 'true'
        }.items()
    )

    lidar_bringup = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('rplidar_ros'),
            'launch',
            'rplidar_s2_launch.py'
        ]),
        launch_arguments={
            'frame_id':'laser_frame'
        }.items()
    )

    # robot_state_publisher = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[{'use_sim_time': False}],
    #     arguments=[os.path.join(pkg_dir, 'urdf', 'vision_system.urdf')]
    # )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        arguments=[os.path.join(pkg_dir, 'urdf', 'catamaran.urdf')]
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

    laser_odometry = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('rf2o_laser_odometry'),
            'launch',
            'rf2o_laser_odometry.launch.py'
        ])
    )

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('slam_toolbox'), 'launch', 'online_sync_launch.py')
        ),
        launch_arguments={
            'slam_params_file': slam_config,
        }.items()
    )

    gmapping = Node(
        package='slam_gmapping',
        executable='slam_gmapping',
        output='screen'
    )


    ld = LaunchDescription()
    ld.add_action(camera_bringup)
    ld.add_action(imu_node)
    ld.add_action(stella_vslam)
    # ld.add_action(lidar_bringup)
    # ld.add_action(robot_state_publisher)
    # ld.add_action(laser_odometry)
    # ld.add_action(slam_toolbox)

    return ld 