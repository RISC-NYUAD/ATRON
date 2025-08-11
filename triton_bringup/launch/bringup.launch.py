#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():
    pkg_dir = get_package_share_directory('triton_bringup')
    slam_config = os.path.join(pkg_dir, 'config', 'slam_toolbox.yaml')
    vslam_config = os.path.join(pkg_dir, 'config', 'vslam.yaml')
    vocab_file = os.path.join(pkg_dir, 'config', 'orb_vocab.fbow')
    robot_localization_config = os.path.join(pkg_dir, 'config', 'robot_localization.yaml')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    # robot_state_publisher = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='screen',
    #     parameters=[{'use_sim_time': use_sim_time}],
    #     arguments=[os.path.join(pkg_dir, 'urdf', 'vision_system.urdf')]
    # )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[os.path.join(pkg_dir, 'urdf', 'catamaran.urdf')]
    )

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

    bottom_view = Node(
        package='triton_bringup',
        executable='bottom_view.py',
        name='bottom_view',
        output='screen'
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
            ]),
            {'use_sim_time': use_sim_time}
        ]
    )

    lidar_node = Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='lidar_node',
        parameters=[{
            'channel_type': 'serial',
            'serial_port': '/dev/ttyUSB0',
            'serial_baudrate': 1000000,
            'frame_id': 'laser_frame',
            'inverted': False,
            'angle_compensate': True,
            'scan_mode': 'DenseBoost'
        }],
        output='screen'
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
        ],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    scan_matcher = Node(
        package='ros2_laser_scan_matcher',
        executable='laser_scan_matcher',
        name='scan_matcher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'base_frame': 'laser_frame',
            'odom_frame': 'odom',
            'map_frame': 'map',
            'laser_frame': 'laser_frame',
            'publish_odom': 'odom',
            'publish_tf': True}],
        output='screen'
    )

    robot_localization = Node(
        package='robot_localization',
        executable='ukf_node',
        name='ukf_node',
        parameters=[
            robot_localization_config],
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

    web_server = Node(
        package='rosboard',
        executable='rosboard_node',
        name='rosboard',
        output='screen'
    )


    ld = LaunchDescription()
    ld.add_action(use_sim_time_arg)
    # ld.add_action(robot_state_publisher)
    ld.add_action(camera_bringup)
    ld.add_action(bottom_view)
    ld.add_action(imu_node)
    ld.add_action(lidar_node)
    # ld.add_action(stella_vslam)
    ld.add_action(scan_matcher)
    # ld.add_action(robot_localization)
    ld.add_action(slam_toolbox)
    # ld.add_action(web_server)

    return ld 