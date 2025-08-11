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
    vslam_config = os.path.join(pkg_dir, 'config', 'vslam.yaml')
    vocab_file = os.path.join(pkg_dir, 'config', 'orb_vocab.fbow')
    
    # --- Argument Declaration ---
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # --- Configuration ---
    # Path to the corrected YAML configuration file
    rtabmap_config_path = os.path.join(pkg_dir, 'config', 'rtabmap.yaml')
    
    # Load the structured parameters from the YAML file
    with open(rtabmap_config_path, 'r') as f:
        params = yaml.safe_load(f)

    # Extract parameters for each node from the loaded YAML
    icp_odometry_params = params['icp_odometry']['ros__parameters']
    rtabmap_params = params['rtabmap']['ros__parameters']
    rtabmap_viz_params = params['rtabmap_viz']['ros__parameters']

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
            {'use_sim_time': True}
        ]
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': True}],
        arguments=[os.path.join(pkg_dir, 'urdf', 'catamaran.urdf')]
    )


    # ICP Odometry Node
    icp_odometry_node = Node(
        package='rtabmap_odom',
        executable='icp_odometry',
        name='icp_odometry',
        output='screen',
        parameters=[icp_odometry_params, {'use_sim_time': use_sim_time}],
        remappings=[('scan', '/scan')],
        emulate_tty=True,
    )

    # RTAB-Map SLAM Node
    rtabmap_node = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[rtabmap_params, {'use_sim_time': use_sim_time}],
        arguments=['-d'],  # '-d' deletes the database on every start
        emulate_tty=True,
    )
    
    # Visualization Node
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=[rtabmap_viz_params, {'use_sim_time': use_sim_time}],
        emulate_tty=True,
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
        parameters=[{'use_sim_time': True}]
    )

    
    # --- Launch Description ---
    ld = LaunchDescription()
    ld.add_action(imu_node)
    ld.add_action(stella_vslam)
    ld.add_action(robot_state_publisher)
    ld.add_action(icp_odometry_node)
    ld.add_action(rtabmap_node)
    ld.add_action(rtabmap_viz_node)
    ld.add_action(slam_toolbox)

    return ld