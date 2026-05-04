"""
ros_gz_bridge.launch.py — Launch Gazebo-ROS2 bridges for drone swarm.

Conditionally starts ros_gz_bridge parameter_bridge processes for:
  - Drone pose data (GZ_TO_ROS, visualisation only)
  - Simulation clock (GZ_TO_ROS, required for use_sim_time)
  - Optional sensor data (depth camera, lidar, IMU)

NOTE: These bridges are for MONITORING and VISUALISATION only.
      PX4 flight control uses uXRCE-DDS, not this bridge.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def launch_setup(context, *args, **kwargs):
    """Build bridge node list based on resolved flags."""
    pkg_share = FindPackageShare('mappo_swarm_bridge').find('mappo_swarm_bridge')

    use_pose_bridge = LaunchConfiguration('use_pose_bridge').perform(context)
    use_clock_bridge = LaunchConfiguration('use_clock_bridge').perform(context)
    use_sensor_bridge = LaunchConfiguration('use_sensor_bridge').perform(context)

    # Allow explicit config file overrides
    pose_cfg = LaunchConfiguration('pose_bridge_config').perform(context)
    clock_cfg = LaunchConfiguration('clock_bridge_config').perform(context)
    sensor_cfg = LaunchConfiguration('sensor_bridge_config').perform(context)

    # Use package defaults if not overridden
    if not pose_cfg:
        pose_cfg = os.path.join(pkg_share, 'config', 'pose_bridge.yaml')
    if not clock_cfg:
        clock_cfg = os.path.join(pkg_share, 'config', 'clock_bridge.yaml')
    if not sensor_cfg:
        sensor_cfg = os.path.join(pkg_share, 'config', 'sensor_bridge.yaml')

    nodes = []

    if use_clock_bridge.lower() == 'true':
        nodes.append(Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_clock_bridge',
            output='screen',
            parameters=[{'config_file': clock_cfg}],
        ))

    if use_pose_bridge.lower() == 'true':
        nodes.append(Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_pose_bridge',
            output='screen',
            parameters=[{'config_file': pose_cfg}],
        ))

    if use_sensor_bridge.lower() == 'true':
        nodes.append(Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='gz_sensor_bridge',
            output='screen',
            parameters=[{'config_file': sensor_cfg}],
        ))

    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'pose_bridge_config',
            default_value='',
            description='Absolute path to pose bridge YAML config '
                        '(default: package config/pose_bridge.yaml)'),
        DeclareLaunchArgument(
            'clock_bridge_config',
            default_value='',
            description='Absolute path to clock bridge YAML config '
                        '(default: package config/clock_bridge.yaml)'),
        DeclareLaunchArgument(
            'sensor_bridge_config',
            default_value='',
            description='Absolute path to sensor bridge YAML config '
                        '(default: package config/sensor_bridge.yaml)'),
        DeclareLaunchArgument(
            'use_pose_bridge',
            default_value='True',
            description='Enable Gazebo model pose -> ROS 2 bridge'),
        DeclareLaunchArgument(
            'use_clock_bridge',
            default_value='True',
            description='Enable Gazebo clock -> ROS 2 bridge'),
        DeclareLaunchArgument(
            'use_sensor_bridge',
            default_value='False',
            description='Enable optional sensor data bridge '
                        '(requires matching PX4 model variant)'),
        OpaqueFunction(function=launch_setup),
    ])
