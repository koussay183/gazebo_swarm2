"""
bridge_sensors.launch.py — Launch sensor bridge only.

Bridges depth camera, lidar, and IMU data from Gazebo to ROS 2.
Requires a PX4 model variant with the corresponding sensors
(gz_x500_depth, gz_x500_lidar, etc.).

Usage:
  ros2 launch mappo_swarm_bridge bridge_sensors.launch.py
  ros2 launch mappo_swarm_bridge bridge_sensors.launch.py \
      sensor_bridge_config:=/path/to/custom_sensor_bridge.yaml
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def launch_setup(context, *args, **kwargs):
    pkg_share = FindPackageShare('mappo_swarm_bridge').find('mappo_swarm_bridge')

    sensor_cfg = LaunchConfiguration('sensor_bridge_config').perform(context)
    if not sensor_cfg:
        sensor_cfg = os.path.join(pkg_share, 'config', 'sensor_bridge.yaml')

    sensor_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='gz_sensor_bridge',
        output='screen',
        parameters=[{'config_file': sensor_cfg}],
    )

    return [sensor_bridge]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'sensor_bridge_config',
            default_value='',
            description='Absolute path to sensor bridge YAML config. '
                        'Defaults to package config/sensor_bridge.yaml. '
                        'NOTE: Default file has all entries commented out — '
                        'you must provide a custom config with entries enabled.'),
        OpaqueFunction(function=launch_setup),
    ])
