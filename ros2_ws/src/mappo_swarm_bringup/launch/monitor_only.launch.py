"""
monitor_only.launch.py — Launch monitoring tools only (no control).

Starts swarm_monitor, RViz2, and the CSV logger. Safe to run alongside
any control stack or independently to observe an already-running system.

Usage:
  ros2 launch mappo_swarm_bringup monitor_only.launch.py num_drones:=4
  ros2 launch mappo_swarm_bringup monitor_only.launch.py \\
      num_drones:=4 use_rviz:=True log_dir:=/tmp/swarm_logs
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def launch_setup(context, *args, **kwargs):
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    ns_prefix = LaunchConfiguration('namespace_prefix').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    use_rviz = LaunchConfiguration('use_rviz').perform(context).lower() == 'true'
    log_dir = LaunchConfiguration('log_dir').perform(context)

    nodes = []

    # Swarm monitor (status table + JSON)
    swarm_monitor = Node(
        package='mappo_swarm_gcs',
        executable='swarm_monitor',
        name='swarm_monitor',
        output='screen',
        parameters=[{
            'num_drones': num_drones,
            'topic_namespace_prefix': ns_prefix,
            'use_sim_time': use_sim_time,
        }]
    )
    nodes.append(swarm_monitor)

    # CSV logger
    swarm_logger = Node(
        package='mappo_swarm_gcs',
        executable='swarm_logger',
        name='swarm_logger',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'log_dir': log_dir,
        }]
    )
    nodes.append(swarm_logger)

    # Optional RViz2
    if use_rviz:
        gcs_share = FindPackageShare('mappo_swarm_gcs').find('mappo_swarm_gcs')
        rviz_config = os.path.join(gcs_share, 'rviz', 'swarm.rviz')
        rviz_node = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[{'use_sim_time': use_sim_time}],
        )
        nodes.append(rviz_node)

    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'num_drones', default_value='4',
            description='Number of drones to monitor'),
        DeclareLaunchArgument(
            'namespace_prefix', default_value='/px4_',
            description='PX4 namespace prefix'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='True',
            description='Use simulation clock'),
        DeclareLaunchArgument(
            'use_rviz', default_value='False',
            description='Launch RViz2'),
        DeclareLaunchArgument(
            'log_dir', default_value='.',
            description='Directory for CSV log files'),
        OpaqueFunction(function=launch_setup),
    ])
