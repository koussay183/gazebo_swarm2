"""
gcs.launch.py — Launch all Ground Control Station nodes.

Starts:
  - mission_supervisor  — CLI mission command interface
  - swarm_monitor       — Real-time status table + JSON publisher
  - emergency_control   — Emergency stop watchdog
  - swarm_logger        — CSV flight data logger
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    ns_prefix = LaunchConfiguration('namespace_prefix').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    log_dir = LaunchConfiguration('log_dir').perform(context)

    common_params = {
        'num_drones': num_drones,
        'topic_namespace_prefix': ns_prefix,
        'use_sim_time': use_sim_time,
    }

    mission_supervisor = Node(
        package='mappo_swarm_gcs',
        executable='mission_supervisor',
        name='mission_supervisor',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        prefix='',  # Run in foreground for stdin access
    )

    swarm_monitor = Node(
        package='mappo_swarm_gcs',
        executable='swarm_monitor',
        name='swarm_monitor',
        output='screen',
        parameters=[common_params],
    )

    emergency_control = Node(
        package='mappo_swarm_gcs',
        executable='emergency_control',
        name='emergency_control',
        output='screen',
        parameters=[common_params],
    )

    swarm_logger = Node(
        package='mappo_swarm_gcs',
        executable='swarm_logger',
        name='swarm_logger',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'log_dir': log_dir,
        }],
    )

    return [mission_supervisor, swarm_monitor, emergency_control, swarm_logger]


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
            description='Use simulation time'),
        DeclareLaunchArgument(
            'log_dir', default_value='.',
            description='Directory for CSV log files'),
        OpaqueFunction(function=launch_setup),
    ])
