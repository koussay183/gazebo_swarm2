"""
autonomy_only.launch.py — Launch only the MAPPO RL node and diagnostics.

Minimal deployment stack: no GCS, no bridge, no RViz.
Suitable for:
  - Headless deployment
  - Integrating with external GCS systems
  - Performance profiling of the RL inference loop

Preconditions:
  - PX4 SITL running
  - uXRCE-DDS agent running (MicroXRCEAgent udp4 -p 8888)
  - ROS 2 workspace sourced

Usage:
  ros2 launch mappo_swarm_bringup autonomy_only.launch.py \\
      num_drones:=4 checkpoint_path:=/path/to/best.pt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def launch_setup(context, *args, **kwargs):
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    checkpoint = LaunchConfiguration('checkpoint_path').perform(context)
    control_mode = LaunchConfiguration('control_mode').perform(context)
    auto_arm = LaunchConfiguration('auto_arm').perform(context).lower() == 'true'
    auto_offboard = LaunchConfiguration('auto_offboard').perform(context).lower() == 'true'
    topic_style = LaunchConfiguration('px4_topic_style').perform(context)
    ns_prefix = LaunchConfiguration('namespace_prefix').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    control_rate = float(LaunchConfiguration('control_rate').perform(context))

    px4_pkg = FindPackageShare('mappo_swarm_px4').find('mappo_swarm_px4')
    params_file = os.path.join(px4_pkg, 'config', 'mappo_params.yaml')

    mappo_node = Node(
        package='mappo_swarm_px4',
        executable='mappo_px4_rl_node',
        name='mappo_px4_rl_node',
        output='screen',
        parameters=[
            params_file,
            {
                'num_drones': num_drones,
                'checkpoint_path': checkpoint,
                'control_mode': control_mode,
                'auto_arm': auto_arm,
                'auto_offboard': auto_offboard,
                'px4_topic_style': topic_style,
                'topic_namespace_prefix': ns_prefix,
                'use_sim_time': use_sim_time,
                'control_rate': control_rate,
            }
        ]
    )

    diagnostics_node = Node(
        package='mappo_swarm_px4',
        executable='swarm_diagnostics',
        name='swarm_diagnostics',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return [mappo_node, diagnostics_node]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'num_drones', default_value='1',
            description='Number of drones'),
        DeclareLaunchArgument(
            'checkpoint_path', default_value='',
            description='Absolute path to best.pt actor checkpoint'),
        DeclareLaunchArgument(
            'control_mode', default_value='velocity',
            description='velocity or position'),
        DeclareLaunchArgument(
            'auto_arm', default_value='False',
            description='Auto-arm drones (default False — safe)'),
        DeclareLaunchArgument(
            'auto_offboard', default_value='False',
            description='Auto-enable offboard mode (default False — safe)'),
        DeclareLaunchArgument(
            'px4_topic_style', default_value='auto',
            description='auto | namespaced | global'),
        DeclareLaunchArgument(
            'namespace_prefix', default_value='/px4_',
            description='PX4 namespace prefix'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='True',
            description='Use simulation clock'),
        DeclareLaunchArgument(
            'control_rate', default_value='30.0',
            description='RL control loop frequency in Hz'),
        OpaqueFunction(function=launch_setup),
    ])
