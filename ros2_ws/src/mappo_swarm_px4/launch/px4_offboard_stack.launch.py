"""
px4_offboard_stack.launch.py — Launch offboard manager in minimal mode.

Suitable for manual testing without RL inference. Launches the MAPPO node
with auto_arm=False and auto_offboard=False so operators can verify
topic connectivity before enabling autonomous control.

Use Case:
  - Verify PX4 topics are visible from ROS 2.
  - Test offboard heartbeat publishing manually.
  - Debug telemetry subscriptions.

Usage:
  ros2 launch mappo_swarm_px4 px4_offboard_stack.launch.py num_drones:=1
  ros2 launch mappo_swarm_px4 px4_offboard_stack.launch.py \
      num_drones:=4 px4_topic_style:=namespaced
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    pkg_share_cmd = 'ros2 pkg prefix mappo_swarm_px4'

    num_drones = LaunchConfiguration('num_drones').perform(context)
    px4_topic_style = LaunchConfiguration('px4_topic_style').perform(context)
    ns_prefix = LaunchConfiguration('namespace_prefix').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)

    # Offboard stack: RL node with safe defaults (no auto arm/offboard)
    offboard_node = Node(
        package='mappo_swarm_px4',
        executable='mappo_px4_rl_node',
        name='mappo_px4_offboard_stack',
        output='screen',
        parameters=[{
            'num_drones': int(num_drones),
            'checkpoint_path': '',        # No RL — demo fallback
            'control_mode': 'velocity',
            'auto_arm': False,            # ALWAYS FALSE in this stack
            'auto_offboard': False,       # ALWAYS FALSE in this stack
            'px4_topic_style': px4_topic_style,
            'topic_namespace_prefix': ns_prefix,
            'use_sim_time': use_sim_time.lower() == 'true',
            'control_rate': 10.0,         # Low rate for manual testing
        }]
    )

    # State listener for manual inspection
    state_listener = Node(
        package='mappo_swarm_px4',
        executable='px4_state_listener',
        name='px4_state_listener',
        output='screen',
        parameters=[{
            'num_drones': int(num_drones),
            'topic_namespace_prefix': ns_prefix,
            'px4_topic_style': px4_topic_style,
            'use_sim_time': use_sim_time.lower() == 'true',
            'print_rate': 0.5,
        }]
    )

    return [offboard_node, state_listener]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'num_drones',
            default_value='1',
            description='Number of drones to manage'),
        DeclareLaunchArgument(
            'px4_topic_style',
            default_value='auto',
            description='PX4 topic naming: auto | namespaced | global'),
        DeclareLaunchArgument(
            'namespace_prefix',
            default_value='/px4_',
            description='PX4 namespace prefix'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation time'),
        OpaqueFunction(function=launch_setup),
    ])
