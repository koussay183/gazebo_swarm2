"""
mappo_px4_rl.launch.py — Launch MAPPO RL control node with diagnostics.

Declares args for all key parameters and loads mappo_params.yaml as
the base configuration, overriding specific values from launch args.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    """Build the node list from resolved launch arguments."""

    pkg_share = FindPackageShare('mappo_swarm_px4').find('mappo_swarm_px4')
    params_file = f'{pkg_share}/config/mappo_params.yaml'

    num_drones = LaunchConfiguration('num_drones').perform(context)
    checkpoint_path = LaunchConfiguration('checkpoint_path').perform(context)
    control_mode = LaunchConfiguration('control_mode').perform(context)
    auto_arm = LaunchConfiguration('auto_arm').perform(context)
    auto_offboard = LaunchConfiguration('auto_offboard').perform(context)
    px4_topic_style = LaunchConfiguration('px4_topic_style').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context)

    mappo_node = Node(
        package='mappo_swarm_px4',
        executable='mappo_px4_rl_node',
        name='mappo_px4_rl_node',
        output='screen',
        parameters=[
            params_file,
            {
                'num_drones': int(num_drones),
                'checkpoint_path': checkpoint_path,
                'control_mode': control_mode,
                'auto_arm': auto_arm.lower() == 'true',
                'auto_offboard': auto_offboard.lower() == 'true',
                'px4_topic_style': px4_topic_style,
                'use_sim_time': use_sim_time.lower() == 'true',
            }
        ]
    )

    diagnostics_node = Node(
        package='mappo_swarm_px4',
        executable='swarm_diagnostics',
        name='swarm_diagnostics',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time.lower() == 'true'}]
    )

    return [mappo_node, diagnostics_node]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'num_drones',
            default_value='1',
            description='Number of drones in the swarm'),
        DeclareLaunchArgument(
            'checkpoint_path',
            default_value='',
            description='Absolute path to best.pt actor checkpoint'),
        DeclareLaunchArgument(
            'control_mode',
            default_value='velocity',
            description='Control mode: velocity or position'),
        DeclareLaunchArgument(
            'auto_arm',
            default_value='False',
            description='Automatically arm drones when ready (DANGER: set False for safety)'),
        DeclareLaunchArgument(
            'auto_offboard',
            default_value='False',
            description='Automatically enable offboard mode (DANGER: set False for safety)'),
        DeclareLaunchArgument(
            'px4_topic_style',
            default_value='auto',
            description='PX4 topic naming: auto | namespaced | global'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='True',
            description='Use simulation time clock'),
        OpaqueFunction(function=launch_setup),
    ])
