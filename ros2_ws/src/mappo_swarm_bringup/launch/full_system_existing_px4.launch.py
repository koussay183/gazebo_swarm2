"""
full_system_existing_px4.launch.py — Full swarm system bringup.

PRECONDITION: PX4 SITL + Gazebo Harmonic must ALREADY be running.

Start PX4 manually BEFORE this launch:
  Terminal 1: cd ~/PX4-Autopilot && make px4_sitl gz_x500
  Terminal 2: MicroXRCEAgent udp4 -p 8888  (or use start_uxrce_agent argument)

This launch file starts:
  1. uXRCE-DDS agent (optional — if start_uxrce_agent:=True)
  2. Gazebo-ROS2 bridge (optional — if use_ros_gz_bridge:=True)
  3. MAPPO RL control node + diagnostics (mappo_swarm_px4)
  4. GCS nodes: monitor, supervisor, emergency control, logger (mappo_swarm_gcs)
  5. RViz2 (optional — if use_rviz:=True)

Safety reminder:
  auto_arm and auto_offboard BOTH default to False.
  Do NOT set True until you have verified all checkpoints CP0-CP8.

Usage:
  # Minimal (verify topics only):
  ros2 launch mappo_swarm_bringup full_system_existing_px4.launch.py

  # With 4 drones and checkpoint:
  ros2 launch mappo_swarm_bringup full_system_existing_px4.launch.py \\
      num_drones:=4 \\
      checkpoint_path:=/home/user/checkpoints/best.pt \\
      px4_topic_style:=namespaced

  # Start uXRCE agent from this launch:
  ros2 launch mappo_swarm_bringup full_system_existing_px4.launch.py \\
      start_uxrce_agent:=True

  # Enable RViz:
  ros2 launch mappo_swarm_bringup full_system_existing_px4.launch.py \\
      use_rviz:=True
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def launch_setup(context, *args, **kwargs):
    # Resolve all arguments
    start_uxrce = LaunchConfiguration('start_uxrce_agent').perform(context).lower() == 'true'
    use_bridge = LaunchConfiguration('use_ros_gz_bridge').perform(context).lower() == 'true'
    use_rviz = LaunchConfiguration('use_rviz').perform(context).lower() == 'true'
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    checkpoint = LaunchConfiguration('checkpoint_path').perform(context)
    control_mode = LaunchConfiguration('control_mode').perform(context)
    auto_arm = LaunchConfiguration('auto_arm').perform(context).lower() == 'true'
    auto_offboard = LaunchConfiguration('auto_offboard').perform(context).lower() == 'true'
    topic_style = LaunchConfiguration('px4_topic_style').perform(context)
    ns_prefix = LaunchConfiguration('namespace_prefix').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'

    nodes = []

    # ------------------------------------------------------------------ #
    # 1. Optional: uXRCE-DDS agent
    # ------------------------------------------------------------------ #
    if start_uxrce:
        uxrce_agent = ExecuteProcess(
            cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
            name='uxrce_dds_agent',
            output='screen',
        )
        nodes.append(uxrce_agent)

    # ------------------------------------------------------------------ #
    # 2. Optional: Gazebo-ROS2 bridge
    # ------------------------------------------------------------------ #
    if use_bridge:
        from launch.actions import IncludeLaunchDescription
        from launch.launch_description_sources import PythonLaunchDescriptionSource
        bridge_pkg = FindPackageShare('mappo_swarm_bridge').find('mappo_swarm_bridge')
        bridge_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bridge_pkg, 'launch', 'ros_gz_bridge.launch.py')),
            launch_arguments={
                'use_pose_bridge': 'True',
                'use_clock_bridge': 'True',
                'use_sensor_bridge': 'False',
            }.items())
        nodes.append(bridge_launch)

    # ------------------------------------------------------------------ #
    # 3. MAPPO RL + diagnostics
    # ------------------------------------------------------------------ #
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
            }
        ]
    )
    nodes.append(mappo_node)

    diagnostics_node = Node(
        package='mappo_swarm_px4',
        executable='swarm_diagnostics',
        name='swarm_diagnostics',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )
    nodes.append(diagnostics_node)

    # ------------------------------------------------------------------ #
    # 4. GCS nodes
    # ------------------------------------------------------------------ #
    from launch.actions import IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    gcs_pkg = FindPackageShare('mappo_swarm_gcs').find('mappo_swarm_gcs')
    gcs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gcs_pkg, 'launch', 'gcs.launch.py')),
        launch_arguments={
            'num_drones': str(num_drones),
            'namespace_prefix': ns_prefix,
            'use_sim_time': str(use_sim_time),
        }.items())
    nodes.append(gcs_launch)

    # ------------------------------------------------------------------ #
    # 5. Optional: RViz2
    # ------------------------------------------------------------------ #
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
            'start_uxrce_agent', default_value='False',
            description='Start MicroXRCEAgent (set True if not running externally)'),
        DeclareLaunchArgument(
            'use_ros_gz_bridge', default_value='True',
            description='Start Gazebo-ROS2 bridge for pose + clock'),
        DeclareLaunchArgument(
            'use_rviz', default_value='False',
            description='Start RViz2 visualisation'),
        DeclareLaunchArgument(
            'num_drones', default_value='1',
            description='Number of drones'),
        DeclareLaunchArgument(
            'checkpoint_path', default_value='',
            description='Absolute path to best.pt (empty = demo mode)'),
        DeclareLaunchArgument(
            'control_mode', default_value='velocity',
            description='velocity or position'),
        DeclareLaunchArgument(
            'auto_arm', default_value='False',
            description='DANGER: auto-arm drones (keep False until CP8)'),
        DeclareLaunchArgument(
            'auto_offboard', default_value='False',
            description='DANGER: auto-offboard mode (keep False until CP7)'),
        DeclareLaunchArgument(
            'px4_topic_style', default_value='auto',
            description='auto | namespaced | global'),
        DeclareLaunchArgument(
            'namespace_prefix', default_value='/px4_',
            description='PX4 namespace prefix'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='True',
            description='Use simulation clock'),
        OpaqueFunction(function=launch_setup),
    ])
