"""
px4_external_sim_assumed.launch.py — Template for external PX4 SITL integration.

IMPORTANT: This launch file assumes PX4 SITL and Gazebo Harmonic are ALREADY
running externally. It will NOT start PX4 or Gazebo.

=============================================================================
REQUIRED EXTERNAL SETUP (in separate terminals BEFORE running this launch):
=============================================================================

  Terminal 1: Start PX4 SITL (single drone)
    cd ~/PX4-Autopilot
    make px4_sitl gz_x500

  Terminal 1: Start PX4 SITL (4 drones — namespaced)
    cd ~/PX4-Autopilot
    PX4_UXRCE_DDS_NS=px4_0 PX4_GZ_MODEL_NAME=x500_0 make px4_sitl gz_x500 &
    PX4_UXRCE_DDS_NS=px4_1 PX4_GZ_MODEL_NAME=x500_1 make px4_sitl gz_x500 &
    # (repeat for px4_2, px4_3 ...)

  Terminal 2: Start uXRCE-DDS Agent
    MicroXRCEAgent udp4 -p 8888

  Terminal 3: (Optional) Start QGroundControl
    ./QGroundControl.AppImage

  Terminal 4: Verify PX4 topics are bridged
    ros2 topic list | grep px4
    # Should show: /px4_0/fmu/out/vehicle_status etc.

=============================================================================
THEN run this launch:
=============================================================================

  ros2 launch mappo_swarm_bringup px4_external_sim_assumed.launch.py \\
      num_drones:=1 checkpoint_path:=/path/to/best.pt

=============================================================================

This launch starts:
  1. (Optional) uXRCE-DDS agent — set start_uxrce_agent:=True if not running
  2. MAPPO autonomy stack (RL node + diagnostics)
  3. GCS stack (monitor, supervisor, emergency control, logger)

Safety:
  auto_arm and auto_offboard default to False.
  Enable step by step following the CP0-CP10 test plan.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
import os


def launch_setup(context, *args, **kwargs):
    start_uxrce = LaunchConfiguration('start_uxrce_agent').perform(context).lower() == 'true'
    num_drones = int(LaunchConfiguration('num_drones').perform(context))
    checkpoint = LaunchConfiguration('checkpoint_path').perform(context)
    topic_style = LaunchConfiguration('px4_topic_style').perform(context)
    ns_prefix = LaunchConfiguration('namespace_prefix').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    auto_arm = LaunchConfiguration('auto_arm').perform(context).lower() == 'true'
    auto_offboard = LaunchConfiguration('auto_offboard').perform(context).lower() == 'true'

    nodes = []

    # ------------------------------------------------------------------ #
    # 1. Optional uXRCE-DDS agent
    #    Set start_uxrce_agent:=True if you haven't started it manually.
    #    If already running, leave False to avoid port conflicts.
    # ------------------------------------------------------------------ #
    if start_uxrce:
        nodes.append(ExecuteProcess(
            cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
            name='uxrce_dds_agent',
            output='screen',
        ))

    # ------------------------------------------------------------------ #
    # 2. MAPPO autonomy stack
    # ------------------------------------------------------------------ #
    bringup_pkg = FindPackageShare('mappo_swarm_bringup').find('mappo_swarm_bringup')
    autonomy_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'autonomy_only.launch.py')),
        launch_arguments={
            'num_drones': str(num_drones),
            'checkpoint_path': checkpoint,
            'px4_topic_style': topic_style,
            'namespace_prefix': ns_prefix,
            'use_sim_time': str(use_sim_time),
            'auto_arm': str(auto_arm),
            'auto_offboard': str(auto_offboard),
        }.items())
    nodes.append(autonomy_launch)

    # ------------------------------------------------------------------ #
    # 3. GCS stack
    # ------------------------------------------------------------------ #
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

    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'start_uxrce_agent', default_value='False',
            description='Start MicroXRCEAgent (False if already running)'),
        DeclareLaunchArgument(
            'num_drones', default_value='1',
            description='Number of drones (must match PX4 SITL instances)'),
        DeclareLaunchArgument(
            'checkpoint_path', default_value='',
            description='Absolute path to best.pt (empty = demo mode)'),
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
            'auto_arm', default_value='False',
            description='DANGER: auto-arm (keep False until CP8 verified)'),
        DeclareLaunchArgument(
            'auto_offboard', default_value='False',
            description='DANGER: auto-offboard (keep False until CP7 verified)'),
        OpaqueFunction(function=launch_setup),
    ])
