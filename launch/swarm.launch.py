"""
ROS2 Python launch file for Gazebo Harmonic swarm drone simulation.
Launches: Gazebo Harmonic (gz sim), ros_gz_bridge (4 drones + set_pose), optional RViz2.
"""
import os
import tempfile

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

try:
    from ament_index_python.packages import get_package_share_directory
    HAS_AMENT = True
except ImportError:
    HAS_AMENT = False


# ─── Helpers ───────────────────────────────────────────────────────────────

def get_world_path():
    """Return absolute path to the SDF world file."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.dirname(script_dir)
    return os.path.join(project_dir, "worlds", "swarm_world.sdf")


def get_models_path():
    """Return absolute path to the models directory."""
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_dir = os.path.dirname(script_dir)
    return os.path.join(project_dir, "models")


def write_bridge_yaml(num_drones: int = 4) -> str:
    """Write a temporary YAML config file for ros_gz_bridge and return its path.

    cmd_vel entries removed: PX4 SITL controls motors directly via gz-transport.
    Only pose (read position) and set_pose service (teleport/reset) are needed.
    """
    yaml_lines = ["- ros_topic_name: /world/swarm_world/set_pose",
                  "  gz_service_name: /world/swarm_world/set_pose",
                  "  ros_type_name: ros_gz_interfaces/srv/SetEntityPose",
                  "  gz_request_type: gz.msgs.Entity",
                  "  gz_response_type: gz.msgs.Boolean",
                  "  direction: ros_to_gz",
                  ""]

    for i in range(num_drones):
        yaml_lines += [
            f"- ros_topic_name: /x500_{i}/pose",
            f"  gz_topic_name: /model/x500_{i}/pose",
            f"  ros_type_name: geometry_msgs/msg/Pose",
            f"  gz_type_name: gz.msgs.Pose",
            f"  direction: gz_to_ros",
            "",
        ]

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_swarm_bridge.yaml", delete=False
    )
    tmp.write("\n".join(yaml_lines))
    tmp.flush()
    tmp.close()
    return tmp.name


# ─── Launch setup function ─────────────────────────────────────────────────

def launch_setup(context, *args, **kwargs):
    headless    = LaunchConfiguration("headless").perform(context)
    use_rviz    = LaunchConfiguration("use_rviz").perform(context)
    num_drones  = int(LaunchConfiguration("num_drones").perform(context))

    world_path  = get_world_path()
    models_path = get_models_path()
    bridge_yaml = write_bridge_yaml(num_drones)

    actions = []

    # ── Environment variables ──────────────────────────────────────────────
    actions.append(
        SetEnvironmentVariable(
            name="GZ_SIM_RESOURCE_PATH",
            value=models_path + ":" + os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        )
    )
    actions.append(
        SetEnvironmentVariable(
            name="GAZEBO_MODEL_PATH",
            value=models_path + ":" + os.environ.get("GAZEBO_MODEL_PATH", ""),
        )
    )

    # ── Gazebo Harmonic ────────────────────────────────────────────────────
    gz_args = [world_path]
    if headless.lower() in ("true", "1", "yes"):
        gz_args = ["-s", world_path]   # server only (headless)

    gz_sim = ExecuteProcess(
        cmd=["gz", "sim"] + gz_args,
        output="screen",
        name="gz_sim",
    )
    actions.append(gz_sim)

    # ── PX4 SITL instances ─────────────────────────────────────────────────
    px4_path = os.environ.get("PX4_PATH", "/opt/px4/PX4-Autopilot")
    px4_bin = os.path.join(px4_path, "build/px4_sitl_default/bin/px4")

    if os.path.isfile(px4_bin):
        for i in range(num_drones):
            px4_instance = TimerAction(
                period=3.0 + i * 1.0,  # Démarrage décalé pour éviter conflits
                actions=[
                    ExecuteProcess(
                        cmd=[px4_bin, "-i", str(i), "-d"],
                        cwd=px4_path,
                        output="screen",
                        name=f"px4_instance_{i}",
                        additional_env={
                            "PX4_SYS_AUTOSTART": "4001",
                            "PX4_GZ_STANDALONE": "1",
                            "PX4_GZ_MODEL_NAME": f"x500_{i}",
                            "PX4_SIM_HOSTNAME": "localhost",
                        },
                    )
                ],
            )
            actions.append(px4_instance)
    else:
        print(f"[swarm.launch] PX4 non trouvé à {px4_bin} — démarrer manuellement")

    # ── ROS2 ↔ Gazebo bridge ───────────────────────────────────────────────
    # Flat parameter list for parameter_bridge (one topic/service per arg).
    # cmd_vel removed: PX4 SITL controls motors directly via gz-transport.
    # Only pose topics (read real physics position) and set_pose service are needed.
    bridge_params = []
    for i in range(num_drones):
        bridge_params += [
            f"/model/x500_{i}/pose@geometry_msgs/msg/Pose]gz.msgs.Pose",
        ]
    # set_pose service bridge
    bridge_params += [
        "/world/swarm_world/set_pose@ros_gz_interfaces/srv/SetEntityPose",
    ]

    ros_gz_bridge = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="swarm_gz_bridge",
                output="screen",
                arguments=bridge_params,
                parameters=[{"use_sim_time": True}],
            )
        ],
    )
    actions.append(ros_gz_bridge)

    # ── Optional RViz2 ─────────────────────────────────────────────────────
    if use_rviz.lower() in ("true", "1", "yes"):
        rviz_config = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "swarm.rviz"
        )
        rviz_args = ["-d", rviz_config] if os.path.isfile(rviz_config) else []

        rviz_node = TimerAction(
            period=7.0,
            actions=[
                Node(
                    package="rviz2",
                    executable="rviz2",
                    name="rviz2",
                    output="screen",
                    arguments=rviz_args,
                    parameters=[{"use_sim_time": True}],
                )
            ],
        )
        actions.append(rviz_node)

    return actions


# ─── Generate launch description ──────────────────────────────────────────

def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "headless",
            default_value="false",
            description="Run Gazebo in headless (server-only) mode",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Launch RViz2 alongside the simulation",
        ),
        DeclareLaunchArgument(
            "num_drones",
            default_value="4",
            description="Number of drones in the swarm (must match world SDF)",
        ),
    ]

    return LaunchDescription(declared_args + [OpaqueFunction(function=launch_setup)])
