"""
px4_command_sender.py — Helper functions for sending PX4 VehicleCommand messages.

These are standalone utility functions (not a ROS 2 node). They require a
pre-created VehicleCommand publisher and the parent node (for timestamp).

Usage example:
    from mappo_swarm_px4.px4_command_sender import send_arm_command
    send_arm_command(my_vehicle_cmd_publisher, my_node)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node
    from rclpy.publisher import Publisher

# MAVLink command IDs
MAV_CMD_COMPONENT_ARM_DISARM: int = 400
MAV_CMD_NAV_LAND: int = 21
MAV_CMD_DO_SET_MODE: int = 176
PX4_CUSTOM_MAIN_MODE_OFFBOARD: float = 6.0


def _get_timestamp(node: 'Node') -> int:
    """Return the current time as microseconds for PX4 timestamp field."""
    return int(node.get_clock().now().nanoseconds / 1000)


def _make_vehicle_command(
        node: 'Node',
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
        target_system: int = 1) -> 'object':
    """
    Build a VehicleCommand message.

    Args:
        node: ROS 2 node (used for timestamp).
        command: MAVLink command ID.
        param1..param7: MAVLink command parameters.
        target_system: MAVLink target system (1-indexed).

    Returns:
        Populated VehicleCommand message, or None if px4_msgs unavailable.
    """
    try:
        from px4_msgs.msg import VehicleCommand
    except ImportError:
        node.get_logger().error(
            'px4_msgs not available — cannot build VehicleCommand.')
        return None

    msg = VehicleCommand()
    msg.timestamp = _get_timestamp(node)
    msg.command = command
    msg.param1 = float(param1)
    msg.param2 = float(param2)
    msg.param3 = float(param3)
    msg.param4 = float(param4)
    msg.param5 = float(param5)
    msg.param6 = float(param6)
    msg.param7 = float(param7)
    msg.target_system = target_system
    msg.target_component = 1
    msg.source_system = 255
    msg.source_component = 0
    msg.from_external = True
    return msg


def send_arm_command(pub: 'Publisher', node: 'Node',
                     target_system: int = 1) -> None:
    """
    Publish an arm command (MAV_CMD_COMPONENT_ARM_DISARM param1=1).

    Args:
        pub: Publisher for VehicleCommand topic.
        node: Parent ROS 2 node (for timestamp and logging).
        target_system: MAVLink system ID of target drone (default 1).
    """
    msg = _make_vehicle_command(
        node, MAV_CMD_COMPONENT_ARM_DISARM,
        param1=1.0,
        target_system=target_system)
    if msg is not None:
        pub.publish(msg)
        node.get_logger().info(
            f'Sent ARM command to system {target_system}')


def send_disarm_command(pub: 'Publisher', node: 'Node',
                        target_system: int = 1) -> None:
    """
    Publish a disarm command (MAV_CMD_COMPONENT_ARM_DISARM param1=0).

    Args:
        pub: Publisher for VehicleCommand topic.
        node: Parent ROS 2 node.
        target_system: MAVLink system ID.
    """
    msg = _make_vehicle_command(
        node, MAV_CMD_COMPONENT_ARM_DISARM,
        param1=0.0,
        target_system=target_system)
    if msg is not None:
        pub.publish(msg)
        node.get_logger().info(
            f'Sent DISARM command to system {target_system}')


def send_offboard_mode_command(pub: 'Publisher', node: 'Node',
                               target_system: int = 1) -> None:
    """
    Publish a mode change to PX4 offboard mode.

    Uses MAV_CMD_DO_SET_MODE with:
      param1 = 1 (MAV_MODE_FLAG_CUSTOM_MODE_ENABLED)
      param2 = 6 (CUSTOM_MAIN_MODE_OFFBOARD)

    Args:
        pub: Publisher for VehicleCommand topic.
        node: Parent ROS 2 node.
        target_system: MAVLink system ID.
    """
    msg = _make_vehicle_command(
        node, MAV_CMD_DO_SET_MODE,
        param1=1.0,
        param2=PX4_CUSTOM_MAIN_MODE_OFFBOARD,
        target_system=target_system)
    if msg is not None:
        pub.publish(msg)
        node.get_logger().info(
            f'Sent SET_OFFBOARD_MODE command to system {target_system}')


def send_land_command(pub: 'Publisher', node: 'Node',
                      target_system: int = 1) -> None:
    """
    Publish a land command (MAV_CMD_NAV_LAND).

    Args:
        pub: Publisher for VehicleCommand topic.
        node: Parent ROS 2 node.
        target_system: MAVLink system ID.
    """
    msg = _make_vehicle_command(
        node, MAV_CMD_NAV_LAND,
        target_system=target_system)
    if msg is not None:
        pub.publish(msg)
        node.get_logger().info(
            f'Sent LAND command to system {target_system}')
