"""
emergency_control_node.py — Emergency stop and watchdog controller.

Monitors drone heartbeats and responds to emergency stop commands.
If no heartbeat is received from any drone for 5 seconds, automatically
publishes an emergency stop.

On emergency stop activation:
  - Publishes LAND VehicleCommand to all drones
  - Sets /swarm/emergency_active to True

Published Topics:
  /swarm/emergency_active  (std_msgs/Bool) — True when e-stop is active

Subscribed Topics:
  /swarm/emergency_stop    (std_msgs/Bool) — Trigger from mission supervisor
  /swarm/monitor_status    (std_msgs/String) — Heartbeat source
"""

from __future__ import annotations

import json
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import Bool, String

try:
    from px4_msgs.msg import VehicleCommand, VehicleStatus
    _PX4_MSGS_AVAILABLE = True
except ImportError:
    _PX4_MSGS_AVAILABLE = False

MAV_CMD_NAV_LAND = 21
MAV_CMD_COMPONENT_ARM_DISARM = 400


class EmergencyControlNode(Node):
    """
    Emergency stop and watchdog node.

    Triggers automated landing when:
      1. /swarm/emergency_stop is received
      2. No drone heartbeat for 5 seconds (watchdog timeout)
    """

    HEARTBEAT_TIMEOUT_S: float = 5.0

    def __init__(self) -> None:
        super().__init__('emergency_control')

        self.declare_parameter('num_drones', 4)
        self.declare_parameter('topic_namespace_prefix', '/px4_')
        self.declare_parameter('watchdog_enabled', True)
        self.declare_parameter('use_sim_time', True)

        self._num_drones: int = self.get_parameter('num_drones').value
        self._ns_prefix: str = self.get_parameter(
            'topic_namespace_prefix').value
        self._watchdog_enabled: bool = self.get_parameter(
            'watchdog_enabled').value

        self._emergency_active: bool = False
        self._last_heartbeat_s: float = time.time()

        # Per-drone VehicleCommand publishers
        self._cmd_pubs: Dict[int, Optional[object]] = {}
        if _PX4_MSGS_AVAILABLE:
            for i in range(self._num_drones):
                ns = f'{self._ns_prefix.rstrip("/")}{i}'
                self._cmd_pubs[i] = self.create_publisher(
                    VehicleCommand,
                    f'{ns}/fmu/in/vehicle_command',
                    QoSPresetProfiles.SENSOR_DATA.value)
        else:
            self.get_logger().warn(
                'px4_msgs unavailable — vehicle commands will be logged only.')

        # Publishers
        self._pub_active = self.create_publisher(
            Bool, '/swarm/emergency_active', 10)

        # Subscribers
        self.create_subscription(
            Bool, '/swarm/emergency_stop',
            self._on_emergency_stop, 10)
        self.create_subscription(
            String, '/swarm/monitor_status',
            self._on_monitor_status, 10)

        # 1 Hz watchdog timer
        self.create_timer(1.0, self._watchdog_tick)

        self.get_logger().info(
            f'EmergencyControlNode ready: {self._num_drones} drones, '
            f'watchdog={self._watchdog_enabled} '
            f'(timeout={self.HEARTBEAT_TIMEOUT_S}s)')

    def _on_emergency_stop(self, msg: Bool) -> None:
        """Handle /swarm/emergency_stop."""
        if msg.data:
            self.get_logger().error(
                'Emergency stop received — landing all drones.')
            self._activate_emergency()

    def _on_monitor_status(self, msg: String) -> None:
        """Update heartbeat timestamp from monitor status."""
        self._last_heartbeat_s = time.time()

    def _watchdog_tick(self) -> None:
        """1 Hz watchdog: check for heartbeat timeout."""
        if not self._watchdog_enabled or self._emergency_active:
            return

        elapsed = time.time() - self._last_heartbeat_s
        if elapsed > self.HEARTBEAT_TIMEOUT_S:
            self.get_logger().error(
                f'WATCHDOG: No heartbeat for {elapsed:.1f}s — '
                f'auto-publishing emergency stop!')
            self._activate_emergency()
        elif elapsed > 2.0:
            self.get_logger().warn(
                f'Watchdog: Last heartbeat {elapsed:.1f}s ago.')

        # Publish current state
        msg = Bool()
        msg.data = self._emergency_active
        self._pub_active.publish(msg)

    def _activate_emergency(self) -> None:
        """Activate emergency stop and land all drones."""
        self._emergency_active = True

        active_msg = Bool()
        active_msg.data = True
        self._pub_active.publish(active_msg)

        self._land_all_drones()

    def _land_all_drones(self) -> None:
        """Publish LAND VehicleCommand to all drones."""
        if not _PX4_MSGS_AVAILABLE:
            self.get_logger().error(
                'px4_msgs unavailable — cannot send LAND command. '
                'Manual intervention required!')
            return

        timestamp_us = int(self.get_clock().now().nanoseconds / 1000)

        for drone_id in range(self._num_drones):
            pub = self._cmd_pubs.get(drone_id)
            if pub is None:
                continue

            msg = VehicleCommand()
            msg.timestamp = timestamp_us
            msg.command = MAV_CMD_NAV_LAND
            msg.param1 = 0.0
            msg.param2 = 0.0
            msg.param3 = 0.0
            msg.param4 = float('nan')  # yaw: keep current
            msg.param5 = float('nan')  # lat
            msg.param6 = float('nan')  # lon
            msg.param7 = 0.0           # altitude (land at ground)
            msg.target_system = drone_id + 1
            msg.target_component = 1
            msg.source_system = 255
            msg.source_component = 0
            msg.from_external = True
            pub.publish(msg)

            self.get_logger().error(
                f'  LAND command sent to drone {drone_id}')

    def reset_emergency(self) -> None:
        """Reset emergency state (call after confirming all drones landed)."""
        self._emergency_active = False
        self._last_heartbeat_s = time.time()
        self.get_logger().info('Emergency state reset.')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EmergencyControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('EmergencyControlNode shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
