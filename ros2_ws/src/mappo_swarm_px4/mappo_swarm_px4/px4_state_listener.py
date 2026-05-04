"""
px4_state_listener.py — Standalone ROS 2 node that monitors PX4 telemetry.

Subscribes to VehicleStatus and VehicleLocalPosition for all drones and
prints a live status table every 2 seconds to stdout.

Usage:
    ros2 run mappo_swarm_px4 px4_state_listener --ros-args -p num_drones:=4
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

try:
    from px4_msgs.msg import VehicleStatus, VehicleLocalPosition
    _PX4_MSGS_AVAILABLE = True
except ImportError:
    _PX4_MSGS_AVAILABLE = False

# PX4 arming state names
ARMING_STATE_NAMES: Dict[int, str] = {
    0: 'INIT',
    1: 'STANDBY',
    2: 'ARMED',
    3: 'STANDBY_ERROR',
    4: 'SHUTTINGDOWN',
    5: 'IN_AIR_RESTORE',
}

# PX4 nav state names (abbreviated)
NAV_STATE_NAMES: Dict[int, str] = {
    0: 'MANUAL',
    1: 'ALTCTL',
    2: 'POSCTL',
    3: 'AUTO_MISSION',
    4: 'AUTO_LOITER',
    5: 'AUTO_RTL',
    10: 'ACRO',
    12: 'DESCEND',
    13: 'TERMINATION',
    14: 'OFFBOARD',
    15: 'STAB',
    17: 'AUTO_TAKEOFF',
    18: 'AUTO_LAND',
    19: 'AUTO_FOLLOW',
    20: 'PRECLAND',
    21: 'ORBIT',
    22: 'AUTO_VTOL_TAKEOFF',
}


class DroneSnapshot:
    """Holds the latest telemetry snapshot for one drone."""

    def __init__(self, drone_id: int) -> None:
        self.drone_id = drone_id
        self.nav_state: Optional[int] = None
        self.arming_state: Optional[int] = None
        self.is_armed: bool = False
        self.is_offboard: bool = False
        self.battery_remaining: float = float('nan')
        self.pos_n: float = float('nan')
        self.pos_e: float = float('nan')
        self.pos_d: float = float('nan')
        self.vel_n: float = float('nan')
        self.vel_e: float = float('nan')
        self.vel_d: float = float('nan')


class PX4StateListener(Node):
    """
    Subscribes to PX4 telemetry for all drones and prints a status table.

    Parameters:
        num_drones (int): Number of drones to monitor (default: 4).
        topic_namespace_prefix (str): e.g. '/px4_'
        px4_topic_style (str): 'namespaced' | 'global' | 'auto'
        print_rate (float): Status print frequency in Hz (default: 0.5).
    """

    def __init__(self) -> None:
        super().__init__('px4_state_listener')

        self.declare_parameter('num_drones', 4)
        self.declare_parameter('topic_namespace_prefix', '/px4_')
        self.declare_parameter('px4_topic_style', 'auto')
        self.declare_parameter('print_rate', 0.5)

        self._num_drones: int = self.get_parameter('num_drones').value
        self._ns_prefix: str = self.get_parameter(
            'topic_namespace_prefix').value
        self._topic_style: str = self.get_parameter('px4_topic_style').value
        print_rate: float = self.get_parameter('print_rate').value

        self._snapshots: Dict[int, DroneSnapshot] = {
            i: DroneSnapshot(i) for i in range(self._num_drones)
        }

        if _PX4_MSGS_AVAILABLE:
            self._create_subscribers()
        else:
            self.get_logger().error(
                'px4_msgs not available. Cannot subscribe to PX4 telemetry.')

        period = 1.0 / max(0.1, print_rate)
        self.create_timer(period, self._print_status_table)

        self.get_logger().info(
            f'PX4StateListener monitoring {self._num_drones} drones '
            f'(style={self._topic_style}, prefix={self._ns_prefix})')

    def _create_subscribers(self) -> None:
        """Create VehicleStatus and VehicleLocalPosition subscribers."""
        qos = QoSPresetProfiles.SENSOR_DATA.value
        for drone_id in range(self._num_drones):
            ns = f'{self._ns_prefix.rstrip("/")}{drone_id}'

            vs_topic = f'{ns}/fmu/out/vehicle_status'
            vlp_topic = f'{ns}/fmu/out/vehicle_local_position'

            self.create_subscription(
                VehicleStatus, vs_topic,
                lambda msg, did=drone_id: self._on_status(msg, did),
                qos)
            self.create_subscription(
                VehicleLocalPosition, vlp_topic,
                lambda msg, did=drone_id: self._on_local_pos(msg, did),
                qos)

    def _on_status(self, msg: 'VehicleStatus', drone_id: int) -> None:
        snap = self._snapshots[drone_id]
        snap.nav_state = msg.nav_state
        snap.arming_state = msg.arming_state
        snap.is_armed = (msg.arming_state == 2)
        snap.is_offboard = (msg.nav_state == 14)
        if hasattr(msg, 'battery_warning'):
            snap.battery_remaining = getattr(msg, 'battery_remaining', float('nan'))

    def _on_local_pos(self, msg: 'VehicleLocalPosition', drone_id: int) -> None:
        snap = self._snapshots[drone_id]
        # PX4 VehicleLocalPosition is NED
        snap.pos_n = msg.x
        snap.pos_e = msg.y
        snap.pos_d = msg.z
        snap.vel_n = msg.vx
        snap.vel_e = msg.vy
        snap.vel_d = msg.vz

    def _print_status_table(self) -> None:
        """Print formatted status table to stdout."""
        header = (
            f'\n{"="*90}\n'
            f'  {"ID":<4} {"NAV_STATE":<16} {"ARM_STATE":<14} '
            f'{"ARM":<5} {"OFB":<5} {"BATT":<7} '
            f'{"N(m)":<8} {"E(m)":<8} {"D(m)":<8} '
            f'{"VN":<7} {"VE":<7} {"VD":<7}\n'
            f'{"="*90}'
        )
        rows = []
        for drone_id, snap in self._snapshots.items():
            nav_name = NAV_STATE_NAMES.get(snap.nav_state, str(snap.nav_state)) \
                if snap.nav_state is not None else 'N/A'
            arm_name = ARMING_STATE_NAMES.get(snap.arming_state, str(snap.arming_state)) \
                if snap.arming_state is not None else 'N/A'
            batt = f'{snap.battery_remaining*100:.0f}%' \
                if not math.isnan(snap.battery_remaining) else 'N/A'

            def _fmt(v: float) -> str:
                return f'{v:7.2f}' if not math.isnan(v) else '    N/A'

            row = (
                f'  {drone_id:<4} {nav_name:<16} {arm_name:<14} '
                f'{"Y" if snap.is_armed else "N":<5} '
                f'{"Y" if snap.is_offboard else "N":<5} '
                f'{batt:<7} '
                f'{_fmt(snap.pos_n)} {_fmt(snap.pos_e)} {_fmt(snap.pos_d)} '
                f'{_fmt(snap.vel_n)} {_fmt(snap.vel_e)} {_fmt(snap.vel_d)}'
            )
            rows.append(row)

        print(header)
        for row in rows:
            print(row)
        print()


def main(args=None) -> None:
    """Entry point for px4_state_listener."""
    rclpy.init(args=args)
    node = PX4StateListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('PX4StateListener shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
