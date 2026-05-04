"""
swarm_monitor_node.py — Real-time swarm status monitor.

Subscribes to PX4 telemetry topics for all drones and prints a
formatted status table to stdout every 2 seconds.

Also publishes a JSON status summary to /swarm/monitor_status for
logging and other consumers.

Published Topics:
  /swarm/monitor_status (std_msgs/String) — JSON status of all drones

Subscribed Topics:
  /px4_N/fmu/out/vehicle_status
  /px4_N/fmu/out/vehicle_local_position
"""

from __future__ import annotations

import json
import math
from typing import Dict, Optional, Any

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import String

try:
    from px4_msgs.msg import VehicleStatus, VehicleLocalPosition
    _PX4_MSGS_AVAILABLE = True
except ImportError:
    _PX4_MSGS_AVAILABLE = False

# PX4 state name maps
NAV_STATE_NAMES: Dict[int, str] = {
    0: 'MANUAL', 1: 'ALTCTL', 2: 'POSCTL',
    3: 'MISSION', 4: 'LOITER', 5: 'RTL',
    10: 'ACRO', 12: 'DESCEND', 13: 'TERM',
    14: 'OFFBOARD', 15: 'STAB', 17: 'TAKEOFF',
    18: 'LAND', 22: 'VTOL_TKO',
}
ARM_STATE_NAMES: Dict[int, str] = {
    0: 'INIT', 1: 'STANDBY', 2: 'ARMED',
    3: 'STBY_ERR', 4: 'SHUTDOWN',
}


class DroneMonitorData:
    """Holds aggregated status for one drone."""

    def __init__(self, drone_id: int) -> None:
        self.drone_id = drone_id
        self.nav_state: Optional[int] = None
        self.arming_state: Optional[int] = None
        self.is_armed: bool = False
        self.is_offboard: bool = False
        self.battery_remaining: float = float('nan')
        # Position (NED from PX4)
        self.pos_x: float = float('nan')
        self.pos_y: float = float('nan')
        self.pos_z: float = float('nan')
        # Velocity (NED)
        self.vel_x: float = float('nan')
        self.vel_y: float = float('nan')
        self.vel_z: float = float('nan')
        # Computed
        self.formation: str = 'N/A'
        self.rl_action: str = 'N/A'
        self.safety_ok: bool = True
        self.min_dist: float = float('nan')
        self.last_seen_s: float = 0.0


class SwarmMonitorNode(Node):
    """
    Monitors all drone telemetry and publishes a status summary.

    Prints a formatted table every 2 seconds and publishes JSON status.
    """

    def __init__(self) -> None:
        super().__init__('swarm_monitor')

        self.declare_parameter('num_drones', 4)
        self.declare_parameter('topic_namespace_prefix', '/px4_')
        self.declare_parameter('print_rate', 0.5)
        self.declare_parameter('use_sim_time', True)

        self._num_drones: int = self.get_parameter('num_drones').value
        self._ns_prefix: str = self.get_parameter(
            'topic_namespace_prefix').value
        print_rate: float = self.get_parameter('print_rate').value

        self._data: Dict[int, DroneMonitorData] = {
            i: DroneMonitorData(i) for i in range(self._num_drones)
        }

        self._pub_status = self.create_publisher(
            String, '/swarm/monitor_status', 10)

        if _PX4_MSGS_AVAILABLE:
            self._create_subscribers()
        else:
            self.get_logger().warn(
                'px4_msgs not available — monitor running in stub mode.')

        period = 1.0 / max(0.1, print_rate)
        self.create_timer(period, self._print_and_publish)

        self.get_logger().info(
            f'SwarmMonitorNode watching {self._num_drones} drones')

    def _create_subscribers(self) -> None:
        qos = QoSPresetProfiles.SENSOR_DATA.value
        for drone_id in range(self._num_drones):
            ns = f'{self._ns_prefix.rstrip("/")}{drone_id}'
            self.create_subscription(
                VehicleStatus,
                f'{ns}/fmu/out/vehicle_status',
                lambda msg, did=drone_id: self._on_status(msg, did), qos)
            self.create_subscription(
                VehicleLocalPosition,
                f'{ns}/fmu/out/vehicle_local_position',
                lambda msg, did=drone_id: self._on_local_pos(msg, did), qos)

    def _on_status(self, msg: 'VehicleStatus', drone_id: int) -> None:
        d = self._data[drone_id]
        d.nav_state = msg.nav_state
        d.arming_state = msg.arming_state
        d.is_armed = (msg.arming_state == 2)
        d.is_offboard = (msg.nav_state == 14)
        d.last_seen_s = self.get_clock().now().nanoseconds / 1e9

    def _on_local_pos(self, msg: 'VehicleLocalPosition', drone_id: int) -> None:
        d = self._data[drone_id]
        d.pos_x = msg.x   # NED North
        d.pos_y = msg.y   # NED East
        d.pos_z = msg.z   # NED Down (negative = up)
        d.vel_x = msg.vx
        d.vel_y = msg.vy
        d.vel_z = msg.vz

    def _compute_min_distances(self) -> None:
        """Compute pairwise minimum distances between all drones."""
        ids = list(self._data.keys())
        for i in ids:
            min_d = float('inf')
            di = self._data[i]
            for j in ids:
                if i == j:
                    continue
                dj = self._data[j]
                if math.isnan(di.pos_x) or math.isnan(dj.pos_x):
                    continue
                dist = math.sqrt(
                    (di.pos_x - dj.pos_x) ** 2 +
                    (di.pos_y - dj.pos_y) ** 2 +
                    (di.pos_z - dj.pos_z) ** 2)
                min_d = min(min_d, dist)
            self._data[i].min_dist = min_d if min_d != float('inf') else float('nan')

    def _print_and_publish(self) -> None:
        """Print status table and publish JSON summary."""
        self._compute_min_distances()

        header = (
            f'\n{"="*100}\n'
            f'  {"D":<3} {"NAV":<12} {"ARM":<10} '
            f'{"A":<3} {"O":<3} {"BATT":<6} '
            f'{"N(m)":<7} {"E(m)":<7} {"Alt":<7} '
            f'{"FORM":<10} {"RL_ACT":<12} {"SAFE":<5} {"DIST":<6}\n'
            f'{"="*100}'
        )

        rows = []
        status_list = []

        for drone_id, d in self._data.items():
            nav = NAV_STATE_NAMES.get(d.nav_state, str(d.nav_state)) \
                if d.nav_state is not None else 'N/A'
            arm = ARM_STATE_NAMES.get(d.arming_state, str(d.arming_state)) \
                if d.arming_state is not None else 'N/A'
            batt = f'{d.battery_remaining*100:.0f}%' \
                if not math.isnan(d.battery_remaining) else 'N/A'
            alt = f'{-d.pos_z:.1f}' if not math.isnan(d.pos_z) else 'N/A'

            def _f(v: float) -> str:
                return f'{v:6.1f}' if not math.isnan(v) else '   N/A'

            dist_str = f'{d.min_dist:.1f}m' \
                if not math.isnan(d.min_dist) else 'N/A'

            row = (
                f'  {drone_id:<3} {nav:<12} {arm:<10} '
                f'{"Y" if d.is_armed else "N":<3} '
                f'{"Y" if d.is_offboard else "N":<3} '
                f'{batt:<6} '
                f'{_f(d.pos_x)} {_f(d.pos_y)} {alt:<7} '
                f'{d.formation:<10} {d.rl_action:<12} '
                f'{"OK" if d.safety_ok else "WARN":<5} {dist_str:<6}'
            )
            rows.append(row)

            status_list.append({
                'drone_id': drone_id,
                'nav_state': d.nav_state,
                'arming_state': d.arming_state,
                'armed': d.is_armed,
                'offboard': d.is_offboard,
                'battery': round(d.battery_remaining, 3),
                'pos': {
                    'n': round(d.pos_x, 2) if not math.isnan(d.pos_x) else None,
                    'e': round(d.pos_y, 2) if not math.isnan(d.pos_y) else None,
                    'alt': round(-d.pos_z, 2) if not math.isnan(d.pos_z) else None,
                },
                'formation': d.formation,
                'safety_ok': d.safety_ok,
                'min_dist_m': round(d.min_dist, 2)
                    if not math.isnan(d.min_dist) else None,
            })

        print(header)
        for row in rows:
            print(row)
        print()

        # Publish JSON summary
        status_msg = String()
        status_msg.data = json.dumps({'drones': status_list})
        self._pub_status.publish(status_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SwarmMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('SwarmMonitorNode shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
