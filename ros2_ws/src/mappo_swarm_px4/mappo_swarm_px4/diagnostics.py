"""
diagnostics.py — Swarm diagnostics publisher for MAPPO PX4 nodes.

Publishes diagnostic_msgs/DiagnosticArray to /swarm/diagnostics,
providing per-drone health status, safety events, and RL inference data.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from typing import Any, List, Optional

try:
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    _DIAG_AVAILABLE = True
except ImportError:
    _DIAG_AVAILABLE = False


class SwarmDiagnostics:
    """
    Publishes diagnostic information for the MAPPO drone swarm.

    Attaches to an existing ROS 2 node and provides methods for
    publishing drone status, safety events, and RL inference data.
    """

    TOPIC = '/swarm/diagnostics'

    def __init__(self, node: Node) -> None:
        """
        Initialise diagnostics publisher.

        Args:
            node: Parent ROS 2 node to attach publisher to.
        """
        self._node = node

        if _DIAG_AVAILABLE:
            self._pub = node.create_publisher(
                DiagnosticArray, self.TOPIC, 10)
            self._node.get_logger().info(
                f'SwarmDiagnostics publishing to {self.TOPIC}')
        else:
            self._pub = None
            self._node.get_logger().warn(
                'diagnostic_msgs not available — diagnostics disabled.')

    def _make_status(
            self,
            name: str,
            level: int,
            message: str,
            values: Optional[List['KeyValue']] = None) -> 'DiagnosticStatus':
        """Build a DiagnosticStatus message."""
        status = DiagnosticStatus()
        status.name = name
        status.level = level
        status.message = message
        status.hardware_id = ''
        if values:
            status.values = values
        return status

    def _make_kv(self, key: str, value: Any) -> 'KeyValue':
        """Build a KeyValue pair for DiagnosticStatus."""
        kv = KeyValue()
        kv.key = str(key)
        kv.value = str(value)
        return kv

    def _publish_array(
            self, statuses: List['DiagnosticStatus']) -> None:
        """Publish a DiagnosticArray with the given statuses."""
        if self._pub is None or not _DIAG_AVAILABLE:
            return
        array = DiagnosticArray()
        array.header.stamp = self._node.get_clock().now().to_msg()
        array.status = statuses
        self._pub.publish(array)

    def publish_drone_status(
            self,
            drone_id: int,
            state_data: dict) -> None:
        """
        Publish overall drone health status.

        Args:
            drone_id: Integer drone identifier.
            state_data: Dict with keys: pos_x, pos_y, pos_z, battery,
                        nav_state, arming_state, is_armed, is_offboard.
        """
        if not _DIAG_AVAILABLE:
            return

        battery = state_data.get('battery', float('nan'))
        is_armed = state_data.get('is_armed', False)
        is_offboard = state_data.get('is_offboard', False)

        # Level: OK=0, WARN=1, ERROR=2, STALE=3
        level = 0  # OK
        summary = 'Nominal'
        if battery < 0.2:
            level = 1
            summary = f'Low battery: {battery*100:.0f}%'
        if not is_armed and not is_offboard:
            pass  # Not yet active — OK

        kv_list = [
            self._make_kv('drone_id', drone_id),
            self._make_kv('pos_x', f"{state_data.get('pos_x', 0):.2f}"),
            self._make_kv('pos_y', f"{state_data.get('pos_y', 0):.2f}"),
            self._make_kv('pos_z', f"{state_data.get('pos_z', 0):.2f}"),
            self._make_kv('battery', f'{battery:.3f}'),
            self._make_kv('nav_state', state_data.get('nav_state', -1)),
            self._make_kv('arming_state', state_data.get('arming_state', -1)),
            self._make_kv('armed', is_armed),
            self._make_kv('offboard', is_offboard),
        ]

        status = self._make_status(
            name=f'drone_{drone_id}/status',
            level=level,
            message=summary,
            values=kv_list)

        self._publish_array([status])

    def publish_safety_event(
            self,
            drone_id: int,
            reason: str) -> None:
        """
        Log and publish a safety filter intervention event.

        Args:
            drone_id: Integer drone identifier.
            reason: Human-readable reason for the intervention.
        """
        self._node.get_logger().warn(
            f'[Safety] Drone {drone_id}: {reason}')

        if not _DIAG_AVAILABLE:
            return

        kv_list = [
            self._make_kv('drone_id', drone_id),
            self._make_kv('reason', reason),
        ]
        status = self._make_status(
            name=f'drone_{drone_id}/safety',
            level=DiagnosticStatus.WARN,
            message=f'Safety intervention: {reason}',
            values=kv_list)
        self._publish_array([status])

    def publish_rl_inference(
            self,
            drone_id: int,
            obs_size: int,
            action: List[float],
            reward: float = 0.0) -> None:
        """
        Log RL inference data (debug level).

        Args:
            drone_id: Integer drone identifier.
            obs_size: Size of observation vector.
            action: Scaled action [vx, vy, vz] in m/s.
            reward: Current reward estimate (if available).
        """
        if len(action) >= 3:
            self._node.get_logger().debug(
                f'[RL] Drone {drone_id}: '
                f'obs={obs_size}, '
                f'action=[{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}], '
                f'reward={reward:.3f}')

        if not _DIAG_AVAILABLE:
            return

        kv_list = [
            self._make_kv('drone_id', drone_id),
            self._make_kv('obs_size', obs_size),
            self._make_kv('vx', f'{action[0]:.4f}' if len(action) > 0 else 'N/A'),
            self._make_kv('vy', f'{action[1]:.4f}' if len(action) > 1 else 'N/A'),
            self._make_kv('vz', f'{action[2]:.4f}' if len(action) > 2 else 'N/A'),
            self._make_kv('reward', f'{reward:.4f}'),
        ]
        status = self._make_status(
            name=f'drone_{drone_id}/rl_inference',
            level=DiagnosticStatus.OK,
            message='RL inference OK',
            values=kv_list)
        self._publish_array([status])


def main(args=None) -> None:
    """Standalone diagnostics aggregator node entry point."""
    rclpy.init(args=args)
    node = Node('swarm_diagnostics')
    node.get_logger().info(
        'Swarm diagnostics node running. '
        'Publishing aggregated status to /swarm/diagnostics')

    _ = SwarmDiagnostics(node)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Diagnostics node shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
