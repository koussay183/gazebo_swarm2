"""
logging_node.py — Flight data logger for MAPPO drone swarm.

Subscribes to drone states and diagnostics and writes timestamped
CSV log files. Rotates log files every 10 minutes.

CSV Columns:
  time, drone_id, pos_x, pos_y, pos_z, battery, nav_state,
  rl_action_x, rl_action_y, rl_action_z, safety_intervened

ALSO RECOMMENDED: Use rosbag2 for full topic recording:
  ros2 bag record -a -o swarm_bag

Subscribed Topics:
  /swarm/monitor_status    (std_msgs/String) — JSON drone states
  /swarm/diagnostics       (diagnostic_msgs/DiagnosticArray)
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
import time
from typing import Dict, List, Optional, TextIO

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    from diagnostic_msgs.msg import DiagnosticArray
    _DIAG_AVAILABLE = True
except ImportError:
    _DIAG_AVAILABLE = False

# Log rotation interval in seconds
LOG_ROTATION_INTERVAL_S: float = 600.0  # 10 minutes

CSV_COLUMNS: List[str] = [
    'time_s', 'drone_id',
    'pos_x', 'pos_y', 'pos_z',
    'battery', 'nav_state',
    'rl_action_x', 'rl_action_y', 'rl_action_z',
    'safety_intervened',
]


class LoggingNode(Node):
    """
    Logs swarm flight data to rotating CSV files.

    Files are written to the current directory or a configured log_dir.
    Filename format: swarm_log_YYYYMMDD_HHMMSS.csv
    """

    def __init__(self) -> None:
        super().__init__('swarm_logger')

        self.declare_parameter('log_dir', '.')
        self.declare_parameter('use_sim_time', True)

        self._log_dir: str = self.get_parameter('log_dir').value

        self._csv_file: Optional[TextIO] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._log_start_time: float = time.time()
        self._row_count: int = 0

        # Ensure log directory exists
        os.makedirs(self._log_dir, exist_ok=True)

        self._open_new_log_file()

        # Subscriptions
        self.create_subscription(
            String, '/swarm/monitor_status',
            self._on_monitor_status, 10)

        if _DIAG_AVAILABLE:
            self.create_subscription(
                DiagnosticArray, '/swarm/diagnostics',
                self._on_diagnostics, 10)

        # Log rotation timer (check every 30s)
        self.create_timer(30.0, self._check_rotation)

        self.get_logger().info(
            f'SwarmLoggingNode writing to: {self._log_dir}')
        self.get_logger().info(
            'TIP: For full topic recording, also run:\n'
            '  ros2 bag record -a -o swarm_bag')

    def _open_new_log_file(self) -> None:
        """Open a new timestamped CSV log file."""
        if self._csv_file is not None:
            self._csv_file.close()
            self.get_logger().info(
                f'Closed log file after {self._row_count} rows.')

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self._log_dir, f'swarm_log_{timestamp}.csv')

        self._csv_file = open(filename, 'w', newline='')
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=CSV_COLUMNS)
        self._csv_writer.writeheader()
        self._csv_file.flush()
        self._log_start_time = time.time()
        self._row_count = 0

        self.get_logger().info(f'Opened log file: {filename}')

    def _check_rotation(self) -> None:
        """Rotate log file if older than LOG_ROTATION_INTERVAL_S."""
        elapsed = time.time() - self._log_start_time
        if elapsed >= LOG_ROTATION_INTERVAL_S:
            self.get_logger().info(
                f'Rotating log file after {elapsed:.0f}s / '
                f'{self._row_count} rows.')
            self._open_new_log_file()

    def _on_monitor_status(self, msg: String) -> None:
        """Parse monitor JSON and write one CSV row per drone."""
        if self._csv_writer is None:
            return

        try:
            data = json.loads(msg.data)
            drones = data.get('drones', [])
            now_s = self.get_clock().now().nanoseconds / 1e9

            for d in drones:
                pos = d.get('pos', {})
                row = {
                    'time_s': f'{now_s:.3f}',
                    'drone_id': d.get('drone_id', -1),
                    'pos_x': pos.get('n', ''),
                    'pos_y': pos.get('e', ''),
                    'pos_z': pos.get('alt', ''),
                    'battery': d.get('battery', ''),
                    'nav_state': d.get('nav_state', ''),
                    'rl_action_x': '',
                    'rl_action_y': '',
                    'rl_action_z': '',
                    'safety_intervened': not d.get('safety_ok', True),
                }
                self._csv_writer.writerow(row)
                self._row_count += 1

            self._csv_file.flush()

        except (json.JSONDecodeError, KeyError) as exc:
            self.get_logger().warn(f'Failed to parse monitor status: {exc}')

    def _on_diagnostics(self, msg: 'DiagnosticArray') -> None:
        """Extract RL inference data from DiagnosticArray (debug only)."""
        for status in msg.status:
            if '/rl_inference' not in status.name:
                continue
            # Extract action values for annotation
            kv = {kv.key: kv.value for kv in status.values}
            self.get_logger().debug(
                f'RL [{kv.get("drone_id")}]: '
                f'vx={kv.get("vx")} vy={kv.get("vy")} vz={kv.get("vz")}')

    def __del__(self) -> None:
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LoggingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('LoggingNode shutting down.')
    finally:
        if node._csv_file is not None:
            node._csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
