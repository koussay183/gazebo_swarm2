"""
mission_supervisor_node.py — CLI-driven mission supervisor for drone swarm.

Provides a simple stdin-based command interface for operators to issue
high-level mission commands to the MAPPO swarm. Commands are published
as ROS 2 topics consumed by the RL control node and emergency controller.

Published Topics:
  /swarm/mission_command   (std_msgs/String)  — takeoff, land, patrol, stop, rtb
  /swarm/formation_command (std_msgs/String)  — formation name
  /swarm/emergency_stop    (std_msgs/Bool)    — true = stop
  /swarm/rtb_command       (std_msgs/Bool)    — return to base

CLI Commands (enter in terminal):
  takeoff    — Begin autonomous flight
  land       — Land all drones at current position
  patrol     — Start patrol mode
  stop       — Stop all movement (hover)
  form <n>   — Change formation (v_shape, line, circle, spread)
  rtb        — Return to base / landing zone
  estop      — EMERGENCY STOP (land immediately)
  status     — Print current published topic states
  help       — Show command list
  quit/exit  — Shutdown supervisor
"""

from __future__ import annotations

import sys
import threading
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool


AVAILABLE_FORMATIONS = ['v_shape', 'line', 'circle', 'spread', 'diamond']

HELP_TEXT = """
=== MAPPO Swarm Mission Supervisor ===
Available commands:
  takeoff         Start autonomous flight
  land            Land all drones at current position
  patrol          Begin patrol pattern
  stop            Hover in place (stop all movement)
  form <name>     Change formation [{formations}]
  rtb             Return to base
  estop           EMERGENCY STOP — land all drones immediately
  status          Show last published command
  help            Show this help message
  quit / exit     Shutdown the supervisor node
""".format(formations=', '.join(AVAILABLE_FORMATIONS))


class MissionSupervisorNode(Node):
    """
    CLI-based mission supervisor node.

    Reads commands from stdin in a background thread and publishes
    them to swarm control topics.
    """

    def __init__(self) -> None:
        super().__init__('mission_supervisor')

        self.declare_parameter('use_sim_time', True)

        # Publishers
        self._pub_mission = self.create_publisher(
            String, '/swarm/mission_command', 10)
        self._pub_formation = self.create_publisher(
            String, '/swarm/formation_command', 10)
        self._pub_estop = self.create_publisher(
            Bool, '/swarm/emergency_stop', 10)
        self._pub_rtb = self.create_publisher(
            Bool, '/swarm/rtb_command', 10)

        self._last_command: Optional[str] = None

        # Start CLI input thread
        self._running = True
        self._input_thread = threading.Thread(
            target=self._cli_loop, daemon=True)
        self._input_thread.start()

        print(HELP_TEXT)
        self.get_logger().info(
            'MissionSupervisorNode ready. Type commands in terminal.')

    def _publish_mission(self, cmd: str) -> None:
        """Publish to /swarm/mission_command."""
        msg = String()
        msg.data = cmd
        self._pub_mission.publish(msg)
        self._last_command = cmd
        self.get_logger().info(f'Mission command: {cmd}')

    def _publish_formation(self, formation: str) -> None:
        """Publish to /swarm/formation_command."""
        msg = String()
        msg.data = formation
        self._pub_formation.publish(msg)
        self.get_logger().info(f'Formation command: {formation}')

    def _publish_estop(self, active: bool = True) -> None:
        """Publish to /swarm/emergency_stop."""
        msg = Bool()
        msg.data = active
        self._pub_estop.publish(msg)
        if active:
            self.get_logger().error('EMERGENCY STOP published!')

    def _publish_rtb(self) -> None:
        """Publish return-to-base command."""
        msg = Bool()
        msg.data = True
        self._pub_rtb.publish(msg)
        self._publish_mission('rtb')
        self.get_logger().info('RTB command published.')

    def _cli_loop(self) -> None:
        """Background thread reading stdin for operator commands."""
        try:
            while self._running:
                try:
                    line = input('swarm> ').strip().lower()
                except EOFError:
                    break

                if not line:
                    continue

                parts = line.split()
                cmd = parts[0]

                if cmd in ('quit', 'exit'):
                    self.get_logger().info('Supervisor shutting down.')
                    self._running = False
                    rclpy.shutdown()
                    break

                elif cmd == 'help':
                    print(HELP_TEXT)

                elif cmd == 'takeoff':
                    self._publish_mission('takeoff')

                elif cmd == 'land':
                    self._publish_mission('land')

                elif cmd == 'patrol':
                    self._publish_mission('patrol')

                elif cmd == 'stop':
                    self._publish_mission('stop')

                elif cmd == 'rtb':
                    self._publish_rtb()

                elif cmd == 'estop':
                    print('!! EMERGENCY STOP !!')
                    self._publish_estop(True)
                    self._publish_mission('estop')

                elif cmd == 'form':
                    if len(parts) < 2:
                        print(f'Usage: form <name>  [{", ".join(AVAILABLE_FORMATIONS)}]')
                    else:
                        formation = parts[1]
                        if formation not in AVAILABLE_FORMATIONS:
                            print(f'Unknown formation: {formation}. '
                                  f'Options: {AVAILABLE_FORMATIONS}')
                        else:
                            self._publish_formation(formation)

                elif cmd == 'status':
                    print(f'Last mission command: {self._last_command}')

                else:
                    print(f'Unknown command: {cmd}. Type "help" for options.')

        except Exception as exc:
            self.get_logger().error(f'CLI loop error: {exc}')


def main(args=None) -> None:
    """Entry point for mission_supervisor node."""
    rclpy.init(args=args)
    node = MissionSupervisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('MissionSupervisorNode shutting down.')
    finally:
        node._running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
