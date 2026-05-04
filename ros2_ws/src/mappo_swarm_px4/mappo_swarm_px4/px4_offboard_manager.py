"""
px4_offboard_manager.py — PX4 offboard control mode manager.

Manages publishers and subscribers for a single PX4 drone instance:
OffboardControlMode, TrajectorySetpoint, VehicleCommand, and telemetry
feedback via VehicleStatus / VehicleLocalPosition / VehicleOdometry.

ENU/NED convention:
  PX4 uses NED (North/East/Down) internally.
  This manager converts ENU inputs to NED before publishing.
  ENU X -> NED North (Y in PX4 VLP)
  ENU Y -> NED East  (X in PX4 VLP)
  ENU Z -> NED Down (negated)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles

try:
    from px4_msgs.msg import (
        OffboardControlMode,
        TrajectorySetpoint,
        VehicleCommand,
        VehicleStatus,
        VehicleLocalPosition,
        VehicleOdometry,
        VehicleCommandAck,
    )
    _PX4_MSGS_AVAILABLE = True
except ImportError:
    _PX4_MSGS_AVAILABLE = False

# MAVLink command IDs used by PX4
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_NAV_LAND = 21
MAV_CMD_DO_SET_MODE = 176

# Vehicle modes
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6


class PX4OffboardManager:
    """
    Manages offboard control for a single PX4 drone.

    Provides publisher methods for OffboardControlMode, TrajectorySetpoint,
    and VehicleCommand, plus subscriber callbacks for state telemetry.
    """

    def __init__(
            self,
            node: Node,
            drone_id: int,
            namespace: str) -> None:
        """
        Initialise the offboard manager.

        Args:
            node: Parent ROS 2 node.
            drone_id: Integer drone identifier.
            namespace: ROS 2 namespace prefix, e.g. '/px4_0'.
        """
        self._node = node
        self._drone_id = drone_id
        self._namespace = namespace.rstrip('/')

        # Internal state (updated by telemetry callbacks)
        self._vehicle_status: Optional['VehicleStatus'] = None
        self._local_position: Optional['VehicleLocalPosition'] = None
        self._odometry: Optional['VehicleOdometry'] = None

        if not _PX4_MSGS_AVAILABLE:
            self._node.get_logger().warn(
                f'px4_msgs not available — drone {drone_id} manager is a stub.')
            return

        qos = QoSPresetProfiles.SENSOR_DATA.value

        # ------------------------------------------------------------------ #
        # Publishers
        # ------------------------------------------------------------------ #
        self._pub_offboard_mode = node.create_publisher(
            OffboardControlMode,
            f'{self._namespace}/fmu/in/offboard_control_mode',
            qos)
        self._pub_trajectory = node.create_publisher(
            TrajectorySetpoint,
            f'{self._namespace}/fmu/in/trajectory_setpoint',
            qos)
        self._pub_vehicle_cmd = node.create_publisher(
            VehicleCommand,
            f'{self._namespace}/fmu/in/vehicle_command',
            qos)

        # ------------------------------------------------------------------ #
        # Subscribers
        # ------------------------------------------------------------------ #
        node.create_subscription(
            VehicleStatus,
            f'{self._namespace}/fmu/out/vehicle_status',
            self._on_vehicle_status, qos)
        node.create_subscription(
            VehicleLocalPosition,
            f'{self._namespace}/fmu/out/vehicle_local_position',
            self._on_vehicle_local_position, qos)
        node.create_subscription(
            VehicleOdometry,
            f'{self._namespace}/fmu/out/vehicle_odometry',
            self._on_vehicle_odometry, qos)
        node.create_subscription(
            VehicleCommandAck,
            f'{self._namespace}/fmu/out/vehicle_command_ack',
            self._on_vehicle_command_ack, qos)

        self._node.get_logger().info(
            f'PX4OffboardManager initialised for drone {drone_id} '
            f'at namespace {self._namespace}')

    # ---------------------------------------------------------------------- #
    # Properties
    # ---------------------------------------------------------------------- #

    @property
    def is_armed(self) -> bool:
        """True if the drone is currently armed."""
        if self._vehicle_status is None:
            return False
        return self._vehicle_status.arming_state == 2  # ARMING_STATE_ARMED

    @property
    def is_offboard(self) -> bool:
        """True if the drone is in offboard navigation mode."""
        if self._vehicle_status is None:
            return False
        return self._vehicle_status.nav_state == 14  # NAV_STATE_OFFBOARD

    @property
    def nav_state(self) -> Optional[int]:
        """Current navigation state from VehicleStatus."""
        if self._vehicle_status is None:
            return None
        return self._vehicle_status.nav_state

    @property
    def arming_state(self) -> Optional[int]:
        """Current arming state from VehicleStatus."""
        if self._vehicle_status is None:
            return None
        return self._vehicle_status.arming_state

    @property
    def local_position(self) -> Optional['VehicleLocalPosition']:
        """Latest VehicleLocalPosition message (NED frame)."""
        return self._local_position

    @property
    def odometry(self) -> Optional['VehicleOdometry']:
        """Latest VehicleOdometry message."""
        return self._odometry

    # ---------------------------------------------------------------------- #
    # Telemetry callbacks (also called by parent node to avoid double-sub)
    # ---------------------------------------------------------------------- #

    def _on_vehicle_status(self, msg: 'VehicleStatus') -> None:
        self._vehicle_status = msg

    def _on_vehicle_local_position(self, msg: 'VehicleLocalPosition') -> None:
        self._local_position = msg

    def _on_vehicle_odometry(self, msg: 'VehicleOdometry') -> None:
        self._odometry = msg

    def _on_vehicle_command_ack(self, msg: 'VehicleCommandAck') -> None:
        self._node.get_logger().debug(
            f'VehicleCommandAck drone {self._drone_id}: '
            f'cmd={msg.command}, result={msg.result}')

    # ---------------------------------------------------------------------- #
    # Publish helpers
    # ---------------------------------------------------------------------- #

    def get_timestamp(self) -> int:
        """Return current time as microseconds (PX4 timestamp field)."""
        return int(self._node.get_clock().now().nanoseconds / 1000)

    def publish_offboard_mode(
            self,
            position: bool = False,
            velocity: bool = True,
            acceleration: bool = False) -> None:
        """
        Publish OffboardControlMode heartbeat.

        Must be published at >= 2 Hz before PX4 will accept setpoints.
        """
        if not _PX4_MSGS_AVAILABLE:
            return
        msg = OffboardControlMode()
        msg.timestamp = self.get_timestamp()
        msg.position = position
        msg.velocity = velocity
        msg.acceleration = acceleration
        msg.attitude = False
        msg.body_rate = False
        self._pub_offboard_mode.publish(msg)

    def publish_velocity_setpoint(
            self,
            vx: float,
            vy: float,
            vz: float,
            yaw_rate: float = 0.0) -> None:
        """
        Publish velocity setpoint.

        Args:
            vx: ENU X velocity (East direction) in m/s.
            vy: ENU Y velocity (North direction) in m/s.
            vz: ENU Z velocity (up) in m/s.
            yaw_rate: Yaw rate in rad/s.

        Note: PX4 TrajectorySetpoint uses NED frame:
            vn (North) <- vy (ENU Y)
            ve (East)  <- vx (ENU X)
            vd (Down)  <- -vz (negated ENU Z)
        """
        if not _PX4_MSGS_AVAILABLE:
            return
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_timestamp()
        # NED conversion: [vn, ve, vd] = [vy, vx, -vz]
        msg.velocity = [float(vy), float(vx), float(-vz)]
        msg.position = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float('nan')
        msg.yawspeed = float(yaw_rate)
        self._pub_trajectory.publish(msg)

    def publish_position_setpoint(
            self,
            north: float,
            east: float,
            down: float,
            yaw: float = 0.0) -> None:
        """
        Publish position setpoint in NED frame.

        Args:
            north: North position in metres.
            east: East position in metres.
            down: Down position in metres (negative = altitude).
            yaw: Yaw angle in radians (NED convention).
        """
        if not _PX4_MSGS_AVAILABLE:
            return
        msg = TrajectorySetpoint()
        msg.timestamp = self.get_timestamp()
        msg.position = [float(north), float(east), float(down)]
        msg.velocity = [float('nan'), float('nan'), float('nan')]
        msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float(yaw)
        msg.yawspeed = float('nan')
        self._pub_trajectory.publish(msg)

    def _publish_vehicle_command(
            self,
            command: int,
            param1: float = 0.0,
            param2: float = 0.0,
            param3: float = 0.0,
            param4: float = 0.0,
            param5: float = 0.0,
            param6: float = 0.0,
            param7: float = 0.0) -> None:
        """Publish a MAVLink VehicleCommand message."""
        if not _PX4_MSGS_AVAILABLE:
            return
        msg = VehicleCommand()
        msg.timestamp = self.get_timestamp()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.param4 = float(param4)
        msg.param5 = float(param5)
        msg.param6 = float(param6)
        msg.param7 = float(param7)
        msg.target_system = self._drone_id + 1  # 1-indexed MAVLink system ID
        msg.target_component = 1
        msg.source_system = 255
        msg.source_component = 0
        msg.from_external = True
        self._pub_vehicle_cmd.publish(msg)

    def arm(self) -> None:
        """Send arm command (MAV_CMD_COMPONENT_ARM_DISARM param1=1)."""
        self._node.get_logger().info(
            f'Arming drone {self._drone_id}')
        self._publish_vehicle_command(
            MAV_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def disarm(self) -> None:
        """Send disarm command (MAV_CMD_COMPONENT_ARM_DISARM param1=0)."""
        self._node.get_logger().info(
            f'Disarming drone {self._drone_id}')
        self._publish_vehicle_command(
            MAV_CMD_COMPONENT_ARM_DISARM, param1=0.0)

    def land(self) -> None:
        """Send land command (MAV_CMD_NAV_LAND)."""
        self._node.get_logger().info(
            f'Landing drone {self._drone_id}')
        self._publish_vehicle_command(MAV_CMD_NAV_LAND)

    def set_offboard_mode(self) -> None:
        """
        Send mode change to offboard (MAV_CMD_DO_SET_MODE).

        PX4 custom mode: param1=1 (MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
        param2=6 (CUSTOM_MAIN_MODE_OFFBOARD).
        """
        self._node.get_logger().info(
            f'Setting offboard mode for drone {self._drone_id}')
        self._publish_vehicle_command(
            MAV_CMD_DO_SET_MODE,
            param1=1.0,
            param2=float(PX4_CUSTOM_MAIN_MODE_OFFBOARD))
