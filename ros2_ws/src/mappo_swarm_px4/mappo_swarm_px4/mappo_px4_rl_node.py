"""
mappo_px4_rl_node.py — Main MAPPO RL control node for PX4 drone swarm.

Loads a trained MAPPO actor, builds observations from PX4 telemetry,
runs inference, applies safety filtering, and sends velocity setpoints
via PX4 offboard control mode.
"""

from __future__ import annotations

import traceback
from typing import List, Optional, Dict, Any

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String, Bool

from mappo_swarm_core import (
    OBS_SIZE_PER_DRONE,
    SwarmState,
    ObservationBuilder,
    SafetyFilter,
    load_actor,
)
from .px4_offboard_manager import PX4OffboardManager
from .px4_topic_tools import get_px4_topic, get_all_drone_topics
from .diagnostics import SwarmDiagnostics

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from px4_msgs.msg import VehicleStatus, VehicleLocalPosition, VehicleOdometry
    _PX4_MSGS_AVAILABLE = True
except ImportError:
    _PX4_MSGS_AVAILABLE = False


class MAPPOPX4RLNode(Node):
    """
    Main MAPPO RL control node for PX4 drone swarm.

    Loads trained actor checkpoint, subscribes to PX4 telemetry, runs
    per-drone RL inference, applies safety filtering, and publishes
    offboard control setpoints.
    """

    def __init__(self) -> None:
        super().__init__('mappo_px4_rl_node')

        # ------------------------------------------------------------------ #
        # Declare parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter('num_drones', 1)
        self.declare_parameter('drone_ids', [0])
        self.declare_parameter('checkpoint_path', 'checkpoints/best.pt')
        self.declare_parameter('control_rate', 30.0)
        self.declare_parameter('control_mode', 'velocity')
        self.declare_parameter('max_xy_speed', 2.5)
        self.declare_parameter('max_z_speed', 1.0)
        self.declare_parameter('min_altitude', 2.0)
        self.declare_parameter('max_altitude', 20.0)
        self.declare_parameter('pos_scale', 25.0)
        self.declare_parameter('topic_namespace_prefix', '/px4_')
        self.declare_parameter('px4_topic_style', 'auto')
        self.declare_parameter('default_battery', 1.0)
        self.declare_parameter('use_sim_time', True)
        self.declare_parameter('leader_id', 0)
        self.declare_parameter('auto_arm', False)
        self.declare_parameter('auto_offboard', False)
        self.declare_parameter('geofence_x_min', -25.0)
        self.declare_parameter('geofence_x_max', 25.0)
        self.declare_parameter('geofence_y_min', -25.0)
        self.declare_parameter('geofence_y_max', 25.0)

        # ------------------------------------------------------------------ #
        # Read parameters
        # ------------------------------------------------------------------ #
        self._num_drones: int = self.get_parameter('num_drones').value
        self._drone_ids: List[int] = list(
            self.get_parameter('drone_ids').value)
        self._checkpoint_path: str = self.get_parameter(
            'checkpoint_path').value
        self._control_rate: float = self.get_parameter('control_rate').value
        self._control_mode: str = self.get_parameter('control_mode').value
        self._max_xy_speed: float = self.get_parameter('max_xy_speed').value
        self._max_z_speed: float = self.get_parameter('max_z_speed').value
        self._min_altitude: float = self.get_parameter('min_altitude').value
        self._max_altitude: float = self.get_parameter('max_altitude').value
        self._pos_scale: float = self.get_parameter('pos_scale').value
        self._ns_prefix: str = self.get_parameter(
            'topic_namespace_prefix').value
        self._topic_style: str = self.get_parameter('px4_topic_style').value
        self._default_battery: float = self.get_parameter(
            'default_battery').value
        self._leader_id: int = self.get_parameter('leader_id').value
        self._auto_arm: bool = self.get_parameter('auto_arm').value
        self._auto_offboard: bool = self.get_parameter(
            'auto_offboard').value

        self.get_logger().info(
            f'MAPPOPX4RLNode starting: {self._num_drones} drones, '
            f'ids={self._drone_ids}, mode={self._control_mode}, '
            f'rate={self._control_rate}Hz, '
            f'auto_arm={self._auto_arm}, auto_offboard={self._auto_offboard}')

        # ------------------------------------------------------------------ #
        # 1. Load actor model (non-fatal on failure)
        # ------------------------------------------------------------------ #
        self._actor: Optional[Any] = None
        if self._checkpoint_path:
            try:
                self._actor = load_actor(
                    self._checkpoint_path, OBS_SIZE_PER_DRONE)
                self.get_logger().info(
                    f'Actor loaded from: {self._checkpoint_path}')
            except Exception as exc:
                self.get_logger().error(
                    f'Failed to load actor from {self._checkpoint_path}: '
                    f'{exc}\n{traceback.format_exc()}'
                    '\n=> Running in demo/fallback mode (zero velocity).')
        else:
            self.get_logger().warn(
                'checkpoint_path is empty — running in demo fallback mode.')

        # ------------------------------------------------------------------ #
        # 2. Core state / logic objects
        # ------------------------------------------------------------------ #
        self._swarm_state = SwarmState(self._num_drones)
        self._obs_builder = ObservationBuilder(
            self._num_drones, OBS_SIZE_PER_DRONE, self._pos_scale)
        self._safety_filter = SafetyFilter(
            max_xy_speed=self._max_xy_speed,
            max_z_speed=self._max_z_speed,
            min_altitude=self._min_altitude,
            max_altitude=self._max_altitude,
        )
        self._diagnostics = SwarmDiagnostics(self)

        # ------------------------------------------------------------------ #
        # 3. Per-drone offboard managers + telemetry subscribers
        # ------------------------------------------------------------------ #
        self._managers: Dict[int, PX4OffboardManager] = {}
        self._drone_topics: Dict[int, Dict[str, str]] = get_all_drone_topics(
            self._num_drones, style=self._topic_style,
            prefix=self._ns_prefix, drone_ids=self._drone_ids)

        for drone_id in self._drone_ids:
            ns = f'{self._ns_prefix}{drone_id}'
            mgr = PX4OffboardManager(self, drone_id, ns)
            self._managers[drone_id] = mgr

        if _PX4_MSGS_AVAILABLE:
            self._create_telemetry_subscribers()
        else:
            self.get_logger().warn(
                'px4_msgs not available — telemetry subscribers not created.')

        # ------------------------------------------------------------------ #
        # 4. Control timer
        # ------------------------------------------------------------------ #
        period_s = 1.0 / self._control_rate
        self._control_timer = self.create_timer(period_s, self._control_loop)

        # ------------------------------------------------------------------ #
        # 5. Mission / command subscribers
        # ------------------------------------------------------------------ #
        self.create_subscription(
            String, '/swarm/mission_command',
            self._on_mission_command, 10)
        self.create_subscription(
            String, '/swarm/formation_command',
            self._on_formation_command, 10)
        self.create_subscription(
            String, '/swarm/emergency_stop',
            self._on_emergency_stop, 10)

        self.get_logger().info('MAPPOPX4RLNode initialised successfully.')

    # ---------------------------------------------------------------------- #
    # Telemetry subscriber creation
    # ---------------------------------------------------------------------- #

    def _create_telemetry_subscribers(self) -> None:
        """Create per-drone PX4 telemetry subscriptions."""
        for drone_id in self._drone_ids:
            topics = self._drone_topics.get(drone_id, {})

            vs_topic = topics.get(
                'vehicle_status',
                get_px4_topic(drone_id, 'out/vehicle_status',
                              self._topic_style, self._ns_prefix))
            vlp_topic = topics.get(
                'vehicle_local_position',
                get_px4_topic(drone_id, 'out/vehicle_local_position',
                              self._topic_style, self._ns_prefix))
            vod_topic = topics.get(
                'vehicle_odometry',
                get_px4_topic(drone_id, 'out/vehicle_odometry',
                              self._topic_style, self._ns_prefix))

            self.create_subscription(
                VehicleStatus, vs_topic,
                lambda msg, did=drone_id: self._on_vehicle_status(msg, did),
                rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
            self.create_subscription(
                VehicleLocalPosition, vlp_topic,
                lambda msg, did=drone_id: self._on_vehicle_local_position(
                    msg, did),
                rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)
            self.create_subscription(
                VehicleOdometry, vod_topic,
                lambda msg, did=drone_id: self._on_vehicle_odometry(
                    msg, did),
                rclpy.qos.QoSPresetProfiles.SENSOR_DATA.value)

            self.get_logger().info(
                f'  Drone {drone_id}: status={vs_topic}, '
                f'pos={vlp_topic}, odom={vod_topic}')

    # ---------------------------------------------------------------------- #
    # Telemetry callbacks
    # ---------------------------------------------------------------------- #

    def _on_vehicle_status(self, msg: 'VehicleStatus', drone_id: int) -> None:
        """Update swarm state from VehicleStatus."""
        try:
            state = self._swarm_state.get_drone(drone_id)
            state.nav_state = msg.nav_state
            state.arming_state = msg.arming_state
            state.is_armed = (msg.arming_state == 2)  # ARMING_STATE_ARMED
            state.is_offboard = (msg.nav_state == 14)  # NAV_STATE_OFFBOARD
            self._swarm_state.update_timestamp(drone_id)
            # Forward to offboard manager
            mgr = self._managers.get(drone_id)
            if mgr:
                mgr._on_vehicle_status(msg)
        except Exception as exc:
            self.get_logger().warn(
                f'vehicle_status callback error (drone {drone_id}): {exc}')

    def _on_vehicle_local_position(
            self, msg: 'VehicleLocalPosition', drone_id: int) -> None:
        """Update swarm state from VehicleLocalPosition (NED frame)."""
        try:
            state = self._swarm_state.get_drone(drone_id)
            # PX4 VehicleLocalPosition is NED; convert to ENU for swarm state
            state.pos_x = msg.y   # North -> ENU X
            state.pos_y = msg.x   # East  -> ENU Y
            state.pos_z = -msg.z  # Down  -> ENU Z (altitude)
            state.vel_x = msg.vy
            state.vel_y = msg.vx
            state.vel_z = -msg.vz
            self._swarm_state.update_timestamp(drone_id)
            mgr = self._managers.get(drone_id)
            if mgr:
                mgr._on_vehicle_local_position(msg)
        except Exception as exc:
            self.get_logger().warn(
                f'local_position callback error (drone {drone_id}): {exc}')

    def _on_vehicle_odometry(
            self, msg: 'VehicleOdometry', drone_id: int) -> None:
        """Update swarm state from VehicleOdometry."""
        try:
            mgr = self._managers.get(drone_id)
            if mgr:
                mgr._on_vehicle_odometry(msg)
        except Exception as exc:
            self.get_logger().warn(
                f'odometry callback error (drone {drone_id}): {exc}')

    # ---------------------------------------------------------------------- #
    # Control loop
    # ---------------------------------------------------------------------- #

    def _control_loop(self) -> None:
        """
        Main control loop — called at control_rate Hz.

        For each active drone:
          1. Build observation
          2. Run actor inference (or use fallback)
          3. Apply safety filter
          4. Publish offboard setpoint (if auto_offboard)
          5. Arm if needed (if auto_arm)
        """
        try:
            # Mark stale drones (no telemetry for >1s)
            self._swarm_state.mark_stale_drones(stale_threshold_s=1.0)

            for drone_id in self._drone_ids:
                drone_state = self._swarm_state.get_drone(drone_id)

                # Build observation
                obs = self._obs_builder.build_observation(
                    drone_id, self._swarm_state)

                # Inference
                action = self._run_inference(drone_id, obs)

                # Scale to physical velocities (ENU)
                vx = float(action[0]) * self._max_xy_speed
                vy = float(action[1]) * self._max_xy_speed
                vz = float(action[2]) * self._max_z_speed

                # Safety filter
                vx, vy, vz, intervened = self._safety_filter.filter(
                    drone_id=drone_id,
                    vx=vx, vy=vy, vz=vz,
                    swarm_state=self._swarm_state,
                )

                if intervened:
                    self._diagnostics.publish_safety_event(
                        drone_id, f'Safety filter applied: '
                        f'vx={vx:.2f} vy={vy:.2f} vz={vz:.2f}')

                mgr = self._managers.get(drone_id)
                if mgr is None:
                    continue

                # Publish offboard mode heartbeat + setpoint
                if self._auto_offboard:
                    mgr.publish_offboard_mode(velocity=True)
                    if self._control_mode == 'velocity':
                        mgr.publish_velocity_setpoint(vx, vy, vz)
                    else:
                        # Placeholder: use current pos + delta for position mode
                        state = self._swarm_state.get_drone(drone_id)
                        mgr.publish_position_setpoint(
                            north=state.pos_y,
                            east=state.pos_x,
                            down=-state.pos_z)

                # Auto-arm: request arm only once when vehicle is ready
                if (self._auto_arm
                        and hasattr(drone_state, 'arming_state')
                        and drone_state.arming_state == 1  # STANDBY
                        and not drone_state.is_armed):
                    mgr.arm()

                # Publish diagnostics
                self._diagnostics.publish_rl_inference(
                    drone_id=drone_id,
                    obs_size=len(obs),
                    action=[vx, vy, vz],
                    reward=0.0)

        except Exception as exc:
            self.get_logger().error(
                f'Control loop error: {exc}\n{traceback.format_exc()}')

    def _run_inference(
            self, drone_id: int,
            obs: 'Any') -> List[float]:
        """
        Run actor inference for a single drone.

        Returns action [vx_norm, vy_norm, vz_norm] in [-1, 1].
        Falls back to zero vector on failure or if no actor loaded.
        """
        if self._actor is None or not _TORCH_AVAILABLE:
            return [0.0, 0.0, 0.0]

        try:
            import torch
            import numpy as np
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = self._actor(obs_tensor, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy()
            # Clamp to [-1, 1]
            action_np = np.clip(action_np, -1.0, 1.0)
            return action_np[:3].tolist()
        except Exception as exc:
            self.get_logger().warn(
                f'Inference error (drone {drone_id}): {exc} — using zero.')
            return [0.0, 0.0, 0.0]

    # ---------------------------------------------------------------------- #
    # Mission command callbacks
    # ---------------------------------------------------------------------- #

    def _on_mission_command(self, msg: String) -> None:
        """Handle mission commands from GCS."""
        cmd = msg.data.strip().lower()
        self.get_logger().info(f'Mission command received: {cmd}')
        if cmd == 'estop':
            self._emergency_stop_all()
        elif cmd == 'land':
            for mgr in self._managers.values():
                mgr.land()
        elif cmd == 'arm':
            if self._auto_arm:
                for mgr in self._managers.values():
                    mgr.arm()
        elif cmd == 'disarm':
            for mgr in self._managers.values():
                mgr.disarm()

    def _on_formation_command(self, msg: String) -> None:
        """Handle formation commands from GCS."""
        formation = msg.data.strip()
        self.get_logger().info(f'Formation command: {formation}')
        # Propagate to swarm state for observation builder context
        if hasattr(self._swarm_state, 'set_formation'):
            self._swarm_state.set_formation(formation)

    def _on_emergency_stop(self, msg: String) -> None:
        """Handle emergency stop command."""
        self.get_logger().error(
            f'EMERGENCY STOP received: {msg.data}')
        self._emergency_stop_all()

    def _emergency_stop_all(self) -> None:
        """Land all drones immediately."""
        self.get_logger().error('Executing emergency stop — landing all drones.')
        for drone_id, mgr in self._managers.items():
            try:
                mgr.land()
            except Exception as exc:
                self.get_logger().error(
                    f'Failed to land drone {drone_id}: {exc}')


def main(args=None) -> None:
    """Entry point for mappo_px4_rl_node."""
    rclpy.init(args=args)
    node = MAPPOPX4RLNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down MAPPOPX4RLNode (KeyboardInterrupt).')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
