"""
mappo_swarm_core.swarm_state
Centralised swarm state container for ROS 2 deployment.

All positions and velocities stored in ENU frame (x=East, y=North, z=Up).
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .frame_conversions import ned_position_to_enu, ned_velocity_to_enu, ned_yaw_to_enu


@dataclass
class DroneStateData:
    """State container for a single drone."""

    drone_id: int = 0
    position_enu: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_enu: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_enu: float = 0.0
    battery: float = 100.0
    target_enu: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = False
    is_leader: bool = False
    px4_vehicle_status: dict = field(default_factory=dict)
    nav_state: int = 0
    arming_state: int = 0
    last_update_time: float = 0.0
    health_flags: dict = field(default_factory=dict)
    comm_quality: float = 1.0


class SwarmState:
    """
    Centralised state store for all drones in the swarm.

    Parameters
    ----------
    num_drones : int
        Total number of drones.
    """

    def __init__(self, num_drones: int):
        self.num_drones = num_drones
        self.drones: Dict[int, DroneStateData] = {
            i: DroneStateData(drone_id=i)
            for i in range(num_drones)
        }

    # ------------------------------------------------------------------
    # Update methods
    # ------------------------------------------------------------------

    def update_from_px4_local_position(self, drone_id: int, msg) -> None:
        """
        Update drone state from a PX4 VehicleLocalPosition message (NED).

        Parameters
        ----------
        drone_id : int
            Drone index.
        msg : px4_msgs.msg.VehicleLocalPosition
            PX4 local position message (NED frame).
        """
        if drone_id not in self.drones:
            return
        d = self.drones[drone_id]
        # PX4 VehicleLocalPosition uses NED: x=North, y=East, z=Down
        x_enu, y_enu, z_enu = ned_position_to_enu(msg.x, msg.y, msg.z)
        d.position_enu = np.array([x_enu, y_enu, z_enu], dtype=np.float32)
        vx_enu, vy_enu, vz_enu = ned_velocity_to_enu(msg.vx, msg.vy, msg.vz)
        d.velocity_enu = np.array([vx_enu, vy_enu, vz_enu], dtype=np.float32)
        d.yaw_enu = ned_yaw_to_enu(msg.heading) if hasattr(msg, 'heading') else 0.0
        d.last_update_time = time.time()
        d.active = True

    def update_from_px4_odometry(self, drone_id: int, msg) -> None:
        """
        Update drone state from a PX4 VehicleOdometry message (NED).

        Parameters
        ----------
        drone_id : int
            Drone index.
        msg : px4_msgs.msg.VehicleOdometry
            PX4 odometry message (NED frame).
        """
        if drone_id not in self.drones:
            return
        d = self.drones[drone_id]
        # VehicleOdometry position: NED
        x_enu, y_enu, z_enu = ned_position_to_enu(
            msg.position[0], msg.position[1], msg.position[2])
        d.position_enu = np.array([x_enu, y_enu, z_enu], dtype=np.float32)
        vx_enu, vy_enu, vz_enu = ned_velocity_to_enu(
            msg.velocity[0], msg.velocity[1], msg.velocity[2])
        d.velocity_enu = np.array([vx_enu, vy_enu, vz_enu], dtype=np.float32)
        d.last_update_time = time.time()
        d.active = True

    def update_from_gazebo_pose(
        self, drone_id: int, x: float, y: float, z: float, yaw: float = 0.0
    ) -> None:
        """
        Update drone state directly from a Gazebo pose (ENU assumed).

        Parameters
        ----------
        drone_id : int
            Drone index.
        x, y, z : float
            ENU position.
        yaw : float
            ENU yaw angle (radians).
        """
        if drone_id not in self.drones:
            return
        d = self.drones[drone_id]
        d.position_enu = np.array([x, y, z], dtype=np.float32)
        d.yaw_enu = yaw
        d.last_update_time = time.time()
        d.active = True

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_active_drones(self) -> List[int]:
        """Return list of drone IDs that are currently active."""
        return [i for i, d in self.drones.items() if d.active]

    def get_leader_id(self) -> int:
        """
        Return the ID of the current leader drone.
        Falls back to drone 0 if no leader is set.
        """
        for i, d in self.drones.items():
            if d.is_leader:
                return i
        return 0

    def compute_active_ratio(self) -> float:
        """Return the fraction of drones that are active."""
        if self.num_drones == 0:
            return 0.0
        return len(self.get_active_drones()) / self.num_drones

    def get_neighbor_relative_positions(
        self, drone_id: int
    ) -> Dict[int, np.ndarray]:
        """
        Return ENU relative positions of all other drones with respect to
        drone ``drone_id``.

        Returns
        -------
        dict[int, np.ndarray]
            Mapping from drone_id to relative position vector (ENU).
        """
        if drone_id not in self.drones:
            return {}
        own_pos = self.drones[drone_id].position_enu
        result = {}
        for i, d in self.drones.items():
            if i == drone_id:
                continue
            result[i] = d.position_enu - own_pos
        return result

    def get_min_inter_drone_distance(self) -> float:
        """
        Return the minimum pairwise distance between any two active drones.
        Returns infinity if fewer than 2 active drones.
        """
        active = self.get_active_drones()
        if len(active) < 2:
            return float('inf')
        min_dist = float('inf')
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                dist = float(np.linalg.norm(
                    self.drones[active[i]].position_enu -
                    self.drones[active[j]].position_enu
                ))
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    def mark_stale_drones(self, timeout: float = 1.0) -> List[int]:
        """
        Mark drones whose last update is older than ``timeout`` seconds as
        inactive.

        Returns
        -------
        list[int]
            IDs of drones that were marked stale.
        """
        now = time.time()
        stale = []
        for i, d in self.drones.items():
            if d.active and (now - d.last_update_time) > timeout:
                d.active = False
                stale.append(i)
        return stale
