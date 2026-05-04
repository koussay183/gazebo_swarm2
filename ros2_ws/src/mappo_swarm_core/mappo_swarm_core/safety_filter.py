"""
mappo_swarm_core.safety_filter
Runtime safety envelope for MAPPO velocity commands.

All positions and velocities are in ENU frame (x=East, y=North, z=Up).
"""

import time
import numpy as np


class SafetyFilter:
    """
    Apply safety constraints to a raw MAPPO velocity command before
    forwarding it to the PX4 interface.

    Parameters
    ----------
    max_xy_speed : float
        Maximum horizontal speed (m/s).
    max_z_speed : float
        Maximum vertical speed (m/s).
    min_alt : float
        Minimum altitude (m, ENU z).
    max_alt : float
        Maximum altitude (m, ENU z).
    geofence : dict or None
        Optional dict with keys x_min, x_max, y_min, y_max (metres).
    min_drone_dist : float
        Minimum inter-drone separation (m).  Triggers repulsion if closer.
    max_accel : float
        Maximum acceleration magnitude (m/s^2).  Reserved for future use.
    stale_timeout : float
        If time since last update > stale_timeout, hold position.
    """

    def __init__(
        self,
        max_xy_speed: float = 2.5,
        max_z_speed: float = 1.0,
        min_alt: float = 2.0,
        max_alt: float = 20.0,
        geofence: dict = None,
        min_drone_dist: float = 1.5,
        max_accel: float = 5.0,
        stale_timeout: float = 1.0,
    ):
        self.max_xy_speed = max_xy_speed
        self.max_z_speed = max_z_speed
        self.min_alt = min_alt
        self.max_alt = max_alt
        self.geofence = geofence
        self.min_drone_dist = min_drone_dist
        self.max_accel = max_accel
        self.stale_timeout = stale_timeout

    def filter(
        self,
        drone_id: int,
        cmd_enu: np.ndarray,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        other_positions: "dict[int, np.ndarray]",
        last_update_time: float,
        active_flags: "dict[int, bool]",
    ) -> tuple:
        """
        Apply safety checks to a velocity command.

        Parameters
        ----------
        drone_id : int
            ID of the drone being controlled.
        cmd_enu : np.ndarray shape (3,)
            Requested velocity command [vx, vy, vz] in ENU (m/s).
        current_pos : np.ndarray shape (3,)
            Current ENU position [x, y, z].
        current_vel : np.ndarray shape (3,)
            Current ENU velocity [vx, vy, vz].
        other_positions : dict[int, np.ndarray]
            ENU positions of all other drones.
        last_update_time : float
            Timestamp (time.time()) of the last valid telemetry update.
        active_flags : dict[int, bool]
            Whether each other drone is active.

        Returns
        -------
        (safe_cmd, was_modified, reason) : tuple
            safe_cmd     : np.ndarray (3,) — the filtered velocity command
            was_modified : bool            — True if the command was changed
            reason       : str             — human-readable reason (or '' if unchanged)
        """
        hold = np.zeros(3, dtype=np.float32)
        cmd = np.array(cmd_enu, dtype=np.float32)

        # 1. NaN / Inf check
        if not np.all(np.isfinite(cmd)):
            return hold.copy(), True, "nan_inf"

        was_modified = False
        reasons = []

        # 2. Clamp horizontal speed
        xy_speed = np.linalg.norm(cmd[:2])
        if xy_speed > self.max_xy_speed:
            cmd[:2] = cmd[:2] / xy_speed * self.max_xy_speed
            was_modified = True
            reasons.append("xy_speed_clamped")

        # 3. Clamp vertical speed
        if abs(cmd[2]) > self.max_z_speed:
            cmd[2] = np.sign(cmd[2]) * self.max_z_speed
            was_modified = True
            reasons.append("z_speed_clamped")

        # 4. Geofence check (project position with cmd)
        if self.geofence is not None:
            proj_x = current_pos[0] + cmd[0]
            proj_y = current_pos[1] + cmd[1]
            if proj_x < self.geofence.get("x_min", -1e9) or \
               proj_x > self.geofence.get("x_max", 1e9):
                cmd[0] = 0.0
                was_modified = True
                reasons.append("geofence_x")
            if proj_y < self.geofence.get("y_min", -1e9) or \
               proj_y > self.geofence.get("y_max", 1e9):
                cmd[1] = 0.0
                was_modified = True
                reasons.append("geofence_y")

        # 5. Altitude limits
        current_z = current_pos[2]
        if current_z < self.min_alt and cmd[2] < 0.0:
            cmd[2] = 0.0
            was_modified = True
            reasons.append("min_alt")
        if current_z > self.max_alt and cmd[2] > 0.0:
            cmd[2] = 0.0
            was_modified = True
            reasons.append("max_alt")

        # 6. Inter-drone repulsion
        for other_id, other_pos in other_positions.items():
            if other_id == drone_id:
                continue
            if not active_flags.get(other_id, False):
                continue
            diff = current_pos - other_pos
            dist = np.linalg.norm(diff)
            if dist < self.min_drone_dist and dist > 1e-4:
                repulsion_dir = diff / dist
                repulsion_mag = self.max_xy_speed * (1.0 - dist / self.min_drone_dist)
                cmd[:2] += repulsion_dir[:2] * repulsion_mag
                was_modified = True
                reasons.append(f"repulsion_from_{other_id}")
        # Re-clamp after repulsion
        xy_speed = np.linalg.norm(cmd[:2])
        if xy_speed > self.max_xy_speed:
            cmd[:2] = cmd[:2] / xy_speed * self.max_xy_speed

        # 7. Stale telemetry check (run last to allow others to still modify)
        if (time.time() - last_update_time) > self.stale_timeout:
            return hold.copy(), True, "stale_telemetry"

        reason_str = ",".join(reasons) if reasons else ""
        return cmd, was_modified, reason_str
