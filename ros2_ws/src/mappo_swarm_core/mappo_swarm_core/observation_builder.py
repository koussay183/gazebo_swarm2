"""
mappo_swarm_core.observation_builder
Build per-drone observation vectors for MAPPO inference.

Observation layout (total = OBS_SIZE_PER_DRONE):
  [0:3]              own position  (x/scale, y/scale, z/scale)
  [3:6]              target position  (tx/scale, ty/scale, tz/scale)
  [6:6+3*(N-1)]      neighbor relative positions (0,0,0 if inactive)
  [6+3*(N-1) : +4]   4-nearest obstacle distances (normalised)
  [+4 : +12]         4-nearest obstacle direction vectors (dx, dy each)
  [+12]              battery / BATTERY_FULL
  [+13]              leader flag (1.0 if this drone is the leader)
  [+14]              active ratio (sum(active) / num_drones)
"""

import math
import numpy as np
from .config import OBSTACLES, BATTERY_FULL, OBS_SIZE_PER_DRONE


class ObservationBuilder:
    """
    Build the observation vector for a single drone.

    Parameters
    ----------
    num_drones : int
        Total number of drones in the swarm.
    obs_size : int
        Expected total observation size (must match OBS_SIZE_PER_DRONE).
    pos_scale : float
        Position normalisation scale (default 25.0 metres).
    """

    def __init__(self, num_drones: int, obs_size: int, pos_scale: float = 25.0):
        self.num_drones = num_drones
        self.obs_size = obs_size
        self.pos_scale = pos_scale

    def build(
        self,
        drone_id: int,
        positions_enu: "dict[int, np.ndarray]",
        targets_enu: "dict[int, np.ndarray]",
        batteries: "dict[int, float]",
        leader_id: int,
        active_flags: "dict[int, bool]",
        obstacles: "list[dict]" = None,
    ) -> np.ndarray:
        """
        Build the observation vector for drone ``drone_id``.

        Parameters
        ----------
        drone_id : int
            ID of the drone whose observation we are building.
        positions_enu : dict[int, np.ndarray]
            ENU positions (x, y, z) for each drone.
        targets_enu : dict[int, np.ndarray]
            ENU formation target (x, y, z) for each drone.
        batteries : dict[int, float]
            Battery level [0..BATTERY_FULL] for each drone.
        leader_id : int
            ID of the current leader drone.
        active_flags : dict[int, bool]
            Whether each drone is active/alive.
        obstacles : list[dict] or None
            Obstacle list (defaults to config.OBSTACLES).

        Returns
        -------
        np.ndarray of shape (obs_size,) dtype float32
        """
        if obstacles is None:
            obstacles = OBSTACLES

        obs = []
        scale = self.pos_scale

        # --- Own position ---
        pos = positions_enu.get(drone_id, np.zeros(3))
        obs += [pos[0] / scale, pos[1] / scale, pos[2] / scale]

        # --- Target position ---
        tgt = targets_enu.get(drone_id, np.zeros(3))
        obs += [tgt[0] / scale, tgt[1] / scale, tgt[2] / scale]

        # --- Neighbor relative positions ---
        for j in range(self.num_drones):
            if j == drone_id:
                continue
            if active_flags.get(j, False):
                other = positions_enu.get(j, np.zeros(3))
                obs += [
                    (other[0] - pos[0]) / scale,
                    (other[1] - pos[1]) / scale,
                    (other[2] - pos[2]) / scale,
                ]
            else:
                obs += [0.0, 0.0, 0.0]

        # --- 4-nearest obstacle distances + direction vectors ---
        px, py = pos[0], pos[1]
        obs_infos = []
        for o in obstacles:
            raw_dx = px - o["x"]
            raw_dy = py - o["y"]
            dist_x = max(0.0, abs(raw_dx) - o.get("sx", 2.0) / 2)
            dist_y = max(0.0, abs(raw_dy) - o.get("sy", 2.0) / 2)
            dist = math.sqrt(dist_x**2 + dist_y**2)
            norm = math.sqrt(raw_dx**2 + raw_dy**2) + 1e-6
            obs_infos.append((dist / scale, raw_dx / norm, raw_dy / norm))
        obs_infos.sort(key=lambda x: x[0])
        obs_infos = obs_infos[:4]
        while len(obs_infos) < 4:
            obs_infos.append((1.0, 0.0, 0.0))
        # distances
        for k in range(4):
            obs.append(obs_infos[k][0])
        # direction vectors (dx, dy) for each of the 4
        for k in range(4):
            obs.append(obs_infos[k][1])
            obs.append(obs_infos[k][2])

        # --- Battery ---
        bat = batteries.get(drone_id, BATTERY_FULL)
        obs.append(bat / BATTERY_FULL)

        # --- Leader flag ---
        obs.append(1.0 if drone_id == leader_id else 0.0)

        # --- Active ratio ---
        n_active = sum(1 for v in active_flags.values() if v)
        obs.append(n_active / self.num_drones)

        obs_array = np.array(obs, dtype=np.float32)

        if len(obs_array) != self.obs_size:
            raise ValueError(
                f"ObservationBuilder.build(): built obs has length {len(obs_array)} "
                f"but expected obs_size={self.obs_size}. "
                f"Check num_drones={self.num_drones} and config.OBS_SIZE_PER_DRONE."
            )

        obs_array = np.nan_to_num(obs_array, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs_array
