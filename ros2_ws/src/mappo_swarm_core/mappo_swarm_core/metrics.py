"""
mappo_swarm_core.metrics
Reward and penalty functions extracted from config.py and rl_controller.py.

These functions are used by both the training environment (rl_env.py) and
the deployment controller (mappo_px4_rl_node.py) for consistent reward shaping.
"""

import math
import numpy as np
from .config import (
    METRICS, OBSTACLES, PATROL_ZONE,
    SAFETY_DISTANCE_OBSTACLES, SAFETY_DISTANCE_DRONES,
    BATTERY_FULL,
)

# ---------------------------------------------------------------------------
# Reward configuration (mirror of METRICS in config.py)
# ---------------------------------------------------------------------------

REWARD_CONFIG = {
    "reward_target_close":      METRICS["reward_target_close"],
    "reward_target_near":       METRICS["reward_target_near"],
    "reward_target_gradient":   METRICS["reward_target_gradient"],
    "reward_formation_perfect": METRICS["reward_formation_perfect"],
    "reward_team_max":          METRICS["reward_team_max"],
    "penalty_collision_drone":  METRICS["penalty_collision_drone"],
    "penalty_near_drone":       METRICS["penalty_near_drone"],
    "penalty_collision_obs":    METRICS["penalty_collision_obs"],
    "penalty_out_of_zone":      METRICS["penalty_out_of_zone"],
    "penalty_energy":           METRICS["penalty_energy"],
    "formation_perfect_dist":   METRICS["formation_perfect_dist"],
    "collision_drone_dist":     METRICS["collision_drone_dist"],
    "near_drone_dist":          METRICS["near_drone_dist"],
}


# ---------------------------------------------------------------------------
# Individual penalty/reward functions
# ---------------------------------------------------------------------------

def compute_obstacle_penalty(
    x: float, y: float, z: float,
    obstacles: list = None
) -> float:
    """
    Compute the total obstacle penalty for a position (x, y, z).

    Uses a two-zone model:
      - Soft gradient: dist < SAFETY_DISTANCE_OBSTACLES -> proportional penalty
      - Hard collision: dist < 0.5m -> x3 penalty

    Parameters
    ----------
    x, y, z : float
        ENU position.
    obstacles : list of dict or None
        Obstacle list (defaults to config.OBSTACLES).

    Returns
    -------
    float
        Total penalty (negative or zero).
    """
    if obstacles is None:
        obstacles = OBSTACLES
    M = REWARD_CONFIG
    penalty = 0.0
    for obs in obstacles:
        ox, oy = obs["x"], obs["y"]
        sx, sy, h = obs["sx"], obs["sy"], obs["h"]
        dx_o = max(0.0, abs(x - ox) - sx / 2)
        dy_o = max(0.0, abs(y - oy) - sy / 2)
        dist_obs = math.sqrt(dx_o**2 + dy_o**2)
        if z <= h + 2:
            if dist_obs < 0.5:
                penalty += M["penalty_collision_obs"] * 3.0
            elif dist_obs < SAFETY_DISTANCE_OBSTACLES:
                factor = max(0.0, 1.0 - dist_obs / SAFETY_DISTANCE_OBSTACLES)
                penalty += M["penalty_collision_obs"] * factor
    return penalty


def compute_drone_collision_penalty(
    pos: np.ndarray,
    other_positions: "list[np.ndarray]",
    active: "list[bool]" = None,
) -> float:
    """
    Compute inter-drone collision/proximity penalty.

    Parameters
    ----------
    pos : np.ndarray (3,)
        ENU position of this drone.
    other_positions : list of np.ndarray (3,)
        ENU positions of all other drones.
    active : list of bool or None
        Whether each other drone is active.  If None, all are treated as active.

    Returns
    -------
    float
        Total penalty (negative or zero).
    """
    M = REWARD_CONFIG
    penalty = 0.0
    for j, other_pos in enumerate(other_positions):
        if active is not None and not active[j]:
            continue
        dist = float(np.linalg.norm(pos - other_pos))
        if dist < M["collision_drone_dist"]:
            penalty += M["penalty_collision_drone"]
        elif dist < M["near_drone_dist"]:
            penalty += M["penalty_near_drone"]
    return penalty


def compute_formation_reward(
    pos: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute the formation-following reward for a single drone.

    Parameters
    ----------
    pos : np.ndarray (3,)
        Current ENU position.
    target : np.ndarray (3,)
        Desired formation target in ENU.

    Returns
    -------
    float
        Reward value (positive).
    """
    M = REWARD_CONFIG
    dist = float(np.linalg.norm(pos - target))
    reward = M["reward_target_gradient"] * math.exp(-dist / 4.0)
    if dist < 2.0:
        reward += M["reward_target_close"]
    elif dist < 4.0:
        reward += M["reward_target_near"]
    return reward


def compute_out_of_zone_penalty(
    pos: np.ndarray,
    patrol_zone: dict = None,
    margin: float = 0.5,
) -> float:
    """
    Compute the out-of-zone penalty.

    Parameters
    ----------
    pos : np.ndarray (3,)
        Current ENU position [x, y, z].
    patrol_zone : dict or None
        Zone dict with x_min, x_max, y_min, y_max.
    margin : float
        Inner margin (metres) before penalty is applied.

    Returns
    -------
    float
        Penalty (negative or zero).
    """
    if patrol_zone is None:
        patrol_zone = PATROL_ZONE
    M = REWARD_CONFIG
    x, y = pos[0], pos[1]
    if (x <= patrol_zone["x_min"] + margin or
            x >= patrol_zone["x_max"] - margin or
            y <= patrol_zone["y_min"] + margin or
            y >= patrol_zone["y_max"] - margin):
        return M["penalty_out_of_zone"]
    return 0.0
