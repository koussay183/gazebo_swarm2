"""
PFE - Swarm Drones MARL - ROS 2 adaptation
Leader Election: dynamic leader election for the swarm.

Scoring (total 100 pts):
  Battery     40 pts  — higher battery = better candidate
  Centrality  20 pts  — closer to swarm centre = better
  Reliability 20 pts  — history of no failures
  ID priority 20 pts  — lower ID = slightly preferred (tie-breaker)

Hysteresis: a new leader is only elected if its score exceeds the
current leader's score by at least hysteresis_margin.

Adapted from src/leader_election.py.
Does NOT depend on drone.DroneState — accepts plain state strings or None.
"""

import math
from .config import NUM_DRONES, INITIAL_LEADER, BATTERY_FULL, SPAWN_RADIUS


# State strings considered "failed" / "unavailable"
_DEAD_STATES = frozenset({
    "FAILED", "GROUNDED", "EMERGENCY_LAND", "LANDED", "IDLE", "failed",
    "grounded", "emergency_land", "landed", "idle",
})


class LeaderElection:
    """
    Dynamic leader election for a swarm.

    Parameters
    ----------
    num_drones : int
        Total number of drones.
    hysteresis_margin : float
        A new leader is only accepted if its score exceeds the current
        leader's score by this amount (prevents thrashing).
    use_comm_quality : bool
        If True, incorporate comm_quality from state_extras into scoring.
    use_nav_health : bool
        If True, incorporate nav_health from state_extras into scoring.
    """

    def __init__(
        self,
        num_drones: int = NUM_DRONES,
        hysteresis_margin: float = 5.0,
        use_comm_quality: bool = False,
        use_nav_health: bool = False,
    ):
        self.n = num_drones
        self.leader_id = INITIAL_LEADER
        self.hysteresis_margin = hysteresis_margin
        self.use_comm_quality = use_comm_quality
        self.use_nav_health = use_nav_health
        self.reliability_scores: dict = {i: 1.0 for i in range(num_drones)}

    def elect(self, drone_states: list) -> int:
        """
        Elect the best leader among active drones.

        Parameters
        ----------
        drone_states : list of dict
            Each entry is a dict with keys:
              id       : int   — drone ID
              x, y     : float — ENU position
              battery  : float — battery level [0..BATTERY_FULL]
              state    : str or None — state string (None = active)
              problems : list  — list of active problem strings (can be empty)
              (optional) comm_quality : float in [0, 1]
              (optional) nav_health   : float in [0, 1]

        Returns
        -------
        int
            ID of the elected leader.
        """
        active = [
            d for d in drone_states
            if d.get("state") not in _DEAD_STATES and d.get("state") is not None
            or d.get("state") is None
        ]
        # Filter out drones that are explicitly dead
        active = [d for d in drone_states if d.get("state") not in _DEAD_STATES]
        if not active:
            return self.leader_id

        # Swarm centre
        cx = sum(d["x"] for d in active) / len(active)
        cy = sum(d["y"] for d in active) / len(active)

        best_id = -1
        best_score = -float('inf')
        current_leader_score = -float('inf')

        for d in active:
            if d.get("problems"):
                continue

            drone_id = d["id"]

            # Battery score (0-40)
            bat_score = (d.get("battery", BATTERY_FULL) / BATTERY_FULL) * 40.0

            # Centrality score (0-20)
            dist_center = math.sqrt((d["x"] - cx)**2 + (d["y"] - cy)**2)
            max_dist = SPAWN_RADIUS * 3
            central_score = max(0.0, (1.0 - dist_center / max_dist)) * 20.0

            # Reliability score (0-20)
            reliability = self.reliability_scores.get(drone_id, 1.0)
            rel_score = reliability * 20.0

            # ID priority score (0-20)
            id_score = ((self.n - drone_id) / self.n) * 20.0

            total = bat_score + central_score + rel_score + id_score

            # Optional real-world metrics
            if self.use_comm_quality:
                total += d.get("comm_quality", 1.0) * 10.0
            if self.use_nav_health:
                total += d.get("nav_health", 1.0) * 10.0

            if drone_id == self.leader_id:
                current_leader_score = total

            if total > best_score:
                best_score = total
                best_id = drone_id

        if best_id < 0:
            return self.leader_id

        # Hysteresis: only switch if clearly better
        if best_id != self.leader_id:
            if best_score <= current_leader_score + self.hysteresis_margin:
                return self.leader_id

        self.leader_id = best_id
        return self.leader_id

    def on_drone_problem(self, drone_id: int) -> bool:
        """
        Decrease the reliability score for a drone that reported a problem.

        Returns True if re-election is needed (i.e. the affected drone is
        the current leader).
        """
        self.reliability_scores[drone_id] = (
            self.reliability_scores.get(drone_id, 1.0) * 0.5
        )
        return drone_id == self.leader_id

    def on_drone_recovered(self, drone_id: int) -> None:
        """Partially restore reliability when a drone recovers."""
        current = self.reliability_scores.get(drone_id, 1.0)
        self.reliability_scores[drone_id] = min(1.0, current + 0.2)
