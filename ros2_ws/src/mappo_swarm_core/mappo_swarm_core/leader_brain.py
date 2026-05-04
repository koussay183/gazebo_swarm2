"""
PFE - Swarm Drones MARL - ROS 2 adaptation
LeaderBrain: autonomous decision-making for the swarm leader.

Adapted from src/leader_brain.py.
tick() does NOT depend on a sim object — accepts SwarmState instead.

Decision cycle (every DECISION_INTERVAL seconds):
  1. OBSERVE  — collect state from SwarmState
  2. EVALUATE — state machine: PATROL / THREAT / REGROUP / RTB
  3. DECIDE   — choose formation, assignments, RTB commands
  4. COMMAND  — return a BrainCommand dataclass
"""

import math
import time as _time
from dataclasses import dataclass, field
from typing import List, Dict

from .config import (
    PATROL_ZONE, PATROL_RADIUS, BATTERY_FULL,
)
from .swarm_state import SwarmState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DECISION_INTERVAL   = 2.0   # seconds between decision cycles
BATTERY_RTB         = 20.0  # % -> order follower RTB
BATTERY_LEADER_WARN = 35.0  # % -> prepare handover
BATTERY_LEADER_CRIT = 20.0  # % -> leader returns to base
SECTOR_RADIUS       = 8.0   # m -> threshold for "we are in this sector"

# Optimal formation by number of active drones
FORMATION_BY_N = {
    8: "v_shape",
    7: "v_shape",
    6: "diamond",
    5: "diamond",
    4: "v_shape",
    3: "line",
    2: "line",
    1: "line",
}


# ---------------------------------------------------------------------------
# BrainCommand dataclass
# ---------------------------------------------------------------------------

@dataclass
class BrainCommand:
    """Output of a single LeaderBrain.tick() cycle."""
    mission_state: str = "PATROL"
    desired_formation: str = "v_shape"
    target_assignments: Dict[int, tuple] = field(default_factory=dict)
    rtb_drones: List[int] = field(default_factory=list)
    threat_positions: List[tuple] = field(default_factory=list)
    log_entries: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# LeaderBrain
# ---------------------------------------------------------------------------

class LeaderBrain:
    """
    Autonomous leader brain.

    States:
      PATROL  -> intelligent patrol, adapted formation
      THREAT  -> threat detected, encircle + defensive formation
      REGROUP -> drone loss, reorganise swarm
      RTB     -> leader battery critical, hand over command
    """

    def __init__(self, num_drones: int):
        self.n = num_drones
        self.state = "PATROL"
        self._t_last_decision = 0.0

        self._last_formation = None
        self._last_n_threats = -1
        self._last_n_active = num_drones
        self._rtb_sent: set = set()
        self._regroup_done = False

        self._sectors = self._init_sectors()
        self._sector_visit_t = {i: 0.0 for i in range(len(self._sectors))}

        self._log: List[str] = []

    # -------------------------------------------------------------------------
    # Main entry point
    # -------------------------------------------------------------------------

    def tick(
        self,
        sim_time: float,
        swarm_state: SwarmState,
        formation_mgr,
    ) -> BrainCommand:
        """
        Run one decision cycle.

        Parameters
        ----------
        sim_time : float
            Current simulation / wall-clock time (seconds).
        swarm_state : SwarmState
            Current state of all drones.
        formation_mgr : FormationManager
            Formation manager to update formation choice.

        Returns
        -------
        BrainCommand
            Decisions for this cycle.
        """
        if sim_time - self._t_last_decision < DECISION_INTERVAL:
            return BrainCommand(
                mission_state=self.state,
                desired_formation=self._last_formation or "v_shape",
            )
        self._t_last_decision = sim_time

        obs = self._observe(swarm_state)

        prev_state = self.state
        self.state = self._transition(obs)
        if self.state != prev_state:
            self._emit(f"STATE: {prev_state} -> {self.state}")

        cmd = self._execute(obs, swarm_state, formation_mgr)
        self._last_n_active = obs["n_active"]
        return cmd

    # -------------------------------------------------------------------------
    # 1. OBSERVE
    # -------------------------------------------------------------------------

    def _observe(self, swarm_state: SwarmState) -> dict:
        active_ids = swarm_state.get_active_drones()
        leader_id = swarm_state.get_leader_id()
        leader = swarm_state.drones.get(leader_id)
        leader_battery = leader.battery if leader else BATTERY_FULL

        low_bat = [
            i for i in active_ids
            if swarm_state.drones[i].battery < BATTERY_RTB
            and i not in self._rtb_sent
            and i != leader_id
        ]

        return {
            "active_ids":    active_ids,
            "n_active":      len(active_ids),
            "leader_id":     leader_id,
            "leader_battery": leader_battery,
            "leader_warn":   leader_battery < BATTERY_LEADER_WARN,
            "leader_crit":   leader_battery < BATTERY_LEADER_CRIT,
            "low_bat":       low_bat,
            # Threats are injected externally via threat_positions in the command
            "n_threats":     0,
        }

    # -------------------------------------------------------------------------
    # 2. STATE TRANSITION
    # -------------------------------------------------------------------------

    def _transition(self, obs: dict) -> str:
        if obs["leader_crit"]:
            return "RTB"
        if obs["n_threats"] > 0:
            return "THREAT"
        if self.state == "THREAT" and obs["n_threats"] == 0:
            self._regroup_done = False
            return "REGROUP"
        if obs["n_active"] < self._last_n_active and self.state == "PATROL":
            self._regroup_done = False
            return "REGROUP"
        if self.state == "REGROUP" and not self._regroup_done:
            # Simple heuristic: regroup done once we have waited one decision cycle
            self._regroup_done = True
            return "PATROL"
        if self.state == "RTB" and not obs["leader_crit"]:
            return "PATROL"
        return self.state

    # -------------------------------------------------------------------------
    # 3+4. DECIDE AND COMMAND
    # -------------------------------------------------------------------------

    def _execute(
        self, obs: dict, swarm_state: SwarmState, formation_mgr
    ) -> BrainCommand:
        cmd = BrainCommand(mission_state=self.state)
        cmd.log_entries = list(self._log[-5:])  # last 5 entries

        # RTB for low-battery followers
        for did in obs["low_bat"]:
            self._rtb_sent.add(did)
            cmd.rtb_drones.append(did)
            self._emit(f"RTB ordered -> D{did}  "
                       f"(battery {swarm_state.drones[did].battery:.0f}%)")

        # Formation adaptation
        n = obs["n_active"]
        if self.state == "THREAT":
            desired = "wedge" if n >= 3 else "line"
        else:
            desired = FORMATION_BY_N.get(n, "line")

        if desired != self._last_formation:
            formation_mgr.set_formation(desired)
            self._last_formation = desired
            self._emit(f"Formation -> {desired}  ({n} active drones)")
        cmd.desired_formation = desired

        # Patrol waypoint for leader
        if self.state == "PATROL":
            leader_id = obs["leader_id"]
            leader = swarm_state.drones.get(leader_id)
            if leader is not None:
                self._update_sector_visit(leader.position_enu)
                sector = self._least_visited_sector()
                wps = self._gen_coverage_patrol(sector)
                if wps:
                    cmd.target_assignments[leader_id] = wps[0] + (5.0,)

        # RTB state
        if self.state == "RTB":
            self._emit(f"Leader battery {obs['leader_battery']:.0f}% critical — RTB")

        return cmd

    # -------------------------------------------------------------------------
    # Sector logic (patrol coverage)
    # -------------------------------------------------------------------------

    def _init_sectors(self):
        z = PATROL_ZONE
        cx = (z["x_min"] + z["x_max"]) / 2
        cy = (z["y_min"] + z["y_max"]) / 2
        return [
            {"cx": (z["x_min"] + cx) / 2, "cy": (z["y_min"] + cy) / 2, "name": "SW"},
            {"cx": (z["x_max"] + cx) / 2, "cy": (z["y_min"] + cy) / 2, "name": "SE"},
            {"cx": (z["x_max"] + cx) / 2, "cy": (z["y_max"] + cy) / 2, "name": "NE"},
            {"cx": (z["x_min"] + cx) / 2, "cy": (z["y_max"] + cy) / 2, "name": "NW"},
        ]

    def _update_sector_visit(self, position_enu) -> None:
        t = _time.time()
        for i, s in enumerate(self._sectors):
            dist = math.sqrt(
                (position_enu[0] - s["cx"])**2 + (position_enu[1] - s["cy"])**2)
            if dist < SECTOR_RADIUS:
                self._sector_visit_t[i] = t

    def _least_visited_sector(self) -> dict:
        idx = min(range(len(self._sectors)),
                  key=lambda i: self._sector_visit_t[i])
        return self._sectors[idx]

    def _gen_coverage_patrol(self, priority_sector: dict) -> list:
        """
        Generate 16 circular waypoints starting toward the priority sector.
        """
        z = PATROL_ZONE
        r = min(PATROL_RADIUS, (z["x_max"] - z["x_min"]) / 2 - 3)
        cx = (z["x_min"] + z["x_max"]) / 2
        cy = (z["y_min"] + z["y_max"]) / 2
        n = 16
        start_angle = math.atan2(
            priority_sector["cy"] - cy, priority_sector["cx"] - cx)
        wps = []
        for i in range(n):
            angle = start_angle + 2 * math.pi * i / n
            x = max(z["x_min"] + 3, min(z["x_max"] - 3, r * math.cos(angle)))
            y = max(z["y_min"] + 3, min(z["y_max"] - 3, r * math.sin(angle)))
            wps.append((x, y))
        return wps

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _emit(self, msg: str) -> None:
        print(f"  [BRAIN] {msg}")
        self._log.append(msg)
        if len(self._log) > 50:
            self._log.pop(0)

    def get_log(self, n: int = 15) -> List[str]:
        """Return the last n decisions."""
        return self._log[-n:]

    @property
    def state_label(self) -> str:
        return self.state
