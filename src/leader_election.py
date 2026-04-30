"""
PFE - Swarm Drones MARL - PyBullet
Leader Election : élection dynamique du leader
Critères : batterie, centralité de position, fiabilité (pas de problèmes), priorité ID
"""
import math
from config import *
from drone import DroneState


class LeaderElection:
    def __init__(self, num_drones=NUM_DRONES):
        self.n = num_drones
        self.leader_id = INITIAL_LEADER
        self.reliability_scores = {i: 1.0 for i in range(num_drones)}

    def elect(self, drones):
        """Élit le meilleur leader parmi les drones actifs."""
        best_id = -1
        best_score = -float('inf')

        # Calculer le centre du swarm
        active = [d for d in drones if d.state not in
                  [DroneState.FAILED, DroneState.GROUNDED,
                   DroneState.EMERGENCY_LAND, DroneState.LANDED]]
        if not active:
            return self.leader_id

        cx = sum(d.x for d in active) / len(active)
        cy = sum(d.y for d in active) / len(active)

        for d in active:
            if d.problems:
                continue

            # Score batterie (0-40)
            bat_score = (d.battery / BATTERY_FULL) * 40

            # Score centralité (0-20) : plus proche du centre = mieux
            dist_center = math.sqrt((d.x - cx)**2 + (d.y - cy)**2)
            max_dist = SPAWN_RADIUS * 3
            central_score = max(0, (1 - dist_center / max_dist)) * 20

            # Score fiabilité (0-20) : historique sans problèmes
            reliability = self.reliability_scores.get(d.id, 1.0)
            rel_score = reliability * 20

            # Score priorité ID (0-20) : ID bas = plus prioritaire
            id_score = ((self.n - d.id) / self.n) * 20

            total = bat_score + central_score + rel_score + id_score

            if total > best_score:
                best_score = total
                best_id = d.id

        if best_id >= 0:
            old = self.leader_id
            self.leader_id = best_id
            if old != best_id:
                return best_id

        return self.leader_id

    def on_drone_problem(self, drone_id):
        """Diminue la fiabilité d'un drone."""
        self.reliability_scores[drone_id] *= 0.5
        if drone_id == self.leader_id:
            return True  # Besoin re-élection
        return False

    def on_drone_recovered(self, drone_id):
        self.reliability_scores[drone_id] = min(1.0,
            self.reliability_scores[drone_id] + 0.2)
