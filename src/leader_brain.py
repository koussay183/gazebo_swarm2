"""
PFE - Swarm Drones MARL - PyBullet
LeaderBrain : cerveau décisionnel autonome du leader.

Le leader n'est PAS une simple passerelle — il analyse en continu la
situation, prend des décisions et ORDONNE le swarm.

Cycle de décision (toutes les DECISION_INTERVAL secondes) :
  1. OBSERVER  — collecte l'état de tous les drones, menaces, zone
  2. ÉVALUER   — machine à états : PATROL / THREAT / REGROUP / RTB
  3. DÉCIDER   — choisit formation, stratégie, affectations
  4. ORDONNER  — applique les décisions (cibles, formations, RTB, comm)
"""

import math
import time as _time
from config import *
from drone import DroneState
from communication import Message


# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES DU CERVEAU
# ─────────────────────────────────────────────────────────────────────────────

DECISION_INTERVAL   = 2.0   # secondes entre deux cycles de décision
BATTERY_RTB         = 20.0  # % → le leader ordonne RTB au follower
BATTERY_LEADER_WARN = 35.0  # % → le leader prépare la passation
BATTERY_LEADER_CRIT = 20.0  # % → le leader rentre et transfère le commandement
SECTOR_RADIUS       = 8.0   # m → seuil "on est dans ce secteur"

# Formation optimale selon le nombre de drones actifs
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


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSE PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

class LeaderBrain:
    """
    Cerveau autonome du leader du swarm.

    États internes :
      PATROL  → patrouille intelligente, formation adaptée
      THREAT  → menace détectée, commande l'encerclement + formation défensive
      REGROUP → perte de drones, réorganisation du swarm
      RTB     → batterie du leader critique, passation de commandement
    """

    def __init__(self, num_drones):
        self.n      = num_drones
        self.state  = "PATROL"
        self._t_last_decision = 0.0

        # Mémoire de décision
        self._last_formation  = None
        self._last_n_threats  = -1
        self._last_n_active   = num_drones
        self._rtb_sent        = set()   # IDs auxquels on a ordonné RTB
        self._regroup_done    = False

        # Secteurs de la zone pour couvrir uniformément
        self._sectors          = self._init_sectors()
        self._sector_visit_t   = {i: 0.0 for i in range(len(self._sectors))}

        # Log des décisions (50 entrées)
        self._log = []

    # ═════════════════════════════════════════════════════════════════════════
    #  POINT D'ENTRÉE — appelé depuis sim.step()
    # ═════════════════════════════════════════════════════════════════════════

    def tick(self, sim_time, drones, intruders, formation_mgr, election, comm, sim):
        """
        Cycle complet de décision.
        À appeler à chaque pas de simulation — la décision n'est prise que
        tous les DECISION_INTERVAL secondes.
        """
        if sim_time - self._t_last_decision < DECISION_INTERVAL:
            return
        self._t_last_decision = sim_time

        leader = drones[election.leader_id]
        obs    = self._observe(drones, intruders, election)

        # ── Transition d'état ──────────────────────────────────────────────
        prev_state = self.state
        self.state = self._transition(obs)
        if self.state != prev_state:
            self._emit(f"ÉTAT : {prev_state} → {self.state}")

        # ── Exécution des décisions ────────────────────────────────────────
        self._execute(obs, drones, intruders, formation_mgr, election, comm, sim, leader)

        # Mémoriser pour le prochain cycle
        self._last_n_active = obs["n_active"]

    # ═════════════════════════════════════════════════════════════════════════
    #  1. OBSERVER
    # ═════════════════════════════════════════════════════════════════════════

    def _observe(self, drones, intruders, election):
        dead = {DroneState.FAILED, DroneState.GROUNDED,
                DroneState.EMERGENCY_LAND, DroneState.LANDED, DroneState.IDLE}

        active     = [d for d in drones if d.state not in dead]
        failed     = [d for d in drones if d.state in dead]
        encircling = [d for d in active if d.state == DroneState.ENCIRCLING]
        escorting  = [d for d in active if d.state != DroneState.ENCIRCLING]

        detected   = [i for i in intruders if i.get("detected")]

        leader     = drones[election.leader_id]
        low_bat    = [d for d in active
                      if d.battery < BATTERY_RTB and d.id not in self._rtb_sent
                      and d.id != election.leader_id]

        return {
            "active":      active,
            "failed":      failed,
            "encircling":  encircling,
            "escorting":   escorting,
            "threats":     detected,
            "low_bat":     low_bat,
            "n_active":    len(active),
            "n_failed":    len(failed),
            "n_threats":   len(detected),
            "n_encircling": len(encircling),
            "leader_warn": leader.battery < BATTERY_LEADER_WARN,
            "leader_crit": leader.battery < BATTERY_LEADER_CRIT,
        }

    # ═════════════════════════════════════════════════════════════════════════
    #  2. ÉVALUER — machine à états
    # ═════════════════════════════════════════════════════════════════════════

    def _transition(self, obs):
        # Batterie critique du leader → passation prioritaire
        if obs["leader_crit"]:
            return "RTB"

        # Menace détectée → réponse immédiate
        if obs["n_threats"] > 0:
            return "THREAT"

        # Menace résolue → regroupe avant de reprendre la patrouille
        if self.state == "THREAT" and obs["n_threats"] == 0:
            self._regroup_done = False
            return "REGROUP"

        # Perte de drones depuis le dernier cycle → regroupe
        if obs["n_active"] < self._last_n_active and self.state == "PATROL":
            self._regroup_done = False
            return "REGROUP"

        # Regroup terminé : tous les drones actifs sont en vol normal
        if self.state == "REGROUP":
            all_flying = all(d.state in (DroneState.HOVERING, DroneState.NAVIGATING)
                             for d in obs["active"])
            if all_flying and not self._regroup_done:
                self._regroup_done = True
                return "PATROL"

        # Leader batterie remontée (rechargé) → reprendre
        if self.state == "RTB" and not obs["leader_crit"]:
            return "PATROL"

        return self.state

    # ═════════════════════════════════════════════════════════════════════════
    #  3 + 4. DÉCIDER ET ORDONNER
    # ═════════════════════════════════════════════════════════════════════════

    def _execute(self, obs, drones, intruders, formation_mgr, election, comm, sim, leader):
        lid = election.leader_id

        # ── TOUJOURS : batteries basses des followers ──────────────────────
        self._order_followers_rtb(obs, comm, lid)

        # ── TOUJOURS : adapter la formation au contexte ───────────────────
        self._adapt_formation(obs, formation_mgr, comm, lid, sim)

        # ── SELON L'ÉTAT ──────────────────────────────────────────────────
        if self.state == "PATROL":
            self._do_patrol(obs, sim, leader)

        elif self.state == "THREAT":
            self._do_threat(obs, intruders, election, comm, sim)

        elif self.state == "REGROUP":
            self._do_regroup(obs, sim, leader)

        elif self.state == "RTB":
            self._do_rtb(obs, leader, comm, election, sim)

    # ─── Ordre de retour-base pour les followers à batterie basse ───────────

    def _order_followers_rtb(self, obs, comm, leader_id):
        for d in obs["low_bat"]:
            self._rtb_sent.add(d.id)
            d.return_base()
            comm.send(Message(leader_id, "STATUS_REQUEST",
                              {"target": d.id, "order": "RTB", "reason": "LOW_BATTERY"}))
            self._emit(f"RTB ordonné → D{d.id}  (batterie {d.battery:.0f}%)")

    # ─── Choix de formation adapté au contexte ───────────────────────────────

    def _adapt_formation(self, obs, formation_mgr, comm, leader_id, sim=None):
        # Si l'opérateur a verrouillé une formation, ne pas l'écraser
        # sauf en cas de menace réelle (THREAT) où la sécurité prime
        if getattr(sim, "formation_locked", False) and self.state != "THREAT":
            return

        n = obs["n_active"]

        # Sous menace avec assez de drones pour escorter → formation défensive
        in_threat = obs["n_threats"] > 0
        n_escort  = obs["n_active"] - obs["n_encircling"]
        if in_threat and n_escort >= 2:
            desired = "wedge"
        elif in_threat and n_escort < 2:
            desired = "line"   # trop peu → formation minimale
        else:
            desired = FORMATION_BY_N.get(n, "line")

        if desired != self._last_formation:
            formation_mgr.set_formation(desired)
            self._last_formation = desired
            self._emit(f"Formation → {desired}  ({n} drones actifs, {obs['n_threats']} menace(s))")
            comm.send(Message(leader_id, "FORMATION_CMD",
                              {"formation": desired,
                               "reason": f"{n}_drones_{obs['n_threats']}_threats"}))

    # ─── État PATROL : patrouille intelligente par secteurs ─────────────────

    def _do_patrol(self, obs, sim, leader):
        self._update_sector_visit(leader)

        # Pas en patrouille et drones en attente → lancer une patrouille ciblée
        if not sim.patrolling and leader.state == DroneState.HOVERING:
            sector = self._least_visited_sector()
            wps    = self._gen_coverage_patrol(sector)
            sim.patrol_wps    = wps
            sim.patrol_idx    = 0
            sim.patrolling    = True
            sim.phase         = "PATROLLING"
            sim.custom_target = None
            sim.custom_route  = None
            self._emit(f"Patrouille lancée — couverture secteur {sector['name']}")

    # ─── État THREAT : réponse aux menaces ───────────────────────────────────

    def _do_threat(self, obs, intruders, election, comm, sim):
        if obs["n_threats"] == 0:
            return

        # Arrêter la patrouille immédiatement
        if sim.patrolling:
            sim.patrolling = False
            self._emit("Patrouille interrompue — menace active")

        # Si le nombre de menaces a changé → forcer la réassignation
        if obs["n_threats"] != self._last_n_threats:
            self._last_n_threats = obs["n_threats"]
            self._emit(f"{obs['n_threats']} menace(s) détectée(s) — réassignation du swarm")
            sim.phase = "THREAT_RESPONSE"
            # Forcer la répartition (la logique de détection continue est dans sim)
            sim._detect_intruders_continuous()

        # Broadcast INTRUDER pour chaque menace → tous les drones en sont informés
        lid = election.leader_id
        for intr in obs["threats"]:
            ix, iy = intr["pos"]
            comm.send(Message(lid, "INTRUDER",
                              {"pos": [ix, iy], "order": "ENCIRCLE",
                               "assigned": intr.get("assigned_drones", [])}))

    # ─── État REGROUP : réorganisation après perte ───────────────────────────

    def _do_regroup(self, obs, sim, leader):
        sim.patrolling    = False
        sim.custom_target = None
        sim.custom_route  = None
        sim.phase         = "REGROUPING"

        n = obs["n_active"]
        n_lost = self.n - n
        self._emit(f"Regroupement — {n} drones opérationnels "
                   f"({n_lost} perdu(s))")

        # Remettre en vol normal les drones dont l'encerclement est terminé
        if obs["n_threats"] == 0:
            for d in obs["encircling"]:
                d.state = DroneState.NAVIGATING

    # ─── État RTB : batterie critique du leader ───────────────────────────────

    def _do_rtb(self, obs, leader, comm, election, sim):
        if obs["leader_warn"] and not obs["leader_crit"]:
            self._emit(f"Batterie leader {leader.battery:.0f}% — préparation passation")
            comm.send(Message(leader.id, "ELECTION",
                              {"reason": "LOW_BATTERY", "battery": leader.battery}))

        if obs["leader_crit"]:
            self._emit(f"Leader D{leader.id} critique ({leader.battery:.0f}%) — rentre à la base")
            leader.return_base()
            sim.phase = "LEADER_RTB"

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTEURS ET GÉNÉRATION DE TRAJECTOIRES
    # ═════════════════════════════════════════════════════════════════════════

    def _init_sectors(self):
        z  = PATROL_ZONE
        cx = (z["x_min"] + z["x_max"]) / 2
        cy = (z["y_min"] + z["y_max"]) / 2
        return [
            {"cx": (z["x_min"] + cx) / 2, "cy": (z["y_min"] + cy) / 2, "name": "SW"},
            {"cx": (z["x_max"] + cx) / 2, "cy": (z["y_min"] + cy) / 2, "name": "SE"},
            {"cx": (z["x_max"] + cx) / 2, "cy": (z["y_max"] + cy) / 2, "name": "NE"},
            {"cx": (z["x_min"] + cx) / 2, "cy": (z["y_max"] + cy) / 2, "name": "NW"},
        ]

    def _update_sector_visit(self, leader):
        t = _time.time()
        for i, s in enumerate(self._sectors):
            dist = math.sqrt((leader.x - s["cx"])**2 + (leader.y - s["cy"])**2)
            if dist < SECTOR_RADIUS:
                self._sector_visit_t[i] = t

    def _least_visited_sector(self):
        """Retourne le secteur qui n'a pas été visité depuis le plus longtemps."""
        idx = min(range(len(self._sectors)),
                  key=lambda i: self._sector_visit_t[i])
        return self._sectors[idx]

    def _gen_coverage_patrol(self, priority_sector):
        """
        Génère 16 waypoints circulaires en démarrant dans la direction
        du secteur prioritaire pour couvrir la zone uniformément.
        """
        z   = PATROL_ZONE
        r   = min(PATROL_RADIUS, (z["x_max"] - z["x_min"]) / 2 - 3)
        cx  = (z["x_min"] + z["x_max"]) / 2
        cy  = (z["y_min"] + z["y_max"]) / 2
        n   = 16

        start_angle = math.atan2(
            priority_sector["cy"] - cy,
            priority_sector["cx"] - cx
        )

        wps = []
        for i in range(n):
            angle = start_angle + 2 * math.pi * i / n
            x = max(z["x_min"] + 3, min(z["x_max"] - 3, r * math.cos(angle)))
            y = max(z["y_min"] + 3, min(z["y_max"] - 3, r * math.sin(angle)))
            wps.append((x, y))
        return wps

    # ═════════════════════════════════════════════════════════════════════════
    #  LOG ET INTROSPECTION
    # ═════════════════════════════════════════════════════════════════════════

    def _emit(self, msg):
        print(f"  👑 [CERVEAU] {msg}")
        self._log.append(msg)
        if len(self._log) > 50:
            self._log.pop(0)

    def get_log(self, n=15):
        """Retourne les n dernières décisions du cerveau."""
        return self._log[-n:]

    @property
    def state_label(self):
        return self.state
