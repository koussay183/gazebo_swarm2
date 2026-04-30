"""
PFE - Swarm Drones MARL - Gazebo Harmonic
Drone : mouvement DIRECT via cmd_vel ROS2

Identique à la version PyBullet sauf :
  - Pas de corps PyBullet (mesh visuel géré par le SDF Gazebo)
  - Mouvement via gazebo_interface.set_velocity() au lieu de
    p.resetBasePositionAndOrientation()
  - Position lue depuis gazebo_interface.get_pose()
"""

import math
import config as _cfg
from config import *
from communication import Message
from gazebo_interface import get_gazebo_interface
from px4_mavsdk_interface import get_px4_interface, init_px4


class DroneState:
    IDLE           = "IDLE"
    TAKEOFF        = "TAKEOFF"
    HOVERING       = "HOVERING"
    NAVIGATING     = "NAVIGATING"
    RETURNING      = "RETURNING"
    LANDING        = "LANDING"
    LANDED         = "LANDED"
    ENCIRCLING     = "ENCIRCLING"
    EMERGENCY_LAND = "EMERGENCY_LAND"
    GROUNDED       = "GROUNDED"
    FAILED         = "FAILED"


class Drone:
    def __init__(self, drone_id, comm_channel):
        self.id     = drone_id
        self.comm   = comm_channel
        self.state  = DroneState.IDLE
        self.color  = DRONE_COLORS[drone_id]

        angle  = 2.0 * math.pi * drone_id / NUM_DRONES
        self.x = SPAWN_RADIUS * math.cos(angle)
        self.y = SPAWN_RADIUS * math.sin(angle)
        self.z = _cfg.SPAWN_Z
        self.tx, self.ty, self.tz = self.x, self.y, self.z

        self.battery  = BATTERY_FULL
        self.problems = []
        self.is_marked = False

        self.last_heartbeat_time  = 0.0
        self.neighbors_positions  = {}
        self.inbox_log            = []
        self.total_distance       = 0.0
        self.collisions_avoided   = 0

        # Position de spawn (home NED pour PX4)
        self._spawn_x = SPAWN_RADIUS * math.cos(2.0 * math.pi * drone_id / NUM_DRONES)
        self._spawn_y = SPAWN_RADIUS * math.sin(2.0 * math.pi * drone_id / NUM_DRONES)
        self._px4_initialized = False
        self._low_battery_alerted = False

    # ═══════ COMMUNICATION ═══════════════════════════════════════════════════

    def send_heartbeat(self, sim_time):
        if sim_time - self.last_heartbeat_time < HEARTBEAT_INTERVAL:
            return
        self.last_heartbeat_time = sim_time
        self.comm.send(Message(self.id, "HEARTBEAT", {
            "x": round(self.x, 2), "y": round(self.y, 2), "z": round(self.z, 2),
            "battery": round(self.battery, 1), "state": self.state,
            "problems": self.problems.copy(),
        }))

    def send_alert(self, alert_type, data=None):
        self.comm.send(Message(self.id, "ALERT",
                               {"type": alert_type, **(data or {})}))

    def send_intruder_alert(self, ix, iy):
        self.comm.send(Message(self.id, "INTRUDER", {"x": ix, "y": iy}))

    def process_messages(self):
        for msg in self.comm.receive(self.id):
            if msg.msg_type == "HEARTBEAT":
                self.neighbors_positions[msg.sender_id] = (
                    msg.data["x"], msg.data["y"], msg.data["z"],
                    msg.data["battery"], msg.data["state"])
            elif msg.msg_type == "FORMATION_CMD":
                if "tx" in msg.data:
                    self.tx, self.ty = msg.data["tx"], msg.data["ty"]
                    self.tz = msg.data.get("tz", CRUISE_ALTITUDE)
                    if self.state == DroneState.HOVERING:
                        self.state = DroneState.NAVIGATING
            self.inbox_log.append(msg)
            if len(self.inbox_log) > 50:
                self.inbox_log.pop(0)

    # ═══════ ÉVITEMENT ════════════════════════════════════════════════════════

    def _compute_avoidance(self, other_drones):
        avx, avy, avz = obstacle_avoidance_vector(self.x, self.y, self.z)
        # Inter-drones
        if other_drones:
            for od in other_drones:
                if od.id == self.id or od.state in [
                        DroneState.FAILED, DroneState.GROUNDED, DroneState.LANDED]:
                    continue
                ddx = self.x - od.x
                ddy = self.y - od.y
                ddz = self.z - od.z
                d = math.sqrt(ddx**2 + ddy**2 + ddz**2)
                safe = SAFETY_DISTANCE_DRONES * 1.5
                if 0.0 < d < safe:
                    ratio = max(0.0, 1.0 - d / safe)
                    f = AVOIDANCE_FORCE * ratio
                    avx += (ddx / d) * f
                    avy += (ddy / d) * f
                    avz += (ddz / d) * f * 0.3
                    self.collisions_avoided += 1
        # Bords de zone
        m = 2.0
        if self.x < PATROL_ZONE["x_min"] + m: avx += AVOIDANCE_FORCE * 0.5
        if self.x > PATROL_ZONE["x_max"] - m: avx -= AVOIDANCE_FORCE * 0.5
        if self.y < PATROL_ZONE["y_min"] + m: avy += AVOIDANCE_FORCE * 0.5
        if self.y > PATROL_ZONE["y_max"] - m: avy -= AVOIDANCE_FORCE * 0.5
        return avx, avy, avz

    # ═══════ MOUVEMENT DIRECT (contrôle proportionnel) ════════════════════════

    def update(self, dt, other_drones=None, sim_time=0.0):
        """
        Met à jour la position via dead-reckoning et téléporte le modèle Gazebo
        en temps réel. Synchronisation directe Python → Gazebo à chaque pas.
        """
        if self.state in [DroneState.IDLE, DroneState.LANDED,
                          DroneState.FAILED, DroneState.GROUNDED]:
            return  # Position fixe, pas de téléport nécessaire

        # ── Synchronisation position depuis Gazebo (physique PX4 réelle) ─────
        try:
            gz_x, gz_y, gz_z, _ = get_gazebo_interface().get_pose(self.id)
            self.x, self.y, self.z = gz_x, gz_y, gz_z
        except Exception:
            pass  # Garder la position précédente si Gazebo non prêt

        self.send_heartbeat(sim_time)
        self.process_messages()

        # ── Décollage ─────────────────────────────────────────────────────
        if self.state == DroneState.TAKEOFF:
            # Activer PX4 offboard si pas encore fait
            if not self._px4_initialized:
                try:
                    get_px4_interface().arm_and_start_offboard(self.id)
                    self._px4_initialized = True
                except Exception:
                    pass
            # PX4 gère la montée — surveiller altitude réelle via Gazebo
            if self.z >= _cfg.CRUISE_ALTITUDE - 0.3:
                self.z = _cfg.CRUISE_ALTITUDE
                self.state = DroneState.HOVERING

        # ── Atterrissage ──────────────────────────────────────────────────
        elif self.state in [DroneState.LANDING, DroneState.EMERGENCY_LAND]:
            # PX4 gère la descente via mode Land
            if self.z <= _cfg.SPAWN_Z + 0.3:
                self.z = _cfg.SPAWN_Z
                self._px4_initialized = False
                self.state = (DroneState.GROUNDED
                              if self.state == DroneState.EMERGENCY_LAND
                              else DroneState.LANDED)

        # ── Navigation / Retour / Encerclement ───────────────────────────
        elif self.state in [DroneState.NAVIGATING, DroneState.RETURNING,
                             DroneState.ENCIRCLING]:
            dx = self.tx - self.x
            dy = self.ty - self.y
            dz = self.tz - self.z
            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            if dist > 0.05:
                speed = min(_cfg.MAX_SPEED, max(0.3, dist * 1.5))
                vx = (dx / dist) * speed
                vy = (dy / dist) * speed
                vz = (dz / dist) * speed

                avx, avy, avz = self._compute_avoidance(other_drones)
                # Cap avoidance to MAX_SPEED to prevent instability from overlapping obstacles
                _cap = _cfg.MAX_SPEED
                avx = max(-_cap, min(_cap, avx))
                avy = max(-_cap, min(_cap, avy))
                avz = max(-_cap * 0.5, min(_cap * 0.5, avz))
                vx += avx; vy += avy; vz += avz

                old_x, old_y, old_z = self.x, self.y, self.z
                self.x = max(PATROL_ZONE["x_min"],
                             min(PATROL_ZONE["x_max"], self.x + vx * dt))
                self.y = max(PATROL_ZONE["y_min"],
                             min(PATROL_ZONE["y_max"], self.y + vy * dt))
                self.z = max(_cfg.SPAWN_Z, self.z + vz * dt)
                self.total_distance += math.sqrt(
                    (self.x - old_x)**2 + (self.y - old_y)**2 + (self.z - old_z)**2)

            # RETURNING : atterrir à la base quand arrivé
            if self.state == DroneState.RETURNING and dist < POSITION_TOLERANCE:
                self.state = DroneState.LANDING
                self.tz = _cfg.SPAWN_Z

        # ── Hovering avec évitement passif ────────────────────────────────
        elif self.state == DroneState.HOVERING:
            avx, avy, avz = self._compute_avoidance(other_drones)
            if abs(avx) > 0.1 or abs(avy) > 0.1 or abs(avz) > 0.1:
                vx = avx * 0.3
                vy = avy * 0.3
                vz = avz * 0.3
                self.x = max(PATROL_ZONE["x_min"], min(PATROL_ZONE["x_max"], self.x + vx * dt))
                self.y = max(PATROL_ZONE["y_min"], min(PATROL_ZONE["y_max"], self.y + vy * dt))
                self.z = max(_cfg.SPAWN_Z, self.z + vz * dt)

        # ── Envoyer setpoint de position à PX4 (mode Offboard) ───────────
        # PX4 utilise les moteurs réels pour atteindre la position cible.
        # Conversion ENU (Gazebo) → NED (PX4) automatique.
        try:
            get_px4_interface().send_position_enu(self.id, self.x, self.y, self.z)
        except Exception:
            pass  # PX4 non prêt (mode dégradé)

        self.comm.update_position(self.id, self.x, self.y, self.z)

        # ── Batterie ──────────────────────────────────────────────────────
        if self.state not in [DroneState.IDLE, DroneState.LANDED,
                               DroneState.GROUNDED, DroneState.FAILED]:
            self.battery -= BATTERY_DRAIN_RATE
            if self.battery <= BATTERY_CRITICAL and self.state not in [
                    DroneState.RETURNING, DroneState.EMERGENCY_LAND, DroneState.LANDING]:
                if not self._low_battery_alerted:
                    self._low_battery_alerted = True
                    self.report_problem("LOW_BATTERY")
            if self.battery <= 0:
                self.battery = 0
                self.state = DroneState.FAILED

    # ═══════ COMMANDES ════════════════════════════════════════════════════════

    def takeoff(self):
        if self.state in [DroneState.IDLE, DroneState.LANDED] and not self.problems:
            self._px4_initialized = False  # Sera initialisé dans update()
            self.state = DroneState.TAKEOFF
            self.tz = _cfg.CRUISE_ALTITUDE

    def land(self):
        if self.state not in [DroneState.FAILED, DroneState.GROUNDED, DroneState.LANDED]:
            try:
                get_px4_interface().land(self.id)
            except Exception:
                pass
            self.state = DroneState.LANDING

    def go_to(self, x, y, z=None):
        if self.state in [DroneState.GROUNDED, DroneState.FAILED]:
            return
        self.tx, self.ty = x, y
        self.tz = z if z is not None else _cfg.CRUISE_ALTITUDE
        self.state = DroneState.NAVIGATING

    def return_base(self):
        if self.state not in [DroneState.GROUNDED, DroneState.FAILED]:
            self.state = DroneState.RETURNING
            self.tx, self.ty, self.tz = 0.0, 0.0, _cfg.CRUISE_ALTITUDE

    def report_problem(self, problem):
        if problem not in self.problems:
            self.problems.append(problem)
        self.is_marked = True
        self.send_alert(problem)
        if DRONE_PROBLEMS.get(problem, {}).get("auto_land", True):
            self.state = DroneState.EMERGENCY_LAND

    def repair(self):
        self.problems   = []
        self.is_marked  = False
        self.battery    = max(self.battery, 50.0)
        self._low_battery_alerted = False
        self.state      = DroneState.LANDED
        angle = 2.0 * math.pi * self.id / NUM_DRONES
        self.x = SPAWN_RADIUS * math.cos(angle)
        self.y = SPAWN_RADIUS * math.sin(angle)
        self.z = _cfg.SPAWN_Z
        # Téléporter physiquement dans Gazebo (reset position)
        get_gazebo_interface().teleport(self.id, self.x, self.y, self.z)
        # Resynchroniser le home NED de PX4 après téléportation
        try:
            get_px4_interface().teleport_reset(self.id, self.x, self.y, self.z)
        except Exception:
            pass
        self._px4_initialized = False
        self.comm.update_position(self.id, self.x, self.y, self.z)

    def clear_problems(self):
        self.repair()
