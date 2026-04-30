#!/usr/bin/env python3
"""
PFE - Swarm Drones MARL - Gazebo Harmonic
Simulation complète : formations exactes, intrus mobile,
encerclement centré, oiseaux dynamiques — portée identique à PyBullet.

Remplacements PyBullet → Gazebo :
  - Corps physiques        → modèles SDF (swarm_world.sdf)
  - resetBasePosition()    → gazebo_interface.set_velocity() + teleport()
  - addUserDebugLine/Text  → marqueurs RViz2 (visualization_msgs)
  - stepSimulation()       → Gazebo tourne en continu (pas de step manuel)
  - GUI/caméra             → Gazebo GUI intégré
"""
import sys, os
sys.path.insert(0, "/opt/ros/jazzy/lib/python3.12/site-packages")
os.environ.setdefault("AMENT_PREFIX_PATH", "/opt/ros/jazzy")

import time, math, threading
import numpy as np

# ── Import ROS2 pour marqueurs visuels ──────────────────────────────────────
import rclpy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

import config
from config import *
from drone import Drone, DroneState
from formations import FormationManager
from leader_election import LeaderElection
from communication import CommChannel, Message
from leader_brain import LeaderBrain
from remote_control import RemoteControl
from rl_controller import RLController
from drone_monitor import DroneMonitor
from gazebo_interface import init_gazebo, get_gazebo_interface


class SwarmSimulation:
    def __init__(self):
        # ── Initialiser Gazebo Interface ────────────────────────────────────
        print("  [init] Connexion à Gazebo Harmonic...")
        self.gz = init_gazebo(NUM_DRONES)
        print("  [init] Gazebo connecté")

        # ── Publisher marqueurs RViz2 ────────────────────────────────────────
        self._marker_pub = self.gz.node.create_publisher(
            MarkerArray, "/swarm/markers", 10)
        self._marker_id  = 0

        # ── Swarm logique ────────────────────────────────────────────────────
        self.comm          = CommChannel(NUM_DRONES)
        self.drones        = [Drone(i, self.comm) for i in range(NUM_DRONES)]
        self.formation_mgr = FormationManager(NUM_DRONES)
        self.election      = LeaderElection(NUM_DRONES)
        self.brain         = LeaderBrain(NUM_DRONES)
        self.formation_locked = False

        # ── Intrus mobiles ───────────────────────────────────────────────────
        self.intruders     = []
        self.intruder_speed = 1.0

        # ── Oiseaux (obstacles aériens dynamiques) ───────────────────────────
        self.birds = []
        self._init_birds()

        # ── Patrouille ───────────────────────────────────────────────────────
        self.patrol_altitude = PATROL_ALTITUDE
        self.patrol_radius   = PATROL_RADIUS
        self.patrol_wps      = self._gen_patrol_wps()
        self.patrol_idx      = 0
        self.patrolling      = False
        self.phase           = "INIT"

        # ── Target / Route personnalisé ──────────────────────────────────────
        self.custom_target    = None
        self.custom_route     = None
        self.custom_route_idx = 0
        self._form_decided    = False

        # ── Tracking mission ─────────────────────────────────────────────────
        self._mission_start_time  = None
        self._mission_start_real  = None
        self._mission_origin      = None
        self._mission_dest        = None
        self._mission_traj_type   = None
        self._mission_route_wps   = None
        self._mission_leader_dist = 0.0

        # ── Timing ──────────────────────────────────────────────────────────
        self.dt       = 1.0 / CONTROL_RATE
        self.sim_time = 0.0

        # ── Contrôleur MAPPO ────────────────────────────────────────────────
        self.mode = "rl"
        self.rl   = RLController()

        # Téléporter les drones à leur position initiale dans Gazebo
        for d in self.drones:
            self.gz.teleport(d.id, d.x, d.y, d.z)

        print("  [init] SwarmSimulation prête")

    # ═══════════════════════════════════════════════════════════════════════
    # OISEAUX
    # ═══════════════════════════════════════════════════════════════════════

    def _init_birds(self):
        """6 oiseaux volants — logique identique à PyBullet, visuels via Gazebo SDF."""
        bird_params = [
            {"x":  8.0, "y": -3.0, "z": 5.0, "speed": 2.5, "amp_z": 1.2,
             "freq": 0.3,  "turn": 0.15},
            {"x": -7.0, "y":  9.0, "z": 6.5, "speed": 3.0, "amp_z": 0.8,
             "freq": 0.45, "turn": -0.2},
            {"x": 14.0, "y":  4.0, "z": 4.5, "speed": 2.0, "amp_z": 1.8,
             "freq": 0.25, "turn": 0.1},
            {"x": -3.0, "y":-12.0, "z": 5.5, "speed": 3.5, "amp_z": 0.6,
             "freq": 0.5,  "turn": -0.12},
            {"x":  5.0, "y": 14.0, "z": 7.0, "speed": 2.8, "amp_z": 1.0,
             "freq": 0.35, "turn": 0.18},
            {"x":-12.0, "y": -5.0, "z": 5.0, "speed": 2.2, "amp_z": 1.5,
             "freq": 0.28, "turn": -0.25},
        ]
        rng = np.random.default_rng(42)
        for bp in bird_params:
            self.birds.append({
                "pos":       [bp["x"], bp["y"], bp["z"]],
                "speed":     bp["speed"],
                "amp_z":     bp["amp_z"],
                "freq":      bp["freq"],
                "base_z":    bp["z"],
                "angle_z":   rng.uniform(0, 6.28),
                "dir_angle": rng.uniform(0, 6.28),
                "turn_rate": bp["turn"],
            })
        config.BIRD_POSITIONS = [b["pos"] for b in self.birds]
        print(f"  🐦 {len(self.birds)} oiseaux — évitement aérien actif")

    def _update_birds(self):
        """Déplace chaque oiseau — rebond aux bords, sinusoïde verticale."""
        for idx, bird in enumerate(self.birds):
            bx, by, bz = bird["pos"]
            bird["dir_angle"] += bird["turn_rate"] * self.dt
            bx += math.cos(bird["dir_angle"]) * bird["speed"] * self.dt
            by += math.sin(bird["dir_angle"]) * bird["speed"] * self.dt
            bird["angle_z"] += bird["freq"] * self.dt
            bz = bird["base_z"] + bird["amp_z"] * math.sin(bird["angle_z"])

            if bx < PATROL_ZONE["x_min"] + 4:
                bird["dir_angle"] = math.pi - bird["dir_angle"]; bx = PATROL_ZONE["x_min"] + 4
            elif bx > PATROL_ZONE["x_max"] - 4:
                bird["dir_angle"] = math.pi - bird["dir_angle"]; bx = PATROL_ZONE["x_max"] - 4
            if by < PATROL_ZONE["y_min"] + 4:
                bird["dir_angle"] = -bird["dir_angle"]; by = PATROL_ZONE["y_min"] + 4
            elif by > PATROL_ZONE["y_max"] - 4:
                bird["dir_angle"] = -bird["dir_angle"]; by = PATROL_ZONE["y_max"] - 4
            bz = max(3.0, min(10.0, bz))

            bird["pos"] = [bx, by, bz]
            # Déplacer le modèle oiseau dans Gazebo (bird_{idx})
            self.gz.teleport_model(f"bird_{idx}", bx, by, bz)

        config.BIRD_POSITIONS = [b["pos"] for b in self.birds]

    # ═══════════════════════════════════════════════════════════════════════
    # PATROUILLE
    # ═══════════════════════════════════════════════════════════════════════

    def _gen_patrol_wps(self):
        wps = []
        r = self.patrol_radius
        for i in range(16):
            angle = 2 * math.pi * i / 16
            x = max(PATROL_ZONE["x_min"]+3, min(PATROL_ZONE["x_max"]-3, r * math.cos(angle)))
            y = max(PATROL_ZONE["y_min"]+3, min(PATROL_ZONE["y_max"]-3, r * math.sin(angle)))
            wps.append((x, y))
        return wps

    # ═══════════════════════════════════════════════════════════════════════
    # MARQUEURS RVIZ2 (remplace addUserDebugLine/Text PyBullet)
    # ═══════════════════════════════════════════════════════════════════════

    def _next_mid(self):
        self._marker_id += 1
        return self._marker_id

    def _pub_markers(self, markers: list[Marker]):
        ma = MarkerArray()
        ma.markers = markers
        self._marker_pub.publish(ma)

    def _make_line_marker(self, p1, p2, color, width=0.1, lifetime=0.0, ns="lines"):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = ns; m.id = self._next_mid()
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = width
        m.color = ColorRGBA(r=float(color[0]), g=float(color[1]), b=float(color[2]), a=1.0)
        m.pose.orientation.w = 1.0
        m.points = [Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])),
                    Point(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]))]
        if lifetime > 0:
            from builtin_interfaces.msg import Duration
            m.lifetime = Duration(sec=int(lifetime))
        return m

    def _make_text_marker(self, pos, text, color, size=0.8, ns="texts"):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = ns; m.id = self._next_mid()
        m.type = Marker.TEXT_VIEW_FACING; m.action = Marker.ADD
        m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        m.pose.orientation.w = 1.0
        m.scale.z = size
        m.color = ColorRGBA(r=float(color[0]), g=float(color[1]), b=float(color[2]), a=1.0)
        m.text = text
        return m

    def _make_sphere_marker(self, pos, radius, color, ns="spheres"):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = ns; m.id = self._next_mid()
        m.type = Marker.SPHERE; m.action = Marker.ADD
        m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = radius * 2
        m.color = ColorRGBA(r=float(color[0]), g=float(color[1]), b=float(color[2]), a=0.7)
        return m

    def _publish_status_markers(self):
        """Publie la position + état de chaque drone comme marqueur RViz2."""
        markers = []
        colors = [[1,0,0],[0,0,1],[0,0.8,0],[1,1,0]]
        for d in self.drones:
            c = colors[d.id % len(colors)]
            # Sphère de position
            markers.append(self._make_sphere_marker(
                [d.x, d.y, d.z], 0.35, c, ns=f"drone_{d.id}"))
            # Texte état
            ldr = "LEADER" if d.id == self.election.leader_id else f"D{d.id}"
            markers.append(self._make_text_marker(
                [d.x, d.y, d.z + 0.6],
                f"{ldr}\n{d.battery:.0f}%", c, size=0.5, ns=f"label_{d.id}"))
        # Intrus
        for k, intr in enumerate(self.intruders):
            ix, iy = intr["pos"]
            markers.append(self._make_sphere_marker(
                [ix, iy, CRUISE_ALTITUDE], 0.6, [1,0,0], ns=f"intruder_{k}"))
            markers.append(self._make_text_marker(
                [ix, iy, CRUISE_ALTITUDE + 1.0],
                f"INTRUS ({ix:.1f},{iy:.1f})", [1,0,0], size=0.5, ns=f"intrlabel_{k}"))
        # Oiseaux
        for k, bird in enumerate(self.birds):
            bx, by, bz = bird["pos"]
            markers.append(self._make_sphere_marker(
                [bx, by, bz], 0.3, [0.85,0.85,0.75], ns=f"bird_{k}"))
        self._pub_markers(markers)

    # ═══════════════════════════════════════════════════════════════════════
    # COMMANDES — identiques à la version PyBullet
    # ═══════════════════════════════════════════════════════════════════════

    def cmd_takeoff(self):
        self.phase = "TAKEOFF"
        n = 0
        for d in self.drones:
            if d.state in [DroneState.IDLE, DroneState.LANDED] and not d.problems:
                d.takeoff(); n += 1
        print(f"  🛫 Décollage de {n} drones")

    def cmd_land(self):
        self.phase = "LANDING"; self.patrolling = False
        for d in self.drones:
            if d.state not in [DroneState.FAILED, DroneState.GROUNDED,
                                DroneState.LANDED, DroneState.IDLE]:
                d.land()
        print("  🛬 Atterrissage")

    def cmd_stop(self):
        self.patrolling = False
        for d in self.drones:
            if d.state == DroneState.NAVIGATING:
                d.state = DroneState.HOVERING
        self.phase = "STOPPED"
        print("  🛑 Stop — drones en hover")

    def cmd_formation(self, name):
        if self.formation_mgr.set_formation(name):
            self.formation_locked = True
            self.phase = "FORMATION"
            leader_id = self.election.leader_id
            leader    = self.drones[leader_id]
            yaw       = math.atan2(leader.ty - leader.y, leader.tx - leader.x)
            targets   = self.formation_mgr.get_targets(
                (leader.x, leader.y, leader.z), yaw)
            for did, (tx, ty, tz) in targets.items():
                d = self.drones[did]
                if d.id == leader_id:
                    continue
                if d.state in [DroneState.HOVERING, DroneState.NAVIGATING]:
                    d.tx, d.ty, d.tz = tx, ty, tz
                    d.state = DroneState.NAVIGATING
            # Marqueurs RViz2 formation cible
            markers = []
            for did, (tx, ty, tz) in targets.items():
                markers.append(self._make_sphere_marker(
                    [tx, ty, tz], 0.4, [0.2,1,1], ns="formation_targets"))
            self._pub_markers(markers)
            print(f"  📐 Formation → {name}")
        else:
            print(f"  ❌ '{name}' inconnu. Formations: {', '.join(FORMATION_TYPES)}")

    def cmd_patrol(self):
        self.patrol_wps = self._gen_patrol_wps()
        self.patrolling = True
        self.patrol_idx = 0
        self.phase      = "PATROLLING"
        self.custom_target = None
        self.custom_route  = None
        print(f"  🗺️  Patrouille — r={self.patrol_radius:.0f}m "
              f"alt={self.patrol_altitude:.0f}m "
              f"formation={self.formation_mgr.formation}")
        # Cercle de patrouille en RViz2
        markers = []
        r = self.patrol_radius
        for i in range(32):
            a1 = 2*math.pi*i/32; a2 = 2*math.pi*(i+1)/32
            markers.append(self._make_line_marker(
                [r*math.cos(a1), r*math.sin(a1), 0.2],
                [r*math.cos(a2), r*math.sin(a2), 0.2],
                [0,1,0.5], width=0.08, ns="patrol_circle"))
        self._pub_markers(markers)

    def cmd_target(self, x, y):
        """Envoie tout le swarm vers un point précis."""
        self.patrolling    = False
        self.custom_route  = None
        self.custom_target = (x, y)
        self._form_decided = False
        self.phase         = "TARGET"
        leader = self.drones[self.election.leader_id]
        leader.tx, leader.ty, leader.tz = x, y, config.CRUISE_ALTITUDE
        if leader.state == DroneState.HOVERING:
            leader.state = DroneState.NAVIGATING
        # Followers immédiatement en mouvement
        yaw     = math.atan2(y - leader.y, x - leader.x)
        targets = self.formation_mgr.get_targets((leader.x, leader.y, leader.z), yaw)
        for did, (tx, ty, tz) in targets.items():
            d = self.drones[did]
            if d.id == self.election.leader_id:
                continue
            if d.state in [DroneState.HOVERING, DroneState.NAVIGATING]:
                d.tx, d.ty, d.tz = tx, ty, tz
                d.state = DroneState.NAVIGATING
        # Marqueur cible RViz2
        alt = config.CRUISE_ALTITUDE
        markers = [
            self._make_line_marker([x-2.5, y, 0.3], [x+2.5, y, 0.3], [1,0.5,0], 0.1, ns="target"),
            self._make_line_marker([x, y-2.5, 0.3], [x, y+2.5, 0.3], [1,0.5,0], 0.1, ns="target"),
            self._make_line_marker([leader.x, leader.y, alt], [x, y, alt], [1,0.65,0], 0.06, ns="traj"),
            self._make_text_marker([x, y, alt+1.8], f"CIBLE ({x:.0f},{y:.0f})",
                                   [1,0.8,0], size=0.8, ns="target_label"),
        ]
        self._pub_markers(markers)
        # Tracking mission
        self._mission_start_time  = self.sim_time
        self._mission_start_real  = time.time()
        self._mission_origin      = (leader.x, leader.y)
        self._mission_dest        = (x, y)
        self._mission_traj_type   = "direct"
        self._mission_route_wps   = None
        self._mission_leader_dist = 0.0
        print(f"  🎯 Target → ({x:.1f}, {y:.1f})  [formation {self.formation_mgr.formation}]")

    def cmd_route(self, waypoints):
        """Envoie le swarm suivre un trajet personnalisé."""
        self.patrolling       = False
        self.custom_target    = None
        self.custom_route     = waypoints
        self.custom_route_idx = 0
        self._form_decided    = False
        self.phase            = "ROUTE"
        print(f"  🗺️  Route personnalisée : {len(waypoints)} waypoints")
        for idx, (wx, wy) in enumerate(waypoints):
            print(f"     WP{idx+1}: ({wx:.1f}, {wy:.1f})")
        # Tracking mission
        leader = self.drones[self.election.leader_id]
        self._mission_start_time  = self.sim_time
        self._mission_start_real  = time.time()
        self._mission_origin      = (leader.x, leader.y)
        self._mission_dest        = waypoints[-1] if waypoints else None
        self._mission_traj_type   = "route"
        self._mission_route_wps   = list(waypoints)
        self._mission_leader_dist = 0.0
        # Marqueurs RViz2 route
        alt = config.CRUISE_ALTITUDE
        markers = []
        for i in range(len(waypoints) - 1):
            markers.append(self._make_line_marker(
                [waypoints[i][0], waypoints[i][1], alt],
                [waypoints[i+1][0], waypoints[i+1][1], alt],
                [0.3,0.9,1], 0.06, ns="route"))
            markers.append(self._make_text_marker(
                [waypoints[i][0], waypoints[i][1], alt+1.2],
                f"WP{i+1}", [0.3,0.9,1], size=0.6, ns="route_labels"))
        if waypoints:
            markers.append(self._make_text_marker(
                [waypoints[-1][0], waypoints[-1][1], alt+1.2],
                f"WP{len(waypoints)}", [0.3,0.9,1], size=0.6, ns="route_labels"))
        self._pub_markers(markers)

    # ═══════════════════════════════════════════════════════════════════════
    # INTRUS MOBILES
    # ═══════════════════════════════════════════════════════════════════════

    def cmd_intruder(self, x, y, formation="random", target_drones=None):
        """Lance un intrus et l'encercle immédiatement."""
        wps = self._gen_intruder_waypoints(x, y, formation)
        if target_drones is None:
            already = set()
            for intr in self.intruders:
                if intr.get("assigned_drones"):
                    already.update(intr["assigned_drones"])
            target_drones = [d.id for d in self.drones
                             if d.state not in [DroneState.FAILED, DroneState.GROUNDED,
                                                DroneState.LANDED, DroneState.IDLE]
                             and d.id not in already]
        idx = len(self.intruders)
        intruder = {
            "id":             idx,
            "pos":            [x, y],
            "speed":          self.intruder_speed,
            "waypoints":      wps,
            "wp_idx":         0,
            "formation":      formation,
            "angle":          0.0,
            "detected":       True,
            "start_pos":      [x, y],
            "target_drones":  target_drones,
            "assigned_drones":target_drones,
            "spawn_real":     time.time(),
            "spawn_sim":      self.sim_time,
            "spawn_pos":      [x, y],
        }
        self.intruders.append(intruder)
        # Déplacer le modèle intruder_{idx} dans Gazebo
        self.gz.teleport_model(f"intruder_{idx}", x, y, CRUISE_ALTITUDE)
        # Encerclement immédiat
        self.patrolling    = False
        self.custom_target = None
        self.custom_route  = None
        self.phase         = "ENCIRCLING"
        self._assign_drones_to_intruder(intruder)
        names = [f"D{d}" for d in target_drones]
        print(f"  🔴 INTRUS @ ({x:.0f},{y:.0f}) [{formation}] → {', '.join(names)} encerclent")

    def _assign_drones_to_intruder(self, intruder):
        ix, iy   = intruder["pos"]
        assigned = intruder.get("assigned_drones", [])
        n        = len(assigned)
        r        = ENCIRCLE_RADIUS
        for j, did in enumerate(assigned):
            if did < 0 or did >= NUM_DRONES:
                continue
            d = self.drones[did]
            if d.state in [DroneState.FAILED, DroneState.GROUNDED,
                           DroneState.LANDED, DroneState.IDLE]:
                continue
            angle  = 2.0 * math.pi * j / max(n, 1)
            d.tx   = ix + r * math.cos(angle)
            d.ty   = iy + r * math.sin(angle)
            d.tz   = config.CRUISE_ALTITUDE
            d.state = DroneState.ENCIRCLING

    def _gen_intruder_waypoints(self, x, y, formation):
        wps = []
        if formation == "straight":
            wps = [(x+15, y), (x-15, y), (x+15, y)]
        elif formation == "zigzag":
            for i in range(8):
                wps.append((x + (i-4)*5, y + (5 if i%2==0 else -5)))
        elif formation == "circle":
            r = 8
            for i in range(12):
                a = 2*math.pi*i/12
                wps.append((x + r*math.cos(a), y + r*math.sin(a)))
        elif formation == "evasive":
            for _ in range(10):
                wps.append((
                    np.random.uniform(PATROL_ZONE["x_min"]+5, PATROL_ZONE["x_max"]-5),
                    np.random.uniform(PATROL_ZONE["y_min"]+5, PATROL_ZONE["y_max"]-5),
                ))
        else:  # random
            for _ in range(10):
                wps.append((
                    np.random.uniform(max(PATROL_ZONE["x_min"]+5, x-15),
                                      min(PATROL_ZONE["x_max"]-5, x+15)),
                    np.random.uniform(max(PATROL_ZONE["y_min"]+5, y-15),
                                      min(PATROL_ZONE["y_max"]-5, y+15)),
                ))
        return wps

    def _update_intruder(self):
        for intruder in self.intruders:
            ix, iy = intruder["pos"]
            fm     = intruder["formation"]
            spd    = intruder["speed"]
            if fm == "circle":
                sx, sy = intruder["start_pos"]
                r = 8
                intruder["angle"] += spd * self.dt * 0.1
                ix = sx + r * math.cos(intruder["angle"])
                iy = sy + r * math.sin(intruder["angle"])
            elif intruder["waypoints"]:
                wx, wy = intruder["waypoints"][intruder["wp_idx"]]
                dx, dy = wx - ix, wy - iy
                dist   = math.sqrt(dx**2 + dy**2)
                if dist < 1.0:
                    intruder["wp_idx"] = (intruder["wp_idx"]+1) % len(intruder["waypoints"])
                else:
                    s = spd * self.dt
                    ix += (dx/dist)*s; iy += (dy/dist)*s
            ix = max(PATROL_ZONE["x_min"]+2, min(PATROL_ZONE["x_max"]-2, ix))
            iy = max(PATROL_ZONE["y_min"]+2, min(PATROL_ZONE["y_max"]-2, iy))
            intruder["pos"] = [ix, iy]
            # Déplacer modèle dans Gazebo
            self.gz.teleport_model(f"intruder_{intruder['id']}", ix, iy, CRUISE_ALTITUDE)

    def cmd_clear_intruder(self):
        import datetime
        n = len(self.intruders)
        for intruder in self.intruders:
            now_str  = datetime.datetime.now().strftime("%H:%M:%S")
            elapsed  = time.time() - intruder.get("spawn_real", time.time())
            sim_dur  = self.sim_time - intruder.get("spawn_sim", self.sim_time)
            sx, sy   = intruder.get("spawn_pos", intruder["pos"])
            ix, iy   = intruder["pos"]
            dist_trav= math.sqrt((ix-sx)**2 + (iy-sy)**2)
            assigned = intruder.get("assigned_drones", [])
            drones_s = ", ".join(f"D{d}" for d in assigned) if assigned else "—"
            lid      = self.election.leader_id
            report   = (
                f"{'═'*48}\n"
                f"  RAPPORT NEUTRALISATION — {now_str}\n"
                f"{'─'*48}\n"
                f"  Leader          : Drone {lid}\n"
                f"  Intrus détecté  : ({sx:.1f}, {sy:.1f})\n"
                f"  Position finale : ({ix:.1f}, {iy:.1f})\n"
                f"  Distance intrus : {dist_trav:.1f} m\n"
                f"  Durée encercl.  : {elapsed:.1f}s  (sim {sim_dur:.1f}s)\n"
                f"  Drones assignés : {drones_s}\n"
                f"  Nb drones       : {len(assigned)}/{NUM_DRONES}\n"
                f"  Formation       : ENCIRCLE\n"
                f"  Statut          : NEUTRALISE\n"
                f"{'═'*48}\n"
            )
            print(report)
            if hasattr(self, "monitor") and self.monitor.running:
                try: self.monitor.add_mission_report(report)
                except Exception: pass
            # Cacher le modèle Gazebo loin de la scène
            self.gz.teleport_model(f"intruder_{intruder['id']}", 999, 999, -10)
        self.intruders = []
        self.formation_mgr.intruder_pos = None
        for d in self.drones:
            if d.state == DroneState.ENCIRCLING:
                d.state = DroneState.NAVIGATING
        self.phase = "FORMATION"
        print(f"  ✅ {n} intrus neutralisé(s) → retour formation {self.formation_mgr.formation}")

    # ═══════════════════════════════════════════════════════════════════════
    # PANNES
    # ═══════════════════════════════════════════════════════════════════════

    def cmd_kill(self, did):
        if 0 <= did < NUM_DRONES:
            d = self.drones[did]
            d.report_problem("MOTOR_FAILURE")
            was_leader = (did == self.election.leader_id)
            self.election.on_drone_problem(did)
            if was_leader:
                nl = self.election.elect(self.drones)
                self.formation_mgr.leader_id = nl
                print(f"  👑 Nouveau leader: drone_{nl} [{DRONE_COLOR_NAMES[nl]}]")
            print(f"  💥 drone_{did} → PANNE")

    def cmd_mark(self, did, problem):
        if 0 <= did < NUM_DRONES:
            self.drones[did].report_problem(problem)
            self.election.on_drone_problem(did)
            if did == self.election.leader_id:
                nl = self.election.elect(self.drones)
                self.formation_mgr.leader_id = nl
            print(f"  ⚠️  drone_{did} marqué: {problem}")

    def cmd_unmark(self, did):
        if 0 <= did < NUM_DRONES:
            d = self.drones[did]
            d.repair()
            self.election.on_drone_recovered(did)
            flying = any(dr.state in [DroneState.HOVERING, DroneState.NAVIGATING,
                                       DroneState.ENCIRCLING]
                         for dr in self.drones if dr.id != did)
            if flying:
                d.takeoff()
                print(f"  🔧 drone_{did} RÉPARÉ + REDÉCOLLAGE !")
            else:
                print(f"  🔧 drone_{did} RÉPARÉ")

    def cmd_send(self, did, command):
        if did < 0 or did >= NUM_DRONES:
            print(f"  ❌ drone_{did} n'existe pas"); return
        d   = self.drones[did]
        cmd = command.upper()
        if   cmd == "TAKEOFF": d.takeoff(); print(f"  🛫 drone_{did} décollage")
        elif cmd == "LAND":    d.land();    print(f"  🛬 drone_{did} atterrissage")
        elif cmd == "HOVER":   d.state = DroneState.HOVERING; print(f"  ⏸ drone_{did} hover")
        elif cmd == "RETURN":  d.return_base(); print(f"  🏠 drone_{did} retour base")
        else: print(f"  ❌ {cmd}. Options: takeoff/land/hover/return")

    def cmd_speed(self, value=None):
        if value is None:
            print(f"  🏎️  Vitesse: {config.MAX_SPEED:.1f} m/s")
        else:
            v = max(0.5, min(10.0, value))
            config.MAX_SPEED = v
            print(f"  🏎️  Vitesse → {v:.1f} m/s")

    def cmd_altitude(self, value=None):
        if value is None:
            print(f"  ✈️  Altitude: {config.CRUISE_ALTITUDE:.1f} m"); return
        h = max(2.0, min(25.0, value))
        config.CRUISE_ALTITUDE = h
        n_updated = 0
        for d in self.drones:
            if d.state in [DroneState.HOVERING, DroneState.NAVIGATING,
                           DroneState.ENCIRCLING, DroneState.TAKEOFF]:
                d.tz = h
                if d.state == DroneState.HOVERING:
                    d.state = DroneState.NAVIGATING
                n_updated += 1
        if self.custom_target:
            self.drones[self.election.leader_id].tz = h
        print(f"  ✈️  Altitude → {h:.1f} m  ({n_updated} drones)")

    # ═══════════════════════════════════════════════════════════════════════
    # INFOS
    # ═══════════════════════════════════════════════════════════════════════

    def cmd_status(self):
        leader   = self.election.leader_id
        active   = sum(1 for d in self.drones if d.state not in
                       [DroneState.FAILED, DroneState.GROUNDED,
                        DroneState.LANDED, DroneState.IDLE])
        avg_bat  = sum(d.battery for d in self.drones) / NUM_DRONES
        cs       = self.comm.get_stats()
        print(f"""
╔══════════════════════════════════════════════════════╗
║  Mode  : {self.mode.upper():<12} Phase: {self.phase:<12}           ║
║  Form  : {self.formation_mgr.formation:<12} Leader: D{leader} [{DRONE_COLOR_NAMES[leader]:<8}] ║
║  Actifs: {active}/{NUM_DRONES}  Bat:{avg_bat:3.0f}%  Comm: {cs['delivery_rate']:.0f}%{'':<14}║""")
        if self.intruders:
            for idx, intr in enumerate(self.intruders):
                ix, iy = intr["pos"]
                det    = "OUI" if intr["detected"] else "NON"
                print(f"║  INTRUS#{idx+1}: ({ix:5.1f},{iy:5.1f}) Dét:{det} [{intr['formation']}] ║")
        print(f"╚══════════════════════════════════════════════════════╝")
        for d in self.drones:
            ldr = "👑" if d.id == leader else "  "
            prb = f" [{','.join(d.problems)}]" if d.problems else ""
            print(f"  {ldr}D{d.id} {d.state:<11} bat={d.battery:5.1f}% "
                  f"pos=({d.x:5.1f},{d.y:5.1f},{d.z:4.1f}){prb}")

    def cmd_drone_detail(self, did):
        if 0 <= did < NUM_DRONES:
            d    = self.drones[did]
            dist = math.sqrt((d.x-d.tx)**2+(d.y-d.ty)**2+(d.z-d.tz)**2)
            print(f"\n  Drone_{did} [{DRONE_COLOR_NAMES[did]}] — {d.state}")
            print(f"  Position  : ({d.x:.2f}, {d.y:.2f}, {d.z:.2f})")
            print(f"  Cible     : ({d.tx:.2f}, {d.ty:.2f}, {d.tz:.2f}) dist={dist:.2f}m")
            print(f"  Batterie  : {d.battery:.1f}%")
            print(f"  Évitements: {d.collisions_avoided}")
            print(f"  Distance  : {d.total_distance:.1f}m")
            if d.problems: print(f"  Problèmes : {d.problems}")

    def cmd_comm_status(self):
        cs = self.comm.get_stats()
        print(f"\n  📡 COMMUNICATION")
        print(f"  Envoyés: {cs['total_sent']}  Livrés: {cs['total_delivered']}  "
              f"Hors portée: {cs['total_out_of_range']}")
        print(f"  Taux livraison: {cs['delivery_rate']:.1f}%")

    def cmd_metrics(self):
        print("\n  📊 MÉTRIQUES RL (config.py)")
        for k, v in METRICS.items():
            print(f"    {k:<25} = {v}")

    def cmd_env(self):
        print(f"\n  🌍 ENVIRONNEMENT (Gazebo SDF)")
        print(f"  Zone: {PATROL_ZONE['x_min']}→{PATROL_ZONE['x_max']} x "
              f"{PATROL_ZONE['y_min']}→{PATROL_ZONE['y_max']}")
        print(f"  Bâtiments: {len(OBSTACLES)}, Arbres: {len(TREES)}, Murs: {len(WALLS)}")

    def cmd_explain(self):
        print(f"""
  ARCHITECTURE Gazebo :
    config.py          → Paramètres (vitesse, formations, obstacles...)
    communication.py   → Canal comm (portée {COMM_RANGE}m, heartbeat)
    drone.py           → 1 drone : mouvement cmd_vel, évitement, batterie
    formations.py      → 11 formations géométriques exactes
    leader_election.py → Élection du leader (score batterie+position)
    gazebo_interface.py→ Pont ROS2 ↔ Gazebo (cmd_vel, pose, teleport)
    main.py            → Simulation + commandes

  GAZEBO :
    Monde SDF  : worlds/swarm_world.sdf
    Drone SDF  : models/drone/model.sdf
    Launch     : launch/swarm.launch.py
    Contrôle   : VelocityControl plugin → cmd_vel topics

  COMMUNICATION :
    Heartbeat toutes les {HEARTBEAT_INTERVAL}s → position + batterie + état
    Portée limitée à {COMM_RANGE}m

  INTRUS : 5 mouvements
    random, straight, zigzag, circle, evasive
    Ex: intruder 10 5 zigzag
""")

    # ═══════════════════════════════════════════════════════════════════════
    # DÉTECTION CONTINUE D'INTRUS
    # ═══════════════════════════════════════════════════════════════════════

    def _any_intruder_detected(self):
        return any(i["detected"] for i in self.intruders)

    def _get_active_drone_ids(self):
        return [d.id for d in self.drones
                if d.state not in [DroneState.FAILED, DroneState.GROUNDED,
                                   DroneState.LANDED, DroneState.IDLE]]

    def _detect_intruders_continuous(self):
        detected   = [i for i in self.intruders if i["detected"]]
        undetected = [i for i in self.intruders if not i["detected"]]
        for intruder in undetected:
            ix, iy = intruder["pos"]
            for d in self.drones:
                if d.state in [DroneState.FAILED, DroneState.GROUNDED,
                               DroneState.LANDED, DroneState.IDLE]:
                    continue
                dist = math.sqrt((d.x-ix)**2 + (d.y-iy)**2)
                if dist < SENSOR_RANGE:
                    intruder["detected"] = True
                    detected.append(intruder)
                    print(f"\n  🚨 INTRUS DÉTECTÉ par drone_{d.id} @ ({ix:.1f},{iy:.1f})")
                    d.send_intruder_alert(ix, iy)
                    break
        if not detected:
            return
        active_ids  = self._get_active_drone_ids()
        n_intruders = len(detected)
        n_active    = len(active_ids)
        if n_intruders == 0 or n_active == 0:
            return
        base  = n_active // n_intruders
        extra = n_active % n_intruders
        idx   = 0
        for k, intruder in enumerate(detected):
            count    = base + (1 if k < extra else 0)
            assigned = active_ids[idx:idx+count]
            idx += count
            intruder["assigned_drones"] = assigned
            ix, iy   = intruder["pos"]
            n_a      = len(assigned)
            r        = ENCIRCLE_RADIUS
            for j, did in enumerate(assigned):
                angle  = 2.0 * math.pi * j / max(n_a, 1)
                d      = self.drones[did]
                d.tx   = ix + r * math.cos(angle)
                d.ty   = iy + r * math.sin(angle)
                d.tz   = config.CRUISE_ALTITUDE
                if d.state != DroneState.ENCIRCLING:
                    d.state = DroneState.ENCIRCLING
        self.phase = "ENCIRCLING"
        if self.patrolling:
            self.patrolling = False

    # ═══════════════════════════════════════════════════════════════════════
    # RAPPORT MISSION (identique à PyBullet)
    # ═══════════════════════════════════════════════════════════════════════

    def _leader_decide_formation(self, arrival_x, arrival_y):
        import datetime
        leader_id  = self.election.leader_id
        leader     = self.drones[leader_id]
        active_ids = self._get_active_drone_ids()
        n_active   = len(active_ids)
        from leader_brain import FORMATION_BY_N
        chosen = FORMATION_BY_N.get(n_active, "v_shape")
        if not self.formation_locked:
            self.formation_mgr.set_formation(chosen)
        else:
            chosen = self.formation_mgr.formation
        yaw     = math.atan2(arrival_y - leader.y, arrival_x - leader.x)
        targets = self.formation_mgr.get_targets((arrival_x, arrival_y, leader.z), yaw)
        for did, (tx, ty, tz) in targets.items():
            d = self.drones[did]
            if d.id == leader_id: continue
            if d.state in [DroneState.FAILED, DroneState.GROUNDED,
                           DroneState.LANDED, DroneState.IDLE]: continue
            d.tx, d.ty, d.tz = tx, ty, tz
            if d.state == DroneState.HOVERING:
                d.state = DroneState.NAVIGATING
        self._send_mission_report(chosen, arrival_x, arrival_y)

    def _send_mission_report(self, formation_chosen, ax, ay):
        import datetime
        if self._mission_start_time is None:
            return
        now_str    = datetime.datetime.now().strftime("%H:%M:%S")
        elapsed_r  = time.time() - (self._mission_start_real or time.time())
        elapsed_s  = self.sim_time - (self._mission_start_time or self.sim_time)
        origin     = self._mission_origin or (0, 0)
        dest       = self._mission_dest or (ax, ay)
        straight   = math.sqrt((dest[0]-origin[0])**2 + (dest[1]-origin[1])**2)
        efficiency = (straight / max(self._mission_leader_dist, 0.1)) * 100
        active_ids = self._get_active_drone_ids()
        lid        = self.election.leader_id
        report = (
            f"{'═'*52}\n"
            f"  RAPPORT MISSION — {now_str}\n"
            f"{'─'*52}\n"
            f"  Leader          : Drone {lid} [{DRONE_COLOR_NAMES[lid]}]\n"
            f"  Origine         : ({origin[0]:.1f}, {origin[1]:.1f})\n"
            f"  Destination     : ({dest[0]:.1f}, {dest[1]:.1f})\n"
            f"  Distance vol    : {self._mission_leader_dist:.1f} m\n"
            f"  Efficacité      : {efficiency:.0f}%\n"
            f"  Durée réelle    : {elapsed_r:.1f} s\n"
            f"  Durée sim       : {elapsed_s:.1f} s\n"
            f"  Formation arrivée: {formation_chosen}\n"
            f"  Drones actifs   : {len(active_ids)}/{NUM_DRONES}\n"
            f"{'═'*52}\n"
        )
        print(report)
        if hasattr(self, "monitor") and self.monitor.running:
            try: self.monitor.add_mission_report(report)
            except Exception: pass
        self._mission_start_time = None

    # ═══════════════════════════════════════════════════════════════════════
    # BOUCLE PRINCIPALE — step() identique à PyBullet
    # ═══════════════════════════════════════════════════════════════════════

    def step(self):
        self.sim_time += self.dt

        leader = self.drones[self.election.leader_id]

        # ═══ 0. CIBLES LEADER/FOLLOWERS avant d.update() ═══
        if not self._any_intruder_detected():
            if self.custom_target:
                leader.tx, leader.ty, leader.tz = (
                    self.custom_target[0], self.custom_target[1], config.CRUISE_ALTITUDE)
                if leader.state == DroneState.HOVERING:
                    leader.state = DroneState.NAVIGATING
                ldx    = self.custom_target[0] - leader.x
                ldy    = self.custom_target[1] - leader.y
                ldist  = max(0.1, math.sqrt(ldx**2 + ldy**2))
                ahead  = min(ldist, 4.0)
                form_c = (leader.x + (ldx/ldist)*ahead,
                          leader.y + (ldy/ldist)*ahead,
                          leader.z)
                yaw_pre = math.atan2(ldy, ldx)
                pre_tgt = self.formation_mgr.get_targets(form_c, yaw_pre)
                for fdid, (ftx, fty, ftz) in pre_tgt.items():
                    fd = self.drones[fdid]
                    if fd.id == leader.id: continue
                    if fd.state in [DroneState.FAILED, DroneState.GROUNDED,
                                    DroneState.LANDED, DroneState.IDLE,
                                    DroneState.EMERGENCY_LAND]: continue
                    fd.tx, fd.ty, fd.tz = ftx, fty, ftz
                    if fd.state == DroneState.HOVERING:
                        fd.state = DroneState.NAVIGATING

            elif self.custom_route and self.custom_route_idx < len(self.custom_route):
                wp = self.custom_route[self.custom_route_idx]
                leader.tx, leader.ty, leader.tz = wp[0], wp[1], config.CRUISE_ALTITUDE
                if leader.state == DroneState.HOVERING:
                    leader.state = DroneState.NAVIGATING
                ldx   = wp[0] - leader.x
                ldy   = wp[1] - leader.y
                ldist = max(0.1, math.sqrt(ldx**2 + ldy**2))
                ahead = min(ldist, 4.0)
                form_c  = (leader.x + (ldx/ldist)*ahead,
                           leader.y + (ldy/ldist)*ahead, leader.z)
                yaw_pre = math.atan2(ldy, ldx)
                pre_tgt = self.formation_mgr.get_targets(form_c, yaw_pre)
                for fdid, (ftx, fty, ftz) in pre_tgt.items():
                    fd = self.drones[fdid]
                    if fd.id == leader.id: continue
                    if fd.state in [DroneState.FAILED, DroneState.GROUNDED,
                                    DroneState.LANDED, DroneState.IDLE,
                                    DroneState.EMERGENCY_LAND]: continue
                    fd.tx, fd.ty, fd.tz = ftx, fty, ftz
                    if fd.state == DroneState.HOVERING:
                        fd.state = DroneState.NAVIGATING

        # Sauvegarder position leader avant update
        _prev_lx, _prev_ly = leader.x, leader.y

        # ═══ 1. UPDATE PHYSIQUE DE TOUS LES DRONES ═══
        for d in self.drones:
            d.update(self.dt, self.drones, self.sim_time)

        # Accumuler distance leader
        if self._mission_start_time is not None and (self.custom_target or self.custom_route):
            _ldr = self.drones[self.election.leader_id]
            self._mission_leader_dist += math.sqrt(
                (_ldr.x - _prev_lx)**2 + (_ldr.y - _prev_ly)**2)

        # ═══ 1b. CERVEAU DU LEADER ═══
        if not self.custom_target and not self.custom_route:
            self.brain.tick(
                self.sim_time, self.drones, self.intruders,
                self.formation_mgr, self.election, self.comm, self)

        leader = self.drones[self.election.leader_id]

        # ═══ 1c. MOUVEMENT LEADER (target / route / patrol) ═══
        if not self._any_intruder_detected():
            if self.custom_target:
                leader.tx, leader.ty, leader.tz = (
                    self.custom_target[0], self.custom_target[1], config.CRUISE_ALTITUDE)
                if leader.state == DroneState.HOVERING:
                    leader.state = DroneState.NAVIGATING
                dist = math.sqrt((leader.x - self.custom_target[0])**2 +
                                 (leader.y - self.custom_target[1])**2)
                if dist < 2.0 and not self._form_decided:
                    ax, ay = self.custom_target
                    print(f"  ✅ Target atteint ({ax:.1f},{ay:.1f})")
                    self._form_decided = True
                    self.custom_target = None
                    self.phase         = "FORMATION"
                    self._leader_decide_formation(ax, ay)

            elif self.custom_route:
                wp = self.custom_route[self.custom_route_idx]
                leader.tx, leader.ty, leader.tz = wp[0], wp[1], config.CRUISE_ALTITUDE
                if leader.state == DroneState.HOVERING:
                    leader.state = DroneState.NAVIGATING
                dist = math.sqrt((leader.x - wp[0])**2 + (leader.y - wp[1])**2)
                if dist < 2.0:
                    self.custom_route_idx += 1
                    if self.custom_route_idx >= len(self.custom_route):
                        self.custom_route_idx = 0
                    nwp = self.custom_route[self.custom_route_idx]
                    print(f"  📍 WP{self.custom_route_idx+1} → ({nwp[0]:.1f},{nwp[1]:.1f})")

            elif self.patrolling and self.patrol_wps:
                wp = self.patrol_wps[self.patrol_idx]
                leader.tx, leader.ty, leader.tz = wp[0], wp[1], self.patrol_altitude
                if leader.state == DroneState.HOVERING:
                    leader.state = DroneState.NAVIGATING
                dist = math.sqrt((leader.x - wp[0])**2 + (leader.y - wp[1])**2)
                if dist < 2.5:
                    self.patrol_idx = (self.patrol_idx + 1) % len(self.patrol_wps)

        # ═══ 2. INTRUS & OISEAUX ═══
        self._update_intruder()
        self._detect_intruders_continuous()
        self._update_birds()

        # ═══ 3. FORMATION FOLLOWERS ═══
        encircling_ids = set()
        for intr in self.intruders:
            if intr["detected"] and "assigned_drones" in intr:
                encircling_ids.update(intr["assigned_drones"])

        if leader.state in [DroneState.HOVERING, DroneState.NAVIGATING]:
            if self.custom_target:
                ldx   = self.custom_target[0] - leader.x
                ldy   = self.custom_target[1] - leader.y
                ldist = max(0.1, math.sqrt(ldx**2 + ldy**2))
                ahead = min(ldist, 4.0)
                form_c = (leader.x + (ldx/ldist)*ahead,
                          leader.y + (ldy/ldist)*ahead, leader.z)
            elif self.custom_route and self.custom_route_idx < len(self.custom_route):
                wp    = self.custom_route[self.custom_route_idx]
                ldx   = wp[0] - leader.x
                ldy   = wp[1] - leader.y
                ldist = max(0.1, math.sqrt(ldx**2 + ldy**2))
                ahead = min(ldist, 4.0)
                form_c = (leader.x + (ldx/ldist)*ahead,
                          leader.y + (ldy/ldist)*ahead, leader.z)
            else:
                form_c = (leader.x, leader.y, leader.z)

            yaw      = math.atan2(leader.ty - leader.y, leader.tx - leader.x)
            form_tgt = self.formation_mgr.get_targets(form_c, yaw)
            for fdid, (ftx, fty, ftz) in form_tgt.items():
                if fdid in encircling_ids: continue
                fd = self.drones[fdid]
                if fd.id == self.election.leader_id: continue
                if fd.state in [DroneState.FAILED, DroneState.GROUNDED,
                                DroneState.LANDED, DroneState.IDLE,
                                DroneState.EMERGENCY_LAND]: continue
                fd.tx, fd.ty, fd.tz = ftx, fty, ftz
                if fd.state == DroneState.HOVERING:
                    fd.state = DroneState.NAVIGATING

        # ═══ 4. CONTRÔLEUR MAPPO ═══
        rl_targets = self.rl.compute_actions(self.drones, self.formation_mgr, self.election)
        leader_id  = self.election.leader_id
        for did, (tx, ty, tz) in rl_targets.items():
            if did in encircling_ids: continue
            d = self.drones[did]
            if d.state in [DroneState.FAILED, DroneState.GROUNDED,
                           DroneState.LANDED, DroneState.IDLE]: continue
            if self.custom_target or self.custom_route:
                if d.state == DroneState.HOVERING:
                    d.state = DroneState.NAVIGATING
                continue
            if self.patrolling and did == leader_id:
                if d.state == DroneState.HOVERING:
                    d.state = DroneState.NAVIGATING
                continue
            d.tx, d.ty, d.tz = tx, ty, tz
            if d.state == DroneState.HOVERING:
                d.state = DroneState.NAVIGATING

        # ═══ 5. MARQUEURS RVIZ2 (toutes les 0.5 s) ═══
        if int(self.sim_time * 2) != int((self.sim_time - self.dt) * 2):
            self._publish_status_markers()

        # ═══ 6. FEED MONITOR ═══
        if hasattr(self, "monitor") and self.monitor.running:
            self.monitor.notify_step(self.drones, self.sim_time)

    # ═══════════════════════════════════════════════════════════════════════
    # BOUCLE RUN + COMMANDES TERMINAL
    # ═══════════════════════════════════════════════════════════════════════

    def run(self):
        self.remote = RemoteControl(self)
        threading.Thread(target=self._command_loop, daemon=True).start()
        threading.Thread(target=self._sim_loop, daemon=True).start()
        print("\n" + "="*55)
        print("  🚁 SIMULATION DRONE SWARM — Gazebo Harmonic")
        print("  📺 Télécommande + 📡 Moniteur ouverts")
        print("  Tapez 'help' pour les commandes")
        print("="*55 + "\n")
        self.monitor = DroneMonitor(self)
        self.monitor.start_main()

    def _sim_loop(self):
        """Boucle simulation @ CONTROL_RATE Hz dans un thread séparé."""
        period = 1.0 / CONTROL_RATE
        while True:
            t0 = time.time()
            self.step()
            elapsed = time.time() - t0
            sleep   = period - elapsed
            if sleep > 0:
                time.sleep(sleep)

    def _command_loop(self):
        time.sleep(1); self._show_help()
        while True:
            try:
                raw   = input("\n🎮 swarm> ").strip()
                if not raw: continue
                parts = raw.split(); cmd = parts[0].lower()

                if   cmd in ["takeoff","t"]:   self.cmd_takeoff()
                elif cmd in ["land","l"]:       self.cmd_land()
                elif cmd in ["stop"]:           self.cmd_stop()
                elif cmd in ["form","f"]:
                    if len(parts) > 1:
                        self.cmd_formation(parts[1])
                        flying = any(d.state in [DroneState.HOVERING, DroneState.NAVIGATING]
                                     for d in self.drones)
                        if flying and not self.patrolling and not self._any_intruder_detected():
                            self.cmd_patrol()
                    else:
                        print(f"  Formations: {', '.join(FORMATION_TYPES)}")
                elif cmd in ["patrol","p"]:     self.cmd_patrol()
                elif cmd in ["target","tg"]:
                    if len(parts) >= 3:
                        self.cmd_target(float(parts[1]), float(parts[2]))
                    else:
                        print("  Usage: target <x> <y>")
                elif cmd in ["route","rt"]:
                    if len(parts) >= 2:
                        wps = []
                        for pair in parts[1:]:
                            xy = pair.split(",")
                            if len(xy) == 2:
                                wps.append((float(xy[0]), float(xy[1])))
                        if wps:
                            self.cmd_route(wps)
                        else:
                            print("  Usage: route x1,y1 x2,y2 ...")
                    else:
                        print("  Usage: route x1,y1 x2,y2 x3,y3 ...")
                elif cmd in ["intruder","i"]:
                    if len(parts) >= 3:
                        fm = parts[3] if len(parts) >= 4 else "random"
                        self.cmd_intruder(float(parts[1]), float(parts[2]), fm)
                    else:
                        print("  Usage: intruder <x> <y> [random|straight|zigzag|circle|evasive]")
                elif cmd in ["clear_intruder","ci"]: self.cmd_clear_intruder()
                elif cmd in ["ispeed"]:
                    if len(parts) > 1:
                        self.intruder_speed = max(0.5, min(5.0, float(parts[1])))
                        print(f"  🔴 Vitesse intrus → {self.intruder_speed:.1f} m/s")
                elif cmd in ["send"]:
                    if len(parts) >= 3:
                        self.cmd_send(int(parts[1]), parts[2])
                    else:
                        print("  Usage: send <id> <takeoff|land|hover|return>")
                elif cmd in ["goto","g"]:
                    if len(parts) >= 4:
                        self.drones[int(parts[1])].go_to(float(parts[2]), float(parts[3]))
                        print(f"  drone_{parts[1]} → ({parts[2]}, {parts[3]})")
                    else:
                        print("  Usage: goto <id> <x> <y>")
                elif cmd in ["kill","k"]:
                    if len(parts) > 1:
                        t = parts[1]
                        self.cmd_kill(self.election.leader_id if t == "leader" else int(t))
                    else:
                        print("  Usage: kill <id|leader>")
                elif cmd in ["mark","m"]:
                    if len(parts) >= 3:
                        self.cmd_mark(int(parts[1]), parts[2].upper())
                    else:
                        print("  Usage: mark <id> <problème>")
                elif cmd in ["unmark","u"]:
                    if len(parts) > 1:
                        self.cmd_unmark(int(parts[1]))
                    else:
                        print("  Usage: unmark <id>")
                elif cmd in ["status","s"]:   self.cmd_status()
                elif cmd in ["drone","d"]:
                    if len(parts) > 1:
                        self.cmd_drone_detail(int(parts[1]))
                    else:
                        print("  Usage: drone <id>")
                elif cmd in ["bat"]:           self._show_batteries()
                elif cmd in ["pos"]:           self._show_positions()
                elif cmd in ["prob"]:          self._show_problems()
                elif cmd in ["speed","v"]:
                    if len(parts) > 1:
                        self.cmd_speed(float(parts[1]))
                    else:
                        self.cmd_speed()
                elif cmd in ["alt","altitude","a"]:
                    if len(parts) > 1:
                        self.cmd_altitude(float(parts[1]))
                    else:
                        self.cmd_altitude()
                elif cmd in ["comm"]:          self.cmd_comm_status()
                elif cmd in ["metrics"]:       self.cmd_metrics()
                elif cmd in ["env"]:           self.cmd_env()
                elif cmd in ["explain"]:       self.cmd_explain()
                elif cmd in ["palt"]:
                    if len(parts) > 1:
                        self.patrol_altitude = max(2.0, min(25.0, float(parts[1])))
                        print(f"  🛫 Altitude patrouille → {self.patrol_altitude:.1f}m")
                elif cmd in ["prad"]:
                    if len(parts) > 1:
                        self.patrol_radius = max(3.0, min(23.0, float(parts[1])))
                        self.patrol_wps    = self._gen_patrol_wps()
                        self.patrol_idx    = 0
                        print(f"  🗺️  Rayon patrouille → {self.patrol_radius:.1f}m")
                elif cmd in ["brain","br"]:
                    log = self.brain.get_log(15)
                    print(f"\n  👑 CERVEAU LEADER — État : {self.brain.state_label}")
                    print(f"  Formation actuelle : {self.formation_mgr.formation}")
                    print("  Décisions récentes :")
                    for entry in log:
                        print(f"    • {entry}")
                    if not log:
                        print("    (aucune décision — décollage requis)")
                elif cmd in ["rlinfo","ri"]:
                    info    = self.rl.get_last_info()
                    trained = "OUI" if info["trained"] else f"NON ({info['mode']})"
                    print(f"\n  🧠 RL INFO — Sous-mode: {info['mode']}")
                    print(f"  Modèle entraîné  : {trained}")
                    if info["rewards"]:
                        for did2, r in sorted(info["rewards"].items()):
                            bar = "+"*int(max(0,r)) if r > 0 else "-"*int(min(10,abs(r)))
                            print(f"    D{did2}: {r:+7.3f}  {bar}")
                elif cmd in ["reward","rw"]:   self.rl.print_reward_function()
                elif cmd in ["rlstatus","rs"]: self.rl.check_training_status()
                elif cmd in ["mode"]:
                    print(f"  🧠 Mode: MAPPO [{'RÉSEAU' if self.rl.is_trained() else 'DÉMO'}]")
                elif cmd in ["help","h"]:      self._show_help()
                elif cmd in ["quit","q"]:
                    print("  👋"); os._exit(0)
                else:
                    print(f"  ❌ '{cmd}' inconnu. Tapez 'help'")
            except (ValueError, IndexError) as e:
                print(f"  ❌ Erreur: {e}")
            except EOFError:
                break

    def _show_batteries(self):
        for d in self.drones:
            bar = "█"*int(d.battery/5) + "░"*(20-int(d.battery/5))
            ldr = "👑" if d.id == self.election.leader_id else "  "
            print(f"  {ldr}D{d.id}: [{bar}] {d.battery:5.1f}%")

    def _show_positions(self):
        for d in self.drones:
            ldr = "👑" if d.id == self.election.leader_id else "  "
            print(f"  {ldr}D{d.id}: ({d.x:7.1f},{d.y:7.1f},{d.z:5.1f}) [{d.state}]")

    def _show_problems(self):
        any_p = False
        for d in self.drones:
            if d.problems:
                any_p = True
                print(f"  🔴 D{d.id}: {','.join(d.problems)} [{d.state}]")
        if not any_p:
            print("  ✅ Aucun problème")

    def _show_help(self):
        print("""
╔═══════════════════════════════════════════════════════════════════╗
║                 🎮 COMMANDES — Gazebo Swarm                       ║
╠═══════════════════════════════════════════════════════════════════╣
║  takeoff(t)  land(l)  stop  form <type>(f)  patrol(p)            ║
║  target <x> <y>(tg)  route x1,y1 x2,y2 ...(rt)                  ║
║  send <id> <cmd>    goto <id> <x> <y>(g)   drone <id>(d)         ║
║  intruder <x> <y> [mouvement](i)   clear_intruder(ci)            ║
║  ispeed <v>  speed <v>(v)  alt <h>(a)                            ║
║  kill <id|leader>(k)   mark <id> <prob>(m)  unmark <id>(u)       ║
║  status(s)   bat   pos   prob   comm   env   explain              ║
║  palt <h>   prad <r>   brain(br)   rlinfo(ri)   rlstatus(rs)     ║
║  mode   reward(rw)   help(h)   quit(q)                            ║
║                                                                   ║
║  Formations: v_shape line circle triangle diamond grid            ║
║    column echelon_left echelon_right wedge encircle               ║
╚═══════════════════════════════════════════════════════════════════╝""")


if __name__ == "__main__":
    import os
    sim = SwarmSimulation()
    sim.run()
