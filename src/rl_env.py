"""
PFE - Swarm Drones MARL
Environnement RL avec CURRICULUM LEARNING (5 niveaux)

Niveau 1 : 1 drone  → suivre une trajectoire simple, pas d'obstacles
Niveau 2 : 2 drones → follower maintient distance, leader fixe
Niveau 3 : 3 drones → formation triangle, leader fixe
Niveau 4 : N drones → leader dynamique, formation se réorganise
Niveau 5 : N drones → obstacles + communication limitée + tout

Chaque niveau ajoute de la complexité progressivement.

CORRECTIONS ÉVITEMENT OBSTACLES (v2) :
  1. obs_size étendu : +8 floats pour vecteurs direction obstacles (dx,dy normalisés)
     → le drone sait maintenant DANS QUELLE DIRECTION est chaque obstacle
  2. _find_safe_target() : les cibles de formation sont poussées hors des obstacles
     → supprime le signal contradictoire (aller vers la cible = foncer dans l'obstacle)
  3. Pénalité obstacle à deux zones :
     - gradient doux (signal anticipatif) pour dist < SAFETY_DISTANCE_OBSTACLES
     - pénalité hard x3 pour dist < 0.5m (collision réelle)
  4. Push physique dans step() : le drone ne peut plus traverser un obstacle
     → collision impossible même si la politique est sous-optimale
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import config as _cfg
from config import *


class SwarmFormationEnv(gym.Env):
    """Environnement multi-agent avec curriculum learning."""

    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, num_drones=NUM_DRONES, render_mode=None,
                 target_formation="circle", use_pybullet=False, level=1):
        super().__init__()

        self.max_drones = num_drones
        self.render_mode = render_mode
        self.target_formation = target_formation
        self.use_pybullet = use_pybullet
        self.client = None

        # ══════ CURRICULUM ══════
        self.level = level
        self.n = self._get_num_drones_for_level(level)

        # ══════ ESPACES ══════
        # Observation par drone :
        #   position (3) + cible (3) + voisins (3*(N-1)) + obstacles (4 dist + 4*2 dir) + bat (1) + infos (2)
        # FIX: ajout des vecteurs direction obstacles (8 floats) pour permettre le contournement directionnel
        self.obs_size = 3 + 3 + 3 * (num_drones - 1) + 4 + 8 + 1 + 2

        # Action par drone : vx, vy, vz dans [-1, 1]
        self.action_size = 3

        # Espaces Gymnasium (taille max, on remplit de 0 pour les drones inactifs)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_drones, self.obs_size), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_drones, self.action_size), dtype=np.float32)

        # ══════ ÉTAT ══════
        self.positions = np.zeros((num_drones, 3), dtype=np.float32)
        self.velocities = np.zeros((num_drones, 3), dtype=np.float32)
        self.batteries = np.full(num_drones, BATTERY_FULL, dtype=np.float32)
        self.alive = np.zeros(num_drones, dtype=bool)  # Seulement self.n actifs
        self.formation_targets = np.zeros((num_drones, 3), dtype=np.float32)

        # Leader
        self.leader_id = 0
        self.leader_change_count = 0

        # Communication simulée
        self.comm_active = np.ones((num_drones, num_drones), dtype=bool)

        # Compteurs
        self.step_count = 0
        self.max_steps = self._get_max_steps_for_level(level)
        self.episode_count = 0

        # Normalisation
        self.pos_scale = 25.0

        # Obstacles actifs selon le niveau
        self.active_obstacles = self._get_obstacles_for_level(level)

        # Stats par épisode
        self.episode_collisions = 0
        self.episode_formations_reached = 0

        if self.use_pybullet:
            self._init_pybullet()

    # ═══════════════════════════════════
    # HELPERS SÉCURITÉ OBSTACLES
    # ═══════════════════════════════════

    def _dist_to_obstacle(self, px, py, obs_data):
        """Distance depuis un point (px,py) jusqu'à la face la plus proche d'un obstacle."""
        dx = max(0.0, abs(px - obs_data["x"]) - obs_data["sx"] / 2)
        dy = max(0.0, abs(py - obs_data["y"]) - obs_data["sy"] / 2)
        return math.sqrt(dx**2 + dy**2)

    def _point_in_obstacle(self, px, py, margin=1.5):
        """Retourne True si (px,py) est dans un obstacle (+ marge de sécurité)."""
        for obs_data in self.active_obstacles:
            if self._dist_to_obstacle(px, py, obs_data) < margin:
                return True
        return False

    def _find_safe_target(self, px, py, pz, max_tries=20):
        """
        FIX: Si (px,py) tombe dans un obstacle, déplace le point vers
        une position sûre en le poussant depuis le centre de l'obstacle.
        """
        if not self.active_obstacles or not self._point_in_obstacle(px, py):
            return np.array([px, py, pz], dtype=np.float32)

        for _ in range(max_tries):
            # Trouver l'obstacle le plus proche
            min_dist = float("inf")
            nearest = None
            for obs_data in self.active_obstacles:
                d = self._dist_to_obstacle(px, py, obs_data)
                if d < min_dist:
                    min_dist = d
                    nearest = obs_data
            if nearest is None:
                break
            # Pousser le point hors de l'obstacle (direction depuis le centre)
            dx = px - nearest["x"]
            dy = py - nearest["y"]
            norm = math.sqrt(dx**2 + dy**2) + 1e-6
            push = nearest["sx"] / 2 + nearest["sy"] / 2 + 2.0  # demi-largeur + marge
            px = nearest["x"] + (dx / norm) * push
            py = nearest["y"] + (dy / norm) * push
            # Clip dans la zone de patrouille
            px = float(np.clip(px, PATROL_ZONE["x_min"] + 1, PATROL_ZONE["x_max"] - 1))
            py = float(np.clip(py, PATROL_ZONE["y_min"] + 1, PATROL_ZONE["y_max"] - 1))
            if not self._point_in_obstacle(px, py):
                break

        return np.array([px, py, pz], dtype=np.float32)

    # ═══════════════════════════════════
    # CURRICULUM LEARNING
    # ═══════════════════════════════════

    def set_level(self, level):
        """Change le niveau du curriculum."""
        self.level = min(level, 5)
        self.n = self._get_num_drones_for_level(self.level)
        self.max_steps = self._get_max_steps_for_level(self.level)
        self.active_obstacles = self._get_obstacles_for_level(self.level)

    def _get_num_drones_for_level(self, level):
        """Nombre de drones actifs par niveau."""
        if level == 1: return 1
        elif level == 2: return 2
        elif level == 3: return 3
        elif level == 4: return self.max_drones
        elif level == 5: return self.max_drones
        return self.max_drones

    def _get_max_steps_for_level(self, level):
        """Durée max d'un épisode par niveau."""
        if level == 1: return 400   # augmenté : drone peut être à 9m, besoin de 180+ steps
        elif level == 2: return 500
        elif level == 3: return 600
        elif level == 4: return 700
        elif level == 5: return 700
        return 700

    def _get_obstacles_for_level(self, level):
        """Obstacles actifs par niveau."""
        if level <= 3: return []          # Niveaux 1-3 : pas d'obstacles
        elif level == 4: return OBSTACLES[:5]  # Niveau 4 : 5 obstacles (introduction douce)
        return OBSTACLES                  # Niveau 5 : tous les obstacles

    def _get_formations_for_level(self, level):
        """Formations disponibles par niveau."""
        if level == 1: return ["point"]  # 1 drone = juste un point cible
        elif level == 2: return ["line"]  # 2 drones = follower derrière
        elif level == 3: return ["triangle", "line", "v_shape"]
        elif level == 4: return ["circle", "v_shape", "line", "grid", "diamond",
                                  "triangle", "wedge"]
        elif level == 5: return ["circle", "v_shape", "line", "grid", "diamond",
                                  "triangle", "column", "echelon_left",
                                  "echelon_right", "wedge", "encircle"]
        return FORMATION_TYPES

    def _get_leader_mode_for_level(self, level):
        """Mode du leader par niveau."""
        if level <= 3: return "fixed"  # Leader toujours drone_0
        return "dynamic"  # Leader peut changer

    # ═══════════════════════════════════
    # PYBULLET VISUALISATION
    # ═══════════════════════════════════

    def _init_pybullet(self):
        import pybullet as p
        import pybullet_data
        self.p = p
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        else:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

        # ── Sol vert ──
        ground = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeVisualShape(ground, -1, rgbaColor=[0.3, 0.5, 0.3, 1],
                            physicsClientId=self.client)

        # ── Bordure zone de patrouille (lignes bleues) ──
        z = PATROL_ZONE
        corners = [(z["x_min"], z["y_min"]), (z["x_max"], z["y_min"]),
                   (z["x_max"], z["y_max"]), (z["x_min"], z["y_max"])]
        for i in range(4):
            x1, y1 = corners[i]; x2, y2 = corners[(i + 1) % 4]
            p.addUserDebugLine([x1, y1, 0.1], [x2, y2, 0.1], [0, 0.5, 1], 2, 0,
                               physicsClientId=self.client)

        # ── Base centrale ──
        bv = p.createVisualShape(p.GEOM_CYLINDER, radius=2, length=0.1,
                                 rgbaColor=[0.3, 0.3, 0.3, 0.8],
                                 physicsClientId=self.client)
        p.createMultiBody(0, -1, bv, [0, 0, 0.05], physicsClientId=self.client)
        p.addUserDebugText("BASE", [0, 0, 0.3], [1, 1, 0], 1.2,
                           physicsClientId=self.client)

        # ── Lac (rectangle bleu plat, zone Sud-Centre) ──
        lake_vis = p.createVisualShape(p.GEOM_BOX,
                                       halfExtents=[6.0, 4.0, 0.05],
                                       rgbaColor=[0.1, 0.4, 0.9, 0.85],
                                       physicsClientId=self.client)
        p.createMultiBody(0, -1, lake_vis, [-2.0, -10.0, 0.02],
                          physicsClientId=self.client)
        p.addUserDebugText("LAC", [-2.0, -10.0, 0.3], [0.2, 0.6, 1.0], 1.0,
                           physicsClientId=self.client)

        # ── Délimitation zone militaire (cadre rouge, Sud-Ouest) ──
        mil_corners = [(-25, -25), (-8, -25), (-8, -5), (-25, -5)]
        for i in range(4):
            mx1, my1 = mil_corners[i]; mx2, my2 = mil_corners[(i + 1) % 4]
            p.addUserDebugLine([mx1, my1, 0.15], [mx2, my2, 0.15],
                               [0.8, 0.1, 0.1], 2, 0, physicsClientId=self.client)
        p.addUserDebugText("ZONE MILITAIRE", [-20, -23, 0.5],
                           [1, 0.2, 0.2], 1.1, physicsClientId=self.client)

        # ── Bâtiments avec labels ──
        self.obs_bodies = []
        for obs in OBSTACLES:
            col = p.createCollisionShape(p.GEOM_BOX,
                halfExtents=[obs["sx"] / 2, obs["sy"] / 2, obs["h"] / 2],
                physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX,
                halfExtents=[obs["sx"] / 2, obs["sy"] / 2, obs["h"] / 2],
                rgbaColor=obs.get("color", [0.5, 0.5, 0.5, 0.9]),
                physicsClientId=self.client)
            body = p.createMultiBody(0, col, vis,
                [obs["x"], obs["y"], obs["h"] / 2],
                physicsClientId=self.client)
            self.obs_bodies.append(body)
            p.addUserDebugText(obs["name"],
                [obs["x"], obs["y"], obs["h"] + 0.3],
                [1, 1, 1], 0.7, physicsClientId=self.client)

        # ── Arbres (tronc + feuillage) ──
        for t in TREES:
            trunk = p.createVisualShape(p.GEOM_CYLINDER,
                radius=t["r"] * 0.3, length=t["h"] * 0.6,
                rgbaColor=[0.4, 0.25, 0.1, 1], physicsClientId=self.client)
            p.createMultiBody(0, -1, trunk, [t["x"], t["y"], t["h"] * 0.3],
                              physicsClientId=self.client)
            foliage = p.createVisualShape(p.GEOM_SPHERE,
                radius=t["r"] + 0.8,
                rgbaColor=[0.1, 0.5, 0.1, 0.9], physicsClientId=self.client)
            p.createMultiBody(0, -1, foliage, [t["x"], t["y"], t["h"] * 0.7],
                              physicsClientId=self.client)

        # ── Murs ──
        for wall in WALLS:
            wx = (wall["x1"] + wall["x2"]) / 2
            wy = (wall["y1"] + wall["y2"]) / 2
            wsx = abs(wall["x2"] - wall["x1"]) + 0.3
            wsy = abs(wall["y2"] - wall["y1"]) + 0.3
            wh = wall["h"]
            vis = p.createVisualShape(p.GEOM_BOX,
                halfExtents=[wsx / 2, wsy / 2, wh / 2],
                rgbaColor=wall.get("color", [0.6, 0.6, 0.6, 0.8]),
                physicsClientId=self.client)
            p.createMultiBody(0, -1, vis, [wx, wy, wh / 2],
                              physicsClientId=self.client)

        # ── Drones (modèle complet : corps + bras + moteurs + sphère ID) ──
        self.drone_bodies = []
        for i in range(self.max_drones):
            color = DRONE_COLORS[i]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.04],
                                         physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.04],
                                      rgbaColor=color, physicsClientId=self.client)
            arm_pos = [[0.2, -0.2, 0], [0.2, 0.2, 0], [-0.2, -0.2, 0], [-0.2, 0.2, 0]]
            lc, lv, lp, lo, lm, li, lpar, ljt, lja = [], [], [], [], [], [], [], [], []
            for apos in arm_pos:
                ac = p.createCollisionShape(p.GEOM_BOX,
                    halfExtents=[0.12, 0.01, 0.005], physicsClientId=self.client)
                av = p.createVisualShape(p.GEOM_BOX,
                    halfExtents=[0.12, 0.01, 0.005], rgbaColor=[0.2, 0.2, 0.2, 1],
                    physicsClientId=self.client)
                lc.append(ac); lv.append(av); lp.append(apos); lo.append([0, 0, 0, 1])
                lm.append(0.05); li.append([0, 0, 0]); lpar.append(0)
                ljt.append(p.JOINT_FIXED); lja.append([0, 0, 0])
                mc = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03,
                                            physicsClientId=self.client)
                mv = p.createVisualShape(p.GEOM_SPHERE, radius=0.03,
                    rgbaColor=[0.1, 0.1, 0.1, 1], physicsClientId=self.client)
                lc.append(mc); lv.append(mv); lp.append([apos[0], apos[1], 0.05])
                lo.append([0, 0, 0, 1]); lm.append(0.02); li.append([0, 0, 0])
                lpar.append(0); ljt.append(p.JOINT_FIXED); lja.append([0, 0, 0])
            ic = p.createCollisionShape(p.GEOM_SPHERE, radius=0.06,
                                        physicsClientId=self.client)
            iv = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=color,
                                     physicsClientId=self.client)
            lc.append(ic); lv.append(iv); lp.append([0, 0, 0.12])
            lo.append([0, 0, 0, 1]); lm.append(0.01); li.append([0, 0, 0])
            lpar.append(0); ljt.append(p.JOINT_FIXED); lja.append([0, 0, 0])
            body = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                basePosition=[0, 0, -10],
                linkMasses=lm, linkCollisionShapeIndices=lc, linkVisualShapeIndices=lv,
                linkPositions=lp, linkOrientations=lo, linkInertialFramePositions=li,
                linkInertialFrameOrientations=lo, linkParentIndices=lpar,
                linkJointTypes=ljt, linkJointAxis=lja, physicsClientId=self.client)
            self.drone_bodies.append(body)

        # ── Marqueurs de cibles (sphères blanches) ──
        self.target_bodies = []
        for i in range(self.max_drones):
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,
                                      rgbaColor=[1, 1, 1, 0.3],
                                      physicsClientId=self.client)
            body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                     basePosition=[0, 0, -10],
                                     physicsClientId=self.client)
            self.target_bodies.append(body)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(40, 45, -60, [0, 0, 0],
                                          physicsClientId=self.client)

    # ═══════════════════════════════════
    # FORMATION TARGETS
    # ═══════════════════════════════════

    def _compute_formation_targets(self, center=(0, 0, 5)):
        cx, cy, cz = center
        s = FORMATION_SPACING
        targets = np.zeros((self.max_drones, 3), dtype=np.float32)
        n = self.n  # Nombre de drones actifs

        if self.target_formation == "point" or n == 1:
            # Niveau 1 : un seul point cible
            targets[0] = [cx, cy, cz]

        elif self.target_formation == "circle":
            for i in range(n):
                angle = 2 * math.pi * i / n
                r = s * 1.5
                targets[i] = [cx + r * math.cos(angle),
                              cy + r * math.sin(angle), cz]

        elif self.target_formation == "v_shape":
            targets[0] = [cx, cy, cz]
            left, right = 0, 0
            for i in range(1, n):
                if left <= right:
                    left += 1
                    targets[i] = [cx - s * left, cy - s * left, cz]
                else:
                    right += 1
                    targets[i] = [cx - s * right, cy + s * right, cz]

        elif self.target_formation == "line":
            for i in range(n):
                targets[i] = [cx - s * i, cy, cz]

        elif self.target_formation == "grid":
            cols = max(1, math.ceil(math.sqrt(n)))
            for i in range(n):
                row = i // cols
                col = i % cols
                targets[i] = [cx - row * s, cy + (col - cols / 2) * s, cz]

        elif self.target_formation == "diamond":
            positions = [
                (0, 0), (-s, -s), (-s, s), (-2*s, 0),
                (-2*s, -2*s), (-2*s, 2*s), (-3*s, -s), (-3*s, s)]
            for i in range(min(n, len(positions))):
                targets[i] = [cx + positions[i][0], cy + positions[i][1], cz]

        elif self.target_formation == "triangle":
            targets[0] = [cx, cy, cz]
            idx = 0
            for i in range(1, n):
                idx += 1
                row, count = 0, 0
                while count + row + 2 <= idx:
                    count += row + 2
                    row += 1
                col = idx - count
                cols = row + 2
                targets[i] = [cx - (row + 1) * s,
                              cy + (col - (cols - 1) / 2.0) * s, cz]

        elif self.target_formation == "column":
            for i in range(n):
                targets[i] = [cx - s * i, cy, cz]

        elif self.target_formation == "echelon_left":
            targets[0] = [cx, cy, cz]
            for i in range(1, n):
                targets[i] = [cx - s * i, cy - s * i * 0.5, cz]

        elif self.target_formation == "echelon_right":
            targets[0] = [cx, cy, cz]
            for i in range(1, n):
                targets[i] = [cx - s * i, cy + s * i * 0.5, cz]

        elif self.target_formation == "wedge":
            targets[0] = [cx, cy, cz]
            left, right = 0, 0
            for i in range(1, n):
                if left <= right:
                    left += 1
                    targets[i] = [cx - s * left * 0.7, cy - s * left * 0.7, cz]
                else:
                    right += 1
                    targets[i] = [cx - s * right * 0.7, cy + s * right * 0.7, cz]

        elif self.target_formation == "encircle":
            for i in range(n):
                angle = 2 * math.pi * i / n
                r = ENCIRCLE_RADIUS
                targets[i] = [cx + r * math.cos(angle),
                              cy + r * math.sin(angle), cz]

        self.formation_targets = targets

        # FIX: s'assurer qu'aucune cible ne tombe dans un obstacle
        if self.active_obstacles:
            for i in range(n):
                safe = self._find_safe_target(
                    targets[i, 0], targets[i, 1], targets[i, 2])
                targets[i] = safe
        self.formation_targets = targets
        return targets

    # ═══════════════════════════════════
    # GYM INTERFACE
    # ═══════════════════════════════════

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_count += 1
        self.episode_collisions = 0
        self.episode_formations_reached = 0
        self._prev_dists = None  # Reset pour éviter le bonus spurieux au 1er pas

        # Appliquer le niveau si passé dans options
        if options and "level" in options:
            self.set_level(options["level"])

        # Drones actifs
        self.alive = np.zeros(self.max_drones, dtype=bool)
        self.alive[:self.n] = True
        self.batteries = np.full(self.max_drones, BATTERY_FULL, dtype=np.float32)
        self.positions = np.zeros((self.max_drones, 3), dtype=np.float32)
        self.velocities = np.zeros((self.max_drones, 3), dtype=np.float32)

        # Positions initiales
        for i in range(self.n):
            angle = 2 * math.pi * i / max(1, self.n) + np.random.uniform(-0.3, 0.3)
            r = SPAWN_RADIUS + np.random.uniform(-1, 1)
            self.positions[i] = [r * math.cos(angle),
                                 r * math.sin(angle),
                                 _cfg.CRUISE_ALTITUDE + np.random.uniform(-0.5, 0.5)]

        # Drones inactifs hors vue
        for i in range(self.n, self.max_drones):
            self.positions[i] = [0, 0, -10]

        # Leader
        self.leader_id = 0
        self.leader_change_count = 0

        # Communication (niveau 5 = portée limitée)
        self.comm_active = np.ones((self.max_drones, self.max_drones), dtype=bool)
        if self.level >= 5:
            self._update_comm_range()

        # Formation
        if options and "formation" in options and options["formation"]:
            self.target_formation = options["formation"]
        else:
            formations = self._get_formations_for_level(self.level)
            self.target_formation = np.random.choice(formations)

        # Centre aléatoire (plus simple aux niveaux bas)
        if self.level <= 2:
            center = (np.random.uniform(-3, 3), np.random.uniform(-3, 3), _cfg.CRUISE_ALTITUDE)
        else:
            center = (np.random.uniform(-8, 8), np.random.uniform(-8, 8), _cfg.CRUISE_ALTITUDE)

        self._compute_formation_targets(center)

        # Centre de formation indépendant (suit les waypoints)
        self._formation_center = np.array([center[0], center[1], _cfg.CRUISE_ALTITUDE],
                                          dtype=np.float32)
        self._leader_just_changed = False

        # Waypoints de patrouille pour le centre de formation
        self._gen_leader_waypoints()

        # Visualisation
        if self.use_pybullet:
            for i in range(self.max_drones):
                pos = self.positions[i].tolist() if self.alive[i] else [0, 0, -10]
                self.p.resetBasePositionAndOrientation(
                    self.drone_bodies[i], pos, [0, 0, 0, 1],
                    physicsClientId=self.client)
                tpos = self.formation_targets[i].tolist() if self.alive[i] else [0, 0, -10]
                self.p.resetBasePositionAndOrientation(
                    self.target_bodies[i], tpos, [0, 0, 0, 1],
                    physicsClientId=self.client)
            # Montrer/cacher obstacles
            for idx, body in enumerate(self.obs_bodies):
                if idx < len(self.active_obstacles):
                    obs = OBSTACLES[idx]
                    self.p.resetBasePositionAndOrientation(
                        body, [obs["x"], obs["y"], obs["h"]/2], [0, 0, 0, 1],
                        physicsClientId=self.client)
                else:
                    self.p.resetBasePositionAndOrientation(
                        body, [0, 0, -20], [0, 0, 0, 1],
                        physicsClientId=self.client)

        obs = self._get_obs()
        info = {"formation": self.target_formation, "level": self.level,
                "num_drones": self.n}
        return obs, info

    def step(self, actions):
        self.step_count += 1
        actions = np.clip(actions, -1, 1)

        # Appliquer les actions (seulement drones actifs)
        for i in range(self.n):
            if not self.alive[i]:
                continue
            self.velocities[i] = actions[i] * _cfg.MAX_SPEED
            self.positions[i] += self.velocities[i] * (1.0 / CONTROL_RATE)

            # Limiter à la zone
            self.positions[i, 0] = np.clip(self.positions[i, 0],
                                           PATROL_ZONE["x_min"], PATROL_ZONE["x_max"])
            self.positions[i, 1] = np.clip(self.positions[i, 1],
                                           PATROL_ZONE["y_min"], PATROL_ZONE["y_max"])
            self.positions[i, 2] = np.clip(self.positions[i, 2], 1.0, 15.0)

            # FIX: push physique — empêche les drones de traverser les obstacles
            # Si le drone pénètre dans un obstacle, on le repousse vers l'extérieur
            if self.active_obstacles:
                for obs_data in self.active_obstacles:
                    ox, oy = obs_data["x"], obs_data["y"]
                    half_sx = obs_data["sx"] / 2
                    half_sy = obs_data["sy"] / 2
                    px, py, pz = self.positions[i]
                    # Vérifier si le drone est à l'intérieur de la boîte (+ marge 0.3m)
                    inside_x = abs(px - ox) < half_sx + 0.3
                    inside_y = abs(py - oy) < half_sy + 0.3
                    if inside_x and inside_y and pz <= obs_data["h"] + 1.5:
                        # Calculer le vecteur de sortie le plus court
                        overlap_x = (half_sx + 0.3) - abs(px - ox)
                        overlap_y = (half_sy + 0.3) - abs(py - oy)
                        if overlap_x < overlap_y:
                            sign_x = 1.0 if px > ox else -1.0
                            self.positions[i, 0] = ox + sign_x * (half_sx + 0.3)
                        else:
                            sign_y = 1.0 if py > oy else -1.0
                            self.positions[i, 1] = oy + sign_y * (half_sy + 0.3)

            # Batterie
            self.batteries[i] -= BATTERY_DRAIN_RATE * 10
            if self.batteries[i] <= 0:
                self.alive[i] = False

        # ═══ LEADER MOBILE — bouge vers des waypoints ═══
        if self.level >= 2 and self.n > 1:
            self._move_leader_target()
            # Recalculer les cibles de formation autour du leader
            leader_pos = self.positions[self.leader_id]
            self._compute_formation_targets(
                (leader_pos[0], leader_pos[1], _cfg.CRUISE_ALTITUDE))

        # Leader dynamique (niveau 4+)
        if self.level >= 4 and self.step_count % 100 == 0:
            self._maybe_change_leader()

        # Communication limitée (niveau 5)
        if self.level >= 5:
            self._update_comm_range()

        # Récompenses
        rewards = self._compute_rewards()

        # Terminaison
        # terminated = état final RÉEL (but atteint)
        # truncated  = dépassement du temps limite
        avg_dist = self._avg_distance_to_targets()
        perfect_dist = METRICS.get("formation_perfect_dist", 1.5)

        terminated = False
        truncated = self.step_count >= self.max_steps

        if avg_dist < perfect_dist:
            terminated = True
            rewards[:self.n] += METRICS.get("reward_formation_perfect", 50.0)
            self.episode_formations_reached += 1

        # Visualisation
        if self.use_pybullet:
            for i in range(self.n):
                if self.alive[i]:
                    self.p.resetBasePositionAndOrientation(
                        self.drone_bodies[i], self.positions[i].tolist(),
                        [0, 0, 0, 1], physicsClientId=self.client)
            # Mettre à jour les cibles visuelles
            for i in range(self.n):
                if self.alive[i]:
                    self.p.resetBasePositionAndOrientation(
                        self.target_bodies[i],
                        self.formation_targets[i].tolist(),
                        [0, 0, 0, 1], physicsClientId=self.client)

        obs = self._get_obs()
        info = {
            "avg_dist": avg_dist, "formation": self.target_formation,
            "step": self.step_count, "level": self.level,
            "num_drones": self.n, "collisions": self.episode_collisions,
            "leader_changes": self.leader_change_count,
        }
        return obs, rewards, terminated, truncated, info

    def _move_leader_target(self):
        """Bouge le point central de la formation vers des waypoints."""
        if not hasattr(self, '_leader_waypoints'):
            self._gen_leader_waypoints()

        wp = self._leader_waypoints[self._leader_wp_idx]
        leader_pos = self.positions[self.leader_id]
        dx = wp[0] - leader_pos[0]
        dy = wp[1] - leader_pos[1]
        dist = math.sqrt(dx**2 + dy**2)

        if dist < 2.0:
            self._leader_wp_idx = (self._leader_wp_idx + 1) % len(self._leader_waypoints)

    def _gen_leader_waypoints(self):
        """Génère des waypoints de patrouille pour le leader."""
        self._leader_waypoints = []
        z = PATROL_ZONE
        for y in range(int(z["y_min"]) + 8, int(z["y_max"]) - 7, 10):
            self._leader_waypoints.append((z["x_max"] - 8, float(y)))
            self._leader_waypoints.append((z["x_min"] + 8, float(y)))
        if not self._leader_waypoints:
            self._leader_waypoints = [(0, 0)]
        self._leader_wp_idx = 0

    # ═══════════════════════════════════
    # LEADER DYNAMIQUE (Niveau 4+)
    # ═══════════════════════════════════

    def _maybe_change_leader(self):
        """Simule un changement de leader (batterie faible ou aléatoire)."""
        if not self.alive[self.leader_id] or self.batteries[self.leader_id] < 30:
            # Choisir le drone avec la meilleure batterie
            best_id = 0
            best_bat = -1
            for i in range(self.n):
                if self.alive[i] and self.batteries[i] > best_bat:
                    best_bat = self.batteries[i]
                    best_id = i
            if best_id != self.leader_id:
                self.leader_id = best_id
                self.leader_change_count += 1
                # Recalculer les cibles autour du nouveau leader
                center = (self.positions[self.leader_id, 0],
                          self.positions[self.leader_id, 1],
                          _cfg.CRUISE_ALTITUDE)
                self._compute_formation_targets(center)

    # ═══════════════════════════════════
    # COMMUNICATION LIMITÉE (Niveau 5)
    # ═══════════════════════════════════

    def _update_comm_range(self):
        """Met à jour quels drones peuvent communiquer (portée limitée)."""
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    self.comm_active[i, j] = True
                else:
                    dist = np.linalg.norm(self.positions[i] - self.positions[j])
                    self.comm_active[i, j] = dist <= COMM_RANGE

    # ═══════════════════════════════════
    # OBSERVATIONS
    # ═══════════════════════════════════

    def _get_obs(self):
        obs = np.zeros((self.max_drones, self.obs_size), dtype=np.float32)

        for i in range(self.n):
            if not self.alive[i]:
                continue
            idx = 0

            # Position propre normalisée
            obs[i, idx:idx+3] = self.positions[i] / self.pos_scale
            idx += 3

            # Position cible normalisée
            obs[i, idx:idx+3] = self.formation_targets[i] / self.pos_scale
            idx += 3

            # Positions des autres drones (relatives)
            for j in range(self.max_drones):
                if j == i:
                    continue
                if j < self.n and self.alive[j]:
                    # Niveau 5 : si hors portée comm, position bruitée
                    if self.level >= 5 and not self.comm_active[i, j]:
                        noise = np.random.uniform(-0.1, 0.1, 3)
                        rel_pos = (self.positions[j] - self.positions[i]) / self.pos_scale + noise
                    else:
                        rel_pos = (self.positions[j] - self.positions[i]) / self.pos_scale
                else:
                    rel_pos = np.zeros(3)
                obs[i, idx:idx+3] = rel_pos
                idx += 3

            # FIX: Distance + direction vers les 4 obstacles les plus proches
            # Format: [d0, d1, d2, d3, dx0, dy0, dx1, dy1] (12 floats total)
            # Les vecteurs direction permettent au drone d'apprendre à contourner
            if self.active_obstacles:
                obs_infos = []  # (dist, dx_norm, dy_norm)
                for obs_data in self.active_obstacles:
                    px_i = self.positions[i, 0]
                    py_i = self.positions[i, 1]
                    raw_dx = px_i - obs_data["x"]
                    raw_dy = py_i - obs_data["y"]
                    dist_x = max(0.0, abs(raw_dx) - obs_data["sx"] / 2)
                    dist_y = max(0.0, abs(raw_dy) - obs_data["sy"] / 2)
                    dist = math.sqrt(dist_x**2 + dist_y**2)
                    # Vecteur direction normalisé (du drone vers l'obstacle, inversé = de l'obstacle vers le drone)
                    norm = math.sqrt(raw_dx**2 + raw_dy**2) + 1e-6
                    dx_norm = raw_dx / norm  # positif = drone à droite de l'obstacle
                    dy_norm = raw_dy / norm
                    obs_infos.append((dist / self.pos_scale, dx_norm, dy_norm))
                # Trier par distance croissante, garder les 4 plus proches
                obs_infos.sort(key=lambda x: x[0])
                obs_infos = obs_infos[:4]
                while len(obs_infos) < 4:
                    obs_infos.append((1.0, 0.0, 0.0))
                # Écrire distances
                obs[i, idx:idx+4] = [x[0] for x in obs_infos]
                idx += 4
                # Écrire vecteurs direction (dx, dy) pour chacun des 4
                for k in range(4):
                    obs[i, idx] = obs_infos[k][1]
                    obs[i, idx+1] = obs_infos[k][2]
                    idx += 2
            else:
                obs[i, idx:idx+4] = [1.0, 1.0, 1.0, 1.0]  # Pas d'obstacles
                idx += 4
                obs[i, idx:idx+8] = [0.0] * 8  # Directions nulles
                idx += 8

            # Batterie normalisée
            obs[i, idx] = self.batteries[i] / BATTERY_FULL
            idx += 1

            # Info niveau : est-ce que je suis le leader + ratio drones actifs
            obs[i, idx] = 1.0 if i == self.leader_id else 0.0
            idx += 1
            obs[i, idx] = self.n / self.max_drones
            idx += 1

        return obs

    # ═══════════════════════════════════
    # RÉCOMPENSES
    # ═══════════════════════════════════

    def _compute_rewards(self):
        rewards = np.zeros(self.max_drones, dtype=np.float32)
        M = METRICS

        for i in range(self.n):
            if not self.alive[i]:
                rewards[i] = -1.0
                continue

            # 1. Distance à la cible — RÉCOMPENSE PRINCIPALE
            dist = np.linalg.norm(self.positions[i] - self.formation_targets[i])

            # Récompense exponentielle : signal fort même à grande distance
            # exp(-dist/4) : à 0m=1.0, à 2m=0.61, à 4m=0.37, à 8m=0.13
            rewards[i] += M["reward_target_gradient"] * math.exp(-dist / 4.0)

            if dist < 2.0:
                rewards[i] += M["reward_target_close"]
            elif dist < 4.0:
                rewards[i] += M["reward_target_near"]

            # Bonus de progression : récompense si on se rapproche
            if self._prev_dists is not None and i < len(self._prev_dists):
                if dist < self._prev_dists[i]:
                    rewards[i] += 2.0  # Bonus progression renforcé (était 1.0)

            # 2. Collision inter-drones (pénalité modérée)
            if self.n > 1:
                for j in range(self.n):
                    if j == i or not self.alive[j]:
                        continue
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    if d < M["collision_drone_dist"]:
                        rewards[i] += M["penalty_collision_drone"]
                        self.episode_collisions += 1
                    elif d < M["near_drone_dist"]:
                        rewards[i] += M["penalty_near_drone"]

            # 3. Pénalité obstacles — deux zones :
            #    Zone danger (dist < SAFETY_DISTANCE_OBSTACLES) : gradient doux
            #    Zone collision (dist < 0.5m) : pénalité hard forte
            # FIX: séparation claire signal anticipatif vs signal collision réelle
            if self.active_obstacles:
                for obs_data in self.active_obstacles:
                    ox, oy = obs_data["x"], obs_data["y"]
                    sx, sy, h = obs_data["sx"], obs_data["sy"], obs_data["h"]
                    px, py, pz = self.positions[i]
                    dx_o = max(0.0, abs(px - ox) - sx / 2)
                    dy_o = max(0.0, abs(py - oy) - sy / 2)
                    dist_obs = math.sqrt(dx_o**2 + dy_o**2)
                    if pz <= h + 2:  # Hauteur pertinente (pas en survol)
                        if dist_obs < 0.5:
                            # Collision réelle : pénalité hard (force l'apprentissage)
                            rewards[i] += M["penalty_collision_obs"] * 3.0
                            self.episode_collisions += 1
                        elif dist_obs < SAFETY_DISTANCE_OBSTACLES:
                            # Zone danger : gradient doux (signal anticipatif)
                            factor = max(0.0, 1.0 - dist_obs / SAFETY_DISTANCE_OBSTACLES)
                            rewards[i] += M["penalty_collision_obs"] * factor
                            if dist_obs < 0.5:  # déjà compté ci-dessus, mais sécurité
                                self.episode_collisions += 1

            # 4. Hors zone
            if (self.positions[i, 0] <= PATROL_ZONE["x_min"] + 0.5 or
                self.positions[i, 0] >= PATROL_ZONE["x_max"] - 0.5 or
                self.positions[i, 1] <= PATROL_ZONE["y_min"] + 0.5 or
                self.positions[i, 1] >= PATROL_ZONE["y_max"] - 0.5):
                rewards[i] += M["penalty_out_of_zone"]

            # 5. Énergie (très faible)
            speed = np.linalg.norm(self.velocities[i])
            rewards[i] += M["penalty_energy"] * speed

        # Sauvegarder les distances pour le bonus de progression
        self._prev_dists = np.array([
            np.linalg.norm(self.positions[i] - self.formation_targets[i])
            for i in range(self.n)])

        # 6. Récompense d'équipe — BONUS FORT si tous proches
        if self.n > 1:
            avg_dist = self._avg_distance_to_targets()
            team_reward = max(0, M["reward_team_max"] * (1.0 - avg_dist / 10.0))
            rewards[:self.n] += team_reward / self.n

        return rewards

    def _avg_distance_to_targets(self):
        dists = []
        for i in range(self.n):
            if self.alive[i]:
                d = np.linalg.norm(self.positions[i] - self.formation_targets[i])
                dists.append(d)
        return np.mean(dists) if dists else 0.0

    def close(self):
        if self.use_pybullet and self.client is not None:
            self.p.disconnect(physicsClientId=self.client)
