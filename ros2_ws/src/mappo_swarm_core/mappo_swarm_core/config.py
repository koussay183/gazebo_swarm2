"""
PFE - Swarm Drones MARL - Gazebo Harmonic
Configuration centralisee (ROS 2 deployment version)

NOTE: NUM_DRONES can be overridden at runtime via ROS 2 parameters in mappo_px4_rl_node.py
"""
import math

# =============================== GAZEBO ===============================
GZ_WORLD_NAME = "swarm_world"
GZ_MODEL_PREFIX = "x500_"

# =============================== SIMULATION ===============================
SIM_TIMESTEP = 1.0 / 240.0
GRAVITY = -9.81
GUI_MODE = True
CONTROL_RATE = 50

# =============================== SWARM ===============================
# 4 drones : formations plus faciles (moins d'entre-collisions qu'avec 8).
NUM_DRONES = 4
DRONE_COLORS = [
    [1, 0, 0, 1], [0, 0, 1, 1], [0, 0.8, 0, 1], [1, 1, 0, 1],
]
DRONE_COLOR_NAMES = ["Rouge", "Bleu", "Vert", "Jaune"]

# Taille d'observation par drone (rl_env / MAPPO / rl_controller) : coherente avec NUM_DRONES
# 3 (pos) + 3 (cible) + 3*(N-1) (voisins) + 4 (obs) + 1 (bat) + 2 (leader + actifs)
OBS_SIZE_PER_DRONE = 3 + 3 + 3 * (NUM_DRONES - 1) + 4 + 8 + 1 + 2  # +8 = vecteurs direction obstacles (4 x dx,dy)

# =============================== VOL ===============================
SPAWN_RADIUS = 3.0
SPAWN_Z      = 0.18   # hauteur centre drone au sol (patins = 168mm sous le centre)
CRUISE_ALTITUDE = 5.0
MAX_SPEED = 2.5           # Vitesse calme pour simulation visible
MIN_SPEED = 0.5
TAKEOFF_SPEED = 3.5
LANDING_SPEED = 1.5
POSITION_TOLERANCE = 0.3
PATROL_ALTITUDE = 4.0      # Altitude de patrouille (ajustable via telecommande)
PATROL_RADIUS   = 20.0     # Rayon du circuit de patrouille circulaire

# =============================== ZONE ===============================
PATROL_ZONE = {"x_min": -25, "x_max": 25, "y_min": -25, "y_max": 25}
PATROL_STEP = 10.0

# =============================== FORMATIONS ===============================
FORMATION_TYPES = [
    "v_shape", "line", "circle", "triangle", "diamond", "grid",
    "column", "echelon_left", "echelon_right", "wedge", "encircle",
]
DEFAULT_FORMATION = "v_shape"
FORMATION_SPACING = 3.5

# =============================== LEADER ===============================
INITIAL_LEADER = 0
HEARTBEAT_TIMEOUT = 3.0

# =============================== BATTERIE ===============================
BATTERY_FULL = 100.0
BATTERY_DRAIN_RATE = 0.001
BATTERY_LOW = 20.0
BATTERY_CRITICAL = 10.0

# =============================== COMMUNICATION ===============================
COMM_RANGE = 30.0
COMM_RATE = 10
HEARTBEAT_INTERVAL = 0.5

# =============================== PROBLEMES ===============================
DRONE_PROBLEMS = {
    "LOW_BATTERY":    {"auto_land": True,  "severity": "WARNING"},
    "SENSOR_FAILURE": {"auto_land": True,  "severity": "CRITICAL"},
    "MOTOR_FAILURE":  {"auto_land": True,  "severity": "CRITICAL"},
    "COMM_LOST":      {"auto_land": True,  "severity": "CRITICAL"},
    "GPS_ERROR":      {"auto_land": False, "severity": "WARNING"},
    "OVERHEATING":    {"auto_land": True,  "severity": "WARNING"},
}

# =============================== INTRUS ===============================
SENSOR_RANGE = 12.0
ENCIRCLE_RADIUS = 5.0

# =============================== EVITEMENT ===============================
SAFETY_DISTANCE_DRONES = 1.5
SAFETY_DISTANCE_OBSTACLES = 2.5
BIRD_SAFETY_DISTANCE = 3.5      # Distance d'evitement oiseaux (3D)
AVOIDANCE_FORCE = 2.0
AVOIDANCE_VERTICAL_FACTOR = 0.5

# NOTE: BIRD_POSITIONS has been removed — dynamic bird state is not needed in
# ROS 2 deployment. Bird avoidance should be handled via sensor topics if needed.

# =============================== ENVIRONNEMENT ===============================
OBSTACLES = [
    # Zone Nord-Est --- quartier commercial   (drones a 4m -> batiments 5-10m)
    {"x": 10,  "y": 12,  "sx": 4, "sy": 4, "h": 7,  "color": [0.6,0.3,0.1,0.9], "name": "Batiment_A"},
    {"x": 15,  "y": 8,   "sx": 3, "sy": 3, "h": 8,  "color": [0.5,0.4,0.3,0.9], "name": "Immeuble_B"},
    {"x": 8,   "y": 18,  "sx": 5, "sy": 2, "h": 5,  "color": [0.4,0.5,0.4,0.9], "name": "Entrepot_C"},
    {"x": 20,  "y": 15,  "sx": 3, "sy": 5, "h": 10, "color": [0.5,0.3,0.3,0.9], "name": "Tour_Commerce"},
    {"x": 18,  "y": 20,  "sx": 4, "sy": 3, "h": 6,  "color": [0.4,0.4,0.3,0.9], "name": "Parking_NE"},
    # Zone Nord-Ouest --- zone industrielle
    {"x":-18,  "y": 18,  "sx": 5, "sy": 3, "h": 5,  "color": [0.3,0.5,0.3,0.9], "name": "Hangar_D"},
    {"x":-10,  "y": 15,  "sx": 3, "sy": 3, "h": 9,  "color": [0.4,0.3,0.5,0.9], "name": "Tour_E"},
    {"x": -5,  "y": 8,   "sx": 2, "sy": 2, "h": 10, "color": [0.5,0.2,0.2,0.9], "name": "Antenne_F"},
    {"x":-22,  "y": 10,  "sx": 3, "sy": 6, "h": 6,  "color": [0.3,0.4,0.4,0.9], "name": "Usine_NO"},
    {"x":-15,  "y": 22,  "sx": 4, "sy": 2, "h": 5,  "color": [0.4,0.5,0.3,0.9], "name": "Entrepot_NO"},
    # Zone Sud-Est --- zone residentielle
    {"x": 18,  "y":-15,  "sx": 3, "sy": 3, "h": 9,  "color": [0.4,0.4,0.6,0.9], "name": "Tour_G"},
    {"x": 12,  "y": -8,  "sx": 4, "sy": 3, "h": 6,  "color": [0.6,0.5,0.3,0.9], "name": "Bureau_H"},
    {"x": 6,   "y":-15,  "sx": 3, "sy": 4, "h": 7,  "color": [0.3,0.4,0.5,0.9], "name": "Depot_I"},
    {"x": 22,  "y": -8,  "sx": 3, "sy": 3, "h": 7,  "color": [0.5,0.3,0.4,0.9], "name": "Maison_SE1"},
    {"x": 15,  "y":-20,  "sx": 4, "sy": 4, "h": 5,  "color": [0.4,0.4,0.5,0.9], "name": "Ecole_SE"},
    # Zone Sud-Ouest --- zone militaire
    {"x":-12,  "y": -8,  "sx": 3, "sy": 5, "h": 6,  "color": [0.5,0.5,0.5,0.9], "name": "Batiment_J"},
    {"x":-20,  "y":-12,  "sx": 4, "sy": 3, "h": 7,  "color": [0.4,0.3,0.3,0.9], "name": "Usine_K"},
    {"x": -8,  "y":-18,  "sx": 3, "sy": 2, "h": 8,  "color": [0.3,0.3,0.5,0.9], "name": "Silo_L"},
    {"x":-20,  "y":-20,  "sx": 5, "sy": 5, "h": 5,  "color": [0.3,0.4,0.3,0.9], "name": "Caserne_SO"},
    {"x":-15,  "y":-18,  "sx": 3, "sy": 3, "h": 8,  "color": [0.4,0.3,0.4,0.9], "name": "Radar_SO"},
    # Zone Centre
    {"x": 0,   "y": 0,   "sx": 2, "sy": 2, "h": 6,  "color": [0.6,0.4,0.2,0.9], "name": "Kiosque_Centre"},
    {"x": 5,   "y":-5,   "sx": 3, "sy": 2, "h": 5,  "color": [0.5,0.5,0.3,0.9], "name": "Marche_Centre"},
    {"x":-5,   "y":-3,   "sx": 2, "sy": 3, "h": 7,  "color": [0.4,0.3,0.5,0.9], "name": "Mosquee"},
]

TREES = [
    {"x": 5,   "y": 5,   "r": 0.5, "h": 3},
    {"x":-3,   "y": 12,  "r": 0.6, "h": 4},
    {"x": 20,  "y": 20,  "r": 0.5, "h": 3},
    {"x": 15,  "y": 0,   "r": 0.4, "h": 4},
    {"x": 22,  "y": -5,  "r": 0.5, "h": 3},
    {"x": -5,  "y": -5,  "r": 0.5, "h": 3},
    {"x":-15,  "y":-15,  "r": 0.6, "h": 5},
    {"x": 0,   "y":-10,  "r": 0.4, "h": 3},
    {"x":-20,  "y": 5,   "r": 0.5, "h": 4},
    {"x":-15,  "y": 0,   "r": 0.6, "h": 5},
    {"x": 10,  "y": -2,  "r": 0.5, "h": 4},
    {"x":-8,   "y": 20,  "r": 0.7, "h": 5},
    {"x": 3,   "y": 15,  "r": 0.4, "h": 3},
    {"x":-22,  "y":-5,   "r": 0.6, "h": 4},
    {"x": 8,   "y":-22,  "r": 0.5, "h": 3},
    {"x":-3,   "y":-15,  "r": 0.5, "h": 4},
    {"x": 22,  "y": 10,  "r": 0.4, "h": 3},
    {"x":-10,  "y": -22, "r": 0.6, "h": 5},
]

WALLS = [
    {"x1": -8, "y1": 3,  "x2": -8, "y2": 10, "h": 2,   "color": [0.6,0.6,0.6,0.8], "name": "Mur_Ouest_1"},
    {"x1": 5,  "y1":-8,  "x2": 12, "y2":-8,  "h": 2,   "color": [0.6,0.6,0.6,0.8], "name": "Mur_Sud_1"},
    {"x1":-15, "y1":-5,  "x2": -8, "y2":-5,  "h": 1.5, "color": [0.5,0.5,0.5,0.8], "name": "Barriere_1"},
    {"x1": 8,  "y1": 5,  "x2": 8,  "y2": 12, "h": 1.5, "color": [0.5,0.5,0.5,0.8], "name": "Cloture_E"},
    {"x1":-20, "y1":-8,  "x2":-12, "y2":-8,  "h": 2,   "color": [0.4,0.4,0.4,0.8], "name": "Mur_Militaire"},
    {"x1": 15, "y1":-12, "x2": 22, "y2":-12, "h": 1.5, "color": [0.5,0.5,0.5,0.8], "name": "Barriere_SE"},
]

# =============================== METRIQUES RL ===============================
METRICS = {
    "reward_target_close": 10.0, "reward_target_near": 5.0,
    "reward_target_gradient": 5.0, "reward_formation_perfect": 50.0,
    "reward_team_max": 10.0,
    "penalty_collision_drone": -3.0, "penalty_near_drone": -0.5,
    "penalty_collision_obs": -8.0, "penalty_out_of_zone": -1.0,
    "penalty_energy": -0.005,
    "formation_perfect_dist": 1.5,
    "collision_drone_dist": SAFETY_DISTANCE_DRONES,
    "near_drone_dist": SAFETY_DISTANCE_DRONES * 2,
}


def obstacle_avoidance_vector(x, y, z):
    """
    Compute the combined avoidance force vector at position (x, y, z).
    Returns (fx, fy, fz) — a repulsion vector away from nearby obstacles.
    """
    fx, fy, fz = 0.0, 0.0, 0.0
    for obs in OBSTACLES:
        ox, oy, sx, sy, h = obs["x"], obs["y"], obs["sx"], obs["sy"], obs["h"]
        dx, dy = x - ox, y - oy
        dist_x = max(0, abs(dx) - sx / 2)
        dist_y = max(0, abs(dy) - sy / 2)
        dist = math.sqrt(dist_x**2 + dist_y**2)
        if dist < SAFETY_DISTANCE_OBSTACLES and z <= h + 2:
            ratio = max(0.0, 1.0 - dist / SAFETY_DISTANCE_OBSTACLES)
            force = AVOIDANCE_FORCE * ratio
            if abs(dx) > 0.01: fx += (dx / abs(dx)) * force
            if abs(dy) > 0.01: fy += (dy / abs(dy)) * force
            if z < h + 2: fz += force * AVOIDANCE_VERTICAL_FACTOR
    for tree in TREES:
        tx_t, ty_t, tr, th = tree["x"], tree["y"], tree["r"], tree["h"]
        dx, dy = x - tx_t, y - ty_t
        dist = math.sqrt(dx**2 + dy**2) - tr
        if dist < SAFETY_DISTANCE_OBSTACLES and z <= th + 1:
            ratio = max(0.0, 1.0 - dist / SAFETY_DISTANCE_OBSTACLES)
            force = AVOIDANCE_FORCE * ratio
            norm = math.sqrt(dx**2 + dy**2)
            if norm > 0.01:
                fx += (dx / norm) * force
                fy += (dy / norm) * force
            if z < th + 1: fz += force * AVOIDANCE_VERTICAL_FACTOR
    for wall in WALLS:
        wx = (wall["x1"] + wall["x2"]) / 2
        wy = (wall["y1"] + wall["y2"]) / 2
        wsx = abs(wall["x2"] - wall["x1"]) + 0.5
        wsy = abs(wall["y2"] - wall["y1"]) + 0.5
        wh = wall["h"]
        dx, dy = x - wx, y - wy
        dist_x = max(0, abs(dx) - wsx / 2)
        dist_y = max(0, abs(dy) - wsy / 2)
        dist = math.sqrt(dist_x**2 + dist_y**2)
        if dist < SAFETY_DISTANCE_OBSTACLES and z <= wh + 1:
            ratio = max(0.0, 1.0 - dist / SAFETY_DISTANCE_OBSTACLES)
            force = AVOIDANCE_FORCE * ratio
            if abs(dx) > 0.01: fx += (dx / abs(dx)) * force
            if abs(dy) > 0.01: fy += (dy / abs(dy)) * force
            if z < wh + 1: fz += force * AVOIDANCE_VERTICAL_FACTOR
    return fx, fy, fz
