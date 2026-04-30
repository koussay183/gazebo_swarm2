"""
PFE - Swarm Drones MARL - Gazebo Harmonic + PX4 SITL
Interface Gazebo : pont ROS2 ↔ Gazebo

Gère toute la communication ROS2 ↔ Gazebo :
  - Abonnement aux topics de pose pour lire les positions réelles (physique PX4)
  - Service SetEntityPose pour les téléportations (init, repair, reset)
  - Spin ROS2 dans un thread séparé

Note : cmd_vel supprimé — PX4 SITL contrôle les moteurs directement
via gz-transport. Les setpoints de position sont envoyés par MAVSDK
(voir px4_mavsdk_interface.py).
"""

import sys
import os
import math
import time
import threading

# ── ROS2 / Python 3.12 ──────────────────────────────────────────────────────
sys.path.insert(0, "/opt/ros/jazzy/lib/python3.12/site-packages")
os.environ.setdefault("AMENT_PREFIX_PATH", "/opt/ros/jazzy")
os.environ.setdefault("ROS_DISTRO", "jazzy")

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from ros_gz_interfaces.srv import SetEntityPose
from ros_gz_interfaces.msg import Entity

from config import NUM_DRONES, GZ_WORLD_NAME, GZ_MODEL_PREFIX, SPAWN_RADIUS
import config as _cfg


# Singleton
_interface: "GazeboInterface | None" = None


def get_gazebo_interface() -> "GazeboInterface":
    global _interface
    if _interface is None:
        raise RuntimeError("GazeboInterface not initialized. Call init_gazebo() first.")
    return _interface


def init_gazebo(num_drones: int = NUM_DRONES) -> "GazeboInterface":
    global _interface
    if _interface is None:
        _interface = GazeboInterface(num_drones)
    return _interface


def shutdown_gazebo():
    global _interface
    if _interface:
        _interface.shutdown()
        _interface = None


# ────────────────────────────────────────────────────────────────────────────

class GazeboInterface:
    """
    Pont Python ↔ Gazebo via ROS2 (mode PX4 SITL).

    Utilisation :
        gz = init_gazebo()
        x, y, z, yaw = gz.get_pose(0)      # position drone 0 (physique PX4 réelle)
        gz.teleport(0, 3.0, 0.0, 0.25)     # repos au sol (reset / repair)

    Les commandes de vol sont gérées par px4_mavsdk_interface via MAVSDK-Python.
    """

    def __init__(self, num_drones: int = NUM_DRONES):
        self.n = num_drones
        self._lock = threading.Lock()

        # ── Initialisation ROS2 ──────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("swarm_gz_interface")

        # ── Poses courantes (x, y, z, yaw) – initialisées au spawn ──────
        self._poses: dict[int, list[float]] = {}
        for i in range(num_drones):
            angle = 2.0 * math.pi * i / num_drones
            self._poses[i] = [
                SPAWN_RADIUS * math.cos(angle),
                SPAWN_RADIUS * math.sin(angle),
                0.25,
                0.0,
            ]

        # ── Subscribers pose ─────────────────────────────────────────────
        # Topic complet /model/drone_i/pose : correspond au topic publié
        # par le plugin PosePublisher dans Gazebo.
        for i in range(num_drones):
            topic = f"/model/{GZ_MODEL_PREFIX}{i}/pose"
            self.node.create_subscription(
                Pose, topic,
                lambda msg, idx=i: self._on_pose(idx, msg),
                10,
            )

        # ── Service SetEntityPose ────────────────────────────────────────
        svc_name = f"/world/{GZ_WORLD_NAME}/set_pose"
        self._set_pose_cli = self.node.create_client(SetEntityPose, svc_name)

        # ── Spin thread ──────────────────────────────────────────────────
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        # Attendre que Gazebo soit prêt (service disponible)
        self._wait_for_service(timeout=30.0)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_pose(self, drone_id: int, msg: Pose):
        """Met à jour la pose d'un drone depuis Gazebo."""
        q = msg.orientation
        # yaw = atan2(2(qw*qz + qx*qy), 1-2(qy²+qz²))
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        with self._lock:
            self._poses[drone_id] = [
                msg.position.x,
                msg.position.y,
                msg.position.z,
                yaw,
            ]

    def _spin_loop(self):
        """Thread de spin ROS2."""
        while self._running and rclpy.ok():
            rclpy.spin_once(self.node, timeout_sec=0.005)

    def _wait_for_service(self, timeout: float = 30.0):
        """Attend que le service set_pose soit disponible."""
        t0 = time.time()
        while not self._set_pose_cli.wait_for_service(timeout_sec=1.0):
            if time.time() - t0 > timeout:
                print(f"[GazeboInterface] WARNING: service {self._set_pose_cli.srv_name} "
                      "not available after timeout, continuing without teleport.")
                return
        print(f"[GazeboInterface] Service {self._set_pose_cli.srv_name} ready.")

    # ── API publique ──────────────────────────────────────────────────────────

    def get_pose(self, drone_id: int) -> tuple[float, float, float, float]:
        """Retourne (x, y, z, yaw) du drone."""
        with self._lock:
            p = self._poses[drone_id]
            return p[0], p[1], p[2], p[3]

    def teleport(self, drone_id: int, x: float, y: float, z: float, yaw: float = 0.0):
        """
        Téléporte un drone (position directe via service Gazebo).
        Utilisé pour : initialisation, repair, reset.
        """
        req = SetEntityPose.Request()
        req.entity.name = f"{GZ_MODEL_PREFIX}{drone_id}"
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        # Convertir yaw en quaternion
        req.pose.orientation.w = float(math.cos(yaw / 2.0))
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = float(math.sin(yaw / 2.0))
        # Mettre à jour la pose locale immédiatement
        with self._lock:
            self._poses[drone_id] = [x, y, z, yaw]
        # Appel async (non bloquant)
        if self._set_pose_cli.service_is_ready():
            self._set_pose_cli.call_async(req)

    def teleport_model(self, model_name: str, x: float, y: float, z: float,
                       yaw: float = 0.0):
        """Téléporte n'importe quel modèle Gazebo (intrus, oiseaux…)."""
        req = SetEntityPose.Request()
        req.entity.name = model_name
        req.entity.type = Entity.MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        req.pose.orientation.w = float(math.cos(yaw / 2.0))
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = float(math.sin(yaw / 2.0))
        if self._set_pose_cli.service_is_ready():
            self._set_pose_cli.call_async(req)

    def update_birds(self, bird_positions: list[list[float]]):
        """Met à jour la liste des positions d'oiseaux dans config."""
        _cfg.BIRD_POSITIONS = bird_positions

    def shutdown(self):
        """Arrête proprement l'interface."""
        self._running = False
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
