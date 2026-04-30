"""
PFE - Swarm Drones MARL - PX4 SITL
Interface PX4 via MAVSDK-Python

Remplace le contrôle cinématique par téléportation par des commandes
PX4 réelles (arm / takeoff / offboard position setpoints / land).

Architecture :
    Python Swarm → px4_mavsdk_interface → MAVSDK-Python → PX4 SITL UDP → gz-transport → Gazebo

Ports MAVLink PX4 SITL (un port par instance) :
    Instance 0 → udp://:14540
    Instance 1 → udp://:14541
    Instance 2 → udp://:14542
    Instance 3 → udp://:14543

Conversion de coordonnées (Gazebo ENU → PX4 NED) :
    NED North = Gazebo Y
    NED East  = Gazebo X
    NED Down  = -Gazebo Z

    Les setpoints NED sont relatifs au home de chaque drone (position de spawn).
"""

import asyncio
import threading
import math
import time

from config import NUM_DRONES, SPAWN_RADIUS, SPAWN_Z, CRUISE_ALTITUDE

# ── Singleton ────────────────────────────────────────────────────────────────
_px4_interface: "PX4MavsdkInterface | None" = None


def get_px4_interface() -> "PX4MavsdkInterface":
    global _px4_interface
    if _px4_interface is None:
        raise RuntimeError("PX4MavsdkInterface non initialisée. Appeler init_px4() d'abord.")
    return _px4_interface


def init_px4(num_drones: int = NUM_DRONES) -> "PX4MavsdkInterface":
    global _px4_interface
    if _px4_interface is None:
        _px4_interface = PX4MavsdkInterface(num_drones)
    return _px4_interface


def shutdown_px4():
    global _px4_interface
    if _px4_interface:
        _px4_interface.shutdown()
        _px4_interface = None


# ── Positions de spawn (NED home par drone) ───────────────────────────────────

def _spawn_position(drone_id: int, num_drones: int = NUM_DRONES):
    """Retourne la position de spawn (x, y, z) en coordonnées Gazebo ENU."""
    angle = 2.0 * math.pi * drone_id / num_drones
    return (
        SPAWN_RADIUS * math.cos(angle),   # Gazebo X → NED East
        SPAWN_RADIUS * math.sin(angle),   # Gazebo Y → NED North
        SPAWN_Z,                           # Gazebo Z → NED Down = -SPAWN_Z ≈ 0
    )


# ── Interface principale ──────────────────────────────────────────────────────

class PX4MavsdkInterface:
    """
    Interface MAVSDK multi-drone pour essaim PX4 SITL.

    Chaque drone est contrôlé via une connexion MAVSDK distincte (UDP MAVLink).
    Le contrôle s'effectue en mode Offboard avec des setpoints de position NED.

    Utilisation :
        px4 = init_px4(num_drones=4)
        px4.arm_and_start_offboard(drone_id=0)
        px4.send_position_ned(0, north=5.0, east=2.0, down=-5.0)
        px4.land(0)
    """

    def __init__(self, num_drones: int = NUM_DRONES):
        self.n = num_drones
        self._drones = {}           # drone_id → mavsdk.System
        self._offboard = {}         # drone_id → bool
        self._armed = {}            # drone_id → bool
        self._last_setpoint = {}    # drone_id → (N, E, D, yaw)
        self._lock = threading.Lock()

        # Positions de spawn (home NED)
        self._spawn = {i: _spawn_position(i, num_drones) for i in range(num_drones)}

        # Boucle asyncio dédiée dans un thread séparé
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        # Connexion à tous les drones
        future = asyncio.run_coroutine_threadsafe(self._connect_all(), self._loop)
        try:
            future.result(timeout=30.0)
        except TimeoutError:
            print("[PX4] ATTENTION: Connexion timeout — PX4 SITL démarré ?")

    # ── Boucle asyncio ────────────────────────────────────────────────────────

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect_all(self):
        """Connexion séquentielle à toutes les instances PX4 SITL."""
        try:
            from mavsdk import System
        except ImportError:
            print("[PX4] ERREUR: mavsdk non installé. 'pip install mavsdk'")
            return

        for i in range(self.n):
            port = 14540 + i
            print(f"[PX4] Connexion drone {i} → udp://:{port} ...")
            drone = System()
            await drone.connect(system_address=f"udp://:{port}")

            # Attendre la connexion (timeout 10s par drone)
            connected = False
            async for state in drone.core.connection_state():
                if state.is_connected:
                    connected = True
                    break

            if connected:
                print(f"[PX4] Drone {i} connecté (port {port})")
            else:
                print(f"[PX4] Drone {i} non connecté (port {port}) — mode dégradé")

            with self._lock:
                self._drones[i] = drone
                self._offboard[i] = False
                self._armed[i] = False
                sx, sy, sz = self._spawn[i]
                self._last_setpoint[i] = (sy, sx, -sz, 0.0)  # Home NED

            # Lancer le monitoring de statut en tâche de fond
            asyncio.ensure_future(self._monitor_armed(i))

    async def _monitor_armed(self, drone_id: int):
        """Surveille l'état armé du drone en continu."""
        try:
            async for armed in self._drones[drone_id].telemetry.armed():
                with self._lock:
                    self._armed[drone_id] = armed
        except Exception:
            pass

    # ── API publique synchrone ────────────────────────────────────────────────

    def arm_and_start_offboard(self, drone_id: int):
        """Arme le drone et active le mode Offboard (non bloquant)."""
        asyncio.run_coroutine_threadsafe(
            self._arm_offboard(drone_id), self._loop
        )

    def send_position_ned(self, drone_id: int,
                          north: float, east: float, down: float,
                          yaw_deg: float = 0.0):
        """
        Envoie un setpoint de position NED absolu (relatif au home du drone).
        Non bloquant — PX4 va se déplacer vers ce point.
        """
        with self._lock:
            self._last_setpoint[drone_id] = (north, east, down, yaw_deg)
        asyncio.run_coroutine_threadsafe(
            self._set_ned(drone_id, north, east, down, yaw_deg), self._loop
        )

    def send_position_enu(self, drone_id: int,
                          gz_x: float, gz_y: float, gz_z: float,
                          yaw_deg: float = 0.0):
        """
        Envoie un setpoint en coordonnées Gazebo ENU (conversion → NED automatique).
        gz_x, gz_y, gz_z : position cible dans le repère monde Gazebo.
        """
        sx, sy, _ = self._spawn[drone_id]
        north = gz_y - sy   # ENU Y → NED North
        east  = gz_x - sx   # ENU X → NED East
        down  = -gz_z        # ENU Z → NED Down (négatif = altitude)
        self.send_position_ned(drone_id, north, east, down, yaw_deg)

    def land(self, drone_id: int):
        """Commande d'atterrissage PX4 (mode Land)."""
        asyncio.run_coroutine_threadsafe(
            self._land(drone_id), self._loop
        )

    def disarm(self, drone_id: int):
        """Désarme le drone (forcer si nécessaire)."""
        asyncio.run_coroutine_threadsafe(
            self._drones[drone_id].action.disarm(), self._loop
        )

    def is_armed(self, drone_id: int) -> bool:
        with self._lock:
            return self._armed.get(drone_id, False)

    def is_offboard(self, drone_id: int) -> bool:
        with self._lock:
            return self._offboard.get(drone_id, False)

    def teleport_reset(self, drone_id: int, gz_x: float, gz_y: float, gz_z: float):
        """
        Réinitialise le home NED après une téléportation Gazebo.
        À appeler après get_gazebo_interface().teleport() pour resynchroniser PX4.
        """
        with self._lock:
            self._spawn[drone_id] = (gz_x, gz_y, gz_z)
            self._last_setpoint[drone_id] = (0.0, 0.0, -gz_z, 0.0)
        self._offboard[drone_id] = False

    def shutdown(self):
        """Arrêt propre : land tous les drones armés."""
        for i in range(self.n):
            if self.is_armed(i):
                self.land(i)
        time.sleep(2.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3.0)

    # ── Coroutines async privées ──────────────────────────────────────────────

    async def _arm_offboard(self, drone_id: int):
        """Arme le drone et active le mode Offboard."""
        if drone_id not in self._drones:
            return
        drone = self._drones[drone_id]
        try:
            # Envoyer un premier setpoint AVANT d'activer offboard (obligatoire PX4)
            from mavsdk.offboard import PositionNedYaw
            sx, sy, sz = self._spawn[drone_id]
            initial_setpoint = PositionNedYaw(0.0, 0.0, -CRUISE_ALTITUDE, 0.0)
            await drone.offboard.set_position_ned(initial_setpoint)

            # Armer
            await drone.action.arm()
            print(f"[PX4] Drone {drone_id} armé")

            # Activer Offboard
            await drone.offboard.start()
            with self._lock:
                self._offboard[drone_id] = True
            print(f"[PX4] Drone {drone_id} mode Offboard activé")

        except Exception as e:
            print(f"[PX4] Erreur arm/offboard drone {drone_id}: {e}")

    async def _set_ned(self, drone_id: int,
                       north: float, east: float, down: float, yaw_deg: float):
        """Envoie un setpoint PositionNedYaw à PX4."""
        if not self._offboard.get(drone_id, False):
            return
        if drone_id not in self._drones:
            return
        try:
            from mavsdk.offboard import PositionNedYaw
            await self._drones[drone_id].offboard.set_position_ned(
                PositionNedYaw(float(north), float(east), float(down), float(yaw_deg))
            )
        except Exception as e:
            print(f"[PX4] Erreur setpoint drone {drone_id}: {e}")

    async def _land(self, drone_id: int):
        """Atterrissage via mode Land PX4."""
        if drone_id not in self._drones:
            return
        drone = self._drones[drone_id]
        try:
            # Désactiver offboard d'abord
            if self._offboard.get(drone_id, False):
                await drone.offboard.stop()
                with self._lock:
                    self._offboard[drone_id] = False
            await drone.action.land()
            print(f"[PX4] Drone {drone_id} atterrissage en cours...")
        except Exception as e:
            print(f"[PX4] Erreur atterrissage drone {drone_id}: {e}")
