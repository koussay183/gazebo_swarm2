#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Swarm Drones MARL — Gazebo Harmonic + PX4 SITL ==="

# Source ROS2 Jazzy
source /opt/ros/jazzy/setup.bash

# Fix DDS shared memory issues in Docker
export FASTDDS_DEFAULT_PROFILES_FILE="$SCRIPT_DIR/fastdds_no_shm.xml"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Add models path (inclut les modèles x500_i générés)
export GZ_SIM_RESOURCE_PATH="$SCRIPT_DIR/models:$GZ_SIM_RESOURCE_PATH"
export GAZEBO_MODEL_PATH="$SCRIPT_DIR/models:$GAZEBO_MODEL_PATH"

# ── [1/5] Générer les modèles x500 par instance ───────────────────────────
echo "[1/5] Génération des modèles x500_0..x500_3 pour PX4 SITL..."
bash "$SCRIPT_DIR/scripts/generate_px4_models.sh" 4
echo "      Modèles générés dans $SCRIPT_DIR/models/"

# ── [2/5] Démarrer Gazebo Harmonic ────────────────────────────────────────
if ! pgrep -x "gz" > /dev/null 2>&1; then
    echo "[2/5] Démarrage de Gazebo Harmonic..."
    gz sim "$SCRIPT_DIR/worlds/swarm_world.sdf" &
    GZ_PID=$!
    sleep 8
    echo "      Gazebo PID: $GZ_PID"
else
    echo "[2/5] Gazebo déjà en cours d'exécution."
fi

# ── [3/5] Démarrer les instances PX4 SITL ────────────────────────────────
PX4_PATH="${PX4_PATH:-/opt/px4/PX4-Autopilot}"
PX4_BIN="$PX4_PATH/build/px4_sitl_default/bin/px4"
PX4_AUTOSTART=4001  # x500 configuration

if [ -f "$PX4_BIN" ]; then
    echo "[3/5] Démarrage des instances PX4 SITL..."
    for i in 0 1 2 3; do
        echo "      → PX4 instance $i (modèle x500_$i, port MAVLink $((14540+i)))"
        PX4_SYS_AUTOSTART=$PX4_AUTOSTART \
        PX4_GZ_STANDALONE=1 \
        PX4_GZ_MODEL_NAME=x500_$i \
        PX4_SIM_HOSTNAME=localhost \
        "$PX4_BIN" -i $i -d &
        PX4_PIDS[$i]=$!
        sleep 2  # Délai entre instances pour éviter conflits gz-transport
    done
    echo "      PX4 PIDs: ${PX4_PIDS[*]}"
    sleep 3  # Attendre l'initialisation EKF2
else
    echo "[3/5] ATTENTION: PX4 non trouvé à $PX4_BIN"
    echo "      Installer PX4: cd /opt/px4 && git clone --recursive https://github.com/PX4/PX4-Autopilot.git"
    echo "      Puis: cd PX4-Autopilot && make px4_sitl -j$(nproc)"
    echo "      Mode dégradé: les drones seront contrôlés sans PX4 réel."
fi

# ── [4/5] Démarrer le bridge ROS2-Gazebo ─────────────────────────────────
# Uniquement topics de pose (lecture position) + service set_pose (téléportation)
# cmd_vel supprimé : PX4 contrôle les moteurs via gz-transport directement
echo "[4/5] Démarrage du bridge ROS2-Gazebo (pose + set_pose)..."
ros2 run ros_gz_bridge parameter_bridge \
    /model/x500_0/pose@geometry_msgs/msg/Pose\]gz.msgs.Pose \
    /model/x500_1/pose@geometry_msgs/msg/Pose\]gz.msgs.Pose \
    /model/x500_2/pose@geometry_msgs/msg/Pose\]gz.msgs.Pose \
    /model/x500_3/pose@geometry_msgs/msg/Pose\]gz.msgs.Pose \
    /world/swarm_world/set_pose@ros_gz_interfaces/srv/SetEntityPose &
BRIDGE_PID=$!
echo "      Bridge PID: $BRIDGE_PID"
sleep 3

# ── [5/5] Démarrer la simulation Python ──────────────────────────────────
echo "[5/5] Démarrage de la simulation Python (swarm MARL + MAVSDK)..."
cd "$SCRIPT_DIR/src"
exec python3.12 main.py
