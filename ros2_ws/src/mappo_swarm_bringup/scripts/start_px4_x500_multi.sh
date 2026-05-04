#!/bin/bash
# Multi-instance PX4 SITL for N drones with namespaced uXRCE-DDS topics
#
# Usage: ./start_px4_x500_multi.sh [N]
#   N = number of drones (default: 4)
#
# Each instance uses:
#   PX4_UXRCE_DDS_NS=px4_N  -> topics appear as /px4_N/fmu/...
#   PX4_GZ_MODEL_NAME=x500_N -> Gazebo model name
#   Instance ID (-i N)        -> different port/config per instance
#
# After all instances start, launch the uXRCE agent:
#   MicroXRCEAgent udp4 -p 8888
#
# Then verify topics:
#   ros2 topic list | grep px4
#   ros2 topic echo /px4_0/fmu/out/vehicle_status
#
# NOTE: Multi-instance requires PX4 built with SITL support.
# NOTE: Gazebo will launch N model instances in one world.
# NOTE: Each instance spawns at a different position automatically.

set -e

NUM_DRONES="${1:-4}"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
BUILD_DIR="$PX4_DIR/build/px4_sitl_default"
PX4_BIN="$BUILD_DIR/bin/px4"

echo "=== Multi-drone PX4 SITL: $NUM_DRONES drones ==="

if [ ! -d "$PX4_DIR" ]; then
    echo "ERROR: PX4-Autopilot not found at $PX4_DIR"
    echo "Clone: git clone --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot"
    exit 1
fi

# Build first if binary doesn't exist
if [ ! -f "$PX4_BIN" ]; then
    echo "Building PX4 SITL (first time — this may take several minutes)..."
    cd "$PX4_DIR"
    make px4_sitl_default
fi

echo "Launching $NUM_DRONES PX4 instances..."
echo "(Press Ctrl+C to stop all instances)"

PIDS=()

for i in $(seq 0 $((NUM_DRONES - 1))); do
    echo "  Starting drone $i (namespace px4_$i, model x500_$i)..."
    PX4_SYS_AUTOSTART=4001 \
    PX4_GZ_MODEL_NAME="x500_$i" \
    PX4_UXRCE_DDS_NS="px4_$i" \
    PX4_SIM_HOSTNAME=localhost \
        "$PX4_BIN" -i "$i" -d \
        -s "$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/px4-rc.simulator" &

    PIDS+=($!)
    sleep 2  # Give each instance time to initialise before starting next
done

echo ""
echo "All $NUM_DRONES PX4 instances started. PIDs: ${PIDS[*]}"
echo "Now start the uXRCE agent: MicroXRCEAgent udp4 -p 8888"
echo ""
echo "Waiting for instances... (Ctrl+C to stop all)"

# Wait for all background processes
wait "${PIDS[@]}"
