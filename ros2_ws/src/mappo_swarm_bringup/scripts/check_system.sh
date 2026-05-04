#!/bin/bash
# Comprehensive system check for MAPPO drone swarm deployment
# Run this before starting any flight operations.
#
# Usage: ./check_system.sh
#        ./check_system.sh --verbose

set -u

VERBOSE=0
if [[ "${1:-}" == "--verbose" ]]; then
    VERBOSE=1
fi

echo "=== Swarm System Check ==="
echo "Date: $(date)"
echo ""

PASS=0
FAIL=0
WARN=0

check() {
    local label="$1"
    local cmd="$2"
    local is_warn="${3:-0}"

    if eval "$cmd" 2>/dev/null 1>/dev/null; then
        echo "  [OK]   $label"
        ((PASS++))
    else
        if [ "$is_warn" -eq 1 ]; then
            echo "  [WARN] $label"
            ((WARN++))
        else
            echo "  [FAIL] $label"
            ((FAIL++))
        fi
    fi
}

# ---- ROS 2 Environment ------------------------------------------------
echo "--- ROS 2 Environment ---"
check "ROS 2 sourced (AMENT_PREFIX_PATH set)" \
    "[ -n '${AMENT_PREFIX_PATH:-}' ]"
check "ROS 2 Jazzy" \
    "[ '${ROS_DISTRO:-}' = 'jazzy' ]"
check "ros2 CLI available" \
    "command -v ros2"

# ---- Simulation Tools ------------------------------------------------
echo ""
echo "--- Simulation Tools ---"
check "Gazebo (gz) installed" \
    "command -v gz"
check "PX4-Autopilot directory exists" \
    "[ -d '${HOME}/PX4-Autopilot' ]"
check "PX4 SITL binary built" \
    "[ -f '${HOME}/PX4-Autopilot/build/px4_sitl_default/bin/px4' ]" 1

# ---- uXRCE-DDS Agent -------------------------------------------------
echo ""
echo "--- uXRCE-DDS Agent ---"
check "MicroXRCEAgent available" \
    "command -v MicroXRCEAgent"
check "uXRCE agent port 8888 not already in use" \
    "! ss -tuln 2>/dev/null | grep -q ':8888'" 1

# ---- ROS 2 Packages --------------------------------------------------
echo ""
echo "--- ROS 2 Packages ---"
check "px4_msgs available" \
    "ros2 pkg list 2>/dev/null | grep -q '^px4_msgs$'"
check "ros_gz_bridge available" \
    "ros2 pkg list 2>/dev/null | grep -q 'ros_gz_bridge'"
check "ros_gz_interfaces available" \
    "ros2 pkg list 2>/dev/null | grep -q 'ros_gz_interfaces'"
check "mappo_swarm_core built" \
    "ros2 pkg list 2>/dev/null | grep -q 'mappo_swarm_core'"
check "mappo_swarm_px4 built" \
    "ros2 pkg list 2>/dev/null | grep -q 'mappo_swarm_px4'"
check "mappo_swarm_bridge built" \
    "ros2 pkg list 2>/dev/null | grep -q 'mappo_swarm_bridge'"
check "mappo_swarm_gcs built" \
    "ros2 pkg list 2>/dev/null | grep -q 'mappo_swarm_gcs'"
check "mappo_swarm_bringup built" \
    "ros2 pkg list 2>/dev/null | grep -q 'mappo_swarm_bringup'"

# ---- Python Dependencies ---------------------------------------------
echo ""
echo "--- Python Dependencies ---"
check "torch importable" \
    "python3 -c 'import torch'"
check "numpy importable" \
    "python3 -c 'import numpy'"
check "rclpy importable" \
    "python3 -c 'import rclpy'"

# ---- ML Checkpoint --------------------------------------------------
echo ""
echo "--- ML Checkpoint ---"
BEST_PT=$(find "$HOME" -name "best.pt" -not -path "*/\.*" 2>/dev/null | head -1)
check "best.pt found" \
    "[ -n '$BEST_PT' ]"

if [ -n "$BEST_PT" ]; then
    check "best.pt readable" \
        "[ -r '$BEST_PT' ]"
    if [ $VERBOSE -eq 1 ]; then
        echo "    -> Found at: $BEST_PT"
        SIZE=$(du -h "$BEST_PT" 2>/dev/null | cut -f1)
        echo "    -> Size: $SIZE"
    fi
fi

# ---- QGroundControl (optional) -------------------------------------
echo ""
echo "--- QGroundControl (optional) ---"
check "QGroundControl AppImage present" \
    "find $HOME -name 'QGroundControl.AppImage' -maxdepth 4 2>/dev/null | grep -q ." 1
check "qgroundcontrol binary in PATH" \
    "command -v qgroundcontrol" 1

# ---- Summary -------------------------------------------------------
echo ""
echo "========================================"
echo "Results: ${PASS} OK | ${WARN} warnings | ${FAIL} failures"
echo "========================================"

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "ACTION REQUIRED: Fix the [FAIL] items above before proceeding."
    echo "See README.md for installation instructions."
    exit 1
elif [ $WARN -gt 0 ]; then
    echo ""
    echo "Warnings present. Review [WARN] items — they may affect some features."
    exit 0
else
    echo ""
    echo "All checks passed. System ready for MAPPO swarm deployment."
    exit 0
fi
