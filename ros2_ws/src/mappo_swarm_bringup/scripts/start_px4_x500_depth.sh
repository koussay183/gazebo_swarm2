#!/bin/bash
# Start single PX4 SITL with depth camera (Intel D455 model)
# Uses gz_x500_depth variant which includes a forward-facing depth camera.
#
# Usage: ./start_px4_x500_depth.sh
#
# After running: enable sensor bridge in mappo_swarm_bridge to access
# depth camera images via ROS 2.
set -e

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"

if [ ! -d "$PX4_DIR" ]; then
    echo "ERROR: PX4-Autopilot not found at $PX4_DIR"
    echo "Clone with:"
    echo "  git clone --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot"
    exit 1
fi

echo "Starting PX4 SITL (X500 + depth camera) from $PX4_DIR ..."
cd "$PX4_DIR"
make px4_sitl gz_x500_depth
