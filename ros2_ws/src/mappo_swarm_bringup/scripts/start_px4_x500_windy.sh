#!/bin/bash
# Start PX4 SITL with windy world (wind disturbance simulation)
# Uses the 'windy' Gazebo world for testing wind rejection.
#
# Usage: ./start_px4_x500_windy.sh
#
# Note: Wind parameters are defined in the Gazebo 'windy' world file.
# To change wind strength, modify the world or override via gz topic.
set -e

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"

if [ ! -d "$PX4_DIR" ]; then
    echo "ERROR: PX4-Autopilot not found at $PX4_DIR"
    echo "Clone with:"
    echo "  git clone --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot"
    exit 1
fi

echo "Starting PX4 SITL (X500, windy world) from $PX4_DIR ..."
cd "$PX4_DIR"
PX4_GZ_WORLD=windy make px4_sitl gz_x500
