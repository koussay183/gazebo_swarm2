#!/bin/bash
# Start single PX4 SITL with Gazebo Harmonic (official X500 model)
# PX4-Autopilot must be at ~/PX4-Autopilot (or set PX4_DIR env var)
#
# Usage: ./start_px4_x500_single.sh
#        PX4_DIR=/opt/PX4-Autopilot ./start_px4_x500_single.sh
#
# After running: check QGroundControl connects and
#   ros2 topic echo /fmu/out/vehicle_status
# (global style, no namespace on single drone)
set -e

PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"

if [ ! -d "$PX4_DIR" ]; then
    echo "ERROR: PX4-Autopilot not found at $PX4_DIR"
    echo "Clone with:"
    echo "  git clone --recursive https://github.com/PX4/PX4-Autopilot.git ~/PX4-Autopilot"
    exit 1
fi

echo "Starting PX4 SITL (single X500) from $PX4_DIR ..."
cd "$PX4_DIR"
make px4_sitl gz_x500
