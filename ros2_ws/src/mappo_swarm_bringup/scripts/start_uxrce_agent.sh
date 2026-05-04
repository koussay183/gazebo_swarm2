#!/bin/bash
# Start uXRCE-DDS Agent — required for PX4 <-> ROS 2 communication
#
# The uXRCE-DDS Agent bridges PX4's internal uORB topics to ROS 2
# via DDS-XRCE protocol over UDP.
#
# Installation options:
#   Option A (snap): sudo snap install micro-xrce-dds-agent
#   Option B (pip):  pip install micro-xrce-dds-agent
#   Option C (build from source):
#     git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
#     cd Micro-XRCE-DDS-Agent && mkdir build && cd build
#     cmake .. && make -j4 && sudo make install
#
# Usage: ./start_uxrce_agent.sh [port]
#   port = UDP port (default: 8888, must match PX4 config)
#
# PX4 default port: 8888 (configured via PX4_UXRCE_DDS_PORT)
# ============================================================

PORT="${1:-8888}"

echo "Starting MicroXRCEAgent on UDP port $PORT ..."

if ! command -v MicroXRCEAgent &> /dev/null; then
    echo "ERROR: MicroXRCEAgent not found."
    echo ""
    echo "Install with one of:"
    echo "  sudo snap install micro-xrce-dds-agent"
    echo "  pip install micro-xrce-dds-agent"
    echo "  (or build from: https://github.com/eProsima/Micro-XRCE-DDS-Agent)"
    exit 1
fi

MicroXRCEAgent udp4 -p "$PORT"
