# mappo_swarm_bringup

Top-level system bringup package for the MAPPO multi-agent drone swarm.

## Test Plan: CP0 - CP10

Work through checkpoints in order. Never skip ahead.

### CP0 — Source ROS 2 and Verify Environment

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# Verify
echo $ROS_DISTRO          # should output: jazzy
ros2 pkg list | grep mappo  # should list all 5 mappo packages
```

Run the automated check:
```bash
bash ~/ros2_ws/src/mappo_swarm_bringup/scripts/check_system.sh
```

All items must pass before proceeding.

### CP1 — Build Workspace

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select \
    mappo_swarm_core mappo_swarm_training \
    mappo_swarm_px4 mappo_swarm_bridge \
    mappo_swarm_gcs mappo_swarm_bringup

# Check for errors
colcon build --event-handlers console_direct+

source install/setup.bash
```

### CP2 — Start PX4 SITL + Gazebo

Open a dedicated terminal (do not close during testing):

```bash
# Single drone (simplest):
cd ~/PX4-Autopilot
make px4_sitl gz_x500

# OR with depth camera:
make px4_sitl gz_x500_depth

# OR in windy conditions:
PX4_GZ_WORLD=windy make px4_sitl gz_x500
```

Wait until you see `Ready for takeoff!` in the PX4 console.
Gazebo window should show the X500 model on the ground.

### CP3 — Start QGroundControl (Optional but Recommended)

```bash
./QGroundControl.AppImage
```

QGC connects automatically. Verify:
- Vehicle shown on map
- Arming status: Disarmed
- Flight mode: Manual or Stabilized

### CP4 — Start uXRCE-DDS Agent

Open another terminal:

```bash
MicroXRCEAgent udp4 -p 8888
```

Wait for connection messages. You should see:
```
[1704067200.000][RTPS Participant Registered]
```

### CP5 — Verify PX4 Topics in ROS 2

```bash
# List available topics
ros2 topic list | grep fmu

# Single drone (global style):
ros2 topic echo /fmu/out/vehicle_status --no-arr

# Multi-drone (namespaced style):
ros2 topic echo /px4_0/fmu/out/vehicle_status --no-arr
```

Expected output includes `nav_state`, `arming_state` fields.
If no topics appear: check uXRCE agent is running and connected.

### CP6 — Launch Gazebo-ROS2 Bridge

```bash
ros2 launch mappo_swarm_bridge ros_gz_bridge.launch.py

# Verify clock is bridged:
ros2 topic echo /clock --once
```

If clock bridge is working, ROS 2 nodes with `use_sim_time: true` will
receive simulation time.

### CP7 — Launch Autonomy (Safe Mode — No Arming)

```bash
ros2 launch mappo_swarm_bringup full_system_existing_px4.launch.py \
    num_drones:=1 \
    checkpoint_path:=/path/to/best.pt \
    auto_arm:=False \
    auto_offboard:=False
```

Verify:
- Node starts without errors
- Diagnostics publishing: `ros2 topic echo /swarm/diagnostics`
- Monitor shows drone status: `ros2 topic echo /swarm/monitor_status`
- If no checkpoint: node logs "demo fallback mode" — this is OK

### CP8 — Enable Offboard Mode (Still No Arming)

In a new terminal, manually switch PX4 to offboard (or via QGC):

```bash
# Via QGC: switch mode to Offboard manually
# OR via PX4 console (inside PX4 SITL terminal):
#   commander mode offboard
```

Then relaunch with offboard publishing enabled:
```bash
ros2 launch mappo_swarm_bringup full_system_existing_px4.launch.py \
    auto_offboard:=True \
    auto_arm:=False
```

Verify: `/px4_0/fmu/in/offboard_control_mode` publishing at 10+ Hz.

### CP9 — Enable Arming (CAUTION: Drone will attempt to arm)

Only proceed if:
- CP6 verified (bridge working)
- CP7 verified (autonomy running without errors)
- CP8 verified (offboard mode switches cleanly)
- You are in simulation (never try this near real hardware without safety observer)

```bash
ros2 launch mappo_swarm_bringup full_system_existing_px4.launch.py \
    checkpoint_path:=/path/to/best.pt \
    auto_offboard:=True \
    auto_arm:=True
```

Verify in Gazebo: drone arms (props spin) and lifts off.

### CP10 — Validate Safety Filter

Test safety limits while in flight:

```bash
# Send emergency stop
ros2 topic pub /swarm/emergency_stop std_msgs/msg/Bool "data: true" --once

# Verify drone lands in Gazebo
```

Test formation commands:
```bash
ros2 topic pub /swarm/formation_command std_msgs/msg/String \
    "data: 'v_shape'" --once
```

---

## Debug Checklist

### PX4 Topics Missing (`ros2 topic list` shows no /fmu topics)

1. Is uXRCE-DDS agent running? (`ps aux | grep MicroXRCE`)
2. Did PX4 SITL fully start? (look for "Ready for takeoff!" in PX4 terminal)
3. Is the port correct? (default 8888; check `PX4_UXRCE_DDS_PORT`)
4. For namespaced topics: was `PX4_UXRCE_DDS_NS` set when PX4 started?
5. Try: `MicroXRCEAgent udp4 -p 8888 -v 6` for verbose logging

### Offboard Mode Fails

1. Is offboard_control_mode being published at >= 2 Hz?
   `ros2 topic hz /px4_0/fmu/in/offboard_control_mode`
2. Is trajectory_setpoint also being published?
   `ros2 topic hz /px4_0/fmu/in/trajectory_setpoint`
3. PX4 requires both before accepting offboard switch.
4. Check QGC: if OFFBOARD_COMMAND_LOST appears, rate is too low.

### Wrong Flight Direction (drone flies in wrong direction)

ENU/NED confusion. Check:
- MAPPO policy outputs ENU (East-North-Up) velocities
- `px4_offboard_manager.py` converts: vn=vy, ve=vx, vd=-vz
- If drone flies south when commanded north: swap vn/ve in offboard manager
- If drone descends when commanded up: check vd = -vz sign

### MAPPO Model Fails to Load

1. Verify checkpoint exists: `ls -la /path/to/best.pt`
2. Check PyTorch version compatibility:
   `python3 -c "import torch; print(torch.__version__)"`
3. Check model was trained with same OBS_SIZE_PER_DRONE as in config
4. Node falls back to demo mode (zero velocity) on load failure
5. Check log: `ros2 node info /mappo_px4_rl_node`

### Bridge Fails (ros_gz_bridge errors)

1. Is Gazebo running? (`gz sim --version`)
2. Do model names match? Check Gazebo model names:
   `gz topic -l | grep model`
3. Update `config/pose_bridge.yaml` with correct model names.
4. Check bridge version compatibility with your Gazebo Harmonic install.

---

## Launch Files Summary

| File | Use Case |
|------|----------|
| `full_system_existing_px4.launch.py` | Full stack (PX4 already running) |
| `autonomy_only.launch.py` | RL node + diagnostics only |
| `monitor_only.launch.py` | Status monitoring, no control |
| `px4_external_sim_assumed.launch.py` | Template with explanatory comments |

## Scripts

All scripts require Unix line endings (LF not CRLF).
Make executable: `chmod +x scripts/*.sh`

| Script | Purpose |
|--------|---------|
| `start_px4_x500_single.sh` | Single drone SITL |
| `start_px4_x500_depth.sh` | Depth camera variant |
| `start_px4_x500_windy.sh` | Windy world testing |
| `start_px4_x500_multi.sh` | Multi-drone (N instances) |
| `start_uxrce_agent.sh` | Start uXRCE-DDS bridge agent |
| `check_system.sh` | Pre-flight system check |
