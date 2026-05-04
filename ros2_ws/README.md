# MAPPO Swarm ROS 2 Workspace

Multi-agent drone swarm with MAPPO reinforcement learning, PX4 SITL, and Gazebo Harmonic.

---

## Architecture: 5 Packages

```
ros2_ws/src/
├── mappo_swarm_core      — Core library: MAPPO network, safety, formations, state
├── mappo_swarm_training  — Training scripts (Gymnasium, PPO/MAPPO)
├── mappo_swarm_px4       — PX4 offboard control ROS 2 nodes
├── mappo_swarm_bridge    — Gazebo <-> ROS 2 bridge configurations
├── mappo_swarm_gcs       — Ground control station nodes
└── mappo_swarm_bringup   — Top-level launch files and scripts
```

---

## Data Flow

```
MAPPO Actor (PyTorch)
       |
       | action [vx, vy, vz] ENU, normalized [-1, 1]
       v
Safety Filter (mappo_swarm_core.SafetyFilter)
       |
       | scaled + clamped velocities, ENU
       v
PX4OffboardManager (ENU -> NED conversion)
       |
       | OffboardControlMode + TrajectorySetpoint (NED)
       v
uXRCE-DDS Agent (UDP port 8888)
       |
       | uORB messages
       v
PX4 SITL (Autopilot + controller)
       |
       | motor commands
       v
Gazebo Harmonic (physics simulation)
       |
       | model pose, sensor data
       v
ros_gz_bridge -> ROS 2 topics (VISUALISATION ONLY)
```

---

## ENU / NED Frame Convention

| Frame | Standard | X    | Y     | Z    | Used by |
|-------|----------|------|-------|------|---------|
| ENU   | ROS/Gazebo | East | North | Up   | MAPPO, ROS 2, Gazebo |
| NED   | Aviation/PX4 | North | East | Down | PX4 internally |

**Conversion in `px4_offboard_manager.py`:**

```
TrajectorySetpoint.velocity[0] = NED North = ENU Y (vy input)
TrajectorySetpoint.velocity[1] = NED East  = ENU X (vx input)
TrajectorySetpoint.velocity[2] = NED Down  = -ENU Z (-vz input)
```

---

## Quick Start

### 1. Build

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 2. System Check

```bash
bash src/mappo_swarm_bringup/scripts/check_system.sh
```

### 3. Start PX4

```bash
cd ~/PX4-Autopilot && make px4_sitl gz_x500
```

### 4. Start uXRCE Agent

```bash
MicroXRCEAgent udp4 -p 8888
```

### 5. Launch Full Stack

```bash
ros2 launch mappo_swarm_bringup full_system_existing_px4.launch.py \
    checkpoint_path:=/path/to/best.pt \
    num_drones:=1
```

---

## Full Test Plan: CP0 - CP10

See `src/mappo_swarm_bringup/README.md` for the complete step-by-step test plan.

| Checkpoint | Action |
|-----------|--------|
| CP0 | Source ROS 2, run check_system.sh |
| CP1 | Build workspace with colcon |
| CP2 | Start PX4 SITL + Gazebo |
| CP3 | Start QGroundControl (optional) |
| CP4 | Start uXRCE-DDS agent |
| CP5 | Verify PX4 topics in ROS 2 |
| CP6 | Launch Gazebo-ROS2 bridge |
| CP7 | Launch autonomy (safe mode, no arming) |
| CP8 | Enable offboard mode (still no arming) |
| CP9 | Enable arming (CAUTION: drone arms) |
| CP10 | Validate safety filter and emergency stop |

---

## best.pt Compatibility

The `best.pt` actor checkpoint must be compatible with:
- `OBS_SIZE_PER_DRONE` from `mappo_swarm_core/config.py`
- Actor input dimension: `OBS_SIZE_PER_DRONE * num_drones`
- Actor output dimension: 3 (vx, vy, vz, normalized to [-1, 1])

If the model fails to load, the node falls back to **demo mode** (zero velocity).
Check logs: `ros2 node info /mappo_px4_rl_node`

The checkpoint path is set via:
```bash
ros2 launch ... checkpoint_path:=/absolute/path/to/best.pt
```

Or in `mappo_swarm_px4/config/mappo_params.yaml`.

---

## Debug Quick Reference

| Problem | First Check |
|---------|------------|
| No /fmu topics | Is uXRCE agent running? |
| Offboard rejected | Is heartbeat at >= 2 Hz? |
| Drone flies wrong way | ENU/NED conversion in offboard manager |
| Model load fails | OBS_SIZE_PER_DRONE match? torch version? |
| Clock issues | Is clock bridge running? |
| Emergency stop ignored | Is emergency_control node running? |

---

## Safety Defaults

- `auto_arm: false` — Never enabled automatically
- `auto_offboard: false` — Never enabled automatically
- Safety filter bounds: ±2.5 m/s XY, ±1.0 m/s Z, 2-20m altitude
- Geofence: ±25m XY

Always verify these settings in `config/mappo_params.yaml` before flight.
