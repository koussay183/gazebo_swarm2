# mappo_swarm_px4

PX4 offboard control integration for the MAPPO multi-agent drone swarm.

## Overview

This package bridges the trained MAPPO actor network (from `mappo_swarm_core`) to
real or simulated PX4-based drones via the uXRCE-DDS middleware.

## ENU / NED Convention

| Frame | Used by | X | Y | Z |
|-------|---------|---|---|---|
| ENU   | ROS 2 / Gazebo / MAPPO policy | East | North | Up |
| NED   | PX4 internally / TrajectorySetpoint | North | East | Down |

**Conversion rules used in this package:**

- `TrajectorySetpoint.velocity[0]` = NED North = ENU Y input
- `TrajectorySetpoint.velocity[1]` = NED East  = ENU X input
- `TrajectorySetpoint.velocity[2]` = NED Down  = -ENU Z input

The MAPPO policy outputs ENU velocities. `px4_offboard_manager.py` converts
to NED before publishing.

## Topic Naming Styles

PX4 uXRCE-DDS supports two namespace configurations:

### Namespaced (recommended for multi-drone)

Each PX4 instance is launched with `PX4_UXRCE_DDS_NS=px4_N`:

```
/px4_0/fmu/in/offboard_control_mode
/px4_0/fmu/out/vehicle_status
/px4_1/fmu/in/offboard_control_mode
...
```

### Global (single drone only)

No namespace set:

```
/fmu/in/offboard_control_mode
/fmu/out/vehicle_status
```

Set `px4_topic_style` parameter: `auto` | `namespaced` | `global`

## Safety Defaults

- `auto_arm: false` — drones will NOT arm automatically
- `auto_offboard: false` — offboard mode will NOT be enabled automatically

**Always verify these are false before launching near people or property.**

Enable only after confirming:
1. Gazebo simulation is stable
2. PX4 telemetry topics are visible (`ros2 topic echo /px4_0/fmu/out/vehicle_status`)
3. QGroundControl shows expected drone state
4. uXRCE-DDS agent is running (`MicroXRCEAgent udp4 -p 8888`)

## Nodes

| Node | Executable | Description |
|------|------------|-------------|
| MAPPOPX4RLNode | `mappo_px4_rl_node` | Main RL control node |
| PX4StateListener | `px4_state_listener` | Live status table printer |
| SwarmDiagnostics | `swarm_diagnostics` | Diagnostic publisher |

## Launch Files

```bash
# Full RL stack (safe defaults)
ros2 launch mappo_swarm_px4 mappo_px4_rl.launch.py \
    num_drones:=4 checkpoint_path:=/path/to/best.pt

# Offboard testing without RL
ros2 launch mappo_swarm_px4 px4_offboard_stack.launch.py num_drones:=1
```

## Parameters

See `config/mappo_params.yaml` for full parameter documentation.
Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_drones` | 1 | Number of drones |
| `checkpoint_path` | "" | Path to best.pt |
| `control_rate` | 30.0 | Hz |
| `auto_arm` | false | DANGER: keep false |
| `auto_offboard` | false | DANGER: keep false |
| `px4_topic_style` | auto | Topic naming style |
