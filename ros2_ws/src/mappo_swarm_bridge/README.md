# mappo_swarm_bridge

Gazebo Harmonic to ROS 2 bridge configuration for the MAPPO drone swarm.

## What Gets Bridged and Why

### Clock Bridge (always enabled)

Bridges `/clock` from Gazebo to ROS 2.

**Required** when `use_sim_time: true` in any ROS 2 node. Without this bridge,
nodes with sim time will wait indefinitely for a clock message and their timers
will never fire.

### Pose Bridge (enabled by default)

Bridges `/model/x500_N/pose` (Gazebo) -> `/x500_N/gz/pose` (ROS 2).

**Purpose:** Visualisation in RViz, recording ground-truth trajectories,
and debugging position errors. This data is NOT fed back into the MAPPO
policy or PX4 — the policy uses `VehicleLocalPosition` from PX4 via uXRCE-DDS.

### Sensor Bridge (disabled by default)

Bridges depth camera images, point clouds, lidar scans, and IMU data.

**Only needed if:**
- You are using a PX4 model variant with sensors (gz_x500_depth, gz_x500_lidar)
- You want to record sensor data for analysis
- You are developing a perception pipeline

**Not needed for:** MAPPO RL deployment (policy uses PX4 state estimation).

## Architecture

```
Gazebo Harmonic                   ROS 2 Jazzy
--------------------              ---------------
/model/x500_0/pose  --> bridge --> /x500_0/gz/pose
/model/x500_1/pose  --> bridge --> /x500_1/gz/pose
/clock              --> bridge --> /clock
                                       |
                              RViz2 / logging nodes
```

## Control Path (NOT this package)

PX4 flight control uses a separate path via uXRCE-DDS:

```
PX4 SITL <--> MicroXRCEAgent <--> ROS 2 topics (/px4_N/fmu/...)
```

This bridge package does not touch the control path.

## Topic Naming Note

Gazebo model names depend on how PX4 SITL was launched:
- `make px4_sitl gz_x500` -> model name: `x500`
- Multi-instance with `PX4_GZ_MODEL_NAME=x500_0` -> model name: `x500_0`

Update `config/pose_bridge.yaml` if your model names differ.

## Launch

```bash
# Start pose + clock bridges (default)
ros2 launch mappo_swarm_bridge ros_gz_bridge.launch.py

# Clock only (minimal)
ros2 launch mappo_swarm_bridge ros_gz_bridge.launch.py use_pose_bridge:=False

# Sensors only
ros2 launch mappo_swarm_bridge bridge_sensors.launch.py \
    sensor_bridge_config:=/path/to/my_sensor_bridge.yaml
```
