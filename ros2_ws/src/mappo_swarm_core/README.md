# mappo_swarm_core

Core library for the MAPPO multi-agent drone swarm (ROS 2 Jazzy, ament_python).

This package is a **pure library** — it has no executable nodes. It is imported
by deployment nodes (e.g. `mappo_px4_rl_node.py`) and by the training package
(`mappo_swarm_training`).

---

## ENU / NED Rule

**All autonomy logic uses ENU** (x=East, y=North, z=Up).

Only the PX4 interface layer (`frame_conversions.py`) converts to/from NED
(x=North, y=East, z=Down) when reading or publishing PX4 uORB topics via
`px4_msgs`.

---

## Module Reference

| Module | Description |
|---|---|
| `config.py` | All shared constants (NUM_DRONES, OBS_SIZE_PER_DRONE, OBSTACLES, METRICS, etc.). `NUM_DRONES` can be overridden at runtime via ROS 2 parameters in `mappo_px4_rl_node.py`. |
| `mappo_network.py` | MAPPO Actor/Critic neural networks. Exact copy preserved for `best.pt` compatibility. |
| `model_loader.py` | `load_actor(checkpoint_path, obs_size)` — loads and validates a trained Actor. Raises `RuntimeError` on obs_size mismatch. |
| `observation_builder.py` | `ObservationBuilder` class — builds per-drone observation vectors from live swarm state. |
| `frame_conversions.py` | ENU<->NED position/velocity/yaw conversions and `clamp_angle()`. |
| `safety_filter.py` | `SafetyFilter` — applies speed clamping, altitude limits, geofence, inter-drone repulsion, and stale-telemetry hold. |
| `swarm_state.py` | `DroneStateData` dataclass + `SwarmState` container. Accepts PX4 messages and Gazebo poses. |
| `formations.py` | `FormationManager` — geometric formation targets for all 11 formation types. Includes `validate_formation_targets()` and `get_targets_safe()`. |
| `leader_election.py` | `LeaderElection` — dynamic leader election with battery/centrality/reliability/ID scoring and hysteresis. |
| `leader_brain.py` | `LeaderBrain` — autonomous mission state machine (PATROL/THREAT/REGROUP/RTB). `tick()` returns a `BrainCommand` dataclass. |
| `communication_protocol.py` | `Message` / `CommChannel` — inter-drone communication simulation with range checking. |
| `metrics.py` | Reward/penalty helper functions shared between training and deployment. |

---

## Dependencies

- ROS 2 Jazzy
- `rclpy`, `std_msgs`, `geometry_msgs`, `nav_msgs`, `sensor_msgs`, `diagnostic_msgs`
- `px4_msgs`, `ros_gz_interfaces`
- Python: `torch`, `numpy`
