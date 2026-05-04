# mappo_swarm_gcs

Ground Control Station (GCS) nodes for the MAPPO drone swarm.

## Nodes

### mission_supervisor

CLI-based operator interface. Type commands in the terminal after launch.

```
takeoff         Start autonomous flight
land            Land all drones
patrol          Begin patrol pattern
stop            Hover in place
form <name>     Change formation [v_shape, line, circle, spread, diamond]
rtb             Return to base
estop           EMERGENCY STOP — land all immediately
status          Show last command
help            Show help
quit / exit     Shutdown
```

### swarm_monitor

Prints a live status table every 2s and publishes JSON to `/swarm/monitor_status`.

Columns: D | NAV | ARM | A | O | BATT | N | E | Alt | FORM | RL_ACT | SAFE | DIST

### emergency_control

- Subscribes to `/swarm/emergency_stop`
- 5-second watchdog: auto-estop if no drone heartbeat
- Publishes LAND commands to all drones on activation
- Publishes `/swarm/emergency_active`

### swarm_logger

Writes timestamped CSV files with per-drone state, rotated every 10 minutes.

**Also use rosbag2 for full topic recording:**

```bash
ros2 bag record -a -o swarm_bag
```

## Launch

```bash
ros2 launch mappo_swarm_gcs gcs.launch.py num_drones:=4
```

## Topics Published

| Topic | Type | Description |
|-------|------|-------------|
| `/swarm/mission_command` | String | Operator mission commands |
| `/swarm/formation_command` | String | Formation name changes |
| `/swarm/emergency_stop` | Bool | Emergency stop trigger |
| `/swarm/rtb_command` | Bool | Return to base |
| `/swarm/emergency_active` | Bool | E-stop active status |
| `/swarm/monitor_status` | String | JSON drone status |
