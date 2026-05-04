"""
mappo_swarm_core — Core library for MAPPO multi-agent drone swarm.

Exposes the main classes and functions for use in deployment nodes and training.
"""

from .config import (
    NUM_DRONES, OBS_SIZE_PER_DRONE, MAX_SPEED, FORMATION_TYPES,
    FORMATION_SPACING, BATTERY_FULL, COMM_RANGE, PATROL_ZONE,
    OBSTACLES, TREES, WALLS, METRICS, SAFETY_DISTANCE_DRONES,
    SAFETY_DISTANCE_OBSTACLES, INITIAL_LEADER, CONTROL_RATE,
    CRUISE_ALTITUDE, SPAWN_RADIUS, SPAWN_Z, obstacle_avoidance_vector,
)
from .mappo_network import Actor, Critic, MAPPOAgent, SharedCritic
from .model_loader import load_actor
from .observation_builder import ObservationBuilder
from .frame_conversions import (
    enu_position_to_ned, ned_position_to_enu,
    enu_velocity_to_ned, ned_velocity_to_enu,
    enu_yaw_to_ned, ned_yaw_to_enu, clamp_angle,
)
from .safety_filter import SafetyFilter
from .swarm_state import DroneStateData, SwarmState
from .formations import FormationManager
from .leader_election import LeaderElection
from .leader_brain import LeaderBrain, BrainCommand
from .communication_protocol import Message, CommChannel, MESSAGE_TYPES, validate_message
from .metrics import (
    REWARD_CONFIG, compute_obstacle_penalty, compute_drone_collision_penalty,
    compute_formation_reward, compute_out_of_zone_penalty,
)

__all__ = [
    # config
    'NUM_DRONES', 'OBS_SIZE_PER_DRONE', 'MAX_SPEED', 'FORMATION_TYPES',
    'FORMATION_SPACING', 'BATTERY_FULL', 'COMM_RANGE', 'PATROL_ZONE',
    'OBSTACLES', 'TREES', 'WALLS', 'METRICS', 'SAFETY_DISTANCE_DRONES',
    'SAFETY_DISTANCE_OBSTACLES', 'INITIAL_LEADER', 'CONTROL_RATE',
    'CRUISE_ALTITUDE', 'SPAWN_RADIUS', 'SPAWN_Z', 'obstacle_avoidance_vector',
    # networks
    'Actor', 'Critic', 'MAPPOAgent', 'SharedCritic',
    # model loader
    'load_actor',
    # observation
    'ObservationBuilder',
    # frames
    'enu_position_to_ned', 'ned_position_to_enu',
    'enu_velocity_to_ned', 'ned_velocity_to_enu',
    'enu_yaw_to_ned', 'ned_yaw_to_enu', 'clamp_angle',
    # safety
    'SafetyFilter',
    # swarm state
    'DroneStateData', 'SwarmState',
    # formations
    'FormationManager',
    # leader
    'LeaderElection', 'LeaderBrain', 'BrainCommand',
    # communication
    'Message', 'CommChannel', 'MESSAGE_TYPES', 'validate_message',
    # metrics
    'REWARD_CONFIG', 'compute_obstacle_penalty', 'compute_drone_collision_penalty',
    'compute_formation_reward', 'compute_out_of_zone_penalty',
]
