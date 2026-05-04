"""
mappo_swarm_training — MAPPO training package for the drone swarm.

Training is completely separate from PX4/Gazebo deployment.
Only the trained Actor weights (best.pt) are used at deployment.
"""

from .rl_env import SwarmFormationEnv
from .curriculum import CurriculumScheduler

__all__ = [
    'SwarmFormationEnv',
    'CurriculumScheduler',
]
