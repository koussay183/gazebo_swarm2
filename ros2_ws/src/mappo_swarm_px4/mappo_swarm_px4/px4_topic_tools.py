"""
px4_topic_tools.py — Utilities for flexible PX4 topic naming.

PX4 with uXRCE-DDS supports two topic naming styles:

  Style A — namespaced (multi-drone, PX4_UXRCE_DDS_NS set per instance):
    /px4_0/fmu/in/offboard_control_mode
    /px4_0/fmu/out/vehicle_status

  Style B — global (single drone, no namespace):
    /fmu/in/offboard_control_mode
    /fmu/out/vehicle_status

  Style 'auto' — uses namespaced for drone_id >= 0, else global.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Known PX4 uXRCE-DDS topics grouped by direction
KNOWN_TOPICS: Dict[str, List[str]] = {
    'in': [
        'offboard_control_mode',
        'trajectory_setpoint',
        'vehicle_command',
    ],
    'out': [
        'vehicle_status',
        'vehicle_local_position',
        'vehicle_odometry',
        'vehicle_command_ack',
    ],
}


def get_px4_topic(
        drone_id: int,
        topic_suffix: str,
        style: str = 'auto',
        prefix: str = '/px4_') -> str:
    """
    Build a fully-qualified PX4 ROS 2 topic name.

    Args:
        drone_id: Integer drone index (0-based).
        topic_suffix: PX4 topic path after fmu/, e.g. 'out/vehicle_status'.
        style: Naming style: 'namespaced', 'global', or 'auto'.
        prefix: Namespace prefix for namespaced style (default '/px4_').

    Returns:
        Full ROS 2 topic string.

    Examples:
        get_px4_topic(0, 'out/vehicle_status')
        -> '/px4_0/fmu/out/vehicle_status'

        get_px4_topic(0, 'out/vehicle_status', style='global')
        -> '/fmu/out/vehicle_status'
    """
    topic_suffix = topic_suffix.lstrip('/')
    prefix_clean = prefix.rstrip('/')

    if style == 'namespaced':
        return f'{prefix_clean}{drone_id}/fmu/{topic_suffix}'
    elif style == 'global':
        return f'/fmu/{topic_suffix}'
    else:
        # 'auto': use namespaced for valid drone_id (>= 0)
        if drone_id >= 0:
            return f'{prefix_clean}{drone_id}/fmu/{topic_suffix}'
        else:
            return f'/fmu/{topic_suffix}'


def get_all_drone_topics(
        num_drones: int,
        style: str = 'auto',
        prefix: str = '/px4_',
        drone_ids: Optional[List[int]] = None) -> Dict[int, Dict[str, str]]:
    """
    Build a mapping of drone_id -> topic_name for all known PX4 topics.

    Args:
        num_drones: Total number of drones.
        style: Topic naming style ('namespaced', 'global', 'auto').
        prefix: Namespace prefix (default '/px4_').
        drone_ids: Optional explicit list of drone IDs. If None, uses 0..num_drones-1.

    Returns:
        Dict mapping drone_id -> {topic_key: full_topic_name, ...}

    Example return for drone 0 with namespaced style::

        {
            0: {
                'offboard_control_mode': '/px4_0/fmu/in/offboard_control_mode',
                'trajectory_setpoint':   '/px4_0/fmu/in/trajectory_setpoint',
                'vehicle_command':       '/px4_0/fmu/in/vehicle_command',
                'vehicle_status':        '/px4_0/fmu/out/vehicle_status',
                'vehicle_local_position': '/px4_0/fmu/out/vehicle_local_position',
                'vehicle_odometry':      '/px4_0/fmu/out/vehicle_odometry',
                'vehicle_command_ack':   '/px4_0/fmu/out/vehicle_command_ack',
            }
        }
    """
    if drone_ids is None:
        drone_ids = list(range(num_drones))

    result: Dict[int, Dict[str, str]] = {}
    for drone_id in drone_ids:
        topics: Dict[str, str] = {}
        for direction, names in KNOWN_TOPICS.items():
            for name in names:
                suffix = f'{direction}/{name}'
                topics[name] = get_px4_topic(
                    drone_id, suffix, style=style, prefix=prefix)
        result[drone_id] = topics

    return result


def describe_topic_style(style: str) -> str:
    """Return human-readable description of a topic naming style."""
    descriptions = {
        'namespaced': (
            'Namespaced: each drone uses /px4_N/fmu/... '
            '(requires PX4_UXRCE_DDS_NS=px4_N per instance)'),
        'global': (
            'Global: all drones share /fmu/... '
            '(single-drone only, no namespace)'),
        'auto': (
            'Auto: namespaced for drone_id >= 0, global otherwise'),
    }
    return descriptions.get(style, f'Unknown style: {style}')
