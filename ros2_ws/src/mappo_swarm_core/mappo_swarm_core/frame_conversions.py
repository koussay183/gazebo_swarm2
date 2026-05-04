"""
mappo_swarm_core.frame_conversions
Coordinate-frame conversion utilities for PX4/ROS 2 drone swarm.

Frame definitions
-----------------
ENU (East-North-Up):
  x = East
  y = North
  z = Up
  Used by: all autonomy logic, MAPPO observations, formation targets.

NED (North-East-Down):
  x = North
  y = East
  z = Down
  Used by: PX4 firmware, VehicleLocalPosition, VehicleOdometry, TrajectorySetpoint.

Rule: All autonomy logic (MAPPO, formations, leader brain) operates in ENU.
      Only the PX4 interface layer converts to/from NED when publishing or
      reading PX4 uORB topics via px4_msgs.
"""

import math


# ---------------------------------------------------------------------------
# Position conversions
# ---------------------------------------------------------------------------

def enu_position_to_ned(x: float, y: float, z: float) -> tuple:
    """
    Convert ENU position to NED position.

    ENU (x=East, y=North, z=Up) -> NED (x=North, y=East, z=Down)
    """
    north = y
    east = x
    down = -z
    return north, east, down


def ned_position_to_enu(north: float, east: float, down: float) -> tuple:
    """
    Convert NED position to ENU position.

    NED (x=North, y=East, z=Down) -> ENU (x=East, y=North, z=Up)
    """
    x = east
    y = north
    z = -down
    return x, y, z


# ---------------------------------------------------------------------------
# Velocity conversions
# ---------------------------------------------------------------------------

def enu_velocity_to_ned(vx: float, vy: float, vz: float) -> tuple:
    """
    Convert ENU velocity to NED velocity.

    ENU (vx=East, vy=North, vz=Up) -> NED (vn=North, ve=East, vd=Down)
    """
    vn = vy
    ve = vx
    vd = -vz
    return vn, ve, vd


def ned_velocity_to_enu(vn: float, ve: float, vd: float) -> tuple:
    """
    Convert NED velocity to ENU velocity.

    NED (vn=North, ve=East, vd=Down) -> ENU (vx=East, vy=North, vz=Up)
    """
    vx = ve
    vy = vn
    vz = -vd
    return vx, vy, vz


# ---------------------------------------------------------------------------
# Yaw conversions
# ---------------------------------------------------------------------------

def enu_yaw_to_ned(yaw_enu: float) -> float:
    """
    Convert ENU yaw angle to NED yaw angle.

    ENU yaw: measured counter-clockwise from East (x-axis).
    NED yaw: measured clockwise from North (x-axis).

    Formula: yaw_ned = pi/2 - yaw_enu  (normalised to [-pi, pi])
    """
    yaw_ned = math.pi / 2.0 - yaw_enu
    return clamp_angle(yaw_ned)


def ned_yaw_to_enu(yaw_ned: float) -> float:
    """
    Convert NED yaw angle to ENU yaw angle.

    NED yaw: measured clockwise from North (x-axis).
    ENU yaw: measured counter-clockwise from East (x-axis).

    Formula: yaw_enu = pi/2 - yaw_ned  (normalised to [-pi, pi])
    """
    yaw_enu = math.pi / 2.0 - yaw_ned
    return clamp_angle(yaw_enu)


# ---------------------------------------------------------------------------
# Angle utilities
# ---------------------------------------------------------------------------

def clamp_angle(angle: float) -> float:
    """
    Normalise an angle to the range [-pi, pi].

    Parameters
    ----------
    angle : float
        Angle in radians.

    Returns
    -------
    float
        Equivalent angle in [-pi, pi].
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
