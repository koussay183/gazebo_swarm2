"""
PFE - Swarm Drones MARL - ROS 2 adaptation
Inter-drone communication protocol.

Adapted from src/communication.py.
"""

import time
import math
from .config import COMM_RANGE, NUM_DRONES


# All valid message types
MESSAGE_TYPES = [
    "HEARTBEAT",
    "POSITION",
    "FORMATION_CMD",
    "ALERT",
    "INTRUDER",
    "ACK",
    "ELECTION",
    "STATUS_REQUEST",
    "STATUS_RESPONSE",
]


class Message:
    """A message between drones."""

    def __init__(self, sender_id, msg_type, data=None, target_id=None):
        self.sender_id = sender_id       # ID of the sending drone
        self.msg_type = msg_type         # Message type (string)
        self.data = data or {}           # Message payload (dict)
        self.target_id = target_id       # None = broadcast, int = unicast
        self.timestamp = time.time()     # Timestamp

    def __repr__(self):
        target = f"->drone_{self.target_id}" if self.target_id is not None else "->ALL"
        return f"[{self.msg_type}] drone_{self.sender_id}{target} | {self.data}"


def validate_message(msg) -> bool:
    """
    Validate that a Message has all required fields and a known type.

    Parameters
    ----------
    msg : Message
        The message to validate.

    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    if not isinstance(msg, Message):
        return False
    if msg.msg_type not in MESSAGE_TYPES:
        return False
    if not isinstance(msg.sender_id, int):
        return False
    if not isinstance(msg.data, dict):
        return False
    return True


class CommChannel:
    """
    Central communication channel.
    Simulates wireless communication between drones.

    How it works:
    1. A drone calls send() to enqueue a message.
    2. The channel checks range (distance < COMM_RANGE).
    3. Messages are stored in each reachable drone's inbox.
    4. Drones call receive() to read their messages.
    """

    def __init__(self, num_drones=NUM_DRONES):
        self.n = num_drones
        self.inboxes = {i: [] for i in range(num_drones)}
        self.positions = {i: (0.0, 0.0, 0.0) for i in range(num_drones)}
        self.last_heartbeat = {i: time.time() for i in range(num_drones)}
        self.total_sent = 0
        self.total_delivered = 0
        self.total_out_of_range = 0
        self.message_log = []

    def update_position(self, drone_id: int, x: float, y: float, z: float) -> None:
        """Update a drone's position (call at each step)."""
        self.positions[drone_id] = (x, y, z)

    def _distance(self, id1: int, id2: int) -> float:
        p1 = self.positions[id1]
        p2 = self.positions[id2]
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def _in_range(self, id1: int, id2: int) -> bool:
        return self._distance(id1, id2) <= COMM_RANGE

    def send(self, message: Message) -> None:
        """Send a message, checking range for each recipient."""
        self.total_sent += 1

        if message.target_id is not None:
            # Unicast
            if self._in_range(message.sender_id, message.target_id):
                self.inboxes[message.target_id].append(message)
                self.total_delivered += 1
            else:
                self.total_out_of_range += 1
        else:
            # Broadcast
            for i in range(self.n):
                if i == message.sender_id:
                    continue
                if self._in_range(message.sender_id, i):
                    self.inboxes[i].append(message)
                    self.total_delivered += 1
                else:
                    self.total_out_of_range += 1

        if message.msg_type == "HEARTBEAT":
            self.last_heartbeat[message.sender_id] = time.time()

        self.message_log.append(message)
        if len(self.message_log) > 100:
            self.message_log.pop(0)

    def receive(self, drone_id: int) -> list:
        """Return and clear all pending messages for a drone."""
        messages = self.inboxes[drone_id]
        self.inboxes[drone_id] = []
        return messages

    def is_alive(self, drone_id: int, timeout: float = 3.0) -> bool:
        """Check if a drone has sent a heartbeat recently."""
        return (time.time() - self.last_heartbeat[drone_id]) < timeout

    def get_neighbors(self, drone_id: int) -> list:
        """Return list of drone IDs within communication range."""
        return [
            i for i in range(self.n)
            if i != drone_id and self._in_range(drone_id, i)
        ]

    def get_stats(self) -> dict:
        """Communication statistics."""
        return {
            "total_sent": self.total_sent,
            "total_delivered": self.total_delivered,
            "total_out_of_range": self.total_out_of_range,
            "delivery_rate": (self.total_delivered / max(1, self.total_sent)) * 100,
        }
