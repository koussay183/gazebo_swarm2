"""
PFE - Swarm Drones MARL
Système de Communication Inter-Drones

Chaque drone peut :
  - Envoyer des messages broadcast (à tous) ou unicast (à un drone)
  - Recevoir des messages des drones à portée (COMM_RANGE)
  - Envoyer des heartbeats périodiques ("je suis vivant")
  - Détecter les drones qui ne répondent plus

Types de messages :
  HEARTBEAT       → position + batterie + état (automatique)
  POSITION        → broadcast de position
  FORMATION_CMD   → leader envoie l'ordre de formation
  ALERT           → signaler un problème
  INTRUDER        → signaler un intrus détecté
  ACK             → accusé de réception
  ELECTION        → message d'élection de leader
  STATUS_REQUEST  → demander le statut d'un drone
  STATUS_RESPONSE → répondre avec son statut complet
"""

import time
import math
from config import COMM_RANGE, NUM_DRONES


class Message:
    """Un message entre drones."""
    def __init__(self, sender_id, msg_type, data=None, target_id=None):
        self.sender_id = sender_id        # ID du drone émetteur
        self.msg_type = msg_type          # Type de message (string)
        self.data = data or {}            # Contenu du message (dict)
        self.target_id = target_id        # None = broadcast, int = unicast
        self.timestamp = time.time()      # Horodatage

    def __repr__(self):
        target = f"→drone_{self.target_id}" if self.target_id is not None else "→ALL"
        return f"[{self.msg_type}] drone_{self.sender_id}{target} | {self.data}"


class CommChannel:
    """
    Canal de communication central.
    Simule la communication sans fil entre drones.
    
    Fonctionnement :
    1. Un drone appelle send() pour envoyer un message
    2. Le canal vérifie la portée (distance < COMM_RANGE)
    3. Les messages sont stockés dans la boîte de réception de chaque drone à portée
    4. Les drones appellent receive() pour lire leurs messages
    """

    def __init__(self, num_drones=NUM_DRONES):
        self.n = num_drones
        # Boîte de réception par drone
        self.inboxes = {i: [] for i in range(num_drones)}
        # Positions des drones (mises à jour par le système)
        self.positions = {i: (0, 0, 0) for i in range(num_drones)}
        # Dernier heartbeat reçu de chaque drone
        self.last_heartbeat = {i: time.time() for i in range(num_drones)}
        # Compteurs de messages
        self.total_sent = 0
        self.total_delivered = 0
        self.total_out_of_range = 0
        # Log des messages
        self.message_log = []

    def update_position(self, drone_id, x, y, z):
        """Met à jour la position d'un drone (appelé à chaque step)."""
        self.positions[drone_id] = (x, y, z)

    def _distance(self, id1, id2):
        """Calcule la distance entre deux drones."""
        p1 = self.positions[id1]
        p2 = self.positions[id2]
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

    def _in_range(self, id1, id2):
        """Vérifie si deux drones sont à portée de communication."""
        return self._distance(id1, id2) <= COMM_RANGE

    def send(self, message):
        """Envoie un message. Vérifie la portée pour chaque destinataire."""
        self.total_sent += 1

        if message.target_id is not None:
            # Unicast : un seul destinataire
            if self._in_range(message.sender_id, message.target_id):
                self.inboxes[message.target_id].append(message)
                self.total_delivered += 1
            else:
                self.total_out_of_range += 1
        else:
            # Broadcast : tous les drones à portée
            for i in range(self.n):
                if i == message.sender_id:
                    continue
                if self._in_range(message.sender_id, i):
                    self.inboxes[i].append(message)
                    self.total_delivered += 1
                else:
                    self.total_out_of_range += 1

        # Mettre à jour le heartbeat si c'est un heartbeat
        if message.msg_type == "HEARTBEAT":
            self.last_heartbeat[message.sender_id] = time.time()

        # Log (garder les 100 derniers)
        self.message_log.append(message)
        if len(self.message_log) > 100:
            self.message_log.pop(0)

    def receive(self, drone_id):
        """Récupère tous les messages non lus d'un drone. Vide sa boîte."""
        messages = self.inboxes[drone_id]
        self.inboxes[drone_id] = []
        return messages

    def is_alive(self, drone_id, timeout=3.0):
        """Vérifie si un drone a envoyé un heartbeat récemment."""
        return (time.time() - self.last_heartbeat[drone_id]) < timeout

    def get_neighbors(self, drone_id):
        """Liste des drones à portée de communication."""
        neighbors = []
        for i in range(self.n):
            if i != drone_id and self._in_range(drone_id, i):
                neighbors.append(i)
        return neighbors

    def get_stats(self):
        """Statistiques de communication."""
        return {
            "total_sent": self.total_sent,
            "total_delivered": self.total_delivered,
            "total_out_of_range": self.total_out_of_range,
            "delivery_rate": (self.total_delivered / max(1, self.total_sent)) * 100,
        }
