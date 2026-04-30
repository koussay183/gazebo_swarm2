"""
PFE - Swarm Drones MARL - PyBullet v6
Formations EXACTES géométriques + positions safe
"""
import math
from config import *


class FormationManager:
    def __init__(self, num_drones=NUM_DRONES):
        self.n = num_drones
        self.formation = DEFAULT_FORMATION
        self.spacing = FORMATION_SPACING
        self.leader_id = INITIAL_LEADER
        self.intruder_pos = None

    def set_formation(self, name):
        if name in FORMATION_TYPES:
            self.formation = name
            return True
        return False

    def get_targets(self, leader_pos, leader_yaw=0.0):
        lx, ly, lz = leader_pos
        if self.formation == "encircle":
            if self.intruder_pos:
                return self._encircle_targets(lz)
            else:
                # Pas d'intrus : cercle autour du leader (tous les drones)
                r = FORMATION_SPACING * 1.5
                targets = {}
                for i in range(self.n):
                    a = 2.0 * math.pi * i / self.n
                    tx, ty = self._safe_pos(lx + r * math.cos(a), ly + r * math.sin(a), lz)
                    targets[i] = (tx, ty, lz)
                return targets
        offsets = self._compute_offsets()
        targets = {}
        cos_y, sin_y = math.cos(leader_yaw), math.sin(leader_yaw)
        for did, (dx, dy) in offsets.items():
            rx = dx * cos_y - dy * sin_y
            ry = dx * sin_y + dy * cos_y
            tx, ty = self._safe_pos(lx + rx, ly + ry, lz)
            targets[did] = (tx, ty, lz)
        return targets

    def _encircle_targets(self, alt):
        """TOUS les drones en cercle EXACT autour de l'intrus."""
        ix, iy = self.intruder_pos
        r = ENCIRCLE_RADIUS
        targets = {}
        for i in range(self.n):
            a = 2.0 * math.pi * i / self.n
            tx, ty = self._safe_pos(ix + r * math.cos(a), iy + r * math.sin(a), alt)
            targets[i] = (tx, ty, alt)
        return targets

    def _safe_pos(self, x, y, z):
        """Pousse la position hors des obstacles."""
        for obs in OBSTACLES:
            ox, oy, sx, sy, h = obs["x"], obs["y"], obs["sx"], obs["sy"], obs["h"]
            m = 1.5
            if (ox-sx/2-m <= x <= ox+sx/2+m and oy-sy/2-m <= y <= oy+sy/2+m and z <= h+2):
                dx, dy = x - ox, y - oy
                if abs(dx) >= abs(dy):
                    x = ox + (sx/2+m+0.5) * (1 if dx >= 0 else -1)
                else:
                    y = oy + (sy/2+m+0.5) * (1 if dy >= 0 else -1)
        for t in TREES:
            d = math.sqrt((x-t["x"])**2 + (y-t["y"])**2)
            if d < t["r"]+1.5 and z <= t["h"]+1:
                if d < 0.01: d = 0.01
                x = t["x"] + (x-t["x"])/d * (t["r"]+2.0)
                y = t["y"] + (y-t["y"])/d * (t["r"]+2.0)
        x = max(PATROL_ZONE["x_min"]+1, min(PATROL_ZONE["x_max"]-1, x))
        y = max(PATROL_ZONE["y_min"]+1, min(PATROL_ZONE["y_max"]-1, y))
        return x, y

    def _compute_offsets(self):
        n, s = self.n, self.spacing
        offsets = {i: (0.0, 0.0) for i in range(n)}
        fol = [i for i in range(n) if i != self.leader_id]

        if self.formation == "v_shape":
            for idx, f in enumerate(fol):
                side = 1 if idx % 2 == 0 else -1
                rank = idx // 2 + 1
                offsets[f] = (-s * rank, side * s * rank)

        elif self.formation == "line":
            for idx, f in enumerate(fol):
                offsets[f] = (-s * (idx + 1), 0.0)

        elif self.formation == "circle":
            r = s * 1.5
            for idx, f in enumerate(fol):
                a = 2.0 * math.pi * idx / len(fol)
                offsets[f] = (r * math.cos(a), r * math.sin(a))

        elif self.formation == "triangle":
            # Rangées : 1, 2, 3, 4 drones
            idx = 0
            for f in fol:
                idx += 1
                row, cnt = 0, 0
                while cnt + row + 2 <= idx:
                    cnt += row + 2; row += 1
                col = idx - cnt
                cols = row + 2
                offsets[f] = (-(row+1)*s, (col - (cols-1)/2.0)*s)

        elif self.formation == "diamond":
            # Losange : leader devant, followers en losange autour
            # Fonctionne pour n=2..8 drones
            diamond_fol = [
                (-s, 0),        # derrière
                (0, -s),        # droite
                (0,  s),        # gauche
                (-2*s, 0),      # loin derrière
                (-s, -s),       # arrière-droite
                (-s,  s),       # arrière-gauche
                (-2*s, -s),
            ]
            for idx, f in enumerate(fol):
                if idx < len(diamond_fol):
                    offsets[f] = diamond_fol[idx]

        elif self.formation == "grid":
            cols = math.ceil(math.sqrt(n))
            idx = 0
            for f in fol:
                idx += 1
                r, c = idx // cols, idx % cols
                offsets[f] = (-r*s, (c - cols/2)*s)

        elif self.formation == "column":
            for idx, f in enumerate(fol):
                offsets[f] = (-s * (idx + 1), 0.0)

        elif self.formation == "echelon_left":
            for idx, f in enumerate(fol):
                offsets[f] = (-s*(idx+1), -s*(idx+1)*0.5)

        elif self.formation == "echelon_right":
            for idx, f in enumerate(fol):
                offsets[f] = (-s*(idx+1), s*(idx+1)*0.5)

        elif self.formation == "wedge":
            for idx, f in enumerate(fol):
                side = 1 if idx % 2 == 0 else -1
                rank = idx // 2 + 1
                offsets[f] = (-s*rank*0.7, side*s*rank*0.7)

        return offsets
