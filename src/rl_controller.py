"""
PFE - Swarm Drones MARL
Contrôleur RL — mode entraîné (réseau MAPPO) ou mode démo (politique gloutonne)

Mode DÉMO (sans checkpoints/best.pt) :
  - Politique gloutonne 1-étape : évalue des candidats et choisit le meilleur
  - Affiche les récompenses (même fonction que rl_env.py)
  - Démontre l'évitement d'obstacles par récompense négative

Mode ENTRAÎNÉ (avec checkpoints/best.pt) :
  - Réseau Actor MAPPO entraîné
  - Politique optimisée par apprentissage par renforcement
"""

import torch
import numpy as np
import math
import os

from config import *


class RLController:
    """Contrôleur RL — toujours prêt (démo si pas de modèle entraîné)."""

    def __init__(self, checkpoint_path="checkpoints/best.pt"):
        self.actor = None
        self.obs_size = OBS_SIZE_PER_DRONE
        self.pos_scale = 25.0

        # Suivi des infos RL
        self.last_rewards = {}          # drone_id -> valeur reward
        self.last_actions = {}          # drone_id -> (dx, dy, dz)
        self.last_penalty_reasons = {}  # drone_id -> [str, ...] causes des pénalités
        self.rl_applied_to = []         # IDs des drones contrôlés par RL
        self._step_counter = 0

        self.checkpoint_path = checkpoint_path
        self.training_needed = False
        self.training_episodes = 0
        self.training_level = 0
        self._rl_fallback_warned = False

        if os.path.exists(checkpoint_path):
            self._load_model(checkpoint_path)
            self.mode_label = "RÉSEAU MAPPO"
        else:
            self.training_needed = True
            self.mode_label = "DÉMO GLOUTON"
            self._show_training_needed()

    def _show_training_needed(self):
        """Affiche un message clair indiquant qu'un entraînement est nécessaire."""
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  ⚠️  MODÈLE RL NON ENTRAÎNÉ — MODE DÉMO ACTIF               ║
  ║                                                              ║
  ║  Le mode RL fonctionne en mode DÉMO GLOUTON :               ║
  ║    → Politique heuristique basée sur la fonction reward      ║
  ║    → Évitement d'obstacles par maximisation de récompense    ║
  ║    → Pas de mémoire ni d'apprentissage                       ║
  ║                                                              ║
  ║  Pour entraîner un réseau MAPPO complet :                    ║
  ║    python3 train_mappo.py                 (curriculum 1→5)   ║
  ║    python3 train_mappo.py --level 5       (niveau avancé)    ║
  ║    python3 train_mappo.py --episodes 3000 (intensif)         ║
  ║                                                              ║
  ║  Durée estimée : 10–30 min selon GPU/CPU                     ║
  ╚══════════════════════════════════════════════════════════════╝
""")

    def _load_model(self, path):
        try:
            from mappo_network import Actor
            checkpoint = torch.load(path, weights_only=False)
            self.actor = Actor(self.obs_size, 3, hidden_size=128)
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor.eval()
            level = checkpoint.get("level", "?")
            ep = checkpoint.get("episode", "?")
            reward = checkpoint.get("best_reward", None)
            self.training_episodes = ep if isinstance(ep, int) else 0
            self.training_level = level if isinstance(level, int) else 0
            print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  🧠 MODÈLE RL CHARGÉ                                         ║
  ║  Fichier  : {path:<46}║
  ║  Niveau   : {str(level):<46}║
  ║  Épisodes : {str(ep):<46}║""")
            if reward is not None:
                print(f"  ║  Récompense: {reward:<45.3f}║")
            self._check_training_quality(level, ep, reward)
            print(f"  ╚══════════════════════════════════════════════════════════════╝")
        except Exception as e:
            print(f"  ❌ Erreur chargement RL: {e}")
            self.actor = None
            self.training_needed = True
            self._show_training_needed()

    def _check_training_quality(self, level, ep, reward):
        """Indique si plus d'entraînement est recommandé."""
        needs_more = False
        reasons = []
        if isinstance(level, int) and level < 5:
            needs_more = True
            reasons.append(f"Niveau {level}/5 — entraîne jusqu'au niveau 5 pour l'évitement complet")
        if isinstance(ep, int) and ep < 2000:
            needs_more = True
            reasons.append(f"Seulement {ep} épisodes — recommandé: ≥ 3000")
        if reward is not None and reward < 10.0:
            needs_more = True
            reasons.append(f"Récompense {reward:.1f} faible — réseau pas encore convergé")

        if needs_more:
            print(f"  ║  ⚠️  ENTRAÎNEMENT SUPPLÉMENTAIRE RECOMMANDÉ{'':<17}║")
            for r in reasons:
                txt = r[:57]
                print(f"  ║    → {txt:<54}║")
            print(f"  ║  Commande: python3 train_mappo.py --level 5 --load best{'':6}║")
        else:
            print(f"  ║  ✅ Entraînement suffisant — réseau prêt pour la patrouille{'':<2}║")

    def is_ready(self):
        """Toujours True — mode démo si pas de modèle entraîné."""
        return True

    def is_trained(self):
        """True si un modèle MAPPO entraîné est chargé."""
        return self.actor is not None

    def compute_actions(self, drones, formation_mgr, election):
        """Calcule les actions RL pour tous les drones actifs."""
        self._step_counter += 1
        self.rl_applied_to = []
        if self.actor is not None:
            return self._neural_actions(drones, formation_mgr, election)
        else:
            return self._demo_actions(drones, formation_mgr, election)

    # ═══════════════════════════════════════════════════
    # MODE RÉSEAU DE NEURONES (MAPPO entraîné)
    # ═══════════════════════════════════════════════════

    def _neural_actions(self, drones, formation_mgr, election):
        """Actions via réseau Actor MAPPO."""
        from drone import DroneState
        n = len(drones)
        targets = {}

        for i, d in enumerate(drones):
            if d.state in [DroneState.FAILED, DroneState.GROUNDED,
                           DroneState.LANDED, DroneState.IDLE]:
                continue

            obs = np.zeros(self.obs_size, dtype=np.float32)
            idx = 0
            obs[idx:idx+3] = [d.x/self.pos_scale, d.y/self.pos_scale, d.z/self.pos_scale]
            idx += 3
            obs[idx:idx+3] = [d.tx/self.pos_scale, d.ty/self.pos_scale, d.tz/self.pos_scale]
            idx += 3
            for j, od in enumerate(drones):
                if j == i:
                    continue
                if od.state not in [DroneState.FAILED, DroneState.GROUNDED, DroneState.LANDED]:
                    obs[idx:idx+3] = [(od.x-d.x)/self.pos_scale,
                                       (od.y-d.y)/self.pos_scale,
                                       (od.z-d.z)/self.pos_scale]
                idx += 3  # incrémenté seulement pour les autres drones (j != i)
            # Distances + vecteurs direction vers les 4 obstacles les plus proches
            # Format : [d0,d1,d2,d3, dx0,dy0,dx1,dy1,dx2,dy2,dx3,dy3] → 12 floats
            # (même structure que rl_env._get_obs pour cohérence avec le checkpoint)
            obs_infos = []
            for o in OBSTACLES:
                raw_dx = d.x - o["x"]
                raw_dy = d.y - o["y"]
                dist_x = max(0.0, abs(raw_dx) - o.get("sx", 2.0) / 2)
                dist_y = max(0.0, abs(raw_dy) - o.get("sy", 2.0) / 2)
                dist   = math.sqrt(dist_x**2 + dist_y**2)
                norm   = math.sqrt(raw_dx**2 + raw_dy**2) + 1e-6
                obs_infos.append((dist / self.pos_scale, raw_dx / norm, raw_dy / norm))
            obs_infos.sort(key=lambda x: x[0])
            obs_infos = obs_infos[:4]
            while len(obs_infos) < 4:
                obs_infos.append((1.0, 0.0, 0.0))
            for k in range(4):
                obs[idx + k] = obs_infos[k][0]
            idx += 4
            for k in range(4):
                obs[idx]   = obs_infos[k][1]
                obs[idx+1] = obs_infos[k][2]
                idx += 2
            obs[idx] = d.battery / BATTERY_FULL; idx += 1
            obs[idx] = 1.0 if i == election.leader_id else 0.0; idx += 1
            active = sum(1 for dd in drones if dd.state not in
                         [DroneState.FAILED, DroneState.GROUNDED,
                          DroneState.LANDED, DroneState.IDLE])
            obs[idx] = active / n; idx += 1

            # Sanitize observation (guard against NaN/inf from crashed drones)
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            try:
                with torch.no_grad():
                    action, _ = self.actor.get_action(obs_t, deterministic=True)
                action = action.squeeze(0).detach().numpy()
            except RuntimeError as e:
                print(f"  ⚠️  RL inference error ({e}) — reloading model")
                self._load_model(self.checkpoint_path)
                if self.actor is None:
                    return self._fallback_demo(
                        drones, formation_mgr, election,
                        "modèle absent après rechargement")
                try:
                    with torch.no_grad():
                        action, _ = self.actor.get_action(obs_t, deterministic=True)
                    action = action.squeeze(0).detach().numpy()
                except RuntimeError as e2:
                    return self._fallback_demo(
                        drones, formation_mgr, election,
                        f"checkpoint incompatible avec NUM_DRONES/OBS ({e2})")

            # Look-ahead = 1.5s × vitesse max → cible ~3.75m devant
            # drone.update() s'arrête si dist < 0.05m → 0.025m était insuffisant
            look = 1.5
            new_x = d.x + action[0] * MAX_SPEED * look
            new_y = d.y + action[1] * MAX_SPEED * look
            new_z = d.z + action[2] * MAX_SPEED * look * 0.3
            new_x = max(PATROL_ZONE["x_min"]+1, min(PATROL_ZONE["x_max"]-1, new_x))
            new_y = max(PATROL_ZONE["y_min"]+1, min(PATROL_ZONE["y_max"]-1, new_y))
            new_z = max(2.0, min(20.0, new_z))

            targets[i] = (new_x, new_y, new_z)
            r = self._compute_reward_at(new_x, new_y, new_z, d.tx, d.ty, d.tz, drones, i)
            self.last_rewards[i] = r
            self.last_penalty_reasons[i] = (
                self._get_penalty_reasons(new_x, new_y, new_z, drones, i) if r < 0 else [])
            self.rl_applied_to.append(i)

        if len(targets) == 0:
            flying = sum(
                1 for d in drones
                if d.state not in (DroneState.FAILED, DroneState.GROUNDED,
                                   DroneState.LANDED, DroneState.IDLE))
            if flying > 0:
                return self._fallback_demo(
                    drones, formation_mgr, election,
                    "aucune cible RL (tous exclus ou inférence en échec silencieux)")

        return targets

    def _fallback_demo(self, drones, formation_mgr, election, reason):
        """
        En mode RL, les drones en HOVERING ne suivent pas tx,ty : sans entrées dans
        rl_targets ils restent figés. On bascule en démo gloutonne si le réseau échoue.
        """
        if not self._rl_fallback_warned:
            print(f"  ⚠️  RL réseau inutilisable — {reason}")
            print("  →  Bascule politique DÉMO (gloutonne). Pour MAPPO : python3 train_mappo.py")
            print("  →  Ou retire le checkpoint incompatible (ex. entraîné pour 8 drones / obs 34).")
            self._rl_fallback_warned = True
        self.actor = None
        self.mode_label = "DÉMO GLOUTON (fallback)"
        return self._demo_actions(drones, formation_mgr, election)

    # ═══════════════════════════════════════════════════
    # MODE DÉMO — Politique gloutonne (sans modèle)
    # ═══════════════════════════════════════════════════

    def _demo_actions(self, drones, formation_mgr, election):
        """
        Politique gloutonne 1-étape :
          Pour chaque drone, évalue des positions candidates
          et choisit celle qui maximise la récompense RL.
          Démontre l'évitement d'obstacles et le suivi de formation.
        """
        from drone import DroneState
        targets = {}
        self.last_rewards = {}

        leader = drones[election.leader_id]
        dx_l = leader.tx - leader.x
        dy_l = leader.ty - leader.y
        yaw = math.atan2(dy_l, dx_l) if (abs(dx_l) > 0.01 or abs(dy_l) > 0.01) else 0.0
        formation_targets = formation_mgr.get_targets((leader.x, leader.y, leader.z), yaw)

        for i, d in enumerate(drones):
            if d.state in [DroneState.FAILED, DroneState.GROUNDED,
                           DroneState.LANDED, DroneState.IDLE]:
                continue

            # Cible de formation
            if i == election.leader_id:
                tx, ty, tz = d.tx, d.ty, d.tz
            else:
                ft = formation_targets.get(i)
                tx, ty, tz = ft if ft else (d.tx, d.ty, d.tz)

            # Look-ahead 1.5s → cible loin devant, drone.update() peut naviguer vers elle
            step = MAX_SPEED * 1.5

            # ══ Générer positions candidates ══
            dx, dy = tx - d.x, ty - d.y
            dist_to_target = math.sqrt(dx**2 + dy**2)
            candidates = []

            # Vers la cible (direct)
            if dist_to_target > 0.1:
                scale = min(step, dist_to_target)
                nx = d.x + (dx/dist_to_target) * scale
                ny = d.y + (dy/dist_to_target) * scale
                candidates.append((nx, ny, tz))
                candidates.append((nx, ny, tz + 2.0))   # + montée
                candidates.append((nx, ny, max(tz-1, 2.0)))

            # Rester sur place
            candidates.append((d.x, d.y, tz))

            # Monter pour éviter obstacles hauts
            candidates.append((d.x, d.y, min(d.z + step, 20.0)))

            # 8 directions horizontales (contournement)
            for ang_deg in range(0, 360, 45):
                ang = math.radians(ang_deg)
                nx = d.x + step * math.cos(ang)
                ny = d.y + step * math.sin(ang)
                candidates.append((nx, ny, tz))
                # Aussi en montant
                candidates.append((nx, ny, tz + 1.5))

            # ══ Évaluer chaque candidat ══
            best_reward = float('-inf')
            best_pos = (tx, ty, tz)
            for (cx, cy, cz) in candidates:
                cx = max(PATROL_ZONE["x_min"]+1, min(PATROL_ZONE["x_max"]-1, cx))
                cy = max(PATROL_ZONE["y_min"]+1, min(PATROL_ZONE["y_max"]-1, cy))
                cz = max(2.0, min(20.0, cz))
                r = self._compute_reward_at(cx, cy, cz, tx, ty, tz, drones, i)
                if r > best_reward:
                    best_reward = r
                    best_pos = (cx, cy, cz)

            targets[i] = best_pos
            self.last_rewards[i] = best_reward
            self.last_penalty_reasons[i] = (
                self._get_penalty_reasons(*best_pos, drones, i) if best_reward < 0 else [])
            self.rl_applied_to.append(i)

        return targets

    # ═══════════════════════════════════════════════════
    # FONCTION DE RÉCOMPENSE (identique à rl_env.py)
    # ═══════════════════════════════════════════════════

    def _get_penalty_reasons(self, x, y, z, drones, drone_id):
        """Retourne la liste des causes de pénalités pour la position (x,y,z)."""
        from drone import DroneState
        M = METRICS
        reasons = []

        # Obstacles bâtiments
        for obs in OBSTACLES:
            ox, oy = obs["x"], obs["y"]
            sx, sy, h = obs["sx"], obs["sy"], obs["h"]
            dx_o = max(0, abs(x - ox) - sx / 2)
            dy_o = max(0, abs(y - oy) - sy / 2)
            dist_obs = math.sqrt(dx_o**2 + dy_o**2)
            if dist_obs < SAFETY_DISTANCE_OBSTACLES + 1.5 and z <= h + 1.0:
                reasons.append(f"obstacle '{obs['name'][:8]}' {dist_obs:.1f}m")

        # Hors zone de patrouille
        m = 2.0
        if (x <= PATROL_ZONE["x_min"] + m or x >= PATROL_ZONE["x_max"] - m or
                y <= PATROL_ZONE["y_min"] + m or y >= PATROL_ZONE["y_max"] - m):
            reasons.append("hors zone")

        # Proximité / collision inter-drones
        for j, od in enumerate(drones):
            if j == drone_id:
                continue
            if od.state in [DroneState.FAILED, DroneState.GROUNDED, DroneState.LANDED]:
                continue
            d_drone = math.sqrt((x - od.x)**2 + (y - od.y)**2 + (z - od.z)**2)
            if d_drone < M["collision_drone_dist"]:
                reasons.append(f"collision D{j} ({d_drone:.1f}m)")
            elif d_drone < M["near_drone_dist"]:
                reasons.append(f"trop proche D{j} ({d_drone:.1f}m)")

        return reasons

    def _compute_reward_at(self, x, y, z, target_x, target_y, target_z, drones, drone_id):
        """
        Récompense à la position (x, y, z) pour le drone drone_id.
        Même logique que SwarmFormationEnv._compute_rewards() pour cohérence.
        """
        from drone import DroneState
        M = METRICS
        reward = 0.0

        # 1. Distance à la cible (récompense principale, gradient continu)
        dist = math.sqrt((x-target_x)**2 + (y-target_y)**2 + (z-target_z)**2)
        reward += max(0, M["reward_target_gradient"] * (1.0 - dist / 15.0))
        if dist < 1.5:
            reward += M["reward_target_close"]
        elif dist < 3.0:
            reward += M["reward_target_near"]

        # 2. Pénalité collision obstacles (plus forte si plus proche)
        for obs in OBSTACLES:
            ox, oy = obs["x"], obs["y"]
            sx, sy, h = obs["sx"], obs["sy"], obs["h"]
            dx_o = max(0, abs(x - ox) - sx/2)
            dy_o = max(0, abs(y - oy) - sy/2)
            dist_obs = math.sqrt(dx_o**2 + dy_o**2)
            if dist_obs < SAFETY_DISTANCE_OBSTACLES + 1.5 and z <= h + 1.0:
                factor = max(0, 1.0 - dist_obs / (SAFETY_DISTANCE_OBSTACLES + 1.5))
                reward += M["penalty_collision_obs"] * factor

        # 3. Hors zone
        m = 2.0
        if (x <= PATROL_ZONE["x_min"] + m or x >= PATROL_ZONE["x_max"] - m or
                y <= PATROL_ZONE["y_min"] + m or y >= PATROL_ZONE["y_max"] - m):
            reward += M["penalty_out_of_zone"]

        # 4. Inter-drones (évitement de collision)
        for j, od in enumerate(drones):
            if j == drone_id:
                continue
            if od.state in [DroneState.FAILED, DroneState.GROUNDED, DroneState.LANDED]:
                continue
            d_drone = math.sqrt((x-od.x)**2 + (y-od.y)**2 + (z-od.z)**2)
            if d_drone < M["collision_drone_dist"]:
                reward += M["penalty_collision_drone"]
            elif d_drone < M["near_drone_dist"]:
                reward += M["penalty_near_drone"]

        return reward

    # ═══════════════════════════════════════════════════
    # INFOS POUR AFFICHAGE
    # ═══════════════════════════════════════════════════

    def get_last_info(self):
        """Retourne les informations RL du dernier step."""
        return {
            "mode": self.mode_label,
            "rewards": self.last_rewards.copy(),
            "drones": self.rl_applied_to.copy(),
            "trained": self.is_trained(),
        }

    def get_reward_summary(self):
        """Résumé court de la récompense pour affichage GUI."""
        if not self.last_rewards:
            return "—"
        vals = list(self.last_rewards.values())
        avg = sum(vals) / len(vals)
        mn, mx = min(vals), max(vals)
        return f"moy={avg:+.2f} [min={mn:+.1f} max={mx:+.1f}]"

    def print_reward_function(self):
        """Affiche la fonction de récompense complète."""
        M = METRICS
        print("""
  ╔══════════════════════════════════════════════════════╗
  ║           📊 FONCTION DE RÉCOMPENSE RL               ║
  ╠══════════════════════════════════════════════════════╣
  ║  RÉCOMPENSES POSITIVES :                             ║
  ║    +gradient  : approcher la cible formation         ║
  ║    +close     : distance < 1.5m à la cible           ║
  ║    +near      : distance < 3.0m à la cible           ║
  ║    +team      : bonus si TOUS les drones sont proches ║
  ║    +perfect   : formation parfaite atteinte           ║
  ║                                                      ║
  ║  PÉNALITÉS NÉGATIVES :                               ║
  ║    -obs       : trop proche d'un BÂTIMENT            ║
  ║    -collision : collision inter-drone                 ║
  ║    -near      : zone de sécurité inter-drone          ║
  ║    -zone      : sortie de la zone de patrouille       ║
  ║    -energy    : pénalité vitesse (économie énergie)   ║
  ╚══════════════════════════════════════════════════════╝""")
        print(f"  Valeurs configurées (config.py → METRICS):")
        for k, v in M.items():
            print(f"    {k:<30} = {v}")

    def check_training_status(self):
        """Affiche le statut complet d'entraînement avec recommandations."""
        print(f"\n  ╔══════════════════════════════════════════════════════════════╗")
        print(f"  ║              🧠 STATUT ENTRAÎNEMENT RL                       ║")
        print(f"  ╠══════════════════════════════════════════════════════════════╣")
        if self.actor is not None:
            lvl = self.training_level
            ep  = self.training_episodes
            print(f"  ║  Modèle     : CHARGÉ ({self.checkpoint_path}){'':<20}║")
            print(f"  ║  Niveau     : {lvl}/5{'':<47}║")
            print(f"  ║  Épisodes   : {ep}{'':<47}║")
            print(f"  ║  Mode actif : RÉSEAU MAPPO (réseau de neurones entraîné){'':<4}║")
            print(f"  ╠══════════════════════════════════════════════════════════════╣")
            if lvl < 5:
                print(f"  ║  ⚠️  Recommandation : entraîner jusqu'au niveau 5{'':<12}║")
                print(f"  ║     → python3 train_mappo.py --level 5 --load best{'':<11}║")
            elif ep < 2000:
                print(f"  ║  ⚠️  Recommandation : plus d'épisodes (actuel={ep} < 2000){'':<8}║")
                print(f"  ║     → python3 train_mappo.py --load best{'':<21}║")
            else:
                print(f"  ║  ✅ Entraînement suffisant !{'':<33}║")
        else:
            print(f"  ║  Modèle     : ABSENT (aucun checkpoint trouvé){'':<14}║")
            print(f"  ║  Mode actif : DÉMO GLOUTON (politique heuristique){'':<9}║")
            print(f"  ╠══════════════════════════════════════════════════════════════╣")
            print(f"  ║  ❌ ENTRAÎNEMENT NÉCESSAIRE pour le réseau MAPPO{'':<13}║")
            print(f"  ║                                                              ║")
            print(f"  ║  Commandes disponibles :                                     ║")
            print(f"  ║    python3 train_mappo.py              (départ niveau 1)     ║")
            print(f"  ║    python3 train_mappo.py --level 5    (obstacles complets)  ║")
            print(f"  ║    python3 train_mappo.py --episodes 3000 (intensif)         ║")
            print(f"  ║                                                              ║")
            print(f"  ║  En attendant : mode DÉMO fonctionne (reward visible)       ║")
        print(f"  ╚══════════════════════════════════════════════════════════════╝\n")
