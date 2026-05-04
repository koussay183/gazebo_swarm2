"""
PFE - Swarm Drones MARL
Réseau MAPPO (Multi-Agent PPO)

Architecture :
  - Actor : observe son état → choisit une action (vx, vy, vz)
  - Critic : observe l'état GLOBAL → estime la valeur

CTDE = Centralized Training, Decentralized Execution
  - Entraînement : le critic voit TOUT (toutes les positions)
  - Exécution : chaque drone utilise SEULEMENT son actor local
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class Actor(nn.Module):
    """
    Réseau Actor : observation locale → action (vx, vy, vz)

    Entrée : observation d'UN drone (32 valeurs)
    Sortie : moyenne et écart-type des actions (3 valeurs chacun)

    L'action est échantillonnée d'une distribution gaussienne
    pour explorer pendant l'entraînement.
    """
    def __init__(self, obs_size, action_size, hidden_size=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        # Tête moyenne (mu) et log écart-type (log_std)
        self.mu_head = nn.Linear(hidden_size // 2, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, obs):
        """
        obs : (batch, obs_size)
        Retourne : distribution Normal sur les actions
        """
        x = self.net(obs)
        mu = torch.tanh(self.mu_head(x))  # Actions dans [-1, 1]
        std = torch.exp(self.log_std.clamp(-5, 2))
        return Normal(mu, std, validate_args=False)

    def get_action(self, obs, deterministic=False):
        """
        Choisit une action.
        deterministic=True  → prend la moyenne (pas d'exploration)
        deterministic=False → échantillonne (explore)
        """
        dist = self.forward(obs)
        if deterministic:
            action = torch.clamp(dist.mean, -1, 1)
            return action, torch.zeros(action.shape[0])
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, obs, actions):
        """Évalue les log-probabilités d'actions données."""
        dist = self.forward(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy


class Critic(nn.Module):
    """
    Réseau Critic centralisé : état GLOBAL → valeur

    Entrée : toutes les observations concaténées (obs_size × NUM_DRONES)
    Sortie : 1 valeur (estimation de la récompense future)

    Le critic est CENTRALISÉ : il voit l'état complet du swarm
    pour mieux estimer la valeur. Utilisé SEULEMENT pendant
    l'entraînement, pas pendant l'exécution.
    """
    def __init__(self, global_obs_size, hidden_size=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(global_obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, global_obs):
        """
        global_obs : (batch, global_obs_size)
        Retourne : valeur scalaire
        """
        return self.net(global_obs).squeeze(-1)


class MAPPOAgent:
    """
    Agent MAPPO complet pour un drone.

    Chaque drone a :
    - Son propre Actor (réseau local, décisions individuelles)
    - Un Critic partagé (réseau global, entraînement centralisé)
    """
    def __init__(self, obs_size, action_size, global_obs_size,
                 lr_actor=3e-4, lr_critic=1e-3, hidden_size=128):

        self.actor = Actor(obs_size, action_size, hidden_size)
        self.critic = Critic(global_obs_size, hidden_size * 2)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, obs, deterministic=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob = self.actor.get_action(obs_t, deterministic)
        return action.squeeze(0).numpy(), log_prob.item()

    def evaluate_value(self, global_obs):
        with torch.no_grad():
            global_obs_t = torch.FloatTensor(global_obs).unsqueeze(0)
            value = self.critic(global_obs_t)
        return value.item()


class SharedCritic:
    """
    Un seul Critic partagé par tous les agents.
    Plus efficace en mémoire et en entraînement.
    """
    def __init__(self, global_obs_size, lr=1e-3, hidden_size=256):
        self.critic = Critic(global_obs_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def get_value(self, global_obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(global_obs).unsqueeze(0)
            return self.critic(obs_t).item()

    def get_values_batch(self, global_obs_batch):
        obs_t = torch.FloatTensor(global_obs_batch)
        return self.critic(obs_t)
