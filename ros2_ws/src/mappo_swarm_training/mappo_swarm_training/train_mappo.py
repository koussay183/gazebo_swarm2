"""
PFE - Swarm Drones MARL
Entrainement MAPPO avec CURRICULUM LEARNING

5 Niveaux progressifs :
  Niveau 1 : 1 drone  -> trajectoire simple
  Niveau 2 : 2 drones -> follower maintient distance
  Niveau 3 : 3 drones -> formation triangle
  Niveau 4 : tous les drones -> leader dynamique
  Niveau 5 : tous les drones -> obstacles + comm limitee

Le systeme monte AUTOMATIQUEMENT de niveau quand les criteres sont atteints.

Usage :
    python3 -m mappo_swarm_training.train_mappo                         # Curriculum complet
    python3 -m mappo_swarm_training.train_mappo --level 3               # Commencer au niveau 3
    python3 -m mappo_swarm_training.train_mappo --level 5 --load best   # Reprendre au niveau 5
    python3 -m mappo_swarm_training.train_mappo --test --load best      # Tester
    python3 -m mappo_swarm_training.train_mappo --render                # Avec visualisation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import argparse
import json
from collections import defaultdict

from mappo_swarm_core.config import NUM_DRONES
from mappo_swarm_training.rl_env import SwarmFormationEnv
from mappo_swarm_core.mappo_network import Actor, SharedCritic


class RolloutBuffer:
    """Stocke les trajectoires pour l'entrainement PPO."""

    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.clear()

    def clear(self):
        self.obs = [[] for _ in range(self.num_agents)]
        self.actions = [[] for _ in range(self.num_agents)]
        self.log_probs = [[] for _ in range(self.num_agents)]
        self.rewards = [[] for _ in range(self.num_agents)]
        self.values = [[] for _ in range(self.num_agents)]
        self.dones = [[] for _ in range(self.num_agents)]
        self.global_obs = []

    def add(self, agent_id, obs, action, log_prob, reward, value, done):
        self.obs[agent_id].append(obs)
        self.actions[agent_id].append(action)
        self.log_probs[agent_id].append(log_prob)
        self.rewards[agent_id].append(reward)
        self.values[agent_id].append(value)
        self.dones[agent_id].append(done)

    def add_global_obs(self, global_obs):
        self.global_obs.append(global_obs)

    def compute_gae(self, agent_id, last_value, gamma=0.99, lam=0.95):
        rewards = self.rewards[agent_id]
        values = self.values[agent_id] + [last_value]
        dones = self.dones[agent_id]

        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns


# ============================
# CRITERES DE PASSAGE DE NIVEAU
# ============================

LEVEL_CRITERIA = {
    1: {"min_episodes": 30,  "avg_dist_threshold": 3.0, "success_rate": 0.2},
    2: {"min_episodes": 50,  "avg_dist_threshold": 3.5, "success_rate": 0.2},
    3: {"min_episodes": 80,  "avg_dist_threshold": 4.0, "success_rate": 0.15},
    4: {"min_episodes": 100, "avg_dist_threshold": 5.0, "success_rate": 0.1},
    5: {"min_episodes": 150, "avg_dist_threshold": 6.0, "success_rate": 0.05},
}
# Progression forcee apres N episodes bloques sur un niveau
MAX_EPISODES_PER_LEVEL = 600


class MAPPOTrainer:
    """Entraineur MAPPO avec curriculum learning."""

    def __init__(self, config):
        self.config = config
        self.n = NUM_DRONES
        self.current_level = config.get("start_level", 1)

        # Environnement
        self.env = SwarmFormationEnv(
            num_drones=self.n,
            render_mode="human" if config.get("render", False) else "none",
            use_pybullet=config.get("render", False),
            level=self.current_level,
        )

        obs_size = self.env.obs_size
        action_size = self.env.action_size
        global_obs_size = obs_size * self.n

        # Actor partage
        self.actor = Actor(obs_size, action_size,
                           hidden_size=config.get("hidden_size", 128))
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.get("lr_actor", 3e-4))

        # Critic centralise
        self.critic = SharedCritic(
            global_obs_size, lr=config.get("lr_critic", 3e-4),
            hidden_size=config.get("hidden_size", 128) * 2)

        # Hyperparametres PPO
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_coef = config.get("value_coef", 0.5)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.ppo_epochs = config.get("ppo_epochs", 10)
        self.batch_size = config.get("batch_size", 256)

        # Logging
        self.log = defaultdict(list)
        self.save_dir = config.get("save_dir", "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        # Stats par niveau
        self.level_episodes = 0
        self.level_successes = 0
        self.level_dists = []

    def collect_rollout(self):
        buffer = RolloutBuffer(self.n)
        num_steps = self.env.max_steps

        obs, info = self.env.reset(options={
            "level": self.current_level,
            "formation": self.config.get("formation", None)
        })
        formation = info.get("formation", "unknown")
        n_active = info.get("num_drones", self.n)
        total_rewards = np.zeros(self.n)

        for step in range(num_steps):
            global_obs = obs.flatten()
            buffer.add_global_obs(global_obs)

            actions = np.zeros((self.n, 3))
            for i in range(n_active):
                obs_t = torch.FloatTensor(obs[i]).unsqueeze(0)
                with torch.no_grad():
                    dist = self.actor(obs_t)
                action = dist.sample()
                action = torch.clamp(action, -1, 1)
                log_prob = dist.log_prob(action).sum(dim=-1)

                global_obs_t = torch.FloatTensor(global_obs).unsqueeze(0)
                with torch.no_grad():
                    value = self.critic.critic(global_obs_t).item()

                actions[i] = action.squeeze(0).detach().numpy()
                buffer.add(i, obs[i].copy(), actions[i].copy(),
                           log_prob.item(), 0, value, False)

            for i in range(n_active, self.n):
                buffer.add(i, obs[i].copy(), np.zeros(3), 0.0, 0, 0, False)

            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            done = terminated or truncated

            reward_scale = 50.0
            for i in range(self.n):
                buffer.rewards[i][-1] = rewards[i] / reward_scale
                buffer.dones[i][-1] = float(done)
                total_rewards[i] += rewards[i]

            obs = next_obs
            if done:
                break

            if self.config.get("render", False):
                time.sleep(0.01)

        global_obs = obs.flatten()
        last_value = self.critic.get_value(global_obs)

        return buffer, total_rewards, formation, step + 1, info

    def update(self, buffer):
        actor_losses, critic_losses, entropy_losses = [], [], []
        n_active = self.env.n

        for agent_id in range(n_active):
            if len(buffer.obs[agent_id]) < 2:
                continue

            global_obs = buffer.global_obs[-1]
            last_value = self.critic.get_value(global_obs)
            advantages, returns = buffer.compute_gae(
                agent_id, last_value, self.gamma, self.gae_lambda)

            obs_t = torch.FloatTensor(np.array(buffer.obs[agent_id]))
            actions_t = torch.FloatTensor(np.array(buffer.actions[agent_id]))
            old_log_probs_t = torch.FloatTensor(buffer.log_probs[agent_id])
            advantages_t = torch.FloatTensor(advantages)
            returns_t = torch.FloatTensor(returns)
            global_obs_t = torch.FloatTensor(np.array(
                buffer.global_obs[:len(advantages)]))

            if len(advantages_t) > 1:
                advantages_t = (advantages_t - advantages_t.mean()) / \
                               (advantages_t.std() + 1e-8)

            for epoch in range(self.ppo_epochs):
                indices = np.random.permutation(len(obs_t))
                for start in range(0, len(indices), self.batch_size):
                    end = start + self.batch_size
                    idx = indices[start:end]

                    mb_obs = obs_t[idx]
                    mb_actions = actions_t[idx]
                    mb_old_lp = old_log_probs_t[idx]
                    mb_adv = advantages_t[idx]
                    mb_ret = returns_t[idx]
                    mb_gobs = global_obs_t[idx]

                    new_lp, entropy = self.actor.evaluate(mb_obs, mb_actions)
                    ratio = torch.exp(new_lp - mb_old_lp)

                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                        1 + self.clip_epsilon) * mb_adv
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy_loss = -entropy.mean()

                    values = self.critic.critic(mb_gobs)
                    critic_loss = F.mse_loss(values, mb_ret)

                    total_loss = (actor_loss +
                                  self.value_coef * critic_loss +
                                  self.entropy_coef * entropy_loss)

                    self.actor_optimizer.zero_grad()
                    self.critic.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(),
                                             self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.critic.parameters(),
                                             self.max_grad_norm)
                    self.actor_optimizer.step()
                    self.critic.optimizer.step()

                    actor_losses.append(actor_loss.item())
                    critic_losses.append(critic_loss.item())
                    entropy_losses.append(entropy_loss.item())

        al = np.mean(actor_losses) if actor_losses else 0
        cl = np.mean(critic_losses) if critic_losses else 0
        el = np.mean(entropy_losses) if entropy_losses else 0
        return al, cl, el

    def _check_level_up(self):
        """Verifie si on doit monter de niveau."""
        if self.current_level >= 5:
            return False

        criteria = LEVEL_CRITERIA[self.current_level]

        if self.level_episodes < criteria["min_episodes"]:
            return False

        recent = min(50, len(self.level_dists))
        if recent < 10:
            return False

        recent_dists = self.level_dists[-recent:]
        avg_dist = np.mean(recent_dists)
        success_rate = self.level_successes / max(1, self.level_episodes)

        if (avg_dist < criteria["avg_dist_threshold"] and
                success_rate >= criteria["success_rate"]):
            return True
        return False

    def _level_up(self):
        """Monte d'un niveau."""
        self.current_level += 1
        self.level_episodes = 0
        self.level_successes = 0
        self.level_dists = []
        self.env.set_level(self.current_level)
        print(f"\n  {'='*55}")
        print(f"  LEVEL {self.current_level} REACHED!")
        desc = {
            2: "2 drones — follower maintains distance",
            3: "3 drones — triangle formation",
            4: f"{NUM_DRONES} drones — dynamic leader",
            5: f"{NUM_DRONES} drones — obstacles + limited comm",
        }
        print(f"  {desc.get(self.current_level, '')}")
        print(f"  {'='*55}\n")

    def train(self, num_episodes=3000):
        print("=" * 62)
        print("  MAPPO + CURRICULUM LEARNING")
        print(f"  {self.n} drones max | {num_episodes} episodes")
        print(f"  Start level: {self.current_level}")
        print("=" * 62)

        desc = {
            1: "1 drone — simple trajectory",
            2: "2 drones — follower maintains distance",
            3: "3 drones — triangle formation",
            4: f"{NUM_DRONES} drones — dynamic leader",
            5: f"{NUM_DRONES} drones — obstacles + limited comm",
        }
        print(f"\n  Level {self.current_level}: {desc.get(self.current_level, '')}\n")

        best_avg_reward = -float('inf')
        recent_rewards = []

        for episode in range(1, num_episodes + 1):
            buffer, rewards, formation, steps, info = self.collect_rollout()
            actor_loss, critic_loss, entropy_loss = self.update(buffer)

            ep_reward = rewards[:self.env.n].mean()
            avg_dist = info.get("avg_dist", 0)
            recent_rewards.append(ep_reward)
            if len(recent_rewards) > 100:
                recent_rewards.pop(0)
            avg_reward = np.mean(recent_rewards)

            self.level_episodes += 1
            self.level_dists.append(avg_dist)
            success_threshold = LEVEL_CRITERIA[self.current_level]["avg_dist_threshold"]
            if avg_dist < success_threshold:
                self.level_successes += 1

            self.log["episode"].append(episode)
            self.log["reward"].append(float(ep_reward))
            self.log["avg_reward"].append(float(avg_reward))
            self.log["avg_dist"].append(float(avg_dist))
            self.log["level"].append(self.current_level)
            self.log["formation"].append(formation)
            self.log["actor_loss"].append(float(actor_loss))
            self.log["critic_loss"].append(float(critic_loss))
            self.log["num_drones"].append(self.env.n)

            if episode % 10 == 0:
                sr = self.level_successes / max(1, self.level_episodes)
                print(f"  Ep {episode:4d} | Lv{self.current_level} "
                      f"({self.env.n}d) | R={ep_reward:7.1f} | "
                      f"Avg={avg_reward:7.1f} | Dist={avg_dist:5.2f}m | "
                      f"SR={sr:.0%} | {formation[:12]:<12} | "
                      f"AL={actor_loss:.3f}")

            if avg_reward > best_avg_reward and episode > 50:
                best_avg_reward = avg_reward
                self.save("best")

            if episode % 200 == 0:
                self.save(f"checkpoint_ep{episode}_lv{self.current_level}")

            force_up = (self.current_level < 5 and
                        self.level_episodes >= MAX_EPISODES_PER_LEVEL)
            if self._check_level_up() or force_up:
                if force_up and not self._check_level_up():
                    print(f"\n  Forced progression after {self.level_episodes} "
                          f"episodes at level {self.current_level}")
                self.save(f"level{self.current_level}_complete")
                self._level_up()
                recent_rewards = []

        self.save("final")
        self._save_log()
        print(f"\nTraining complete! Final level: {self.current_level}, "
              f"Best reward: {best_avg_reward:.1f}")
        return self.log

    def save(self, name):
        path = os.path.join(self.save_dir, f"{name}.pt")
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.critic.state_dict(),
            "config": self.config,
            "level": self.current_level,
        }, path)

    def load(self, name):
        path = os.path.join(self.save_dir, f"{name}.pt")
        checkpoint = torch.load(path, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.critic.load_state_dict(checkpoint["critic"])
        if "level" in checkpoint:
            self.current_level = checkpoint["level"]
            self.env.set_level(self.current_level)
        print(f"Model loaded: {path} (level {self.current_level})")

    def _save_log(self):
        path = os.path.join(self.save_dir, "training_log.json")
        log_s = {}
        for key, values in self.log.items():
            log_s[key] = [float(v) if isinstance(v, (np.floating, float))
                          else v for v in values]
        with open(path, "w") as f:
            json.dump(log_s, f, indent=2)

    def test(self, formation=None, level=None, num_episodes=5):
        test_level = level or self.current_level
        print(f"\nTEST — Level {test_level}, {num_episodes} episodes")

        self.env = SwarmFormationEnv(
            num_drones=self.n, render_mode="human",
            use_pybullet=True, level=test_level)

        for ep in range(num_episodes):
            opts = {"level": test_level}
            if formation:
                opts["formation"] = formation
            obs, info = self.env.reset(options=opts)
            form = info["formation"]
            n_active = info["num_drones"]
            total_reward = np.zeros(self.n)

            print(f"\n  Episode {ep+1} — Lv{test_level} ({n_active} drones) "
                  f"— Formation: {form}")

            for step in range(500):
                actions = np.zeros((self.n, 3))
                for i in range(n_active):
                    obs_t = torch.FloatTensor(obs[i]).unsqueeze(0)
                    with torch.no_grad():
                        dist = self.actor(obs_t)
                        action = dist.mean
                    actions[i] = torch.clamp(action, -1, 1).squeeze(0).numpy()

                obs, rewards, terminated, truncated, info = self.env.step(actions)
                total_reward += rewards
                time.sleep(0.02)
                if terminated or truncated:
                    break

            print(f"  -> reward={total_reward[:n_active].mean():.1f}, "
                  f"dist={info.get('avg_dist', 0):.2f}m, "
                  f"collisions={info.get('collisions', 0)}, "
                  f"leader_changes={info.get('leader_changes', 0)}, "
                  f"steps={step+1}")

        self.env.close()


def main():
    parser = argparse.ArgumentParser(
        description="MAPPO Training with Curriculum Learning")
    parser.add_argument("--level", type=int, default=1,
                        help="Start level (1-5)")
    parser.add_argument("--formation", type=str, default=None,
                        help="Specific formation")
    parser.add_argument("--episodes", type=int, default=3000,
                        help="Number of episodes")
    parser.add_argument("--render", action="store_true",
                        help="Enable visualization")
    parser.add_argument("--test", action="store_true",
                        help="Test a model")
    parser.add_argument("--load", type=str, default=None,
                        help="Load a model checkpoint name")
    args = parser.parse_args()

    config = {
        "start_level": args.level,
        "formation": args.formation,
        "render": args.render,
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "ppo_epochs": 10,
        "batch_size": 256,
        "hidden_size": 128,
        "save_dir": "checkpoints",
    }

    trainer = MAPPOTrainer(config)

    if args.load:
        trainer.load(args.load)

    if args.test:
        if not args.load:
            trainer.load("best")
        trainer.test(formation=args.formation, level=args.level)
    else:
        trainer.train(num_episodes=args.episodes)


if __name__ == "__main__":
    main()
