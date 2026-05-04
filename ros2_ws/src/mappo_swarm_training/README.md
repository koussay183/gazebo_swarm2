# mappo_swarm_training

MAPPO training package for the drone swarm (ROS 2 Jazzy, ament_python).

**Training is completely separate from PX4/Gazebo deployment.**

- Training uses a simplified `SwarmFormationEnv` (no real physics, no PX4).
- Deployment uses **only the trained Actor** (`checkpoints/best.pt`).
- The Critic is discarded after training (CTDE: Centralized Training, Decentralized Execution).

---

## Architecture

```
mappo_swarm_training/
  rl_env.py          # SwarmFormationEnv — curriculum training environment
  train_mappo.py     # MAPPOTrainer — PPO + GAE + curriculum scheduling
  evaluate_policy.py # Evaluate best.pt across episodes, outputs CSV
  curriculum.py      # CurriculumScheduler — level advancement logic
  checkpoints/       # Saved .pt checkpoints (best.pt, final.pt, etc.)
  logs/              # Training logs (training_log.json)
```

---

## Training Commands

```bash
# Full curriculum (levels 1 -> 5), 3000 episodes
python3 -m mappo_swarm_training.train_mappo

# Start at level 3
python3 -m mappo_swarm_training.train_mappo --level 3

# Resume at level 5 from best checkpoint
python3 -m mappo_swarm_training.train_mappo --level 5 --load best

# More episodes
python3 -m mappo_swarm_training.train_mappo --episodes 5000

# Test a trained model
python3 -m mappo_swarm_training.train_mappo --test --load best
```

## Evaluation

```bash
python3 -m mappo_swarm_training.evaluate_policy \
    --checkpoint checkpoints/best.pt \
    --episodes 20 \
    --level 5 \
    --output eval_results.csv
```

---

## Curriculum Levels

| Level | Drones | Description |
|---|---|---|
| 1 | 1 | Single drone follows a simple trajectory |
| 2 | 2 | Follower maintains distance from fixed leader |
| 3 | 3 | Triangle formation, fixed leader |
| 4 | N | Dynamic leader, full swarm |
| 5 | N | Dynamic leader + obstacles + limited communication |

---

## Dependencies

- `mappo_swarm_core` (Actor/Critic networks, config)
- Python: `torch`, `numpy`, `gymnasium`
- Optional: `pybullet` (for `--render` visualization during training/testing)
