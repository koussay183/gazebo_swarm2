"""
mappo_swarm_training.evaluate_policy
Evaluate a trained best.pt Actor across multiple episodes.

Usage:
    python3 -m mappo_swarm_training.evaluate_policy \\
        --checkpoint checkpoints/best.pt \\
        --episodes 20 \\
        --level 5 \\
        --output eval_results.csv
"""

import argparse
import csv
import sys
import numpy as np
import torch

from mappo_swarm_core.model_loader import load_actor
from mappo_swarm_core.config import NUM_DRONES, OBS_SIZE_PER_DRONE
from mappo_swarm_training.rl_env import SwarmFormationEnv


def run_evaluation(
    checkpoint_path: str,
    num_episodes: int,
    level: int,
    output_csv: str,
    num_drones: int = NUM_DRONES,
):
    """
    Load the Actor and evaluate it for num_episodes episodes.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt checkpoint.
    num_episodes : int
        Number of evaluation episodes.
    level : int
        Curriculum level to evaluate at (1-5).
    output_csv : str
        Path for CSV output.
    num_drones : int
        Number of drones (overrides config default).

    Returns
    -------
    dict
        Summary statistics.
    """
    obs_size = 3 + 3 + 3 * (num_drones - 1) + 4 + 8 + 1 + 2
    actor = load_actor(checkpoint_path, obs_size=obs_size, action_size=3)

    env = SwarmFormationEnv(
        num_drones=num_drones,
        render_mode="none",
        use_pybullet=False,
        level=level,
    )

    results = []
    print(f"\n  Evaluating {num_episodes} episodes at level {level}...")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  obs_size   : {obs_size}")
    print(f"  drones     : {num_drones}\n")

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(options={"level": level})
        n_active = info.get("num_drones", num_drones)
        total_reward = 0.0
        collisions = 0
        obstacle_violations = 0
        formation_errors = []
        step = 0
        success = False

        for step in range(env.max_steps):
            actions = np.zeros((num_drones, 3), dtype=np.float32)
            for i in range(n_active):
                obs_t = torch.FloatTensor(obs[i]).unsqueeze(0)
                with torch.no_grad():
                    action, _ = actor.get_action(obs_t, deterministic=True)
                actions[i] = action.squeeze(0).numpy()

            obs, rewards, terminated, truncated, info = env.step(actions)
            total_reward += float(rewards[:n_active].mean())
            collisions += info.get("collisions", 0)
            formation_errors.append(info.get("avg_dist", 0.0))

            if terminated:
                success = True
                break
            if truncated:
                break

        mean_formation_error = float(np.mean(formation_errors)) if formation_errors else 0.0
        results.append({
            "episode": ep,
            "total_reward": round(total_reward, 3),
            "formation_error": round(mean_formation_error, 3),
            "collisions": collisions,
            "obstacle_violations": obstacle_violations,
            "length": step + 1,
            "success": int(success),
        })

        status = "OK" if success else "  "
        print(f"  Ep {ep:3d} [{status}]  R={total_reward:8.2f}  "
              f"dist={mean_formation_error:.2f}m  "
              f"col={collisions}  steps={step+1}")

    env.close()

    # Summary
    total_rewards = [r["total_reward"] for r in results]
    formation_errors = [r["formation_error"] for r in results]
    all_collisions = [r["collisions"] for r in results]
    lengths = [r["length"] for r in results]
    successes = [r["success"] for r in results]

    success_rate = sum(successes) / len(successes) * 100
    mean_reward = np.mean(total_rewards)
    mean_formation_error = np.mean(formation_errors)
    total_collisions = sum(all_collisions)
    mean_length = np.mean(lengths)

    print("\n" + "=" * 62)
    print("  EVALUATION SUMMARY")
    print("=" * 62)
    print(f"  Episodes              : {num_episodes}")
    print(f"  Level                 : {level}")
    print(f"  Success rate          : {success_rate:.1f}%")
    print(f"  Mean reward           : {mean_reward:.2f}")
    print(f"  Mean formation error  : {mean_formation_error:.3f} m")
    print(f"  Total collisions      : {total_collisions}")
    print(f"  Mean episode length   : {mean_length:.1f} steps")
    print("=" * 62)

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "total_reward", "formation_error",
                        "collisions", "obstacle_violations", "length", "success"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Results saved to: {output_csv}")

    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "mean_formation_error": mean_formation_error,
        "total_collisions": total_collisions,
        "mean_episode_length": mean_length,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MAPPO Actor")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--level", type=int, default=5,
                        help="Curriculum level to evaluate at (1-5)")
    parser.add_argument("--output", type=str, default="eval_results.csv",
                        help="Output CSV file path")
    parser.add_argument("--drones", type=int, default=NUM_DRONES,
                        help="Number of drones")
    args = parser.parse_args()

    try:
        run_evaluation(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            level=args.level,
            output_csv=args.output,
            num_drones=args.drones,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
