"""
mappo_swarm_core.model_loader
Load a trained Actor checkpoint for deployment inference.
"""

import torch
from .mappo_network import Actor


def load_actor(checkpoint_path: str, obs_size: int, action_size: int = 3) -> Actor:
    """
    Load a trained Actor from a checkpoint file.

    Supports both:
      - checkpoint["actor"] format (saved by MAPPOTrainer.save())
      - raw state_dict format

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt checkpoint file.
    obs_size : int
        Expected observation size (must match the network's input dimension).
    action_size : int
        Number of action dimensions (default: 3 for vx, vy, vz).

    Returns
    -------
    Actor
        The loaded actor in eval mode, ready for deterministic inference.

    Raises
    ------
    RuntimeError
        If the checkpoint's network input dimension does not match obs_size.
    FileNotFoundError
        If checkpoint_path does not exist.
    """
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"  [model_loader] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Determine state dict
    if isinstance(checkpoint, dict) and "actor" in checkpoint:
        state_dict = checkpoint["actor"]
        level = checkpoint.get("level", "?")
        episode = checkpoint.get("episode", "?")
        best_reward = checkpoint.get("best_reward", None)
        keys = list(checkpoint.keys())
    else:
        # Assume raw state dict
        state_dict = checkpoint
        level = "?"
        episode = "?"
        best_reward = None
        keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else []

    print(f"  [model_loader] Checkpoint keys : {keys}")
    if level != "?":
        print(f"  [model_loader] Level          : {level}")
    if episode != "?":
        print(f"  [model_loader] Episode        : {episode}")
    if best_reward is not None:
        print(f"  [model_loader] Best reward    : {best_reward:.3f}")
    print(f"  [model_loader] Expected obs_size: {obs_size}")

    # Detect the actual input dimension from the first linear layer
    first_layer_key = "net.0.weight"
    if first_layer_key in state_dict:
        detected_in_features = state_dict[first_layer_key].shape[1]
        print(f"  [model_loader] Detected model input dim: {detected_in_features}")
        if detected_in_features != obs_size:
            raise RuntimeError(
                f"obs_size mismatch: requested obs_size={obs_size} but checkpoint "
                f"net.0.weight has in_features={detected_in_features}. "
                f"Re-train with the correct NUM_DRONES / OBS_SIZE_PER_DRONE, "
                f"or update config.py to match the checkpoint."
            )
    else:
        print(f"  [model_loader] WARNING: Could not detect input dim (key '{first_layer_key}' missing).")

    # Instantiate and load
    actor = Actor(obs_size, action_size, hidden_size=128)
    actor.load_state_dict(state_dict)
    actor.eval()

    # Sanity-check: run dummy inference
    dummy = torch.zeros(1, obs_size)
    with torch.no_grad():
        action, _ = actor.get_action(dummy, deterministic=True)
    print(f"  [model_loader] Dummy inference OK — output shape: {action.shape}")

    return actor
