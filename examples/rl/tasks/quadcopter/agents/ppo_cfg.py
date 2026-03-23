"""
| File: agents/ppo_cfg.py
| Description: PPO config for the Quadcopter task.
|              Matches Isaac Lab quadcopter_direct PPO config exactly.
|
| Extra fields vs Isaac Lab:
|   seed        — reproducibility (None = non-deterministic)
|   actor_class — None = rsl_rl MLPModel (Isaac Lab default)
|                 any rsl_rl-compatible class to override the actor network
|   actor_kwargs— extra kwargs forwarded to actor_class constructor
"""


from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PPOConfig:

    # experiment
    seed: Optional[int] = None

    # ── actor network ─────────────────────────────────────────
    # actor_class = None  → rsl_rl MLPModel (Isaac Lab default)
    # actor_class = MyNet → custom network compatible with rsl_rl MLPModel:
    #   __init__(obs, obs_groups, obs_set, output_dim,
    #            hidden_dims, activation, distribution_cfg=None, **actor_kwargs)
    actor_class:  Optional[Any]  = None
    actor_kwargs: Optional[dict] = None

    # network architecture
    network_type: str = "mlp"
    actor_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "elu"
    init_noise_std: float = 1.0

    # rollout
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    
    # PPO hiperparameters
    clip_param: float = 0.2
    desired_kl: float = 0.01
    entropy_coef: float = 0.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True

    # optimisation
    learning_rate: float = 5e-4
    schedule: str = "adaptive"
    max_grad_norm: float = 1.0

    # GAE
    gamma: float = 0.99
    lam: float = 0.95

    # clipping (None = disabled)
    clip_obs: float = None
    clip_actions: float = None

    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False

    # training duration
    max_iterations: int = 200
    save_interval: int = 50


# ══════════════════════════════════════════════════════════════════
# PRESETS
# ══════════════════════════════════════════════════════════════════

isaac_lab = PPOConfig()

tuned = PPOConfig(
    actor_hidden_dims=[256, 256, 256],
    critic_hidden_dims=[256, 256, 256],
    entropy_coef=0.005,
    learning_rate=1e-3,
    num_learning_epochs=8,
    max_iterations=1500
)

PRESETS = {
    "isaac_lab": isaac_lab,
    "tuned": tuned,
}