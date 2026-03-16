"""
Hiperparâmetros PPO para HoverEnv.
A rede é definida aqui via network_factory.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal
import torch.nn as nn

from pegasus.simulator.logic.rl.networks.base import BaseActorCritic
from pegasus.simulator.logic.rl.networks.mlp  import MlpActorCritic


@dataclass
class HoverPPOCfg:

    # ── rede — factory(obs_dim, act_dim) → BaseActorCritic ───
    # por default: MLP 256-256-128 com ELU
    network_factory: Callable[[int, int], BaseActorCritic] = field(
        default_factory=lambda: lambda obs_dim, act_dim: MlpActorCritic(
            obs_dim     = obs_dim,
            act_dim     = act_dim,
            hidden_dims = [256, 256, 128],
            activation  = "elu",
            init_noise_std = 1.0,
        )
    )

    # ── rollout ───────────────────────────────────────────────
    n_steps_per_env: int = 24
    n_epochs:        int = 5
    n_minibatches:   int = 4

    # ── otimização ────────────────────────────────────────────
    learning_rate:   float = 1e-3
    lr_schedule:     Literal["fixed", "adaptive"] = "adaptive"
    max_grad_norm:   float = 1.0

    # ── PPO ───────────────────────────────────────────────────
    clip_param:             float = 0.2
    entropy_coef:           float = 0.01
    value_loss_coef:        float = 1.0
    use_clipped_value_loss: bool  = True

    # ── GAE ───────────────────────────────────────────────────
    gamma: float = 0.99
    lam:   float = 0.95

    # ── duração e logging ─────────────────────────────────────
    max_iterations: int = 5000
    save_interval:  int = 100
    log_dir:        str = "logs"
    run_name:       str = "hover_ppo"
