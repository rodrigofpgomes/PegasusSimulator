"""
Hiperparâmetros SAC para HoverEnv.
SAC é off-policy — precisa de replay buffer maior e learning_starts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from pegasus.simulator.logic.rl.networks.base import BaseActorCritic
from pegasus.simulator.logic.rl.networks.mlp  import MlpActorCritic


@dataclass
class HoverSACCfg:

    # ── rede ──────────────────────────────────────────────────
    network_factory: Callable[[int, int], BaseActorCritic] = field(
        default_factory=lambda: lambda obs_dim, act_dim: MlpActorCritic(
            obs_dim     = obs_dim,
            act_dim     = act_dim,
            hidden_dims = [256, 256],
            activation  = "relu",
            init_noise_std = 0.5,
        )
    )

    # ── otimização ────────────────────────────────────────────
    learning_rate:   float = 3e-4
    batch_size:      int   = 256
    gradient_steps:  int   = 1

    # ── replay buffer ─────────────────────────────────────────
    buffer_size:     int = 1_000_000
    learning_starts: int = 1000
    train_freq:      int = 1

    # ── SAC ───────────────────────────────────────────────────
    gamma:          float = 0.99
    tau:            float = 0.005   # soft update do target network
    ent_coef:       str   = "auto"  # alpha automático
    target_entropy: str   = "auto"  # -dim(action)

    # ── duração e logging ─────────────────────────────────────
    total_timesteps: int = 2_000_000
    save_interval:   int = 50_000
    log_dir:         str = "logs"
    run_name:        str = "hover_sac"
