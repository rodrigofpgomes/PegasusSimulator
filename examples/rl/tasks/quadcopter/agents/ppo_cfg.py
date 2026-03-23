"""
PPO Configuration for Quadcopter
Configuração específica de PPO para esta task

Compatível com a arquitetura atual:
- wrapper custom para rsl_rl 2.x
- observações via obs_dict["policy"]
- sem privileged observations separadas
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PPOConfig:
    """
    Configuração PPO para Quadcopter
    Compatível com a arquitetura atual do projeto.
    """

    # ══════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE
    # ══════════════════════════════════════════════════════════════
    network_type: str = "mlp"  # manter "mlp" para equivalência com Isaac Lab

    actor_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "elu"
    init_noise_std: float = 1.0

    # Mantidos por compatibilidade com a tua arquitetura,
    # mesmo não sendo usados no modo MLP.
    hidden_size: int = 256
    num_layers: int = 1

    # ══════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    max_iterations: int = 200

    # ══════════════════════════════════════════════════════════════
    # PPO
    # ══════════════════════════════════════════════════════════════
    clip_param: float = 0.2
    desired_kl: float = 0.01
    entropy_coef: float = 0.0
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True

    # ══════════════════════════════════════════════════════════════
    # OPTIMIZATION
    # ══════════════════════════════════════════════════════════════
    learning_rate: float = 5e-4
    schedule: str = "adaptive"  # Isaac Lab config
    max_grad_norm: float = 1.0

    # ══════════════════════════════════════════════════════════════
    # GAE
    # ══════════════════════════════════════════════════════════════
    gamma: float = 0.99
    lam: float = 0.95

    # ══════════════════════════════════════════════════════════════
    # WRAPPER / IO
    # ══════════════════════════════════════════════════════════════
    clip_obs: float = None
    clip_actions: float = None

    # No snippet do Isaac Lab:
    # actor_obs_normalization=False
    # critic_obs_normalization=False
    #
    # Mantemos estes flags por compatibilidade lógica com a tua arquitetura,
    # mesmo que o wrapper não faça normalização.
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False

    # ══════════════════════════════════════════════════════════════
    # LOGGING / CHECKPOINTS
    # ══════════════════════════════════════════════════════════════
    save_interval: int = 50
    experiment_name: str = "quadcopter_direct"
    run_name: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# PRESETS
# ══════════════════════════════════════════════════════════════════

# Cópia fiel do comportamento que mostraste do Isaac Lab
isaac_lab_exact = PPOConfig()

# Preset alternativo teu para treinos mais longos
tuned = PPOConfig(
    actor_hidden_dims=[256, 256, 256],
    critic_hidden_dims=[256, 256, 256],
    entropy_coef=0.005,
    learning_rate=1e-3,
    num_learning_epochs=8,
    max_iterations=1500,
    experiment_name="quadcopter_ppo",
)

PRESETS = {
    "isaac_lab_exact": isaac_lab_exact,
    "tuned": tuned,
}