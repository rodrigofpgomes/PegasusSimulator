"""
PPO Configuration for Quadcopter
Configuração específica de PPO para esta task

Author: Rodrigo Gomes
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class PPOConfig:
    """
    Configuração PPO para Quadcopter
    Pode escolher tipo de rede aqui
    """
    
    # ══════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE
    # ══════════════════════════════════════════════════════════════
    network_type: str = "mlp"  # "mlp", "lstm", "gru"
    
    # MLP settings
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    init_noise_std: float = 1.0
    
    # LSTM/GRU settings (se network_type != "mlp")
    hidden_size: int = 256
    num_layers: int = 1
    
    # ══════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════
    num_steps_per_env: int = 24
    num_learning_epochs: int = 8
    num_mini_batches: int = 4
    max_iterations: int = 1500
    
    # ══════════════════════════════════════════════════════════════
    # PPO
    # ══════════════════════════════════════════════════════════════
    clip_param: float = 0.2
    desired_kl: float = 0.01
    entropy_coef: float = 0.005  # balanced: enough exploration without dominating reward
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    
    # ══════════════════════════════════════════════════════════════
    # OPTIMIZATION
    # ══════════════════════════════════════════════════════════════
    learning_rate: float = 1e-3
    schedule: str = "adaptive"  # "adaptive", "linear", "fixed"
    max_grad_norm: float = 1.0
    
    # ══════════════════════════════════════════════════════════════
    # GAE
    # ══════════════════════════════════════════════════════════════
    gamma: float = 0.99
    lam: float = 0.95
    
    clip_obs:     float = 100.0   # observation clipping for rsl_rl wrapper
    clip_actions: float = 100.0   # action clipping for rsl_rl wrapper

    # ══════════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════
    save_interval: int = 50
    experiment_name: str = "quadcopter_ppo"
    run_name: str = None


# ══════════════════════════════════════════════════════════════════
# PRESETS (configs prontas)
# ══════════════════════════════════════════════════════════════════

# Isaac Lab exact (MLP 256-256-256)
isaac_lab = PPOConfig(
    network_type="mlp",
    actor_hidden_dims=[256, 256, 256],
    critic_hidden_dims=[256, 256, 256],
    activation="elu",
    init_noise_std=1.0,
    entropy_coef=0.005,
    learning_rate=1e-3,
    schedule="adaptive",
    max_iterations=1500,
)

# Fast training (MLP menor)
fast = PPOConfig(
    network_type="mlp",
    actor_hidden_dims=[128, 128],
    critic_hidden_dims=[128, 128],
    activation="elu",
    learning_rate=3e-4,
    num_learning_epochs=4,
    max_iterations=500,
)

# LSTM (rede recorrente)
lstm = PPOConfig(
    network_type="lstm",
    hidden_size=256,
    num_layers=1,
    learning_rate=1e-3,
    max_iterations=1500,
)

# Large network (MLP grande)
large = PPOConfig(
    network_type="mlp",
    actor_hidden_dims=[512, 512, 256],
    critic_hidden_dims=[512, 512, 256],
    activation="relu",
    learning_rate=5e-4,
    max_iterations=2000,
)


# Dict de presets
PRESETS = {
    "isaac_lab": isaac_lab,
    "fast": fast,
    "lstm": lstm,
    "large": large,
}