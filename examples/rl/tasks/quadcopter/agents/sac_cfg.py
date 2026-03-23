"""
SAC Configuration for Quadcopter
Configuração específica de SAC para esta task

Author: Rodrigo Gomes
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class SACConfig:
    """
    Configuração SAC para Quadcopter
    """
    
    # ══════════════════════════════════════════════════════════════
    # NETWORK ARCHITECTURE
    # ══════════════════════════════════════════════════════════════
    actor_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    
    # ══════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════
    buffer_size: int = 1000000
    batch_size: int = 256
    max_iterations: int = 1500
    learning_starts: int = 10000
    
    # ══════════════════════════════════════════════════════════════
    # SAC
    # ══════════════════════════════════════════════════════════════
    alpha: float = 0.2  # Entropy coefficient
    tau: float = 0.005  # Target network update rate
    gamma: float = 0.99  # Discount factor
    
    # ══════════════════════════════════════════════════════════════
    # OPTIMIZATION
    # ══════════════════════════════════════════════════════════════
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    
    # ══════════════════════════════════════════════════════════════
    # LOGGING
    # ══════════════════════════════════════════════════════════════
    save_interval: int = 50
    experiment_name: str = "quadcopter_sac"
    run_name: str = None


# ══════════════════════════════════════════════════════════════════
# PRESETS
# ══════════════════════════════════════════════════════════════════

default = SACConfig()

fast = SACConfig(
    max_iterations=500,
    buffer_size=100000,
)

PRESETS = {
    "default": default,
    "fast": fast,
}
