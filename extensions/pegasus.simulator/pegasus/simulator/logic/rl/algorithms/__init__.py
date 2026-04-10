"""
RL Algorithms

Available algorithms:
- ppo: Proximal Policy Optimization
- sac: Soft Actor-Critic (future)
- td3: Twin Delayed DDPG (future)

Each algorithm has:
- runner.py: Main training logic
- __init__.py: Exports train() function
"""

from . import ppo
# from . import sac  # Adicionar quando implementar
# from . import td3  # Adicionar quando implementar

__all__ = ["ppo"]
