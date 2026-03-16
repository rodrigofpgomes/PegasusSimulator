"""
Exemplo de rede própria para HoverEnv.
ResidualActorCritic usa skip connections no trunk partilhado.

Para usar em train.py:
    from tasks.hover.networks.custom_net import ResidualActorCritic
    algo_cfg.network_factory = lambda obs_dim, act_dim: ResidualActorCritic(
        obs_dim, act_dim, hidden=256
    )
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
from pegasus.simulator.logic.rl.networks.base import BaseActorCritic


class ResidualBlock(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ELU(),
            nn.Linear(dim, dim),
        )
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))   # skip connection


class ResidualActorCritic(BaseActorCritic):
    """
    Actor-Critic com trunk residual partilhado.
    Útil quando o MLP simples não converge — os skip connections
    facilitam o gradiente em redes mais fundas.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()

        # trunk partilhado entre actor e critic
        self.embed  = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU())
        self.block1 = ResidualBlock(hidden)
        self.block2 = ResidualBlock(hidden)

        # cabeças separadas
        self.actor_head  = nn.Linear(hidden, act_dim)
        self.critic_head = nn.Linear(hidden, 1)

        # std aprendida — inicializa em ~1.0
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def _trunk(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.embed(obs)
        x = self.block1(x)
        x = self.block2(x)
        return x

    def act(self, obs: torch.Tensor):
        x    = self._trunk(obs)
        mean = self.actor_head(x)
        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        a    = dist.sample()
        return a, dist.log_prob(a).sum(dim=-1)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        x    = self._trunk(obs)
        mean = self.actor_head(x)
        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        return (
            self.critic_head(x).squeeze(-1),
            dist.log_prob(actions).sum(dim=-1),
            dist.entropy().sum(dim=-1),
        )

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor_head(self._trunk(obs))
