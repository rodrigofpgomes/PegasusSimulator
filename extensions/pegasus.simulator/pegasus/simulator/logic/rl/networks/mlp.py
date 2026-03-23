"""
MLP Actor-Critic standard.
Usado como rede default quando nenhuma rede é especificada.
"""
from typing import List
import torch
import torch.nn as nn
from torch.distributions import Normal
from .base import BaseActorCritic


class MlpActorCritic(BaseActorCritic):

    def __init__(
        self,
        obs_dim:     int,
        act_dim:     int,
        hidden_dims: List[int] = [256, 256, 128],
        activation:  str       = "elu",
        init_noise_std: float  = 1.0,
    ):
        super().__init__()

        act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

        # ── actor ────────────────────────────────────────────
        actor_layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            actor_layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        actor_layers.append(nn.Linear(in_dim, act_dim))
        self.actor = nn.Sequential(*actor_layers)

        # ── critic ───────────────────────────────────────────
        critic_layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            critic_layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # std da distribuição — aprendida
        self.log_std = nn.Parameter(
            torch.ones(act_dim) * torch.log(torch.tensor(init_noise_std))
        )

    # ── interface BaseActorCritic ─────────────────────────────

    def act(self, obs: torch.Tensor):
        mean = self.actor(obs)
        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        actions   = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        mean = self.actor(obs)
        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        values    = self.critic(obs).squeeze(-1)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy   = dist.entropy().sum(dim=-1)
        return values, log_probs, entropy

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.actor(obs)