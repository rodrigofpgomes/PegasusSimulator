"""
LSTM Actor-Critic.
Útil para tarefas onde a observação atual não é suficiente
(ex: estimação de vento, perturbações não observáveis).
"""
import torch
import torch.nn as nn
from torch.distributions import Normal
from .base import BaseActorCritic


class LstmActorCritic(BaseActorCritic):

    def __init__(
        self,
        obs_dim:         int,
        act_dim:         int,
        hidden_size:     int   = 256,
        num_layers:      int   = 1,
        init_noise_std:  float = 1.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # encoder LSTM partilhado entre actor e critic
        self.lstm = nn.LSTM(
            input_size  = obs_dim,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
        )

        self.actor_head  = nn.Linear(hidden_size, act_dim)
        self.critic_head = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(
            torch.ones(act_dim) * torch.log(torch.tensor(init_noise_std))
        )

        # estado oculto — mantido entre steps dentro do episódio
        # shape: (num_layers, N, hidden_size)
        self._hx: tuple[torch.Tensor, torch.Tensor] | None = None

    # ── gestão do estado oculto ───────────────────────────────

    def reset_hidden(self, env_ids: torch.Tensor | None = None):
        """
        Chamado pelo VecEnv após reset de episódios.
        Se env_ids=None, limpa todos os envs.
        """
        if self._hx is None or env_ids is None:
            self._hx = None
            return
        self._hx[0][:, env_ids, :] = 0.0
        self._hx[1][:, env_ids, :] = 0.0

    def _step_lstm(self, obs: torch.Tensor):
        """Avança LSTM um step. obs: [N, obs_dim]"""
        # LSTM espera [N, seq_len=1, obs_dim]
        out, self._hx = self.lstm(obs.unsqueeze(1), self._hx)
        return out.squeeze(1)  # [N, hidden_size]

    # ── interface BaseActorCritic ─────────────────────────────

    def act(self, obs: torch.Tensor):
        h    = self._step_lstm(obs)
        mean = self.actor_head(h)
        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        actions   = dist.sample()
        log_probs = dist.log_prob(actions).sum(dim=-1)
        return actions, log_probs

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        # durante o update PPO o estado oculto é reconstruído
        # a partir do rollout completo — não do estado atual
        h = self._step_lstm(obs)
        mean = self.actor_head(h)
        std  = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        values    = self.critic_head(h).squeeze(-1)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy   = dist.entropy().sum(dim=-1)
        return values, log_probs, entropy

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self._step_lstm(obs)
            return self.actor_head(h)
