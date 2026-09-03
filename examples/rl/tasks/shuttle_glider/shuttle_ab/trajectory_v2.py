"""
| File: trajectory_v2.py
| Drop-in replacement do trajectory.py original.
|
| Novidades (100% retrocompatível com a API antiga `reset` / `current` / `advance`):
|   * `lookahead(env_ids, step_ids, horizon_steps)` -> posicao de referencia H passos
|     a frente, respeitando o replay ping-pong (nao devolve lixo nas fronteiras).
|   * `set_generator(seed)` -> torna a geracao das trajetorias determinista, para que
|     as condicoes A/B/C sejam avaliadas EXATAMENTE nas mesmas referencias.
|   * `is_langevin(env_ids)` -> permite estratificar as metricas por tipo de trajetoria.
"""

from __future__ import annotations

import torch


class RaptorLikeTrajectory:
    """
    Aproxima a trajetoria RAPTOR:
      - mixture: null trajectory ou Langevin-like
      - duracao episode_steps
      - replay ping-pong: forward, depois backward com vel invertida
    """

    def __init__(
        self,
        num_envs: int,
        episode_steps: int,
        dt: float,
        device: str,
        gamma: float = 1.0,
        omega: float = 2.0,
        sigma: float = 0.5,
        mixture_langevin_prob: float = 0.5,
    ):
        self.num_envs = num_envs
        self.episode_steps = episode_steps
        self.dt = dt
        self.device = device

        self.gamma = gamma
        self.omega = omega
        self.sigma = sigma
        self.mixture_langevin_prob = mixture_langevin_prob

        self.pos_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.vel_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.acc_traj = torch.zeros(num_envs, episode_steps, 3, device=device)
        self.step_counter = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.use_langevin = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Gerador opcional para avaliacao determinista (None = usa o RNG global).
        self._generator: torch.Generator | None = None

    # ------------------------------------------------------------------
    # Determinismo para avaliacao A/B
    # ------------------------------------------------------------------
    def set_generator(self, seed: int | None):
        """Fixa (ou liberta) o RNG usado para gerar as trajetorias.

        Chama isto ANTES de avaliar, com o mesmo `seed` em todas as condicoes,
        para que A, B e C vejam exatamente o mesmo conjunto de referencias.
        """
        if seed is None:
            self._generator = None
            return
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed))
        self._generator = gen

    def _rand(self, *shape):
        return torch.rand(*shape, device=self.device, generator=self._generator)

    def _randn(self, *shape):
        return torch.randn(*shape, device=self.device, generator=self._generator)

    # ------------------------------------------------------------------
    # Geracao
    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor, centers: torch.Tensor):
        """centers: (num_envs, 3), normalmente vehicle._init_pos."""
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()
        self.step_counter[env_ids] = 0

        use_langevin = self._rand(n) < self.mixture_langevin_prob
        self.use_langevin[env_ids] = use_langevin

        # default/null trajectory: referencia fixa no spawn
        self.pos_traj[env_ids] = centers[env_ids].unsqueeze(1)
        self.vel_traj[env_ids] = 0.0
        self.acc_traj[env_ids] = 0.0

        if not use_langevin.any():
            return

        langevin_ids = env_ids[use_langevin]
        m = langevin_ids.numel()

        x = centers[langevin_ids].clone()
        v = torch.zeros(m, 3, device=self.device)

        self.pos_traj[langevin_ids, 0] = x
        self.vel_traj[langevin_ids, 0] = v

        center = centers[langevin_ids]

        for t in range(1, self.episode_steps):
            noise = self._randn(m, 3)
            acc = (
                -self.gamma * v
                - (self.omega ** 2) * (x - center)
                + self.sigma * noise
            )
            v = v + self.dt * acc
            x = x + self.dt * v

            self.pos_traj[langevin_ids, t] = x
            self.vel_traj[langevin_ids, t] = v
            self.acc_traj[langevin_ids, t] = acc

        self.acc_traj[langevin_ids, 0] = self.acc_traj[langevin_ids, 1]

    # ------------------------------------------------------------------
    # Indexacao ping-pong (partilhada por current() e lookahead())
    # ------------------------------------------------------------------
    def _resolve_index(self, full_step: torch.Tensor):
        interval = full_step // self.episode_steps
        progress = full_step % self.episode_steps
        forward = (interval % 2) == 0
        index = torch.where(forward, progress, self.episode_steps - progress - 1)
        return index, forward

    def current(self, env_ids: torch.Tensor | None = None, step_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        env_ids = env_ids.to(dtype=torch.long, device=self.device)

        if step_ids is None:
            full_step = self.step_counter[env_ids]
        else:
            full_step = step_ids.to(dtype=torch.long, device=self.device)

        index, forward = self._resolve_index(full_step)

        pos = self.pos_traj[env_ids, index]
        vel = self.vel_traj[env_ids, index]
        acc = self.acc_traj[env_ids, index]

        vel = torch.where(forward.unsqueeze(1), vel, -vel)

        return pos, vel, acc

    def lookahead(
        self,
        env_ids: torch.Tensor,
        step_ids: torch.Tensor,
        horizon_steps: int,
    ) -> torch.Tensor:
        """Posicao de referencia `horizon_steps` a frente, com replay ping-pong.

        Para a null trajectory devolve o proprio ponto fixo, logo a distancia de
        look-ahead e' 0 e o termo de heading fica automaticamente desativado.
        """
        env_ids = env_ids.to(dtype=torch.long, device=self.device)
        full_step = step_ids.to(dtype=torch.long, device=self.device) + int(horizon_steps)
        index, _ = self._resolve_index(full_step)
        return self.pos_traj[env_ids, index]

    def is_langevin(self, env_ids: torch.Tensor) -> torch.Tensor:
        return self.use_langevin[env_ids.to(dtype=torch.long, device=self.device)]

    def advance(self):
        self.step_counter += 1
