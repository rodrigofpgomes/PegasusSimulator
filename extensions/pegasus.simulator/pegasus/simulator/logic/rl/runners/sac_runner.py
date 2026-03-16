"""
SAC Runner — off-policy, usa replay buffer.
Compatível com qualquer PegasusEnv.
Recebe a rede da cfg.network_factory.
"""
from __future__ import annotations
import os
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer:

    def __init__(self, obs_dim: int, act_dim: int, size: int, n_envs: int, device: str):
        self.size    = size
        self.n_envs  = n_envs
        self.device  = device
        self.ptr     = 0
        self.full    = False

        self.obs     = torch.zeros(size, obs_dim, device=device)
        self.next_obs = torch.zeros(size, obs_dim, device=device)
        self.actions = torch.zeros(size, act_dim, device=device)
        self.rewards = torch.zeros(size, device=device)
        self.dones   = torch.zeros(size, device=device)

    def add(
        self,
        obs:      torch.Tensor,   # [N, obs_dim]
        next_obs: torch.Tensor,
        actions:  torch.Tensor,
        rewards:  torch.Tensor,
        dones:    torch.Tensor,
    ):
        """Adiciona N transições (uma por env) ao buffer."""
        n = obs.shape[0]
        idxs = torch.arange(self.ptr, self.ptr + n) % self.size
        self.obs[idxs]      = obs
        self.next_obs[idxs] = next_obs
        self.actions[idxs]  = actions
        self.rewards[idxs]  = rewards
        self.dones[idxs]    = dones
        self.ptr  = (self.ptr + n) % self.size
        self.full = self.full or (self.ptr + n >= self.size)

    def sample(self, batch_size: int):
        max_idx = self.size if self.full else self.ptr
        idxs    = torch.randint(0, max_idx, (batch_size,), device=self.device)
        return (
            self.obs[idxs],
            self.next_obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.dones[idxs],
        )

    def __len__(self):
        return self.size if self.full else self.ptr


class SACRunner:

    def __init__(self, env, cfg, world=None):
        self.env    = env
        self.cfg    = cfg
        self.device = env.device

        if world is not None:
            env._world = world

        # ── redes ─────────────────────────────────────────────
        # actor — política estocástica
        self.actor = cfg.network_factory(
            env.num_obs, env.num_actions
        ).to(self.device)

        # SAC precisa de dois critics Q independentes (reduz overestimation)
        # reutiliza a mesma factory mas instancia duas vezes
        self.critic1        = self._make_critic(env.num_obs, env.num_actions)
        self.critic2        = self._make_critic(env.num_obs, env.num_actions)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # target networks não treinam — só são atualizadas por soft update
        for p in self.critic1_target.parameters(): p.requires_grad_(False)
        for p in self.critic2_target.parameters(): p.requires_grad_(False)

        # otimizadores separados
        self.actor_opt    = torch.optim.Adam(self.actor.parameters(),   lr=cfg.learning_rate)
        self.critic1_opt  = torch.optim.Adam(self.critic1.parameters(), lr=cfg.learning_rate)
        self.critic2_opt  = torch.optim.Adam(self.critic2.parameters(), lr=cfg.learning_rate)

        # ── alpha (temperatura de entropia) ───────────────────
        if cfg.ent_coef == "auto":
            # target_entropy = -dim(action) se "auto"
            target_ent = (
                -env.num_actions if cfg.target_entropy == "auto"
                else float(cfg.target_entropy)
            )
            self.target_entropy = target_ent
            self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_opt      = torch.optim.Adam([self.log_alpha], lr=cfg.learning_rate)
            self.alpha          = self.log_alpha.exp().item()
        else:
            self.target_entropy = None
            self.log_alpha      = None
            self.alpha          = float(cfg.ent_coef)

        # ── replay buffer ─────────────────────────────────────
        self.buffer = ReplayBuffer(
            obs_dim = env.num_obs,
            act_dim = env.num_actions,
            size    = cfg.buffer_size,
            n_envs  = env.num_envs,
            device  = self.device,
        )

        self.total_steps = 0
        self.writer      = None

    # ── Q network simples (não é actor-critic — só Q) ─────────

    def _make_critic(self, obs_dim: int, act_dim: int) -> nn.Module:
        """Q(s,a) → escalar. Rede simples MLP."""
        return nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),               nn.ReLU(),
            nn.Linear(256, 1),
        ).to(self.device)

    # ── loop principal ────────────────────────────────────────

    def learn(self):
        if self.cfg.log_dir:
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.cfg.log_dir, self.cfg.run_name)
            )

        obs, _ = self.env.reset()
        obs    = obs["policy"]

        start_time   = time.time()
        episode_step = 0

        while self.total_steps < self.cfg.total_timesteps:

            # ── recolhe step ──────────────────────────────────
            with torch.no_grad():
                if self.total_steps < self.cfg.learning_starts:
                    # exploração aleatória no início
                    actions = torch.rand(
                        self.env.num_envs, self.env.num_actions, device=self.device
                    )
                else:
                    actions, _ = self.actor.act(obs)
                    actions    = actions.clamp(0.0, 1.0)

            next_obs_dict, rewards, terminated, truncated, _ = self.env.step(actions)
            next_obs = next_obs_dict["policy"]
            dones    = (terminated | truncated).float()

            self.buffer.add(obs, next_obs, actions, rewards, dones)
            obs               = next_obs
            self.total_steps += self.env.num_envs
            episode_step     += 1

            # ── update ────────────────────────────────────────
            if (len(self.buffer) >= self.cfg.learning_starts
                    and self.total_steps % self.cfg.train_freq == 0):

                for _ in range(self.cfg.gradient_steps):
                    metrics = self._update()

                if self.writer and self.total_steps % 1000 == 0:
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"train/{k}", v, self.total_steps)

            # ── logging periódico ─────────────────────────────
            if self.total_steps % 10_000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[steps {self.total_steps:8d}/{self.cfg.total_timesteps}] "
                    f"alpha={self.alpha:.4f}  elapsed={elapsed:.0f}s"
                )

            # ── checkpoint ────────────────────────────────────
            if self.total_steps % self.cfg.save_interval == 0 and self.cfg.log_dir:
                self.save(os.path.join(
                    self.cfg.log_dir, self.cfg.run_name,
                    f"checkpoint_{self.total_steps}.pt"
                ))

    # ── update SAC ────────────────────────────────────────────

    def _update(self) -> dict:
        obs, next_obs, actions, rewards, dones = self.buffer.sample(self.cfg.batch_size)

        with torch.no_grad():
            # ação do próximo estado
            next_actions, next_logp = self.actor.act(next_obs)

            # Q target
            sa_next = torch.cat([next_obs, next_actions], dim=-1)
            q1_next = self.critic1_target(sa_next).squeeze(-1)
            q2_next = self.critic2_target(sa_next).squeeze(-1)
            q_next  = torch.min(q1_next, q2_next) - self.alpha * next_logp
            q_target = rewards + self.cfg.gamma * (1 - dones) * q_next

        # ── update critics ────────────────────────────────────
        sa = torch.cat([obs, actions], dim=-1)

        q1_loss = F.mse_loss(self.critic1(sa).squeeze(-1), q_target)
        self.critic1_opt.zero_grad()
        q1_loss.backward()
        self.critic1_opt.step()

        q2_loss = F.mse_loss(self.critic2(sa).squeeze(-1), q_target)
        self.critic2_opt.zero_grad()
        q2_loss.backward()
        self.critic2_opt.step()

        # ── update actor ──────────────────────────────────────
        new_actions, log_probs = self.actor.act(obs)
        sa_new  = torch.cat([obs, new_actions], dim=-1)
        q1_new  = self.critic1(sa_new).squeeze(-1)
        q2_new  = self.critic2(sa_new).squeeze(-1)
        q_new   = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_probs - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── update alpha ──────────────────────────────────────
        if self.log_alpha is not None:
            alpha_loss = -(
                self.log_alpha * (log_probs + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)

        # ── soft update dos targets ───────────────────────────
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "q1_loss":    q1_loss.item(),
            "q2_loss":    q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha":      self.alpha,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        tau = self.cfg.tau
        for sp, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    # ── checkpoint ────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor":   self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "cfg":     self.cfg,
        }, path)
        print(f"[SACRunner] guardado em {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        print(f"[SACRunner] carregado de {path}")
