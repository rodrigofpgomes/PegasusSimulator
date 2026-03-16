"""
PPO Runner — on-policy, funciona com qualquer PegasusEnv.
Recebe a rede da cfg.network_factory — não instancia nada internamente.
"""
from __future__ import annotations
import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class PPORunner:

    def __init__(self, env, cfg, world=None):
        self.env    = env
        self.cfg    = cfg
        self.device = env.device

        # injeta referência ao world para o env poder chamar world.step()
        if world is not None:
            env._world = world

        # ── instancia a rede a partir da factory na cfg ───────
        self.network = cfg.network_factory(
            env.num_obs,
            env.num_actions,
        ).to(self.device)

        # se a rede tem estado oculto (LSTM), regista o callback de reset
        if hasattr(self.network, "reset_hidden"):
            env.register_reset_callback(self.network.reset_hidden)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=cfg.learning_rate,
        )

        # ── buffers de rollout ────────────────────────────────
        N  = env.num_envs
        T  = cfg.n_steps_per_env
        obs_dim = env.num_obs
        act_dim = env.num_actions

        self.obs_buf       = torch.zeros(T, N, obs_dim, device=self.device)
        self.act_buf       = torch.zeros(T, N, act_dim, device=self.device)
        self.logp_buf      = torch.zeros(T, N,          device=self.device)
        self.rew_buf       = torch.zeros(T, N,          device=self.device)
        self.val_buf       = torch.zeros(T, N,          device=self.device)
        self.done_buf      = torch.zeros(T, N,          device=self.device)

        # logging
        self.writer     = None
        self.total_steps = 0

    # ── loop principal ────────────────────────────────────────

    def learn(self):
        if self.cfg.log_dir:
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.cfg.log_dir, self.cfg.run_name)
            )

        obs, _ = self.env.reset()
        obs    = obs["policy"]

        start_time = time.time()

        for iteration in range(self.cfg.max_iterations):

            # ── recolhe rollout ───────────────────────────────
            mean_reward = self._collect_rollout(obs)

            # ── calcula vantagens ─────────────────────────────
            advantages, returns = self._compute_gae()

            # ── update PPO ────────────────────────────────────
            mean_loss = self._ppo_update(advantages, returns)

            # ── obs iniciais para próximo rollout ─────────────
            obs = self.obs_buf[-1]

            # ── logging ───────────────────────────────────────
            if self.writer:
                self.writer.add_scalar("train/mean_reward", mean_reward, iteration)
                self.writer.add_scalar("train/loss",        mean_loss,   iteration)

            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{iteration:5d}/{self.cfg.max_iterations}] "
                    f"reward={mean_reward:7.3f}  loss={mean_loss:6.4f}  "
                    f"elapsed={elapsed:.0f}s"
                )

            # ── checkpoint ────────────────────────────────────
            if iteration % self.cfg.save_interval == 0 and self.cfg.log_dir:
                self.save(os.path.join(
                    self.cfg.log_dir, self.cfg.run_name,
                    f"checkpoint_{iteration}.pt"
                ))

    # ── recolha de rollout ────────────────────────────────────

    @torch.no_grad()
    def _collect_rollout(self, obs: torch.Tensor) -> float:
        total_reward = 0.0

        for t in range(self.cfg.n_steps_per_env):
            actions, log_probs = self.network.act(obs)
            values, _, _       = self.network.evaluate(obs, actions)

            next_obs, reward, terminated, truncated, _ = self.env.step(actions)
            next_obs = next_obs["policy"]
            done     = (terminated | truncated).float()

            self.obs_buf[t]  = obs
            self.act_buf[t]  = actions
            self.logp_buf[t] = log_probs
            self.rew_buf[t]  = reward
            self.val_buf[t]  = values
            self.done_buf[t] = done

            obs           = next_obs
            total_reward += reward.mean().item()

        return total_reward / self.cfg.n_steps_per_env

    # ── GAE ───────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_gae(self):
        T  = self.cfg.n_steps_per_env
        N  = self.env.num_envs

        advantages = torch.zeros_like(self.rew_buf)
        last_gae   = torch.zeros(N, device=self.device)

        # bootstrap value do último estado
        last_obs    = self.obs_buf[-1]
        last_val, _, _ = self.network.evaluate(last_obs, self.act_buf[-1])

        for t in reversed(range(T)):
            next_val  = last_val if t == T - 1 else self.val_buf[t + 1]
            next_done = self.done_buf[t]

            delta    = (self.rew_buf[t]
                        + self.cfg.gamma * next_val * (1 - next_done)
                        - self.val_buf[t])
            last_gae = delta + self.cfg.gamma * self.cfg.lam * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + self.val_buf
        # normaliza vantagens
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    # ── update PPO ────────────────────────────────────────────

    def _ppo_update(self, advantages, returns) -> float:
        T, N = self.cfg.n_steps_per_env, self.env.num_envs

        obs_flat     = self.obs_buf.view(T * N, -1)
        act_flat     = self.act_buf.view(T * N, -1)
        logp_old_flat = self.logp_buf.view(T * N)
        adv_flat     = advantages.view(T * N)
        ret_flat     = returns.view(T * N)

        total_loss = 0.0
        batch_size = T * N // self.cfg.n_minibatches

        for _ in range(self.cfg.n_epochs):
            perm = torch.randperm(T * N, device=self.device)

            for start in range(0, T * N, batch_size):
                idx = perm[start: start + batch_size]

                values, log_probs, entropy = self.network.evaluate(
                    obs_flat[idx], act_flat[idx]
                )

                ratio = (log_probs - logp_old_flat[idx]).exp()
                adv   = adv_flat[idx]

                # policy loss com clip
                loss_p1 = -adv * ratio
                loss_p2 = -adv * ratio.clamp(
                    1 - self.cfg.clip_param,
                    1 + self.cfg.clip_param,
                )
                policy_loss = torch.max(loss_p1, loss_p2).mean()

                # value loss
                if self.cfg.use_clipped_value_loss:
                    v_clipped = (self.val_buf.view(T * N)[idx]
                                 + (values - self.val_buf.view(T * N)[idx])
                                 .clamp(-self.cfg.clip_param, self.cfg.clip_param))
                    value_loss = torch.max(
                        (values - ret_flat[idx]) ** 2,
                        (v_clipped - ret_flat[idx]) ** 2,
                    ).mean()
                else:
                    value_loss = ((values - ret_flat[idx]) ** 2).mean()

                loss = (policy_loss
                        + self.cfg.value_loss_coef * value_loss
                        - self.cfg.entropy_coef * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss / (self.cfg.n_epochs * self.cfg.n_minibatches)

    # ── checkpoint ────────────────────────────────────────────

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "network":   self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "cfg":       self.cfg,
        }, path)
        print(f"[PPORunner] guardado em {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        print(f"[PPORunner] carregado de {path}")
