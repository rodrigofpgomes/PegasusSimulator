"""
| File: rsl_rl_wrapper.py
| Description: Adapts PegasusEnv to the rsl_rl 2.x VecEnv interface.
| License: BSD-3-Clause.

Compatível com:
- rsl_rl 2.x
- env custom que devolve obs_dict["policy"]
- PPOConfig com clip_obs / clip_actions
- arquitetura sem privileged observations separadas
- sem dependência de gymnasium
- interface próxima do wrapper do Isaac Lab
"""

from __future__ import annotations

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from typing import Any


class _EnvCfgProxy:
    """Proxy para evitar AttributeError quando o logger tenta ler campos ausentes em env.cfg."""

    def __init__(self, env_cfg):
        self._cfg = env_cfg

    def __getattr__(self, name):
        if self._cfg is None:
            return None
        try:
            return getattr(self._cfg, name)
        except AttributeError:
            return None

    def __repr__(self):
        return repr(self._cfg)


class RslRlVecEnvWrapper(VecEnv):
    """Wrapper compatível com RSL-RL, no estilo Isaac Lab, para envs custom.

    Espera que o env wrapped tenha:
    - num_envs
    - num_obs
    - num_actions
    - device
    - max_episode_length
    - cfg (opcional)
    - episode_length_buf
    - reset() -> (obs_dict, extras)
    - step(actions) -> (obs_dict, rewards, terminated, truncated, extras)

    E que as observações relevantes venham em:
    - obs_dict["policy"]
    """

    def __init__(self, env, clip_obs: float | None = 100.0, clip_actions: float | None = 100.0):
        self.env = env
        self.clip_obs = clip_obs
        self.clip_actions = clip_actions

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        # compat com o teu código anterior
        self.num_obs = self.unwrapped.num_obs
        self.num_privileged_obs = None
        self.num_actions = self.unwrapped.num_actions

        # proxy defensivo para cfg
        self._cfg_proxy = _EnvCfgProxy(getattr(self.unwrapped, "cfg", None))

        # igual ao Isaac Lab: reset inicial
        self.env.reset()

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        return str(self)

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    @property
    def cfg(self):
        return self._cfg_proxy

    @property
    def render_mode(self):
        return getattr(self.env, "render_mode", None)

    @property
    def observation_space(self):
        return getattr(self.env, "observation_space", None)

    @property
    def action_space(self):
        return getattr(self.env, "action_space", None)

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self):
        return getattr(self.env, "unwrapped", self.env)

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    def _ensure_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.as_tensor(value, device=self.device)

    def _clip_tensor(self, x: torch.Tensor, limit: float | None) -> torch.Tensor:
        if limit is None:
            return x
        return torch.clamp(x, -limit, limit)

    def _prepare_obs_dict(self, obs_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Mantém compatibilidade com a tua arquitetura atual: só 'policy'."""
        if "policy" not in obs_dict:
            raise KeyError("Expected obs_dict['policy'] in environment observations.")

        policy = self._ensure_tensor(obs_dict["policy"])
        policy = self._clip_tensor(policy, self.clip_obs)

        return {"policy": policy}

    def _to_tensordict(self, obs_dict: dict[str, torch.Tensor]) -> TensorDict:
        return TensorDict(
            obs_dict,
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _extract_log_data(self, extras: dict) -> dict:
        """Compatibilidade com o logger que já tinhas."""
        log = {}

        for key, val in extras.items():
            if key.startswith("Episode_Reward/") or key.startswith("Episode_Termination/"):
                short = (
                    key.replace("Episode_Reward/", "rew/")
                    .replace("Episode_Termination/", "term/")
                )

                if isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        log[short] = float(val.item())
                    else:
                        log[short] = float(val.float().mean().item())
                elif isinstance(val, (int, float)):
                    log[short] = float(val)

        return log

    # --------------------------------------------------------------------- #
    # Operations - MDP
    # --------------------------------------------------------------------- #

    def seed(self, seed: int = -1) -> int:
        if hasattr(self.unwrapped, "seed"):
            return self.unwrapped.seed(seed)
        return seed

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()

        obs_dict = self._prepare_obs_dict(obs_dict)
        obs_td = self._to_tensordict(obs_dict)

        # compatibilidade com o que já tinhas
        extras = {"observations": obs_dict, **extras}

        return obs_td, extras

    def get_observations(self) -> TensorDict:
        """Igual à lógica do Isaac Lab, mas sem dependências extra."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        elif hasattr(self.unwrapped, "_get_observations"):
            obs_dict = self.unwrapped._get_observations()
        elif hasattr(self.unwrapped, "get_observations"):
            obs_dict = self.unwrapped.get_observations()
        else:
            raise AttributeError(
                "Environment must implement observation_manager.compute(), "
                "_get_observations(), or get_observations()."
            )

        obs_dict = self._prepare_obs_dict(obs_dict)
        return self._to_tensordict(obs_dict)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # igual ao Isaac Lab: clip opcional antes do step
        actions = self._ensure_tensor(actions)
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)

        obs_dict = self._prepare_obs_dict(obs_dict)
        obs_td = self._to_tensordict(obs_dict)

        rewards = self._ensure_tensor(rewards)
        terminated = self._ensure_tensor(terminated).bool()
        truncated = self._ensure_tensor(truncated).bool()

        # igual ao Isaac Lab
        dones = (terminated | truncated).to(dtype=torch.long)

        # igual ao Isaac Lab, mas robusto para cfgs incompletos
        is_finite_horizon = getattr(self.cfg, "is_finite_horizon", False)
        if not is_finite_horizon:
            extras["time_outs"] = truncated

        # compatibilidade com o logger que já tinhas
        log = self._extract_log_data(extras)
        if log:
            extras["log"] = log

        # compatibilidade com o teu wrapper antigo
        extras["observations"] = obs_dict

        return obs_td, rewards, dones, extras

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()