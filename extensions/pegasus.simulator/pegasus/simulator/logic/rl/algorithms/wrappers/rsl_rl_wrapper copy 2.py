"""
| File: rsl_rl_wrapper.py
| Description: Adapts PegasusEnv to the rsl_rl 2.x VecEnv interface.
| License: BSD-3-Clause.

Compatível com:
- rsl_rl 2.x
- env custom que devolve obs_dict["policy"]
- PPOConfig com clip_obs / clip_actions
- arquitetura sem privileged observations separadas
"""

import torch
from tensordict import TensorDict
from typing import Dict, Tuple, Any


class _EnvCfgProxy:
    """Proxy so rsl_rl Logger can read env.cfg without AttributeError."""
    def __init__(self, env_cfg):
        self._cfg = env_cfg

    def __getattr__(self, name):
        try:
            return getattr(self._cfg, name)
        except AttributeError:
            return None

    def __repr__(self):
        return repr(self._cfg)


def _ensure_tensor(value: Any, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.as_tensor(value, device=device)


def _to_tensordict(obs_dict: dict, device: str) -> TensorDict:
    """
    Convert plain dict → TensorDict so rsl_rl can call .to(device).
    Espera pelo menos a chave 'policy'.
    """
    tensor_obs = {
        k: v if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=device)
        for k, v in obs_dict.items()
    }
    return TensorDict(
        tensor_obs,
        batch_size=[tensor_obs["policy"].shape[0]],
        device=device,
    )


class RslRlVecEnvWrapper:
    def __init__(self, env, clip_obs: float = 100.0, clip_actions: float = 100.0):
        self.env = env
        self.clip_obs = clip_obs
        self.clip_actions = clip_actions

        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_privileged_obs = None
        self.num_actions = env.num_actions
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.cfg = _EnvCfgProxy(env.cfg)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def _clip_tensor(self, x: torch.Tensor, limit: float) -> torch.Tensor:
        return torch.clamp(x, -limit, limit)

    def _prepare_obs(self, obs_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Converte observações para tensor no device correto e aplica clipping.
        Mantém apenas o formato compatível com a tua arquitetura atual.
        """
        policy = _ensure_tensor(obs_dict["policy"], self.device)
        policy = self._clip_tensor(policy, self.clip_obs)
        return {"policy": policy}

    def get_observations(self) -> TensorDict:
        """
        Called by learn():
            obs = self.env.get_observations().to(self.device)
        Must return TensorDict so .to() works.
        """
        obs_dict = self.env._get_observations()
        obs = self._prepare_obs(obs_dict)
        return _to_tensordict(obs, self.device)

    def reset(self) -> Tuple[TensorDict, Dict]:
        """
        Esperado que env.reset() devolva:
            obs_dict, extras
        """
        obs_dict, extras = self.env.reset()
        obs = self._prepare_obs(obs_dict)
        td = _to_tensordict(obs, self.device)
        return td, {"observations": obs, **extras}

    def step(self, actions: torch.Tensor) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, Dict]:
        """
        Called by learn():
            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

        Esperado que env.step(actions) devolva:
            obs_dict, rewards, terminated, truncated, extras
        """
        actions = _ensure_tensor(actions, self.device)
        actions = self._clip_tensor(actions, self.clip_actions)

        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)

        obs = self._prepare_obs(obs_dict)
        td = _to_tensordict(obs, self.device)

        rewards = _ensure_tensor(rewards, self.device)
        terminated = _ensure_tensor(terminated, self.device).bool()
        truncated = _ensure_tensor(truncated, self.device).bool()
        dones = terminated | truncated

        # Logging compatível com rsl_rl Logger
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

        # Necessário para bootstrap correto no GAE
        extras["time_outs"] = truncated

        if log:
            extras["log"] = log

        return td, rewards, dones, extras

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()