"""
| File: rsl_rl_wrapper.py
| Description: Adapts PegasusEnv to the rsl_rl 2.x VecEnv interface.
| License: BSD-3-Clause.

rsl_rl 2.x learn() does:
    obs = self.env.get_observations().to(self.device)   ← needs .to()
    obs, rewards, dones, extras = self.env.step(actions)
        ← step returns 4 values, NOT 6 (runner unpacks differently from __init__)

Confirmed from learn() source:
    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
    self.logger.process_env_step(rewards, dones, extras, ...)
        extras["episode"] or extras["log"] → logged per episode

For logging in Isaac Lab style, populate extras["log"] with a flat dict
of scalar values — the Logger accumulates them and prints per episode.
"""
import torch
from tensordict import TensorDict
from typing import Dict, Tuple


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


def _to_tensordict(obs_dict: dict, device: str) -> TensorDict:
    """Convert plain dict → TensorDict so rsl_rl can call .to(device)."""
    return TensorDict(
        {k: v if isinstance(v, torch.Tensor) else torch.as_tensor(v, device=device)
         for k, v in obs_dict.items()},
        batch_size=[obs_dict["policy"].shape[0]],
        device=device,
    )


class RslRlVecEnvWrapper:
    def __init__(self, env, clip_obs: float = 100.0, clip_actions: float = 100.0):
        self.env          = env
        self.clip_obs     = clip_obs
        self.clip_actions = clip_actions

        self.num_envs           = env.num_envs
        self.num_obs            = env.num_obs
        self.num_privileged_obs = None
        self.num_actions        = env.num_actions
        self.device             = env.device
        self.max_episode_length = env.max_episode_length
        self.cfg                = _EnvCfgProxy(env.cfg)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        """
        Called by learn():
            obs = self.env.get_observations().to(self.device)
        Must return TensorDict so .to() works.
        """
        obs_dict = self.env._get_observations()
        return _to_tensordict({"policy": obs_dict["policy"]}, self.device)

    def reset(self) -> Tuple[TensorDict, Dict]:
        obs_dict, extras = self.env.reset()
        td = _to_tensordict({"policy": obs_dict["policy"]}, self.device)
        return td, {"observations": obs_dict, **extras}

    def step(self, actions: torch.Tensor) -> Tuple:
        """
        Called by learn():
            obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
        Returns 4 values — NOT 6 like in __init__ phase.

        extras["log"] → Logger.process_env_step accumulates per-episode scalars.
        """
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)

        td    = _to_tensordict({"policy": obs_dict["policy"]}, self.device)
        dones = terminated | truncated

        # Populate extras["log"] with per-episode metrics for the Logger.
        # Logger accumulates these and prints/saves at the end of each episode.
        # Format: flat dict of scalar tensors, e.g. {"reward/distance": tensor}
        log = {}

        # Episode reward components (set in _reset_idx)
        for key, val in extras.items():
            if key.startswith("Episode_Reward/") or key.startswith("Episode_Termination/"):
                short = key.replace("Episode_Reward/", "rew/").replace("Episode_Termination/", "term/")
                if isinstance(val, torch.Tensor):
                    log[short] = val.item() if val.numel() == 1 else val.mean().item()
                elif isinstance(val, (int, float)):
                    log[short] = float(val)

        # time_outs for GAE bootstrap — must stay in extras
        extras["time_outs"] = truncated
        if log:
            extras["log"] = log

        return td, rewards, dones, extras

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()