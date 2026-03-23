"""
| File: networks/custom_actor.py
| Description: Custom actor networks for use with rsl_rl's OnPolicyRunner.
|
| To plug a network into PPOConfig:
|
|   from tasks.quadcopter.agents.networks.custom_actor import AxisDecoupledActor
|
|   agent_cfg.actor_class  = AxisDecoupledActor
|   agent_cfg.actor_kwargs = {"act_func": True, "bias_z": False}
|
| Any class set as actor_class must follow the RslRlActorWrapper interface:
|   __init__(obs, obs_groups, obs_set, output_dim,
|            hidden_dims, activation, distribution_cfg, **kwargs)
|   forward(obs: TensorDict | Tensor) -> GaussianDistribution
|   act_inference(obs: TensorDict | Tensor) -> Tensor
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict


# ── helpers ───────────────────────────────────────────────────────────────────

def _obs_to_tensor(obs, obs_set: str = "policy") -> torch.Tensor:
    """Extract a plain tensor from a TensorDict or return it unchanged."""
    if isinstance(obs, TensorDict):
        return obs[obs_set]
    return obs


def _build_gaussian_distribution(distribution_cfg: dict | None, act_dim: int,
                                  device: torch.device):
    """
    Instantiate a GaussianDistribution from the rsl_rl distribution_cfg dict.
    Returns None if distribution_cfg is not provided.
    """
    if distribution_cfg is None:
        return None
    try:
        from rsl_rl.modules.distribution import GaussianDistribution
        cls      = distribution_cfg.get("class_name", GaussianDistribution)
        init_std = distribution_cfg.get("init_std", 1.0)
        return cls(act_dim, init_std)
    except ImportError:
        return None


# ── axis-decoupled network ────────────────────────────────────────────────────

class AxisDecoupledPDActorLinear(nn.Module):
    """
    Linear axis-decoupled actor.

    Each output axis (x, y, z) depends only on its corresponding position-error
    and velocity-error pair drawn from the first 6 elements of obs:
        obs[:, 0:6] = [ep_x, ep_y, ep_z, ev_x, ev_y, ev_z]

    Output shape: [B, 3], optionally squashed through tanh.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 act_func: bool = True, bias_z: bool = False):
        super().__init__()

        if obs_dim < 6:
            raise ValueError(f"obs_dim must be >= 6 (got {obs_dim}).")
        if act_dim != 3:
            print(f"[AxisDecoupledPDActorLinear] act_dim={act_dim} != 3 — forcing to 3.")
            act_dim = 3

        self.fcx = nn.Linear(2, 1, bias=False)
        self.fcy = nn.Linear(2, 1, bias=False)
        self.fcz = nn.Linear(2, 1, bias=bias_z)

        nn.init.uniform_(self.fcx.weight, -0.1, 0.1)
        nn.init.uniform_(self.fcy.weight, -0.1, 0.1)
        nn.init.uniform_(self.fcz.weight, -0.1, 0.1)

        # indices into obs: (position_error, velocity_error) per axis
        self.idx_x: Tuple[int, int] = (0, 3)
        self.idx_y: Tuple[int, int] = (1, 4)
        self.idx_z: Tuple[int, int] = (2, 5)

        self.act_func = act_func
        self.in_features = 6    # explicit for ONNX export

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map obs [B, >=6] -> actions [B, 3]."""
        if obs.dim() != 2 or obs.size(1) < 6:
            raise RuntimeError(f"obs must have shape [B, >=6], got {tuple(obs.shape)}.")

        x_in = obs[:, [self.idx_x[0], self.idx_x[1]]]
        y_in = obs[:, [self.idx_y[0], self.idx_y[1]]]
        z_in = obs[:, [self.idx_z[0], self.idx_z[1]]]

        if self.act_func:
            out_x = torch.tanh(self.fcx(x_in))
            out_y = torch.tanh(self.fcy(y_in))
            out_z = torch.tanh(self.fcz(z_in))
        else:
            out_x = self.fcx(x_in)
            out_y = self.fcy(y_in)
            out_z = self.fcz(z_in)

        return torch.cat([out_x, out_y, out_z], dim=1)  # [B, 3]


# ── rsl_rl adapter ────────────────────────────────────────────────────────────

class RslRlActorWrapper(nn.Module):
    """
    Wraps any nn.Module that maps obs_tensor -> action_mean into the interface
    expected by rsl_rl's OnPolicyRunner.

    OnPolicyRunner constructs actors as:
        actor = actor_class(obs, obs_groups, obs_set, output_dim,
                            hidden_dims, activation, distribution_cfg,
                            **actor_kwargs)

    This class absorbs those kwargs and delegates the actual forward pass to the
    inner network (``net_class``), which only needs (obs_dim, act_dim, **net_kwargs).

    forward()         -> GaussianDistribution  (used by PPO update)
    act_inference()   -> Tensor (mean)          (used by play.py)
    """

    def __init__(
        self,
        obs,
        obs_groups: dict,
        obs_set: str,
        output_dim: int,
        hidden_dims: list | None        = None,
        activation: str                 = "elu",
        distribution_cfg: dict | None   = None,
        net_class: type                 = AxisDecoupledPDActorLinear,
        **net_kwargs,
    ):
        super().__init__()

        # Resolve obs_dim: rsl_rl passes obs as a gym Space or int
        if hasattr(obs, "shape"):
            obs_dim = obs.shape[0]
        elif isinstance(obs, (int, tuple)):
            obs_dim = obs if isinstance(obs, int) else obs[0]
        else:
            obs_dim = int(obs)

        self.obs_set   = obs_set
        self.obs_dim   = obs_dim
        self.act_dim   = output_dim

        self.net = net_class(obs_dim=obs_dim, act_dim=output_dim, **net_kwargs)

        self.distribution = _build_gaussian_distribution(
            distribution_cfg, output_dim,
            device=next(self.net.parameters()).device
            if len(list(self.net.parameters())) > 0 else torch.device("cpu"),
        )

    def _extract_obs(self, obs) -> torch.Tensor:
        return _obs_to_tensor(obs, self.obs_set)

    def forward(self, obs):
        """Return a GaussianDistribution over actions (used by PPO loss)."""
        x    = self._extract_obs(obs)
        mean = self.net(x)

        if self.distribution is not None:
            # rsl_rl GaussianDistribution: update(mean) sets the current mean
            self.distribution.update(mean)
            return self.distribution

        # Fallback: return a plain Normal so PPO can still compute log_probs
        std = torch.ones_like(mean)
        return torch.distributions.Normal(mean, std)

    def act_inference(self, obs) -> torch.Tensor:
        """Return deterministic mean action (no gradient)."""
        with torch.no_grad():
            x = self._extract_obs(obs)
            return self.net(x)


# ── convenience alias registered as actor_class ───────────────────────────────

class AxisDecoupledActor(RslRlActorWrapper):
    """
    Ready-to-use actor_class for PPOConfig.

    Sets net_class=AxisDecoupledPDActorLinear and forwards any remaining
    kwargs (act_func, bias_z) to it.

    Usage in ppo_cfg.py:
        from tasks.quadcopter.agents.networks.custom_actor import AxisDecoupledActor

        custom = PPOConfig(
            actor_class  = AxisDecoupledActor,
            actor_kwargs = {"act_func": True, "bias_z": False},
        )
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim,
                 hidden_dims=None, activation="elu",
                 distribution_cfg=None, **net_kwargs):
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            distribution_cfg=distribution_cfg,
            net_class=AxisDecoupledPDActorLinear,
            **net_kwargs,
        )