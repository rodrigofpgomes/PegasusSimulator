"""
| File: agents/sac_cfg.py  (curriculum4)
| Description: SAC config for the Shuttle+EasyGlider hybrid, stage 1.
|
| Changes vs. the RAPTOR-matched quad config:
|   - Wider critics/actor ([256,256]) for the richer 40-D aerodynamic observation.
|   - NO running state_preprocessor. The observation mixes O(10) airspeeds with
|     O(1) rotation entries, so it is normalised STATICALLY inside the env
|     (per-channel physical scales). Toggle env.cfg.use_static_obs_norm to A/B.
"""

import copy
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG

HIDDEN = 256


class Policy(GaussianMixin, Model):
    LOG_PROB_EPSILON = 1e-6

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std, reduction=reduction)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, self.num_actions * 2),
        )

    def compute(self, inputs, role):
        out = self.net(inputs["states"])
        return out[:, :self.num_actions], out[:, self.num_actions:], {}

    def act(self, inputs, role):
        mu, log_std, outputs = self.compute(inputs, role)
        log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)
        dist = torch.distributions.Normal(mu, log_std.exp())
        taken = inputs.get("taken_actions", None)
        if taken is not None:
            u = torch.atanh(taken.clamp(-1 + 1e-6, 1 - 1e-6))
            log_prob = dist.log_prob(u)
        else:
            u = dist.rsample()
            log_prob = dist.log_prob(u)
        log_prob = log_prob - torch.log(1.0 - torch.tanh(u).pow(2) + self.LOG_PROB_EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        outputs["mean_actions"] = torch.tanh(mu)
        return torch.tanh(u), log_prob, outputs


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


def _make_models(obs_space, act_space, device):
    c1, c2 = Critic(obs_space, act_space, device), Critic(obs_space, act_space, device)
    tc1, tc2 = Critic(obs_space, act_space, device), Critic(obs_space, act_space, device)
    tc1.load_state_dict(c1.state_dict()); tc2.load_state_dict(c2.state_dict())
    return {"policy": Policy(obs_space, act_space, device),
            "critic_1": c1, "critic_2": c2, "target_critic_1": tc1, "target_critic_2": tc2}


def _make_cfg(obs_space, device) -> dict:
    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)
    cfg["gradient_steps"] = 1
    cfg["batch_size"] = 512
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005
    cfg["actor_learning_rate"] = 3e-4
    cfg["critic_learning_rate"] = 3e-4
    cfg["entropy_learning_rate"] = 1e-4
    cfg["learn_entropy"] = True
    cfg["initial_entropy_value"] = 0.5
    cfg["target_entropy"] = -float(5)   # -action_dim
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 10_000
    cfg["grad_norm_clip"] = 0
    # Observations are normalised STATICALLY inside the env (per-channel physical
    # scales), so no running state_preprocessor here.
    cfg["state_preprocessor"] = None
    cfg["mixed_precision"] = False
    cfg["experiment"] = {"directory": "", "experiment_name": "", "write_interval": 1000,
                         "checkpoint_interval": 10_000, "store_separately": False,
                         "wandb": False, "wandb_kwargs": {}}
    return cfg


def make_preset(obs_space, act_space, device):
    return {"models": lambda o, a, d: _make_models(o, a, d),
            "cfg": _make_cfg(obs_space, device),
            "timesteps": 20_000_000, "memory_size": 1_000_000, "seed": 10}
