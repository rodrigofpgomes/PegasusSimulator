"""
| File: agents/sac_cfg.py
| Description: SAC config for skrl — Quadcopter task.
|
| Structure required by algorithms/sac.py:
|   PRESETS["<name>"] = {
|       "models":      fn(obs_space, act_space, device) -> dict[str, nn.Module]
|       "cfg":         skrl SAC_DEFAULT_CONFIG dict with overrides
|       "timesteps":   int   — total training timesteps
|       "memory_size": int   — replay buffer size
|       "seed":        int | None
|   }
|
| SAC model keys required by skrl:
|   "policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"
|
| Notes:
|   - SAC is off-policy, so it uses a replay buffer instead of PPO rollouts.
|   - The policy is stochastic (Gaussian).
|   - The critics are deterministic Q-functions over (state, action).
"""
import copy
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler


# ── Default networks ──────────────────────────────────────────────────
# Simple MLP baseline in the same spirit as your PPO config.
# You can swap self.net freely while keeping the same skrl interfaces.


class Policy(GaussianMixin, Model):
    """
    Stochastic policy network for SAC.

    skrl calls:
        act(...) / compute(inputs, role)

    Expected output:
        (mean_actions, log_std, extras_dict)

    inputs["states"] is the flat observation tensor [N, obs_dim].
    """
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64), nn.ELU(),
            nn.Linear(64, 64),                   nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    """
    Deterministic Q-network for SAC.

    Input:
        state and action
    Output:
        scalar Q(s, a)

    skrl expects the critics to consume both states and taken actions.
    """
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 64), nn.ELU(),
            nn.Linear(64, 64),                                       nn.ELU(),
            nn.Linear(64, 1),
        )

    def compute(self, inputs, role):
        states = inputs["states"]
        actions = inputs["taken_actions"]
        x = torch.cat([states, actions], dim=1)
        return self.net(x), {}


def _default_models(obs_space, act_space, device):
    """Factory called by algorithms/sac.py with the correct device."""
    critic_1 = Critic(obs_space, act_space, device)
    critic_2 = Critic(obs_space, act_space, device)

    target_critic_1 = Critic(obs_space, act_space, device)
    target_critic_2 = Critic(obs_space, act_space, device)

    # SAC target critics should start as exact copies
    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())

    return {
        "policy": Policy(obs_space, act_space, device),
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
    }


# ── SAC hyperparameters ───────────────────────────────────────────────
# Based on skrl's SAC_DEFAULT_CONFIG, with practical overrides for your setup.


def _make_cfg() -> dict:
    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)

    # training / replay
    cfg["gradient_steps"] = 1
    cfg["batch_size"] = 256
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005

    # optimiser
    cfg["actor_learning_rate"] = 5e-4
    cfg["critic_learning_rate"] = 5e-4
    cfg["entropy_learning_rate"] = 5e-4

    cfg["learning_rate_scheduler"] = None
    cfg["learning_rate_scheduler_kwargs"] = {}

    # preprocessors — device and size injected at train time by algorithms/sac.py
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": None}

    # replay warmup / exploration
    cfg["random_timesteps"] = 5_000
    cfg["learning_starts"] = 5_000

    # stability
    cfg["grad_norm_clip"] = 1.0

    # entropy / temperature
    cfg["learn_entropy"] = True
    cfg["initial_entropy_value"] = 0.2
    cfg["target_entropy"] = None  # let skrl infer default target entropy

    # reward shaping (equivalent spirit to PPO rewards_shaper_scale = 0.01)
    cfg["rewards_shaper"] = lambda rewards, *args: rewards * 0.01

    # misc
    cfg["mixed_precision"] = False

    # experiment (directory overridden at train time)
    cfg["experiment"] = {
        "directory": "",
        "experiment_name": "",
        "write_interval": 1000,
        "checkpoint_interval": 10_000,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    }

    return cfg


# ── Presets ───────────────────────────────────────────────────────────

PRESETS = {
    "isaac_lab": {
        "models": _default_models,
        "cfg": _make_cfg(),
        "timesteps": 50_000,
        "memory_size": 50_000,
        "seed": None,
    }
}


# ── Custom actor / critic example ─────────────────────────────────────
#
# To swap the actor architecture, subclass Policy and replace self.net:
#
#   class MyPolicy(Policy):
#       def __init__(self, obs_space, act_space, device, **kw):
#           super().__init__(obs_space, act_space, device, **kw)
#           self.net = nn.Sequential(
#               nn.Linear(self.num_observations, 256), nn.ReLU(),
#               nn.Linear(256, 256),                   nn.ReLU(),
#               nn.Linear(256, self.num_actions),
#           )
#
# To swap the critic architecture, subclass Critic and replace self.net:
#
#   class MyCritic(Critic):
#       def __init__(self, obs_space, act_space, device, **kw):
#           super().__init__(obs_space, act_space, device, **kw)
#           self.net = nn.Sequential(
#               nn.Linear(self.num_observations + self.num_actions, 256), nn.ReLU(),
#               nn.Linear(256, 256),                                      nn.ReLU(),
#               nn.Linear(256, 1),
#           )
#
#   def _custom_models(obs_space, act_space, device):
#       critic_1 = MyCritic(obs_space, act_space, device)
#       critic_2 = MyCritic(obs_space, act_space, device)
#       target_critic_1 = MyCritic(obs_space, act_space, device)
#       target_critic_2 = MyCritic(obs_space, act_space, device)
#       target_critic_1.load_state_dict(critic_1.state_dict())
#       target_critic_2.load_state_dict(critic_2.state_dict())
#
#       return {
#           "policy": MyPolicy(obs_space, act_space, device),
#           "critic_1": critic_1,
#           "critic_2": critic_2,
#           "target_critic_1": target_critic_1,
#           "target_critic_2": target_critic_2,
#       }
#
#   PRESETS["custom"] = {
#       "models": _custom_models,
#       "cfg": _make_cfg(),
#       "timesteps": ...,
#       "memory_size": ...,
#       "seed": None,
#   }