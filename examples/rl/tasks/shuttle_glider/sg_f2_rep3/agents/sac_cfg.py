"""
| File: agents/sac_cfg.py
| Description: SAC config for skrl - Quadcopter task.
|
| NOTE: Hyperparameters are still preliminary and subject to tuning.
|
| Structure required by the training pipeline:
|   PRESETS["<name>"] = {
|       "models":    fn(obs_space, act_space, device) -> dict[str, nn.Module]
|       "cfg":       skrl SAC_DEFAULT_CONFIG dict with overrides
|       "timesteps": int - total training timesteps
|       "seed":      int | None
|   }
|
| Models:
|   - Policy: Gaussian actor
|   - Critics: twin Q-networks (critic_1, critic_2)
|   - Target critics: for stability (Polyak averaging)
|
| To use a custom actor:
|   Replace the Policy class or pass a different factory in "models".
|   The Policy must implement skrl's Model interface:
|       compute(inputs, role) -> (output, log_std, extras_dict)
"""

import copy
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler


class Policy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=True,
        clip_log_std=True,
        min_log_std=-5,
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
            nn.Linear(self.num_observations, 128), nn.ELU(),
            nn.Linear(128, 128),                   nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 128), nn.ELU(),
            nn.Linear(128, 128),                                       nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
        return self.net(x), {}


def _default_models(obs_space, act_space, device):
    critic_1 = Critic(obs_space, act_space, device)
    critic_2 = Critic(obs_space, act_space, device)

    target_critic_1 = Critic(obs_space, act_space, device)
    target_critic_2 = Critic(obs_space, act_space, device)

    target_critic_1.load_state_dict(critic_1.state_dict())
    target_critic_2.load_state_dict(critic_2.state_dict())

    return {
        "policy": Policy(obs_space, act_space, device),
        "critic_1": critic_1,
        "critic_2": critic_2,
        "target_critic_1": target_critic_1,
        "target_critic_2": target_critic_2,
    }


def _make_cfg() -> dict:
    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)

    cfg["gradient_steps"] = 1
    cfg["batch_size"] = 512
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005

    cfg["actor_learning_rate"] = 3e-4
    cfg["critic_learning_rate"] = 3e-4
    cfg["entropy_learning_rate"] = 3e-4

    cfg["learning_rate_scheduler"] = None
    cfg["learning_rate_scheduler_kwargs"] = {}

    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": None}

    cfg["random_timesteps"] = 8
    cfg["learning_starts"] = 8

    cfg["grad_norm_clip"] = 1.0

    cfg["learn_entropy"] = True
    cfg["initial_entropy_value"] = 0.2
    cfg["target_entropy"] = None

    cfg["rewards_shaper"] = None
    cfg["mixed_precision"] = False

    cfg["experiment"] = {
        "directory": "",
        "experiment_name": "",
        "write_interval": 1000,
        "checkpoint_interval": 5000,
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    }

    return cfg


PRESETS = {
    "isaac_lab": {
        "models": _default_models,
        "cfg": _make_cfg(),
        "timesteps": 300_000,
        "memory_size": 256,
        "seed": 42,
    }
}