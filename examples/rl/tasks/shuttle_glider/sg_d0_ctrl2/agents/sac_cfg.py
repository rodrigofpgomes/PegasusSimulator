"""
| File: agents/sac_cfg.py (raptor_pretrain)
| Description: SAC config matching the RAPTOR pre-training hyperparameters exactly.

RAPTOR pre-training (src/foundation_policy/pre_training/config.h):
    Algorithm:       SAC
    Actor:           Dense(29->64, ReLU) -> Dense(64->64, ReLU) -> Dense(64->8) [mu+log_std]
    Critic:          Dense(29+4->256, ReLU) -> Dense(256->256, ReLU) -> Dense(256->1)
    Batch size:      128
    Actor LR:        3e-4
    Critic LR:       3e-4
    Alpha LR:        1e-4
    Target entropy:  -2.0  (fixed)
    Gamma:           0.99
    Warmup steps:    N_WARMUP_STEPS=0 (no random), N_WARMUP_STEPS_CRITIC/ACTOR=10000
    Actor interval:  2     (ACTOR_TRAINING_INTERVAL)
    Gradient clip:   disabled
    Weight decay:    disabled (ENABLE_WEIGHT_DECAY = false)
    Step limit:      1 000 000
    Replay buffer:   1 000 000
"""

import copy
import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model
from skrl.agents.torch.sac import SAC_DEFAULT_CONFIG


class Policy(GaussianMixin, Model):
    """Actor: MLP with ReLU, matches RAPTOR teacher actor.
    Outputs 2*ACTION_DIM: first half = mu (pre-tanh), second half = log_std.
    act() overrides GaussianMixin to apply tanh squashing + log_prob correction,
    matching rl-tools SampleAndSquash exactly.
    """

    LOG_PROB_EPSILON = 1e-6  # matches rl-tools LOG_PROBABILITY_EPSILON

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=clip_actions,
                               clip_log_std=clip_log_std,
                               min_log_std=min_log_std,
                               max_log_std=max_log_std,
                               reduction=reduction)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64), nn.ReLU(),
            nn.Linear(64, 64),                   nn.ReLU(),
            nn.Linear(64, self.num_actions * 2),
        )

    def compute(self, inputs, role):
        out = self.net(inputs["states"])
        mu      = out[:, :self.num_actions]
        log_std = out[:, self.num_actions:]
        return mu, log_std, {}

    def act(self, inputs, role):
        mu, log_std, outputs = self.compute(inputs, role)

        # clamp log_std - matches rl-tools [-20, 2] bounds
        log_std = torch.clamp(log_std, self._g_log_std_min, self._g_log_std_max)

        dist = torch.distributions.Normal(mu, log_std.exp())

        taken_actions = inputs.get("taken_actions", None)
        if taken_actions is not None:
            # Inverse tanh to recover pre-squash sample for log_prob computation
            u = torch.atanh(taken_actions.clamp(-1 + 1e-6, 1 - 1e-6))
            log_prob = dist.log_prob(u)
        else:
            u = dist.rsample()
            log_prob = dist.log_prob(u)

        # Tanh squashing correction: -log(1 - tanh²(u) + eps)
        log_prob = log_prob - torch.log(1.0 - torch.tanh(u).pow(2) + self.LOG_PROB_EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        actions = torch.tanh(u)
        outputs["mean_actions"] = torch.tanh(mu)
        return actions, log_prob, outputs


class Critic(DeterministicMixin, Model):
    """Critic: larger MLP matching RAPTOR (256 hidden), takes state + action."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 256), nn.ReLU(),
            nn.Linear(256, 256),                                       nn.ReLU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
        return self.net(x), {}


def _make_models(obs_space, act_space, device):
    c1 = Critic(obs_space, act_space, device)
    c2 = Critic(obs_space, act_space, device)
    tc1 = Critic(obs_space, act_space, device)
    tc2 = Critic(obs_space, act_space, device)
    tc1.load_state_dict(c1.state_dict())
    tc2.load_state_dict(c2.state_dict())
    return {
        "policy":          Policy(obs_space, act_space, device),
        "critic_1":        c1,
        "critic_2":        c2,
        "target_critic_1": tc1,
        "target_critic_2": tc2,
    }


def _make_cfg() -> dict:
    cfg = copy.deepcopy(SAC_DEFAULT_CONFIG)

    # --- core SAC ---
    # Designed for 1 env (--n_envs 1): 1 timestep = 1 transition = 1 gradient step,
    # exactly matching the RAPTOR single-env pre-training setup.
    cfg["gradient_steps"]    = 1
    cfg["batch_size"]        = 128
    cfg["discount_factor"]   = 0.99
    cfg["polyak"]            = 0.005   # tau for target network update

    # --- learning rates (RAPTOR values) ---
    cfg["actor_learning_rate"]   = 3e-4
    cfg["critic_learning_rate"]  = 3e-4
    cfg["entropy_learning_rate"] = 1e-4

    cfg["learning_rate_scheduler"]        = None
    cfg["learning_rate_scheduler_kwargs"] = {}

    # --- entropy / temperature ---
    cfg["learn_entropy"]         = True
    cfg["initial_entropy_value"] = 0.5   # RAPTOR default (SAC DefaultParameters::ALPHA)
    cfg["target_entropy"]        = -2.0   # fixed, as in RAPTOR config

    # --- actor update delay (RAPTOR: ACTOR_TRAINING_INTERVAL=2) ---
    # Passed to _make_sac_with_delay in algorithms/sac.py - not a native skrl parameter.
    cfg["policy_delay"] = 2
    cfg["bootstrap_timeouts"] = True

    # --- warmup ---
    # N_WARMUP_STEPS = 0: no random actions, policy used from step 0
    # N_WARMUP_STEPS_CRITIC = N_WARMUP_STEPS_ACTOR = 10000: learning starts at 10k
    cfg["random_timesteps"] = 0
    cfg["learning_starts"]  = 10_000

    # --- gradient clipping disabled ---
    cfg["grad_norm_clip"] = 0   # 0 = disabled in skrl

    # --- weight decay disabled (RAPTOR: ENABLE_WEIGHT_DECAY = false) ---
    cfg["weight_decay"] = 0.0

    # --- no observation preprocessor (RAPTOR trains on raw observations) ---
    cfg["state_preprocessor"]        = None
    cfg["state_preprocessor_kwargs"] = {}

    cfg["mixed_precision"] = False

    cfg["experiment"] = {
        "directory":        "",
        "experiment_name":  "",
        "write_interval":   1000,
        "checkpoint_interval": 10_000,
        "store_separately": False,
        "wandb":            False,
        "wandb_kwargs":     {},
    }

    return cfg


PRESETS = {
    "isaac_lab": {
        "models":       _make_models,
        "cfg":          _make_cfg(),
        "timesteps":    1_000_000,
        # memory_size = number of transitions (1 env → memory_size == replay buffer capacity).
        # RAPTOR uses 1_000_000. Matches exactly with --n_envs 1.
        "memory_size":  1_000_000,
        "seed":         10,
    }
}
