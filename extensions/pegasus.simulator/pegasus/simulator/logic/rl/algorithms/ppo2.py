"""
| File: algorithms/ppo2.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: RSL-RL-like PPO training using the skrl infrastructure.
"""

import os
import json
import copy
import itertools
import torch
import torch.nn as nn
from skrl import config
from skrl.agents.torch.ppo import PPO
from pegasus.simulator.logic.rl.skrl_pegasus_wrapper import PegasusSkrlWrapper


def _assert_finite(name, tensor):
    """Interrompe o treino se um tensor contiver NaN ou Inf."""
    if not isinstance(tensor, torch.Tensor):
        return

    invalid = ~torch.isfinite(tensor)

    if invalid.any():
        indices = invalid.nonzero(as_tuple=False)[:5].detach().cpu().tolist()
        values = tensor.detach()[invalid][:5].cpu().tolist()

        raise FloatingPointError(f"[PPO2] {name} contém NaN/Inf indices={indices}, values={values}")

# =============================================================================
# PPO2 Agent
# =============================================================================

def _caps_temporal_pairs(rollout_states, terminated, truncated):
    """Pairs from a (time, env, obs) rollout; exclude resets and the last step."""
    if rollout_states.ndim != 3:
        raise ValueError("CAPS expects memory states shaped (time, env, obs)")
    next_states = torch.zeros_like(rollout_states)
    next_states[:-1] = rollout_states[1:]
    valid = torch.zeros(rollout_states.shape[:2], dtype=torch.bool, device=rollout_states.device)
    dones = (terminated.bool() | truncated.bool()).reshape(*rollout_states.shape[:2], -1).any(dim=-1)
    valid[:-1] = ~dones[:-1]
    return next_states.reshape(-1, rollout_states.shape[-1]), valid.reshape(-1)


def _caps_distance(mean, other_mean):
    """Mean L2 distance on the RAW deterministic mean; gradients on both sides."""
    return torch.linalg.vector_norm(mean - other_mean, dim=-1).mean()


class PPO2(PPO):
    """PPO agent reproducing the RSL-RL PPO update using skrl infrastructure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._desired_kl = self.cfg.get("desired_kl", 0.01)
        self._schedule = self.cfg.get("schedule", "adaptive")
        self._use_clipped_value_loss = self.cfg.get("use_clipped_value_loss", True)

        self._min_learning_rate = self.cfg.get("min_learning_rate", 1.0e-5)
        self._max_learning_rate = self.cfg.get("max_learning_rate", 1.0e-2)
        self._learning_rate_factor = self.cfg.get("learning_rate_factor", 1.5)

        # Optional CAPS. Zero weights preserve the original PPO objective.
        self._caps_temporal_scale = float(self.cfg.get("caps_temporal_scale", 0.0))
        self._caps_spatial_scale = float(self.cfg.get("caps_spatial_scale", 0.0))
        self._caps_spatial_std = self.cfg.get("caps_spatial_std", None)
        if self._caps_temporal_scale < 0.0 or self._caps_spatial_scale < 0.0:
            raise ValueError("CAPS weights must be nonnegative")
        if self._caps_spatial_scale > 0.0 and self._caps_spatial_std is None:
            raise ValueError("Specify caps_spatial_std in normalized observation units")

    # =========================================================================
    # Update
    # =========================================================================

    def _update(self, timestep: int, timesteps: int) -> None:
        """Performs one PPO update following the RSL-RL implementation."""

        # ---------------------------------------------------------------------
        # Generalized Advantage Estimation
        # ---------------------------------------------------------------------

        def compute_gae(rewards, dones, values, last_values):
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()

            for i in reversed(range(rewards.shape[0])):
                next_values = values[i + 1] if i < rewards.shape[0] - 1 else last_values
                delta = rewards[i] + self._discount_factor * not_dones[i] * next_values - values[i]
                advantage = delta + self._discount_factor * self._lambda * not_dones[i] * advantage
                advantages[i] = advantage

            returns = advantages + values
            # RSL-RL default: global advantage normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)

            return returns, advantages

        # ---------------------------------------------------------------------
        # Final state value
        # ---------------------------------------------------------------------

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            self.value.train(False)
            last_values, _, _ = self.value.act({"states": self._state_preprocessor(self._current_next_states.float())}, role="value")
            self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        # ---------------------------------------------------------------------
        # Returns and advantages
        # ---------------------------------------------------------------------

        values = self.memory.get_tensor_by_name("values")
        rewards = self.memory.get_tensor_by_name("rewards")
        terminated = self.memory.get_tensor_by_name("terminated")
        truncated = self.memory.get_tensor_by_name("truncated")

        _assert_finite("rewards", rewards)
        _assert_finite("values", values)
        _assert_finite("last_values", last_values)

        returns, advantages = compute_gae(rewards=self.memory.get_tensor_by_name("rewards"), dones=(self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated")), values=values, last_values=last_values)

        _assert_finite("returns", returns)
        _assert_finite("advantages", advantages)

        processed_values = self._value_preprocessor(values, train=True)
        processed_returns = self._value_preprocessor(returns, train=True)

        _assert_finite("values_processed", processed_values)
        _assert_finite("returns_processed", processed_returns)

        self.memory.set_tensor_by_name("values", processed_values)
        self.memory.set_tensor_by_name("returns", processed_returns)
        self.memory.set_tensor_by_name("advantages", advantages)


        # ---------------------------------------------------------------------
        # Flatten rollout
        # ---------------------------------------------------------------------

        states = self.memory.get_tensor_by_name("states", keepdim=False)
        actions = self.memory.get_tensor_by_name("actions", keepdim=False)
        old_log_prob = self.memory.get_tensor_by_name("log_prob", keepdim=False)

        _assert_finite("states", states)
        _assert_finite("actions", actions)
        _assert_finite("old_log_prob", old_log_prob)

        old_values = self.memory.get_tensor_by_name("values", keepdim=False)
        returns = self.memory.get_tensor_by_name("returns", keepdim=False)
        advantages = self.memory.get_tensor_by_name("advantages", keepdim=False)

        _assert_finite("old_values_from_memory", old_values)
        _assert_finite("returns_from_memory", returns)
        _assert_finite("advantages_from_memory", advantages)

        # ---------------------------------------------------------------------
        # Old policy distribution
        # ---------------------------------------------------------------------

        with torch.no_grad():
            old_states = self._state_preprocessor(states, train=False)
            old_mu, old_sigma = self.policy.get_distribution_params(old_states)
            old_mu = old_mu.detach()
            old_sigma = old_sigma.detach()

            _assert_finite("old_mu", old_mu)
            _assert_finite("old_sigma", old_sigma)

            if (old_sigma <= 0).any():
                raise FloatingPointError(
                    "[PPO2] old_sigma contém valores <= 0. "
                    f"min={old_sigma.min().item():.6e}, "
                    f"max={old_sigma.max().item():.6e}"
                )

        # ---------------------------------------------------------------------
        # RSL-RL minibatch permutation
        # ---------------------------------------------------------------------

        caps_next_states = None
        caps_valid = None
        if self._caps_temporal_scale > 0.0:
            caps_next_states, caps_valid = _caps_temporal_pairs(
                self.memory.get_tensor_by_name("states"), terminated, truncated)
            if caps_next_states.shape != states.shape:
                raise ValueError("CAPS rollout/flat state shapes disagree")

        caps_std = None
        if self._caps_spatial_scale > 0.0:
            caps_std = torch.as_tensor(self._caps_spatial_std, device=states.device, dtype=states.dtype)
            if caps_std.ndim != 1 or caps_std.numel() != states.shape[-1]:
                raise ValueError("caps_spatial_std must contain one entry per observation")
            if not torch.isfinite(caps_std).all() or (caps_std < 0.0).any():
                raise ValueError("caps_spatial_std must be finite and nonnegative")

        batch_size = states.shape[0]
        mini_batch_size = batch_size // self._mini_batches
        usable_size = mini_batch_size * self._mini_batches

        # RSL-RL generates one permutation per PPO update and reuses it for every learning epoch.
        indices = torch.randperm(usable_size, requires_grad=False, device=self.device)

        # ---------------------------------------------------------------------
        # Logging accumulators
        # ---------------------------------------------------------------------

        cumulative_policy_loss = 0.0
        cumulative_value_loss = 0.0
        cumulative_entropy = 0.0
        cumulative_kl = 0.0
        cumulative_caps_t = 0.0
        cumulative_caps_s = 0.0
        cumulative_mean_outside = 0.0
        num_updates = 0

        # ---------------------------------------------------------------------
        # Learning epochs
        # ---------------------------------------------------------------------

        for _ in range(self._learning_epochs):
            for i in range(self._mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                sampled_states = self._state_preprocessor(states[batch_idx], train=False)
                sampled_actions = actions[batch_idx]
                sampled_old_log_prob = old_log_prob[batch_idx]
                sampled_old_values = old_values[batch_idx]
                sampled_returns = returns[batch_idx]
                sampled_advantages = advantages[batch_idx]
                sampled_old_mu = old_mu[batch_idx]
                sampled_old_sigma = old_sigma[batch_idx]

                # -------------------------------------------------------------
                # Policy forward pass
                # -------------------------------------------------------------

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    _, next_log_prob, _ = self.policy.act({"states": sampled_states, "taken_actions": sampled_actions}, role="policy")
                    distribution = self.policy.distribution(role="policy")
                    mu = distribution.mean
                    sigma = distribution.stddev

                    _assert_finite("mu", mu)
                    _assert_finite("sigma", sigma)

                    if (sigma <= 0).any():
                        raise FloatingPointError(f"[PPO2] Sigma has negative values: min={sigma.min().item():.6e}, max={sigma.max().item():.6e}")

                    entropy = distribution.entropy().sum(dim=-1).mean()

                    # ---------------------------------------------------------
                    # RSL-RL analytical Gaussian KL
                    # ---------------------------------------------------------

                    with torch.no_grad():
                        kl = torch.sum(torch.log(sigma / sampled_old_sigma + 1.0e-5) + (sampled_old_sigma.square() + (sampled_old_mu - mu).square()) / (2.0 * sigma.square()) - 0.5, dim=-1)
                        kl_mean = kl.mean()

                        # -----------------------------------------------------
                        # Distributed KL reduction
                        # -----------------------------------------------------

                        if config.torch.is_distributed:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= config.torch.world_size

                        # -----------------------------------------------------
                        # RSL-RL adaptive learning rate
                        # -----------------------------------------------------

                        if self._desired_kl is not None and self._schedule == "adaptive":
                            if kl_mean > self._desired_kl * 2.0:
                                self._learning_rate = max(self._min_learning_rate, self._learning_rate / self._learning_rate_factor)
                            elif kl_mean < self._desired_kl / 2.0 and kl_mean > 0.0:
                                self._learning_rate = min(self._max_learning_rate, self._learning_rate * self._learning_rate_factor)

                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self._learning_rate

                    # ---------------------------------------------------------
                    # PPO clipped surrogate loss
                    # ---------------------------------------------------------

                    _assert_finite("next_log_prob", next_log_prob)

                    ratio = torch.exp(next_log_prob - sampled_old_log_prob)

                    _assert_finite("ratio", ratio)  

                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clamp(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    # ---------------------------------------------------------
                    # Critic
                    # ---------------------------------------------------------

                    predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                    # ---------------------------------------------------------
                    # RSL-RL clipped value loss
                    # ---------------------------------------------------------

                    if self._use_clipped_value_loss:
                        value_clipped = sampled_old_values + (predicted_values - sampled_old_values).clamp(-self._ratio_clip, self._ratio_clip)
                        value_losses = (predicted_values - sampled_returns).pow(2)
                        value_losses_clipped = (value_clipped - sampled_returns).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (sampled_returns - predicted_values).pow(2).mean()

                    # ---------------------------------------------------------
                    # CAPS on the deterministic mean (actor and critic are separate)
                    # ---------------------------------------------------------

                    caps_t = mu.sum() * 0.0
                    caps_s = mu.sum() * 0.0
                    if self._caps_temporal_scale > 0.0:
                        valid = caps_valid[batch_idx]
                        if valid.any():
                            next_states_caps = self._state_preprocessor(caps_next_states[batch_idx][valid], train=False)
                            next_mu_caps = self.policy.net(next_states_caps)
                            caps_t = _caps_distance(mu[valid], next_mu_caps)
                    if self._caps_spatial_scale > 0.0:
                        # Perturb ONLY independent observation channels specified by cfg.
                        perturbed_states = sampled_states + torch.randn_like(sampled_states) * caps_std
                        perturbed_mu = self.policy.net(perturbed_states)
                        caps_s = _caps_distance(mu, perturbed_mu)

                    _assert_finite("caps_temporal", caps_t)
                    _assert_finite("caps_spatial", caps_s)

                    # ---------------------------------------------------------
                    # Total PPO loss
                    # ---------------------------------------------------------

                    _assert_finite("policy_loss", policy_loss)
                    _assert_finite("value_loss", value_loss)
                    _assert_finite("entropy", entropy)
                    _assert_finite("kl_mean", kl_mean)

                    loss = policy_loss + self._value_loss_scale * value_loss - self._entropy_loss_scale * entropy
                    if self._caps_temporal_scale > 0.0:
                        loss = loss + self._caps_temporal_scale * caps_t
                    if self._caps_spatial_scale > 0.0:
                        loss = loss + self._caps_spatial_scale * caps_s

                    _assert_finite("loss", loss)

                # -------------------------------------------------------------
                # Optimization
                # -------------------------------------------------------------

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                if config.torch.is_distributed:
                    self.policy.reduce_parameters()
                    if self.policy is not self.value:
                        self.value.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(itertools.chain(self.policy.parameters(), self.value.parameters()), self._grad_norm_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                # -------------------------------------------------------------
                # Logging
                # -------------------------------------------------------------

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                cumulative_entropy += entropy.item()
                cumulative_kl += kl_mean.item()
                cumulative_caps_t += caps_t.detach().item()
                cumulative_caps_s += caps_s.detach().item()
                cumulative_mean_outside += (mu.detach().abs() > 1.0).float().mean().item()
                num_updates += 1

        # ---------------------------------------------------------------------
        # Tracking
        # ---------------------------------------------------------------------

        self.track_data("Loss / Policy loss", cumulative_policy_loss / num_updates)
        self.track_data("Loss / Value loss", cumulative_value_loss / num_updates)
        self.track_data("CAPS / Temporal raw", cumulative_caps_t / num_updates)
        self.track_data("CAPS / Spatial raw", cumulative_caps_s / num_updates)
        self.track_data("CAPS / Temporal weighted", self._caps_temporal_scale * cumulative_caps_t / num_updates)
        self.track_data("CAPS / Spatial weighted", self._caps_spatial_scale * cumulative_caps_s / num_updates)
        self.track_data("Policy / Mean outside action bounds", cumulative_mean_outside / num_updates)
        if caps_valid is not None:
            self.track_data("CAPS / Valid temporal fraction", caps_valid.float().mean().item())
        self.track_data("Policy / Entropy", cumulative_entropy / num_updates)

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        std_values = self.policy.distribution(role="policy").stddev.detach()
        if std_values.ndim > 1:
            std_values = std_values.mean(dim=0)

        for i, std_value in enumerate(std_values):
            self.track_data(f"Policy / Standard deviation action_{i}", std_value.item())

        self.track_data("Learning / KL divergence", cumulative_kl / num_updates)
        self.track_data("Learning / Learning rate", self._learning_rate)


# =============================================================================
# Training
# =============================================================================

def train(env, agent_cfg: dict, log_dir: str, device: str, headless: bool = True, checkpoint: str | None = None):
    """Initializes the PPO2 agent, sets up logging, and starts the training loop."""

    # Deferred imports to avoid early CUDA initialization conflicts with Isaac Sim
    from skrl.memories.torch import RandomMemory
    from skrl.trainers.torch import SequentialTrainer
    from skrl.utils import set_seed

    seed = agent_cfg.get("seed")
    if seed is not None:
        set_seed(seed)
        print(f"[PPO2] Seed set to: {seed}")

    # Environment Wrapping
    wrapped = PegasusSkrlWrapper(env)

    # Model and Configuration Initialization
    models = agent_cfg["models"](wrapped.observation_space, wrapped.action_space, device)
    cfg = _prepare_cfg(agent_cfg["cfg"], device)

    # Inject observation size into preprocessor
    if cfg.get("state_preprocessor") is not None:
        cfg["state_preprocessor_kwargs"]["size"] = wrapped.observation_space

    # Rollout memory (on-policy core)    
    memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=wrapped.num_envs, device=device)

    # Set the base directory for skrl (it will create the timestamped folder inside this)
    cfg["experiment"]["directory"] = log_dir

    # Agent Initialization
    agent = PPO2(models=models, memory=memory, cfg=cfg, observation_space=wrapped.observation_space, action_space=wrapped.action_space, device=device)

    # Load checkpoint if provided
    if checkpoint is not None:
        # weights_only=False allows loading additional data or metadata stored in the checkpoint
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)

        # Check if the format of the data corresponds to the teacher network
        if (isinstance(checkpoint_data, dict) and "actor_net_state_dict" in checkpoint_data):
            # strict=True ensures that the checkpoint and actor architectures match exactly
            models["policy"].net.load_state_dict(checkpoint_data["actor_net_state_dict"], strict=True)

            with torch.no_grad():
                # Reset standard deviation for exploration in the new student policy (lower value)
                models["policy"].std_parameter.fill_(0.2)

            print(f"[PPO2] Initialized actor from teacher checkpoint: {checkpoint}")
        else:
            agent.load(checkpoint)

            print(f"[PPO2] Loaded checkpoint from: {checkpoint}")

    run_dir = agent.experiment_dir
    _save_run_info(run_dir, cfg, models, agent_cfg)
    print(f"\n[PPO2] Logs and checkpoints will be saved to: {run_dir}\n")

    # -------------------------------------------------------------------------
    # RSL-RL: init_at_random_ep_len=True
    # -------------------------------------------------------------------------
    wrapped.reset()

    episode_length_buf = getattr(wrapped.unwrapped,"episode_length_buf", None)
    max_episode_length = getattr(wrapped.unwrapped, "max_episode_length", None)

    #if (isinstance(episode_length_buf, torch.Tensor) and max_episode_length is not None):
    #    episode_length_buf[:] = torch.randint_like(episode_length_buf, high=int(max_episode_length))

    #print("[PPO2] Initial episode lengths randomized (RSL-RL init_at_random_ep_len=True)")

    trainer_cfg = {"timesteps": agent_cfg["timesteps"], "headless": headless, "close_environment_at_exit": False, "environment_info": "log"}
    SequentialTrainer(cfg=trainer_cfg, env=wrapped, agents=agent).train()
    print(f"[PPO2] Training completed. Logs saved to: {run_dir}")


# =============================================================================
# Internal Utilities
# =============================================================================

def _prepare_cfg(cfg: dict, device: str) -> dict:
    """Deep-copies the PPO2 configuration and sets preprocessor devices if required."""
    cfg = copy.deepcopy(cfg)
    for key in ("state_preprocessor_kwargs", "value_preprocessor_kwargs"):
        if isinstance(cfg.get(key), dict):
            cfg[key]["device"] = device
    return cfg


def _save_run_info(run_dir: str, cfg: dict, models: dict, agent_cfg: dict):
    """Saves hyperparameters and model architecture for reproducibility."""

    def _ser(value):
        if isinstance(value, type):
            return value.__name__
        if isinstance(value, dict):
            return {key: _ser(item) for key, item in value.items()}
        if callable(value):
            return str(value)
        return value

    record = {
        "algorithm": "PPO2 (RSL-RL-like via skrl)",
        "seed": agent_cfg.get("seed"),
        "timesteps": agent_cfg["timesteps"],
        "hyperparameters": {key: _ser(value) for key, value in cfg.items() if key != "experiment"},
        "model_architecture": {name: str(model) for name, model in models.items() if isinstance(model, nn.Module)},
    }

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as file:
        json.dump(record, file, indent=2, default=str)