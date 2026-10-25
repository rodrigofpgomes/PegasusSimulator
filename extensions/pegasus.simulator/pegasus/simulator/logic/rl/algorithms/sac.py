"""
| File: algorithms/sac.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: SAC training via skrl - generic and task-agnostic.
|
| Algorithm reference: the delayed actor update follows the SAC training
| schedule used for RAPTOR foundation-policy pre-training. In this configuration,
| the critic is updated every training interval, while the actor is updated
| every two intervals. This scheduling approach is adapted to skrl.
| RAPTOR: https://github.com/rl-tools/raptor
| Pre-training config: https://github.com/rl-tools/rl-tools/blob/e43ae4bcda4556321a63f4eb5dcc826cd637aa39/src/foundation_policy/pre_training/config.h
|
| Relative to skrl 1.4.3 SAC (skrl/agents/torch/sac/sac.py):
|   - The agent stores the physical final observation from Pegasus autoreset
|     transitions instead of the next episode's reset observation.
|   - bootstrap_timeouts can continue the target through truncated episodes;
|     true terminations always stop bootstrapping.
|   - policy_delay updates the actor and entropy coefficient every configured
|     number of critic steps, while critics still update on every step.
|   - Optional weight decay is added to the existing actor/critic optimizers.
|   - Target critics, mixed precision, gradient clipping and skrl logging
|     remain part of the copied update path.
|
| Baseline: https://github.com/Toni-SM/skrl/tree/6054599/skrl/agents/torch/sac
"""

import os
import json
import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from pegasus.simulator.logic.rl.algorithms.transition_contract import terminal_next_states

from pegasus.simulator.logic.rl.skrl_pegasus_wrapper import PegasusSkrlWrapper
from pegasus.simulator.logic.rl.training_export import train_and_export, snapshot_config


def train(env, agent_cfg: dict, log_dir: str, device: str, headless: bool = True, checkpoint: str | None = None):
    """Create replay memory and a Pegasus SAC agent, then start skrl training."""

    # Defer trainer, memory and agent imports until after Isaac Sim initializes
    from skrl.agents.torch.sac import SAC
    from skrl.trainers.torch import SequentialTrainer
    from skrl.memories.torch import RandomMemory
    from skrl.utils import set_seed

    # Setup Seed
    seed = agent_cfg.get("seed")
    if seed is not None:
        set_seed(seed)
        print(f"[SAC] Seed set to: {seed}")

    # Environment Wrapping
    wrapped = PegasusSkrlWrapper(env)

    # Model and Configuration Initialization
    models = agent_cfg["models"](wrapped.observation_space, wrapped.action_space, device)
    cfg = _prepare_cfg(agent_cfg["cfg"], device)

    # Inject observation size into preprocessor (no-op if preprocessor is None)
    if cfg.get("state_preprocessor") is not None:
        cfg["state_preprocessor_kwargs"]["size"] = wrapped.observation_space

    # Store transitions across episodes for off-policy gradient updates
    memory = RandomMemory(memory_size=agent_cfg["memory_size"], num_envs=wrapped.num_envs, device=device)

    # Set the base directory for skrl (it will create the timestamped folder inside this)
    cfg["experiment"]["directory"] = log_dir

    # Keep the terminal-state and timeout contract when actor delay is one
    policy_delay = cfg.pop("policy_delay", 1)
    AgentClass = _make_sac_with_delay(SAC, policy_delay)

    # Agent Initialization
    agent = AgentClass(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
    )

    if checkpoint is not None:
        agent.load(checkpoint)
        print(f"[SAC] Loaded checkpoint from: {checkpoint}")

    # Save our custom config inside the folder skrl just created
    run_dir = agent.experiment_dir
    _save_run_info(run_dir, cfg, models, agent_cfg, policy_delay=policy_delay)

    print(f"\n[SAC] Logs and checkpoints will be saved to: {run_dir}\n")

    trainer_cfg = {
        "timesteps": agent_cfg["timesteps"],
        "headless": headless,
        "close_environment_at_exit": False,
        "environment_info": "log",
    }

    # Start Training
    train_and_export(SequentialTrainer(cfg=trainer_cfg, env=wrapped, agents=agent), agent, run_dir,
                     metadata={"algorithm": "sac", "environment": snapshot_config(env.cfg), "task": agent_cfg.get("task"),
                               "seed": seed, "revision": "aligned-com-review-v2", "resume_checkpoint": checkpoint})

    print(f"[SAC] Training completed. Logs saved to: {run_dir}")


# -------------------------------------------
# Internal Utilities
# -------------------------------------------

def _make_sac_with_delay(BaseSAC, policy_delay: int):
    """Extend skrl SAC with terminal-state replay and delayed actor updates.

    The class is created for each training run because the selected delay is
    part of that run's configuration. A delay of one still uses the Pegasus
    terminal-state and timeout behavior.
    """

    if isinstance(policy_delay, bool) or not isinstance(policy_delay, int) or policy_delay < 1:
        raise ValueError("policy_delay must be a positive integer")

    class SACDelayed(BaseSAC):
        """Update both critics each step and actor/entropy at policy_delay."""

        def __init__(self, *args, **kwargs):
            """Read timeout/decay options and initialize the critic step counter."""
            super().__init__(*args, **kwargs)

            # Count critic updates rather than environment steps; gradient_steps
            # may be greater than one for each environment interaction.
            self._critic_update_counter = 0
            self._policy_delay = policy_delay

            # Time limits and true terminations have different Bellman targets
            self._bootstrap_timeouts = bool(self.cfg.get("bootstrap_timeouts", False))

            # Apply decay in place so any existing scheduler keeps its optimizer
            wd = self.cfg.get("weight_decay", 0.0)
            if wd > 0:
                for optimizer in (self.policy_optimizer, self.critic_optimizer):
                    for group in optimizer.param_groups:
                        group["weight_decay"] = wd

        def record_transition(self, states, actions, rewards, next_states,
                            terminated, truncated, infos, timestep, timesteps) -> None:
            """Replace reset observations before skrl writes the replay sample."""
            next_states = terminal_next_states(next_states, terminated, truncated, infos)

            super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
                                    
        def _update(self, timestep: int, timesteps: int) -> None:
            """Optimize critics every step and actor/entropy at the selected delay.

            The replay buffer already holds physical terminal next states.
            bootstrap_timeouts controls whether a truncated episode contributes
            a discounted target value; a true termination never does.
            """

            # Each gradient step draws a fresh replay minibatch
            for gradient_step in range(self._gradient_steps):
                (
                    sampled_states,
                    sampled_actions,
                    sampled_rewards,
                    sampled_next_states,
                    sampled_terminated,
                    sampled_truncated,
                ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states      = self._state_preprocessor(sampled_states,      train=True)
                    sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                    # Build a soft Bellman target with the lower target-critic Q
                    with torch.no_grad():
                        next_actions, next_log_prob, _ = self.policy.act(
                            {"states": sampled_next_states}, role="policy"
                        )
                        target_q1, _, _ = self.target_critic_1.act(
                            {"states": sampled_next_states, "taken_actions": next_actions},
                            role="target_critic_1",
                        )
                        target_q2, _, _ = self.target_critic_2.act(
                            {"states": sampled_next_states, "taken_actions": next_actions},
                            role="target_critic_2",
                        )
                        target_q  = torch.min(target_q1, target_q2) - self._entropy_coefficient * next_log_prob
                        # Time-limit truncation may bootstrap; termination may not
                        target_values = (
                            sampled_rewards
                            + self._discount_factor
                            * (sampled_terminated.logical_not() if self._bootstrap_timeouts else (sampled_terminated | sampled_truncated).logical_not())
                            * target_q
                        )

                    critic_1_values, _, _ = self.critic_1.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="critic_1"
                    )
                    critic_2_values, _, _ = self.critic_2.act(
                        {"states": sampled_states, "taken_actions": sampled_actions}, role="critic_2"
                    )
                    # Fit both critics to the same detached soft target
                    critic_loss = (
                        F.mse_loss(critic_1_values, target_values)
                        + F.mse_loss(critic_2_values, target_values)
                    ) / 2

                # Apply the critic gradient before deciding whether to update policy
                self.critic_optimizer.zero_grad()
                self.scaler.scale(critic_loss).backward()
                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.critic_optimizer)
                    nn.utils.clip_grad_norm_(
                        itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                        self._grad_norm_clip,
                    )
                self.scaler.step(self.critic_optimizer)

                self._critic_update_counter += 1

                # Actor and entropy updates share the critic-step schedule
                if self._critic_update_counter % self._policy_delay == 0:
                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        actions, log_prob, _ = self.policy.act({"states": sampled_states}, role="policy")
                        q1, _, _ = self.critic_1.act({"states": sampled_states, "taken_actions": actions}, role="critic_1")
                        q2, _, _ = self.critic_2.act({"states": sampled_states, "taken_actions": actions}, role="critic_2")
                        # Favor high Q while retaining the configured entropy
                        policy_loss = (self._entropy_coefficient * log_prob - torch.min(q1, q2)).mean()

                    self.policy_optimizer.zero_grad()
                    self.scaler.scale(policy_loss).backward()
                    if self._grad_norm_clip > 0:
                        self.scaler.unscale_(self.policy_optimizer)
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    self.scaler.step(self.policy_optimizer)

                    # Learn the temperature only on actor-update steps
                    if self._learn_entropy:
                        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                            entropy_loss = -(
                                self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()
                            ).mean()
                        self.entropy_optimizer.zero_grad()
                        self.scaler.scale(entropy_loss).backward()
                        self.scaler.step(self.entropy_optimizer)
                        self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

                    if self.write_interval > 0:
                        self.track_data("Loss / Policy loss", policy_loss.item())
                        if self._learn_entropy:
                            self.track_data("Loss / Entropy loss", entropy_loss.item())
                            self.track_data(
                                "Coefficient / Entropy coefficient", self._entropy_coefficient.item()
                            )

                # Advance each scheduler only when its optimizer was stepped
                self.scaler.update()
                if self._learning_rate_scheduler:
                    self.critic_scheduler.step()
                    if self._critic_update_counter % self._policy_delay == 0:
                        self.policy_scheduler.step()

                # Slowly track both online critics after each critic update
                self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
                self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

                if self.write_interval > 0:
                    self.track_data("Loss / Critic loss", critic_loss.item())
                    self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())
                    self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())
                    self.track_data("Target / Target (mean)", torch.mean(target_values).item())

    # Make the chosen actor delay visible in the generated class name
    SACDelayed.__name__ = f"SACDelayed_delay_{policy_delay}"
    return SACDelayed


def _prepare_cfg(cfg: dict, device: str) -> dict:
    """Copy the task config and place its state preprocessor on the agent device."""
    cfg = copy.deepcopy(cfg)

    # SAC only uses state_preprocessor (no value_preprocessor like PPO)
    if isinstance(cfg.get("state_preprocessor_kwargs"), dict):
        cfg["state_preprocessor_kwargs"]["device"] = device

    return cfg


def _save_run_info(run_dir: str, cfg: dict, models: dict, agent_cfg: dict, policy_delay: int = 1):
    """Record model structure, effective hyperparameters and actor delay."""

    def _ser(v):
        """Recursively serializes config values to JSON-friendly types (classes -> name,
        callables -> str), so the run configuration can be dumped to disk."""
        if isinstance(v, type):
            return v.__name__
        if isinstance(v, dict):
            return {k: _ser(vv) for k, vv in v.items()}
        if callable(v):
            return str(v)
        return v

    record = {
        "algorithm": "SAC (skrl)",
        "seed": agent_cfg.get("seed"),
        "timesteps": agent_cfg["timesteps"],
        "memory_size": agent_cfg["memory_size"],
        "policy_delay": policy_delay,
        "hyperparameters": {
            k: _ser(v) for k, v in cfg.items() if k != "experiment"
        },
        "model_architecture": {
            name: str(model)
            for name, model in models.items()
            if isinstance(model, nn.Module)
        },
    }

    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)
