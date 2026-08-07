"""
| File: algorithms/sac.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: SAC training via skrl - generic and task-agnostic.
"""

import os
import json
import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config as skrl_config

from pegasus.simulator.logic.rl.skrl_pegasus_wrapper import PegasusSkrlWrapper


def train(env, agent_cfg: dict, log_dir: str, device: str, headless: bool = True, checkpoint: str | None = None):
    """Initializes the SAC agent, sets up logging, and starts the training loop."""

    # Deferred imports to avoid early CUDA initialization conflicts with Isaac Sim
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

    # Replay memory (off-policy core)
    memory = RandomMemory(memory_size=agent_cfg["memory_size"], num_envs=wrapped.num_envs, device=device)

    # Set the base directory for skrl (it will create the timestamped folder inside this)
    cfg["experiment"]["directory"] = log_dir

    # Use SACDelayed if policy_delay > 1, otherwise standard SAC
    policy_delay = cfg.pop("policy_delay", 1)
    AgentClass = _make_sac_with_delay(SAC, policy_delay) if policy_delay > 1 else SAC

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
    SequentialTrainer(cfg=trainer_cfg, env=wrapped, agents=agent).train()

    print(f"[SAC] Training completed. Logs saved to: {run_dir}")


# -------------------------------------------
# Internal Utilities
# -------------------------------------------

def _make_sac_with_delay(BaseSAC, policy_delay: int):
    """Returns a SAC subclass that updates the actor only every `policy_delay` critic steps."""

    class SACDelayed(BaseSAC):
        """SAC variant with delayed actor updates (à la TD3) and optional weight decay.

        The critics are updated every gradient step, while the actor and the entropy
        coefficient are only updated once every ``policy_delay`` critic updates.
        """

        def __init__(self, *args, **kwargs):
            """Initializes the delayed-update counter and rebuilds the optimizers with
            weight decay when ``cfg['weight_decay'] > 0`` (not supported natively by skrl)."""
            super().__init__(*args, **kwargs)
            self._critic_update_counter = 0
            self._policy_delay = policy_delay

            self._bootstrap_timeouts = bool(self.cfg.get("bootstrap_timeouts", False))

            # Apply weight decay if specified (skrl SAC doesn't support it natively)
            wd = self.cfg.get("weight_decay", 0.0)
            if wd > 0:
                self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._actor_learning_rate, weight_decay=wd)
                self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self._critic_learning_rate, weight_decay=wd)

        def record_transition(self, states, actions, rewards, next_states,
                            terminated, truncated, infos, timestep, timesteps) -> None:
            
            # If bootstrapping from timeouts, we treat timeouts as non-terminal and replace the next state with the final observation and not the reset observation
            if self._bootstrap_timeouts and infos is not None and "final_observation" in infos:
                done = (terminated | truncated).view(-1)
                if done.any():
                    next_states = next_states.clone()
                    next_states[done] = infos["final_observation"][done].to(next_states.dtype)
            
            super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
                                    
        def _update(self, timestep: int, timesteps: int) -> None:
            """Performs the SAC gradient steps: critic update every step, actor + entropy
            update only every ``policy_delay`` critic updates."""
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

                    # --- critic update (every step) ---
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
                    critic_loss = (
                        F.mse_loss(critic_1_values, target_values)
                        + F.mse_loss(critic_2_values, target_values)
                    ) / 2

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

                # --- actor + entropy update (every policy_delay critic steps) ---
                if self._critic_update_counter % self._policy_delay == 0:
                    with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                        actions, log_prob, _ = self.policy.act({"states": sampled_states}, role="policy")
                        q1, _, _ = self.critic_1.act({"states": sampled_states, "taken_actions": actions}, role="critic_1")
                        q2, _, _ = self.critic_2.act({"states": sampled_states, "taken_actions": actions}, role="critic_2")
                        policy_loss = (self._entropy_coefficient * log_prob - torch.min(q1, q2)).mean()

                    self.policy_optimizer.zero_grad()
                    self.scaler.scale(policy_loss).backward()
                    if self._grad_norm_clip > 0:
                        self.scaler.unscale_(self.policy_optimizer)
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    self.scaler.step(self.policy_optimizer)

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

                self.scaler.update()

                # soft update target critics
                self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
                self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

                if self.write_interval > 0:
                    self.track_data("Loss / Critic loss", critic_loss.item())
                    self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())
                    self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())
                    self.track_data("Target / Target (mean)", torch.mean(target_values).item())

    SACDelayed.__name__ = f"SACDelayed_delay_{policy_delay}"
    return SACDelayed


def _prepare_cfg(cfg: dict, device: str) -> dict:
    """Deep-copies the configuration and sets device for preprocessors."""
    cfg = copy.deepcopy(cfg)

    # SAC only uses state_preprocessor (no value_preprocessor like PPO)
    if isinstance(cfg.get("state_preprocessor_kwargs"), dict):
        cfg["state_preprocessor_kwargs"]["device"] = device

    return cfg


def _save_run_info(run_dir: str, cfg: dict, models: dict, agent_cfg: dict, policy_delay: int = 1):
    """Saves hyperparameters and model architecture to JSON."""

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