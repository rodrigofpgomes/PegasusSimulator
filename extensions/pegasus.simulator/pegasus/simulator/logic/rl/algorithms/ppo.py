"""
| File: algorithms/ppo.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: PPO training via skrl - generic and task-agnostic.
"""

import os
import json
import copy
import torch.nn as nn

from pegasus.simulator.logic.rl.skrl_pegasus_wrapper import PegasusSkrlWrapper


def train(env, agent_cfg: dict, log_dir: str, device: str, headless: bool = True, checkpoint: str | None = None):
    """Initializes the PPO agent, sets up logging, and starts the training loop."""
    
    # Deferred imports to avoid early CUDA initialization conflicts with Isaac Sim
    from skrl.agents.torch.ppo import PPO
    from skrl.trainers.torch import SequentialTrainer
    from skrl.memories.torch import RandomMemory
    from skrl.utils import set_seed

    # Setup Seed
    seed = agent_cfg.get("seed")
    if seed is not None:
        set_seed(seed)
        print(f"[PPO] Seed set to: {seed}")

    # Environment Wrapping
    wrapped = PegasusSkrlWrapper(env)

    # Model and Configuration Initialization
    models = agent_cfg["models"](wrapped.observation_space, wrapped.action_space, device)
    cfg = _prepare_cfg(agent_cfg["cfg"], device)

    # Inject observation size into preprocessor
    if cfg.get("state_preprocessor") is not None:
        cfg["state_preprocessor_kwargs"]["size"] = wrapped.observation_space

    expected_num_envs = agent_cfg.get("expected_num_envs")
    if (expected_num_envs is not None and wrapped.num_envs != expected_num_envs):
        raise ValueError(f"Preset configured for {expected_num_envs} environments, " f"but received {wrapped.num_envs}")

    # Rollout memory (on-policy core)
    memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=wrapped.num_envs, device=device)

    # Set the base directory for skrl (it will create the timestamped folder inside this)
    cfg["experiment"]["directory"] = log_dir

    # Agent Initialization
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
    )

    if checkpoint is not None:
        agent.load(checkpoint)
        print(f"[PPO] Loaded checkpoint from: {checkpoint}")

    # Save the custom config inside the folder skrl just created
    run_dir = agent.experiment_dir
    _save_run_info(run_dir, cfg, models, agent_cfg)
    print(f"\n[PPO] Logs and checkpoints will be saved to: {run_dir}\n")

    trainer_cfg = {
        "timesteps": agent_cfg["timesteps"],
        "headless": headless,
        "close_environment_at_exit": False,
        "environment_info": "log",
    }

    # Start Training
    SequentialTrainer(cfg=trainer_cfg, env=wrapped, agents=agent).train()

    print(f"[PPO] Training completed. Logs saved to: {run_dir}")


# -------------------------------------------
# Internal Utilities
# -------------------------------------------

def _prepare_cfg(cfg: dict, device: str) -> dict:
    """Deep-copies the configuration and overrides the device for preprocessors."""
    cfg = copy.deepcopy(cfg)
    for key in ("state_preprocessor_kwargs", "value_preprocessor_kwargs"):
        if isinstance(cfg.get(key), dict):
            cfg[key]["device"] = device
    return cfg


def _save_run_info(run_dir: str, cfg: dict, models: dict, agent_cfg: dict):
    """Saves hyperparameters and model architecture to a JSON file for reproducibility."""
    
    def _ser(v):
        """Helper to serialize complex objects to JSON strings."""
        if isinstance(v, type): return v.__name__
        if isinstance(v, dict): return {k: _ser(vv) for k, vv in v.items()}
        if callable(v): return str(v)
        return v

    record = {
        "algorithm": "PPO (skrl)",
        "seed": agent_cfg.get("seed"),
        "timesteps": agent_cfg["timesteps"],
        "hyperparameters": {
            k: _ser(v) for k, v in cfg.items() if k != "experiment"
        },
        "model_architecture": {
            name: str(model) for name, model in models.items() if isinstance(model, nn.Module)
        },
    }

    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)