"""
| File: algorithms/sac.py
| Description: SAC training via skrl - generic and task-agnostic.
| License: BSD-3-Clause.
"""

import os
import json
import copy
import torch.nn as nn

from pegasus.simulator.logic.rl.skrl_pegasus_wrapper import PegasusSkrlWrapper


def train(env, agent_cfg: dict, log_dir: str, device: str):
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

    # Inject observation size into preprocessor
    cfg["state_preprocessor_kwargs"]["size"] = wrapped.observation_space

    # Replay memory (off-policy core)
    memory = RandomMemory(memory_size=agent_cfg["memory_size"], num_envs=wrapped.num_envs, device=device)

    # Set the base directory for skrl (it will create the timestamped folder inside this)
    cfg["experiment"]["directory"] = log_dir

    # Agent Initialization
    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space,
        device=device,
    )

    # Save our custom config inside the folder skrl just created
    run_dir = agent.experiment_dir
    _save_run_info(run_dir, cfg, models, agent_cfg)

    print(f"\n[SAC] Logs and checkpoints will be saved to: {run_dir}\n")

    trainer_cfg = {
        "timesteps": agent_cfg["timesteps"],
        "headless": True,
        "close_environment_at_exit": False,
        "environment_info": "log",
    }

    # Start Training
    SequentialTrainer(cfg=trainer_cfg, env=wrapped, agents=agent).train()

    print(f"[SAC] Training completed. Logs saved to: {run_dir}")


# -------------------------------------------
# Internal Utilities
# -------------------------------------------

def _prepare_cfg(cfg: dict, device: str) -> dict:
    """Deep-copies the configuration and sets device for preprocessors."""
    cfg = copy.deepcopy(cfg)

    # SAC only uses state_preprocessor (no value_preprocessor like PPO)
    if isinstance(cfg.get("state_preprocessor_kwargs"), dict):
        cfg["state_preprocessor_kwargs"]["device"] = device

    return cfg


def _save_run_info(run_dir: str, cfg: dict, models: dict, agent_cfg: dict):
    """Saves hyperparameters and model architecture to JSON."""

    def _ser(v):
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