"""
| File: algorithms/ppo.py
| Author: Rodrigo Gomes (rodrigofpgomes@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2026, Rodrigo Gomes. All rights reserved.
| Description: PPO training via skrl - generic and task-agnostic.
"""

import os
import json
import copy
import torch
import torch.nn as nn

from tqdm import tqdm

from pegasus.simulator.logic.rl.skrl_pegasus_wrapper import PegasusSkrlWrapper


def train(env, agent_cfg: dict, log_dir: str, device: str, headless: bool = True, checkpoint: str | None = None):
    """Initializes the PPO agent, sets up logging, and starts the training loop."""
    
    # Deferred imports to avoid early CUDA initialization conflicts with Isaac Sim
    from skrl.agents.torch.ppo import PPO
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
    cfg["state_preprocessor_kwargs"]["size"] = wrapped.observation_space

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

    rollout_steps = cfg["rollouts"]                # steps por env antes do update PPO
    current_stage = None

    timesteps = trainer_cfg["timesteps"]
    environment_info = trainer_cfg["environment_info"]
    headless = trainer_cfg["headless"]

    # initialize trainer
    agent.init(trainer_cfg=trainer_cfg)

    # initial reset
    states, infos = wrapped.reset()

    progress = tqdm(range(trainer_cfg["timesteps"]), desc="PPO training", dynamic_ncols=True)

    # Start Training
    for timestep in progress:
        if timestep % rollout_steps == 0:
            total_env_steps = timestep * wrapped.num_envs
            next_stage = wrapped.unwrapped.curriculum_stage_from_env_steps(total_env_steps)

            if next_stage != current_stage:
                wrapped.unwrapped.set_curriculum_stage(next_stage)
                current_stage = next_stage

                weights = wrapped.unwrapped.get_reward_weights()
                progress.set_postfix({
                    "stage": current_stage,
                    "ep": f'{weights["ep"]:.3g}',
                    "ev": f'{weights["ev"]:.3g}',
                    "u": f'{weights["u"]:.3g}',
                })

                print(
                    f"[PPO] rollout_start={timestep:6d} | "
                    f"env_steps={total_env_steps:10d} | "
                    f"stage={current_stage} | "
                    f"weights={weights}"
                )

        agent.set_running_mode("train")
        agent.pre_interaction(timestep=timestep, timesteps=timesteps)

        with torch.no_grad():
            actions = agent.act(states, timestep=timestep, timesteps=timesteps)[0]

        next_states, rewards, terminated, truncated, infos = wrapped.step(actions)

        if not headless:
            wrapped.render()

        agent.record_transition(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if environment_info in infos:
            for k, v in infos[environment_info].items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    agent.track_data(f"Info / {k}", v.item())

        agent.post_interaction(timestep=timestep, timesteps=timesteps)

        if terminated.any() or truncated.any():
            with torch.no_grad():
                states, infos = wrapped.reset()
        else:
            states = next_states

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