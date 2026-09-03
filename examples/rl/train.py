#!/usr/bin/env python
"""
| File: train.py
| Description: Generic RL training launcher (skrl backend). Discovers tasks and algorithms automatically.
| License: BSD-3-Clause.

Usage:
    python train.py --task quadcopter --algo ppo
    python train.py --task quadcopter --algo ppo --seed 42 --n_envs 512 --headless
"""

import os
import sys
import copy
import argparse
import importlib
import carb

# Isaac Sim imports
from isaacsim import SimulationApp

TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")

def discover_tasks():
    """Finds available tasks under the tasks directory.

    Walks the tree recursively and returns every directory that contains a
    ``*_env.py`` file, as a path relative to ``TASKS_DIR`` using ``/`` as the
    separator (e.g. ``"double_integrator/01_isaac_lab"`` or ``"raptor_pretrain"``).
    This supports the nested task layout and numeric-prefixed package names.
    """
    if not os.path.isdir(TASKS_DIR):
        return []
    tasks = []
    for root, dirs, files in os.walk(TASKS_DIR):
        # Skip private/cache directories
        dirs[:] = [d for d in dirs if not d.startswith("_") and d != "agents"]
        if any(f.endswith("_env.py") for f in files):
            rel = os.path.relpath(root, TASKS_DIR)
            tasks.append(rel.replace(os.sep, "/"))
    return sorted(tasks)

def discover_algos():
    """Finds available algorithms dynamically."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    algo_dir = os.path.join(repo_root, "extensions", "pegasus.simulator", "pegasus", "simulator", "logic", "rl", "algorithms")
    if not os.path.isdir(algo_dir):
        return ["ppo"]

    algos = []
    for f in os.listdir(algo_dir):
        if not f.endswith(".py") or f.startswith("_"):
            continue
        algos.append(f[:-3])

    return sorted(algos) or ["ppo"]


def parse_args():
    """Parses command line arguments for training."""
    tasks = discover_tasks() or ["quadcopter"]
    algos = discover_algos() or ["ppo"]
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task",    required=True, choices=tasks)
    p.add_argument("--algo",    default="ppo", choices=algos)
    p.add_argument("--checkpoint", type=str, default=None, help="Path to agent checkpoint to resume training")
    p.add_argument("--preset",  default="isaac_lab")
    p.add_argument("--n_envs",  type=int,   default=4096)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--device",  default="cuda:0")
    return p.parse_args()

args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

# -----------------------------------
# Post-SimulationApp imports
# -----------------------------------
import numpy as np
import torch
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from pxr import PhysxSchema

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.vehicles.shuttle_glider_batch import ShuttleGliderBatch, ShuttleGliderBatchConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import RLBackend, ResetManager, GoalCfg, InitStateCfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_env(task: str):
    """Loads the target environment and configuration dynamically from *_env.py."""

    task_dir = os.path.join(TASKS_DIR, *task.split("/"))

    # Find *_env.py file
    env_files = [f for f in os.listdir(task_dir) if f.endswith("_env.py")]

    env_file = env_files[0]
    module_name = env_file[:-3]  # remove .py

    # Import dynamically. The task id may contain numeric-prefixed package names
    # (e.g. "double_integrator/01_isaac_lab"), which are not valid as an `import`
    # statement but work through importlib.import_module with the dotted string.
    task_pkg = task.replace("/", ".")
    mod = importlib.import_module(f"tasks.{task_pkg}.{module_name}")

    # Infer class name (CamelCase + Env)
    base_name = module_name.replace("_env", "")
    cls_name = "".join(w.capitalize() for w in base_name.split("_")) + "Env"

    if not hasattr(mod, cls_name):
        raise AttributeError(f"Class '{cls_name}' not found in {module_name}")
    if not hasattr(mod, cls_name + "Cfg"):
        raise AttributeError(f"Config class '{cls_name}Cfg' not found in {module_name}")

    return getattr(mod, cls_name), getattr(mod, cls_name + "Cfg")()


def load_agent_cfg(task: str, algo: str, preset: str) -> dict:
    """Loads the agent configuration for the given preset."""
    task_pkg = task.replace("/", ".")
    PRESETS = getattr(importlib.import_module(f"tasks.{task_pkg}.agents.{algo}_cfg"), "PRESETS")
    if preset not in PRESETS:
        raise KeyError(f"Preset '{preset}' not found.")
    return PRESETS[preset]

def load_train_fn(algo: str):
    """Loads the training function for the specified algorithm."""
    return getattr(importlib.import_module(f"pegasus.simulator.logic.rl.algorithms.{algo}"), "train")

def main():
    EnvClass, env_cfg = load_env(args.task)
    agent_cfg = copy.deepcopy(load_agent_cfg(args.task, args.algo, args.preset))
    train_fn = load_train_fn(args.algo)

    # Apply CLI overrides
    if args.seed is not None:
        agent_cfg["seed"] = args.seed

    device = args.device
    n_envs = args.n_envs

    print("\n" + "="*60)
    print("  RL TRAINING (skrl)")
    print("="*60)
    print(f"  Task:      {args.task}")
    print(f"  Algorithm: {args.algo}")
    print(f"  Preset:    {args.preset}")
    print(f"  Seed:      {agent_cfg.get('seed')}")
    print(f"  Timesteps: {agent_cfg['timesteps']}")
    print(f"  Num envs:  {n_envs}")
    print(f"  Headless:  {args.headless}")
    print("="*60 + "\n")

    # Initialize Simulator
    pg = PegasusInterface()
    pg.set_world_settings(
        physics_dt=env_cfg.sim_dt,
        rendering_dt=env_cfg.sim_dt * env_cfg.decimation,
        device=device,
    )
    pg._world = World(**dict(pg._world_settings))
    world = pg.world

    if "cuda" in device:
        # Increase GPU Found Lost Aggregate Pairs Capacity
        stage = stage_utils.get_current_stage()
        api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(513141)

    # Setup Vehicles
    backend = RLBackend(n_vehicles=n_envs, action_mode=env_cfg.action_mode, device=device)
    physics_cfg = getattr(env_cfg, "vehicle_physics_cfg", None) or {}
    
    if env_cfg.vehicle == "Shuttle_glider" or env_cfg.vehicle == "Shuttle_glider_com":
        vehicle_cfg = ShuttleGliderBatchConfig(
            cfg=physics_cfg,
            n_vehicles=n_envs,
        )
        VehicleClass = ShuttleGliderBatch
    else:
        vehicle_cfg = MultirotorBatchConfig(
            cfg=physics_cfg,
            n_vehicles=n_envs,
        )
        VehicleClass = MultirotorBatch
    
    vehicle_cfg.backends = [backend]

    VehicleClass(
        stage_prefix="/World/quadrotor", usd_file=ROBOTS[env_cfg.vehicle],
        vehicle_batch_id=1, n_vehicles=n_envs, spacing=2.5,
        config=vehicle_cfg,
    )

    env = EnvClass(env_cfg, backend, reset_manager=None)
    env._world = world

    # Start Timeline
    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=not args.headless)

    goal_cfg = GoalCfg(goal_pos_xy_range=env_cfg.goal_pos_xy_range, goal_pos_z_range=env_cfg.goal_pos_z_range)
    init_state_cfg = getattr(env_cfg, "init_state_cfg", None) if getattr(env_cfg, "randomize_init_state", False) else None

    env.reset_manager = ResetManager(vehicles=[backend._vehicle], device=device, goal_cfg=goal_cfg, init_state_cfg=init_state_cfg)
    env.setup()

    # Run Training
    log_dir = os.path.join(TASKS_DIR, *args.task.split("/"), "logs")
    train_fn(env=env, agent_cfg=agent_cfg, log_dir=log_dir, device=device, headless=args.headless, checkpoint=args.checkpoint)

    # Cleanup
    timeline.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()