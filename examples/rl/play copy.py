#!/usr/bin/env python
"""
| File: play.py
| Description: Generic RL inference launcher (skrl). Loads config and weights to run the model.
| License: BSD-3-Clause.
"""

import os
import copy
import sys
import argparse
import importlib
import torch
import numpy as np


# Isaac Sim imports (must be before other Omniverse/Pegasus imports)
from isaacsim import SimulationApp

# Parse arguments to initialize the simulation app
TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")

def discover_tasks():
    """Finds available tasks in the tasks directory."""
    if not os.path.isdir(TASKS_DIR):
        return []
    return sorted(d for d in os.listdir(TASKS_DIR) if os.path.isdir(os.path.join(TASKS_DIR, d)) and not d.startswith("_"))

def parse_args():
    """Parses command line arguments for inference."""
    tasks = discover_tasks() or ["quadcopter"]
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task",       required=True, choices=tasks)
    p.add_argument("--algo",       default="ppo")
    p.add_argument("--preset",     default="isaac_lab")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_envs",     type=int, default=4)
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--headless",   default=False, action="store_true")
    p.add_argument("--compare_lqr", default=False, action="store_true")

    p.add_argument("--record", default=False, action="store_true")
    p.add_argument("--record_dir", default="play_records")
    p.add_argument("--record_every", type=int, default=1)
    p.add_argument("--num_episodes_per_env", type=int, default=0)

    return p.parse_args()

args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})


# -----------------------------------
# Post-SimulationApp imports
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
import isaacsim.core.utils.prims as prim_utils

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor_batch import MultirotorBatch, MultirotorBatchConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.rl import RLBackend, ResetManager

from skrl.envs.wrappers.torch import wrap_env
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.sac import SAC

import isaacsim.core.utils.stage as stage_utils
from pxr import PhysxSchema


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_env(task):
    """Loads the target environment and configuration dynamically from *_env.py."""

    task_dir = os.path.join(TASKS_DIR, task)

    # Find *_env.py file
    env_files = [f for f in os.listdir(task_dir) if f.endswith("_env.py")]

    env_file = env_files[0]
    module_name = env_file[:-3]  # remove .py

    # Import dynamically
    mod = importlib.import_module(f"tasks.{task}.{module_name}")

    # Infer class name (CamelCase + Env)
    base_name = module_name.replace("_env", "")
    cls_name = "".join(w.capitalize() for w in base_name.split("_")) + "Env"

    if not hasattr(mod, cls_name):
        raise AttributeError(f"Class '{cls_name}' not found in {module_name}")
    if not hasattr(mod, cls_name + "Cfg"):
        raise AttributeError(f"Config class '{cls_name}Cfg' not found in {module_name}")

    return getattr(mod, cls_name), getattr(mod, cls_name + "Cfg")()


def load_agent_cfg(task, algo, preset):
    """Loads the agent configuration preset."""
    mod = importlib.import_module(f"tasks.{task}.agents.{algo}_cfg")
    return getattr(mod, "PRESETS")[preset]

def load_algo_class(algo):
    """Returns the skrl Agent class for the selected algorithm."""
    _map = {
        "ppo": ("skrl.agents.torch.ppo", "PPO"),
        "sac": ("skrl.agents.torch.sac", "SAC"),
        "td3": ("skrl.agents.torch.td3", "TD3"),
        "ddpg": ("skrl.agents.torch.ddpg", "DDPG")
    }
    if algo not in _map:
        raise ValueError(f"Unknown algo '{algo}'.")
    mod_name, cls_name = _map[algo]
    return getattr(importlib.import_module(mod_name), cls_name)

def main():
    device = args.device
    n_envs = args.n_envs

    EnvClass, env_cfg = load_env(args.task)
    agent_cfg = load_agent_cfg(args.task, args.algo, args.preset)
    AgentClass = load_algo_class(args.algo)

    # Initialize Simulator and World
    pg = PegasusInterface()
    pg.set_world_settings(
        physics_dt=env_cfg.sim_dt,
        rendering_dt=env_cfg.sim_dt * env_cfg.decimation,
        device=device,
    )
    pg._world = World(**dict(pg._world_settings))
    world = pg.world

    # Load environment and lighting with a curved gridroom. Can be replaced with a flat plane using SIMULATION_ENVIRONMENTS["Flat Plane"].
    pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

    prim_utils.create_prim(
        "/World/Light/DomeLight", "DomeLight",
        position=np.array([1.0, 1.0, 1.0]),
        attributes={"inputs:intensity": 5e3, "inputs:color": (1.0, 1.0, 1.0)},
    )

    if "cuda" in device:
        # Increase GPU Found Lost Aggregate Pairs Capacity
        stage = stage_utils.get_current_stage()
        api = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
        api.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(513141)

    # Setup Vehicles and Backend
    rl_backend = RLBackend(n_vehicles=n_envs, action_mode="direct_force")
    vehicle_cfg = MultirotorBatchConfig(n_vehicles=n_envs)
    vehicle_cfg.backends = [rl_backend]

    rl_vehicle = MultirotorBatch(
        stage_prefix="/World/rl_quadrotor", usd_file=ROBOTS["Iris"],
        vehicle_batch_id=1, n_vehicles=n_envs, spacing=10.0,
        config=vehicle_cfg,
    )

    env = EnvClass(env_cfg, rl_backend, reset_manager=None)
    env._world = world

    lqr_vehicle = None
    lqr_view = None
    has_lqr = args.compare_lqr and hasattr(env, "_compute_discounted_dlqr")
    
    if has_lqr:
        #from examples.rl.utils.lqr_controller_batch import LQRBackend
        from utils.lqr_controller_batch import LQRBackend

        P, K = env._compute_discounted_dlqr()

        lqr_backend = LQRBackend(n_vehicles=n_envs, P=P, K=K, reset_manager=None, action_mode="direct_force")

        lqr_cfg = MultirotorBatchConfig(n_vehicles=n_envs)
        lqr_cfg.backends = [lqr_backend]

        lqr_vehicle = MultirotorBatch(
            stage_prefix="/World/lqr_quadrotor",
            usd_file=ROBOTS["Iris_White"],
            init_pos=rl_vehicle._init_pos, init_orientation=rl_vehicle._init_orientation,
            vehicle_batch_id=2,
            n_vehicles=n_envs,
            config=lqr_cfg,
        )

    # Start simulation timeline
    world.reset()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    world.step(render=False)

    print("RL has _root_prims:", hasattr(rl_vehicle, "_root_prims"))

    if has_lqr:
        print("LQR has _root_prims:", hasattr(lqr_vehicle, "_root_prims"))

        if not hasattr(lqr_vehicle, "_root_prims"):
            raise RuntimeError("LQR vehicle did not initialize. Do not add it to ResetManager.")

    # Create reset manager
    if has_lqr:
        reset_manager = ResetManager(vehicles=[rl_vehicle, lqr_vehicle], device=device)
    else:
        reset_manager = ResetManager(vehicles=[rl_vehicle], device=device)
 
    env.reset_manager = reset_manager

    env.setup()
    env._render_enabled = not args.headless

    if has_lqr:
        lqr_backend.setup(reset_manager)

    # Wrap environment for skrl
    from pegasus.simulator.logic.rl.skrl_pegasus_wrapper import PegasusSkrlWrapper
    wrapped = PegasusSkrlWrapper(env)
    
    cfg = _prepare_cfg(agent_cfg["cfg"], device)
    models = agent_cfg["models"](wrapped.observation_space, wrapped.action_space, device)

    # Update preprocessor sizes based on the wrapped environment's observation space
    cfg["state_preprocessor_kwargs"]["size"] = wrapped.observation_space
    
    # Initialize Agent
    agent = AgentClass(
        models=models, memory=None, cfg=cfg,
        observation_space=wrapped.observation_space,
        action_space=wrapped.action_space, device=device,
    )

    print(f"\nLoading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)
    agent.set_running_mode("eval")
    print("Loaded. Running inference...\n")

    # Evaluation loop
    obs, _ = wrapped.reset()

    while simulation_app.is_running():
        with torch.no_grad():
            actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
        obs, _, terminated, truncated, _ = wrapped.step(actions)
            
    # Cleanup
    timeline.stop()
    simulation_app.close()

# Internal Utility
def _prepare_cfg(cfg: dict, device: str) -> dict:
    """Deep-copies the configuration and overrides the device for preprocessors."""
    cfg = copy.deepcopy(cfg)
    for key in ("state_preprocessor_kwargs", "value_preprocessor_kwargs"):
        if isinstance(cfg.get(key), dict):
            cfg[key]["device"] = device
    return cfg


if __name__ == "__main__":
    main()